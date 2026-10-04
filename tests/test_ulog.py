"""
tests/test_ulog.py

PX4 ULog (.ulg) support (roadmap R3).  Runs LogExtractor end-to-end on a
genuine ULog file written by tests/log_builders.build_ulog().
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mavpose import LogExtractor, validate_mavlink_file
from mavpose.ulog_reader import px4_field_info, raw_ulog_unit
from tests.log_builders import (
    DURATION_S,
    RATE_HZ,
    ULOG_PRESTART_S,
    UNKNOWN_SAMPLES,
    build_tlog,
    build_ulog,
)


@pytest.fixture
def ulog(tmp_path):
    return build_ulog(tmp_path / "flight.ulg")


def _extract(path, **kwargs):
    ex = LogExtractor(str(path), **kwargs)
    return ex, ex.extract_all()


class TestTimeAxis:

    def test_topics_span_the_flight_from_logging_start(self, ulog):
        _, frames = _extract(ulog)
        for mt in ("battery_status", "vehicle_attitude", "sensor_gps"):
            assert frames[mt]["time_s"].min() == pytest.approx(0.0, abs=1e-4), mt
            assert frames[mt]["time_s"].max() == pytest.approx(DURATION_S, abs=1e-4), mt

    def test_time_follows_ulog_timestamps_exactly(self, ulog):
        _, frames = _extract(ulog)
        b = frames["battery_status"]
        steps = b["time_s"].diff().dropna().tolist()
        assert steps == pytest.approx([1 / RATE_HZ] * len(steps), abs=1e-9)

    def test_sample_published_before_logging_keeps_negative_time(self, ulog):
        _, frames = _extract(ulog)
        assert frames["vehicle_status"]["time_s"].min() == pytest.approx(-ULOG_PRESTART_S)

    def test_never_stamped_rows_are_dropped(self, ulog, caplog):
        with caplog.at_level(logging.WARNING):
            _, frames = _extract(ulog)
        assert "commander_state" not in frames
        assert "Dropped 3 ULog row" in caplog.text

    def test_topics_align_with_each_other(self, ulog):
        _, frames = _extract(ulog)
        att, bat = frames["vehicle_attitude"], frames["battery_status"]
        d_ts = (att["timestamp"].iloc[0] - bat["timestamp"].iloc[0]) / 1e6
        assert att["time_s"].iloc[0] - bat["time_s"].iloc[0] == pytest.approx(d_ts, abs=1e-9)


class TestTopicsAndFields:

    def test_multi_instance_topics_are_separate(self, ulog):
        _, frames = _extract(ulog)
        assert frames["battery_status"]["voltage_v"].iloc[0] == pytest.approx(16.2)
        assert frames["battery_status_1"]["voltage_v"].iloc[0] == pytest.approx(25.2)

    def test_array_fields_are_flattened(self, ulog):
        _, frames = _extract(ulog)
        att = frames["vehicle_attitude"]
        assert [f"q[{i}]" for i in range(4)] == [c for c in att.columns if c.startswith("q[")]
        assert att["q[0]"].iloc[0] == pytest.approx(1.0)

    def test_numeric_columns_are_float64(self, ulog):
        _, frames = _extract(ulog)
        assert frames["vehicle_status"]["nav_state"].dtype == np.float64
        assert frames["battery_status"]["timestamp"].dtype == np.float64

    def test_px4_log_messages_become_a_frame(self, ulog):
        _, frames = _extract(ulog)
        msgs = frames["logged_messages"]
        assert msgs["message"].tolist() == ["[commander] Takeoff detected",
                                            "[commander] Low battery"]
        assert msgs["level"].tolist() == ["INFO", "WARNING"]
        assert msgs["time_s"].tolist() == pytest.approx([2.0, 8.0])

    def test_schema_only_matches_extraction(self, ulog):
        ex = LogExtractor(str(ulog))
        schema = ex.schema_only()
        assert schema["battery_status"]["count"] == int(DURATION_S * RATE_HZ) + 1
        assert schema["battery_status_1"]["fields"]["voltage_v"] == "float32"
        assert schema["vehicle_gps_position"]["units"]["lat"] == "deg"
        assert schema["logged_messages"]["count"] == 2
        frames = ex.extract_all()                     # reuses the parsed file
        assert set(frames) == set(schema) - {"commander_state"}


class TestUnits:

    def test_old_integer_gps_converted(self, ulog):
        ex, frames = _extract(ulog)
        gps = frames["vehicle_gps_position"].iloc[-1]
        assert gps["lat"] == pytest.approx(47.3977419)      # degE7 -> deg
        assert gps["alt"] == pytest.approx(498.0)            # mm -> m
        assert ex.units["vehicle_gps_position"]["lat"] == "deg"
        assert ex.raw_units["vehicle_gps_position"]["lat"] == "degE7"

    def test_new_float_gps_unchanged_and_agrees(self, ulog):
        _, frames = _extract(ulog)
        new, old = frames["sensor_gps"].iloc[-1], frames["vehicle_gps_position"].iloc[-1]
        assert new["latitude_deg"] == pytest.approx(old["lat"], abs=1e-7)
        assert new["altitude_msl_m"] == pytest.approx(old["alt"], abs=1e-3)

    def test_units_from_px4_spec_and_suffixes(self, ulog):
        ex, _ = _extract(ulog)
        b = ex.units["battery_status"]
        assert (b["voltage_v"], b["current_a"], b["voltage_cell_v[0]"]) == ("V", "A", "V")
        assert b["temperature"] == "degC"
        assert b["timestamp"] == "us" and b["time_s"] == "s"
        assert "remaining" not in b                          # a 0..1 fraction
        assert ex.units["vehicle_gps_position"]["vel_m_s"] == "m/s"
        assert ex.units["vehicle_attitude"]["rollspeed"] == "rad/s"
        assert "q[0]" not in ex.units["vehicle_attitude"]

    def test_raw_values_when_conversion_disabled(self, ulog):
        ex, frames = _extract(ulog, convert_units=False)
        assert frames["vehicle_gps_position"]["lat"].iloc[0] == 473977419
        assert ex.units["vehicle_gps_position"]["lat"] == "degE7"

    @pytest.mark.parametrize("topic, field, dtype, unit", [
        ("sensor_combined", "accelerometer_m_s2[2]", np.float32, "m/s^2"),
        ("sensor_combined", "gyro_rad[0]", np.float32, "rad"),
        ("battery_status", "discharged_mah", np.float32, "mAh"),
        ("vehicle_global_position", "lat", np.float64, "deg"),
        ("vehicle_gps_position", "lat", np.int32, "degE7"),
        ("vehicle_gps_position", "lat", np.float64, "deg"),
        ("vehicle_attitude", "q[0]", np.float32, None),
        ("some_topic", "nav_state", np.uint8, None),
    ])
    def test_unit_lookup(self, topic, field, dtype, unit):
        assert raw_ulog_unit(topic, field, np.dtype(dtype)) == unit


class TestUnknownMarkers:

    def test_unknown_current_and_empty_cell_slots(self, ulog):
        ex, frames = _extract(ulog)
        b = frames["battery_status"]
        assert b["current_a"].iloc[:UNKNOWN_SAMPLES].isna().all()     # was -1 A
        assert b["current_a"].min() == pytest.approx(12.5)
        assert b["voltage_cell_v[3]"].isna().all()                    # empty slot: 0 V
        assert b["voltage_cell_v[0]"].notna().all()
        assert ex.unknown_counts["battery_status"] == {
            "current_a": UNKNOWN_SAMPLES, "voltage_cell_v[3]": len(b)}
        assert "battery_status_1" not in ex.unknown_counts
        assert ex.stats["unknown_values"] == UNKNOWN_SAMPLES + len(b)

    def test_filtering_can_be_disabled(self, ulog):
        ex, frames = _extract(ulog, filter_unknown=False)
        assert frames["battery_status"]["current_a"].iloc[0] == pytest.approx(-1.0)
        assert ex.unknown_counts == {}

    def test_spec_table_contents(self):
        info = px4_field_info()
        assert info["battery_status"]["current_a"] == {"unit": "A", "invalid": -1}
        assert info["battery_status"]["voltage_v"]["invalid"] == 0
        assert "invalid" not in info.get("cellular_status", {}).get("link_tx_rate", {})


class TestIntegration:

    def test_export_and_llm_summary(self, ulog, tmp_path):
        ex, _ = _extract(ulog)
        summary = ex.export_parquet(["battery_status", "vehicle_gps_position"],
                                    str(tmp_path / "t.parquet"))
        cols = summary["battery_status"]["columns"]
        assert cols["voltage_v"]["unit"] == "V"
        assert cols["current_a"]["unknown"] == UNKNOWN_SAMPLES
        df = pd.read_parquet(tmp_path / "t.parquet")
        assert set(df["msg_type"]) == {"battery_status", "vehicle_gps_position"}

    def test_validator_accepts_ulg(self, ulog):
        validate_mavlink_file(str(ulog))

    def test_truncated_ulog_keeps_what_was_written(self, tmp_path):
        full = build_ulog(tmp_path / "full.ulg")
        cut = build_ulog(tmp_path / "cut.ulg", truncate_to=full.stat().st_size // 2 + 3)
        _, frames = _extract(cut)
        assert 5 < len(frames["battery_status"]) < int(DURATION_S * RATE_HZ) + 1

    def test_garbage_file_raises_clear_error(self, tmp_path):
        bad = tmp_path / "bad.ulg"
        bad.write_bytes(b"this is not a ulog file at all" * 10)
        with pytest.raises(ValueError, match="Could not read PX4 ULog"):
            LogExtractor(str(bad)).extract_all()

    def test_other_formats_unaffected(self, tmp_path):
        _, frames = _extract(build_tlog(tmp_path / "f.tlog"))
        assert "GLOBAL_POSITION_INT" in frames
