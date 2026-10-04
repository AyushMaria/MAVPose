"""
tests/test_real_logs.py

End-to-end tests that run LogExtractor on genuine .tlog and DataFlash .bin
files (built by tests/log_builders.py with pymavlink's own encoder), rather
than on mocked messages.  These cover the P1 data-layer bugs:

  #6  .bin time axis was message count, not TimeUS
  #7  .tlog mixed epoch / boot / sequence clocks in one time_s column
  #8  one bad packet truncated the rest of the log
  #11 raw scaled units (mm, degE7, mV, cA …) reached the LLM
  #27 MAVLink "unknown" markers (-1, UINT16_MAX …) passed through as data
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from tests.log_builders import (
    BOOT_OFFSET_S,
    DURATION_S,
    UNKNOWN_SAMPLES,
    build_dataflash_bin,
    build_tlog,
)
from mavpose.log_extractor import LogExtractor


@pytest.fixture
def tlog(tmp_path):
    return build_tlog(tmp_path / "flight.tlog")


@pytest.fixture
def dataflash(tmp_path):
    return build_dataflash_bin(tmp_path / "flight.bin")


def _extract(path, **kwargs):
    ex = LogExtractor(str(path), **kwargs)
    return ex, ex.extract_all()


# ---------------------------------------------------------------------------
# Time axis (#6, #7)
# ---------------------------------------------------------------------------

class TestTimeAxis:

    def test_tlog_every_type_spans_the_real_flight(self, tlog):
        _, frames = _extract(tlog)
        for mt, df in frames.items():
            assert df["time_s"].min() == pytest.approx(0.0, abs=1e-3), mt
            assert df["time_s"].max() == pytest.approx(DURATION_S, abs=1e-3), mt

    def test_tlog_epoch_and_boot_clocks_do_not_leak_into_time_s(self, tlog):
        """SYSTEM_TIME (epoch) and GPS_RAW_INT (boot µs) share one time axis."""
        _, frames = _extract(tlog)
        st = frames["SYSTEM_TIME"]
        gps = frames["GPS_RAW_INT"]
        for df in (st, gps):
            assert df["time_s"].min() >= -1e-3              # not ~-1.7e9
            assert df["time_s"].max() <= DURATION_S + 1e-3  # not ~1.7e9 or ~52
        # time_s agrees with the vehicle's own Unix clock
        unix_span = (st["time_unix_usec"].max() - st["time_unix_usec"].min()) / 1e6
        assert st["time_s"].max() - st["time_s"].min() == pytest.approx(unix_span, abs=1e-3)

    def test_tlog_native_time_fields_are_kept(self, tlog):
        _, frames = _extract(tlog)
        gpi = frames["GLOBAL_POSITION_INT"]
        assert gpi["time_boot_ms"].iloc[0] == pytest.approx(BOOT_OFFSET_S * 1000)

    def test_bin_time_follows_timeus(self, dataflash):
        _, frames = _extract(dataflash)
        att = frames["ATT"]
        timeus_span = (att["TimeUS"].max() - att["TimeUS"].min()) / 1e6
        assert att["time_s"].max() - att["time_s"].min() == pytest.approx(timeus_span, abs=1e-6)
        assert timeus_span == pytest.approx(DURATION_S, abs=1e-3)

    def test_bin_types_are_aligned_with_each_other(self, dataflash):
        _, frames = _extract(dataflash)
        att, bat = frames["ATT"], frames["BAT"]
        d_timeus = (bat["TimeUS"].iloc[0] - att["TimeUS"].iloc[0]) / 1e6
        d_time_s = bat["time_s"].iloc[0] - att["time_s"].iloc[0]
        # float64 epoch timestamps leave ~0.03 µs of rounding; 1 µs is ample
        assert d_time_s == pytest.approx(d_timeus, abs=1e-6)


# ---------------------------------------------------------------------------
# Units (#11)
# ---------------------------------------------------------------------------

class TestUnits:

    def test_tlog_scaled_fields_are_converted(self, tlog):
        ex, frames = _extract(tlog)
        gpi = frames["GLOBAL_POSITION_INT"].iloc[-1]
        assert gpi["alt"] == pytest.approx(594.0)           # mm -> m
        assert gpi["relative_alt"] == pytest.approx(10.0)   # mm -> m
        assert gpi["lat"] == pytest.approx(-35.3632621)     # degE7 -> deg
        assert gpi["vx"] == pytest.approx(1.5)              # cm/s -> m/s
        assert gpi["hdg"] == pytest.approx(90.0)            # cdeg -> deg
        sys_status = frames["SYS_STATUS"].iloc[-1]
        assert sys_status["voltage_battery"] == pytest.approx(12.5)   # mV -> V
        assert sys_status["current_battery"] == pytest.approx(10.5)   # cA -> A

        u = ex.units["GLOBAL_POSITION_INT"]
        assert (u["alt"], u["lat"], u["vx"], u["hdg"]) == ("m", "deg", "m/s", "deg")
        assert u["time_boot_ms"] == "ms"
        assert ex.units["SYS_STATUS"]["voltage_battery"] == "V"

    def test_tlog_raw_values_when_conversion_disabled(self, tlog):
        ex, frames = _extract(tlog, convert_units=False)
        assert frames["GLOBAL_POSITION_INT"]["alt"].iloc[-1] == 594_000
        assert ex.units["GLOBAL_POSITION_INT"]["alt"] == "mm"

    def test_bin_units_come_from_fmtu_records(self, dataflash):
        ex, frames = _extract(dataflash)
        bat = frames["BAT"].iloc[0]
        assert bat["Volt"] == pytest.approx(12.6)     # raw mV via MULT -> V
        assert bat["Curr"] == pytest.approx(10.5)     # raw cA via MULT -> A
        assert ex.units["BAT"] == {"TimeUS": "µs", "Volt": "V", "Curr": "A", "time_s": "s"}
        assert ex.units["BARO"]["Press"] == "Pa"

    def test_bin_pre_scaled_fields_not_scaled_twice(self, dataflash):
        ex, frames = _extract(dataflash)
        assert frames["ATT"]["Roll"].iloc[0] == pytest.approx(12.34)
        assert ex.units["ATT"]["Roll"] == "deg"

    def test_bin_metadata_records_excluded(self, dataflash):
        _, frames = _extract(dataflash)
        assert not {"FMT", "FMTU", "UNIT", "MULT"} & set(frames)

    def test_units_reach_the_llm_schema(self, tlog, tmp_path):
        ex, _ = _extract(tlog)
        summary = ex.export_parquet(["GLOBAL_POSITION_INT"], str(tmp_path / "t.parquet"))
        cols = summary["GLOBAL_POSITION_INT"]["columns"]
        assert cols["alt"]["unit"] == "m"
        assert cols["alt"]["max"] == pytest.approx(594.0)
        assert cols["time_s"]["unit"] == "s"
        df = pd.read_parquet(tmp_path / "t.parquet")
        assert df["alt"].max() == pytest.approx(594.0)

    def test_schema_only_reports_units(self, tlog):
        schema = LogExtractor(str(tlog)).schema_only()
        assert schema["GLOBAL_POSITION_INT"]["units"]["alt"] == "m"


# ---------------------------------------------------------------------------
# Corrupt data (#8)
# ---------------------------------------------------------------------------

class TestCorruptLogs:

    def test_junk_mid_tlog_keeps_everything_after_it(self, tmp_path, caplog):
        clean_ex, clean = _extract(build_tlog(tmp_path / "clean.tlog"))
        path = build_tlog(tmp_path / "junk.tlog", junk_after_index=20)
        with caplog.at_level(logging.WARNING):
            ex, frames = _extract(path)

        for mt in clean:
            assert len(frames[mt]) == len(clean[mt]), mt
        assert frames["GLOBAL_POSITION_INT"]["time_s"].max() == pytest.approx(DURATION_S)
        assert ex.stats["bad_data"] >= 1
        assert ex.stats["messages"] == clean_ex.stats["messages"]
        assert "corrupt packet" in caplog.text

    def test_truncated_tlog_keeps_what_was_written(self, tmp_path):
        path = build_tlog(tmp_path / "flight.tlog")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2 + 7])   # cut mid-packet
        _, frames = _extract(path)
        assert len(frames["GLOBAL_POSITION_INT"]) > 10


# ---------------------------------------------------------------------------
# Unknown markers (#27)
# ---------------------------------------------------------------------------

class TestUnknownMarkers:

    def test_unknown_current_becomes_nan_not_negative_amps(self, tlog):
        _, frames = _extract(tlog)
        cur = frames["SYS_STATUS"]["current_battery"]
        assert cur.iloc[:UNKNOWN_SAMPLES].isna().all()          # was -0.01 A
        assert cur.iloc[UNKNOWN_SAMPLES:].tolist() == pytest.approx(
            [10.5] * (len(cur) - UNKNOWN_SAMPLES))
        assert cur.min() == pytest.approx(10.5)                  # no fake dip
        rem = frames["SYS_STATUS"]["battery_remaining"]
        assert rem.iloc[:UNKNOWN_SAMPLES].isna().all()
        assert rem.dropna().min() == 80

    def test_gps_accuracy_markers(self, tlog):
        _, frames = _extract(tlog)
        gps = frames["GPS_RAW_INT"]
        assert gps["epv"].isna().all()                           # never provided
        assert gps["eph"].iloc[:UNKNOWN_SAMPLES].isna().all()
        assert gps["eph"].iloc[UNKNOWN_SAMPLES:].eq(100).all()

    def test_real_values_untouched(self, tlog):
        _, frames = _extract(tlog)
        gps = frames["GPS_RAW_INT"]
        assert gps["cog"].eq(90.0).all()                         # 9000 cdeg is real
        assert frames["SYS_STATUS"]["voltage_battery"].notna().all()
        assert frames["GLOBAL_POSITION_INT"].notna().all().all()

    def test_counts_reported(self, tlog, tmp_path):
        ex, frames = _extract(tlog)
        n = len(frames["GPS_RAW_INT"])
        assert ex.unknown_counts["SYS_STATUS"] == {
            "current_battery": UNKNOWN_SAMPLES, "battery_remaining": UNKNOWN_SAMPLES}
        # yaw is 0 throughout: MAVLink's "this GPS does not provide yaw"
        assert ex.unknown_counts["GPS_RAW_INT"] == {
            "eph": UNKNOWN_SAMPLES, "epv": n, "yaw": n}
        assert ex.stats["unknown_values"] == 3 * UNKNOWN_SAMPLES + 2 * n
        summary = ex.export_parquet(["SYS_STATUS"], str(tmp_path / "s.parquet"))
        assert summary["SYS_STATUS"]["columns"]["current_battery"]["unknown"] == UNKNOWN_SAMPLES
        assert summary["SYS_STATUS"]["columns"]["current_battery"]["min"] == pytest.approx(10.5)
        assert "unknown" not in summary["SYS_STATUS"]["columns"]["voltage_battery"]

    def test_all_unknown_column_is_float_nan(self, tlog, tmp_path):
        ex, frames = _extract(tlog)
        assert frames["GPS_RAW_INT"]["epv"].dtype == "float64"
        cols = ex.summary(["GPS_RAW_INT"])["GPS_RAW_INT"]["columns"]
        assert "min" not in cols["epv"]
        assert cols["epv"]["unknown"] == len(frames["GPS_RAW_INT"])

    def test_filtering_can_be_disabled(self, tlog):
        ex, frames = _extract(tlog, filter_unknown=False)
        assert frames["SYS_STATUS"]["current_battery"].iloc[0] == pytest.approx(-0.01)
        assert frames["GPS_RAW_INT"]["epv"].iloc[0] == 65535   # unitless, unscaled
        assert ex.unknown_counts == {}
        assert ex.stats["unknown_values"] == 0

    def test_dataflash_not_filtered(self, dataflash):
        ex, _ = _extract(dataflash)
        assert ex.unknown_counts == {}
