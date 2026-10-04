"""
tests/test_log_extractor.py

Unit tests for mavpose/log_extractor.py.

These tests inject a fake mavutil.mavlink_connection that yields
synthetic messages, to exercise edge cases precisely.  End-to-end tests
against real .tlog / .bin files live in test_real_logs.py.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mavpose.log_extractor import LogExtractor, _to_float, resolve_unit


# ---------------------------------------------------------------------------
# Helpers — synthetic MAVLink message factory
# ---------------------------------------------------------------------------

def _make_msg(
    msg_type: str,
    fields: dict,
    ts_usec: int | None = 0,
    units: dict | None = None,
) -> MagicMock:
    """
    Return a MagicMock that quacks like a pymavlink MAVLink_message.

    ``ts_usec`` becomes pymavlink's ``_timestamp`` (seconds); pass None for
    a message without a timestamp.  ``units`` mimics ``fieldunits_by_name``.
    """
    msg = MagicMock()
    msg.get_type.return_value = msg_type
    msg.get_fieldnames.return_value = list(fields.keys())
    for k, v in fields.items():
        setattr(msg, k, v)
    msg._timestamp = None if ts_usec is None else ts_usec / 1_000_000
    msg.fieldunits_by_name = dict(units or {})
    return msg


def _fake_connection(messages: list):
    """Return a context-managed fake mavlink_connection."""
    conn = MagicMock()
    # recv_match returns messages one by one, then None
    conn.recv_match.side_effect = messages + [None]
    return conn


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_int(self):       assert _to_float(42) == 42.0
    def test_float(self):     assert _to_float(3.14) == pytest.approx(3.14)
    def test_string_num(self): assert _to_float("1.5") == pytest.approx(1.5)
    def test_none(self):       assert _to_float(None) is None
    def test_string(self):     assert _to_float("abc") is None


class TestResolveUnit:
    def test_scaled_mavlink_units(self):
        assert resolve_unit("mm") == (pytest.approx(1e-3), "m")
        assert resolve_unit("degE7") == (pytest.approx(1e-7), "deg")
        assert resolve_unit("cdeg") == (pytest.approx(1e-2), "deg")
        assert resolve_unit("mV") == (pytest.approx(1e-3), "V")
        assert resolve_unit("cA") == (pytest.approx(1e-2), "A")
        assert resolve_unit("cm/s") == (pytest.approx(1e-2), "m/s")
        assert resolve_unit("d%") == (pytest.approx(0.1), "%")

    def test_natural_units_unchanged(self):
        assert resolve_unit("m") == (1.0, "m")
        assert resolve_unit("rad/s") == (1.0, "rad/s")

    def test_time_units_never_rescaled(self):
        assert resolve_unit("ms") == (1.0, "ms")
        assert resolve_unit("us") == (1.0, "us")

    def test_dataflash_factor_form(self):
        assert resolve_unit("1e-07 deg") == (pytest.approx(1e-7), "deg")
        assert resolve_unit("0.01 m") == (pytest.approx(0.01), "m")

    def test_empty(self):
        assert resolve_unit("") == (1.0, None)
        assert resolve_unit(None) == (1.0, None)


# ---------------------------------------------------------------------------
# LogExtractor
# ---------------------------------------------------------------------------

class TestLogExtractorInit:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LogExtractor("/nonexistent/flight.tlog")

    def test_accepts_existing_file(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")
        ex = LogExtractor(str(f))
        assert ex.log_path == str(f)


class TestSchemaOnly:
    def test_returns_schema_dict(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("HEARTBEAT", {"type": 6, "autopilot": 3}),
            _make_msg("HEARTBEAT", {"type": 6, "autopilot": 3}),
            _make_msg("GPS_RAW_INT", {"lat": 473_977_000, "lon": 85_450_000}),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            schema = LogExtractor(str(f)).schema_only()

        assert "HEARTBEAT" in schema
        assert schema["HEARTBEAT"]["count"] == 2
        assert "type" in schema["HEARTBEAT"]["fields"]
        assert "GPS_RAW_INT" in schema
        assert schema["GPS_RAW_INT"]["count"] == 1

    def test_skips_bad_data(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        bad = MagicMock()
        bad.get_type.return_value = "BAD_DATA"

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection([bad])):
            schema = LogExtractor(str(f)).schema_only()

        assert "BAD_DATA" not in schema


class TestExtractAll:
    def test_produces_dataframe_per_type(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("ATTITUDE", {"roll": 0.1, "pitch": 0.05}, ts_usec=1_000_000),
            _make_msg("ATTITUDE", {"roll": 0.2, "pitch": 0.10}, ts_usec=2_000_000),
            _make_msg("SYS_STATUS", {"voltage_battery": 12100}, ts_usec=1_500_000),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            extractor = LogExtractor(str(f))
            frames = extractor.extract_all()

        assert "ATTITUDE" in frames
        assert "SYS_STATUS" in frames
        assert len(frames["ATTITUDE"]) == 2
        assert "time_s" in frames["ATTITUDE"].columns
        assert "msg_type" in frames["ATTITUDE"].columns

    def test_time_s_starts_near_zero(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("HEARTBEAT", {"type": 6}, ts_usec=1_000_000_000),
            _make_msg("HEARTBEAT", {"type": 6}, ts_usec=1_001_000_000),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            frames = LogExtractor(str(f)).extract_all()

        times = frames["HEARTBEAT"]["time_s"].tolist()
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(1.0)

    def test_all_types_share_one_clock(self, tmp_path):
        """Regression (#7): time_s must not depend on per-message time fields."""
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            # Epoch-based field on one type, boot-based on another, none on a third
            _make_msg("SYSTEM_TIME", {"time_unix_usec": 1_700_000_000_000_000},
                      ts_usec=5_000_000),
            _make_msg("GPS_RAW_INT", {"time_usec": 12_000_000}, ts_usec=5_500_000),
            _make_msg("VFR_HUD", {"alt": 10.0}, ts_usec=6_000_000),
        ]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            frames = LogExtractor(str(f)).extract_all()

        assert frames["SYSTEM_TIME"]["time_s"].iloc[0] == pytest.approx(0.0)
        assert frames["GPS_RAW_INT"]["time_s"].iloc[0] == pytest.approx(0.5)
        assert frames["VFR_HUD"]["time_s"].iloc[0] == pytest.approx(1.0)
        # native time fields are kept as ordinary columns
        assert frames["GPS_RAW_INT"]["time_usec"].iloc[0] == 12_000_000

    def test_messages_without_timestamp_are_dropped(self, tmp_path, caplog):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("NAMED_VALUE_FLOAT", {"value": 1.0}, ts_usec=None),
            _make_msg("NAMED_VALUE_FLOAT", {"value": 2.0}, ts_usec=1_000_000),
        ]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            frames = LogExtractor(str(f)).extract_all()

        assert frames["NAMED_VALUE_FLOAT"]["value"].tolist() == [2.0]
        assert "without a timestamp" in caplog.text

    def test_converts_units_from_metadata(self, tmp_path):
        """Regression (#11): scaled MAVLink integers become natural units."""
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        units = {"time_boot_ms": "ms", "lat": "degE7", "alt": "mm", "hdg": "cdeg"}
        msgs = [_make_msg(
            "GLOBAL_POSITION_INT",
            {"time_boot_ms": 5000, "lat": -353_632_610, "alt": 584_090, "hdg": 9000},
            ts_usec=1_000_000, units=units,
        )]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            ex = LogExtractor(str(f))
            row = ex.extract_all()["GLOBAL_POSITION_INT"].iloc[0]

        assert row["alt"] == pytest.approx(584.09)
        assert row["lat"] == pytest.approx(-35.363261)
        assert row["hdg"] == pytest.approx(90.0)
        assert row["time_boot_ms"] == 5000          # time fields untouched
        assert ex.units["GLOBAL_POSITION_INT"]["alt"] == "m"
        assert ex.units["GLOBAL_POSITION_INT"]["lat"] == "deg"
        assert ex.raw_units["GLOBAL_POSITION_INT"]["alt"] == "mm"

    def test_convert_units_can_be_disabled(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [_make_msg("GLOBAL_POSITION_INT", {"alt": 584_090},
                          ts_usec=1_000_000, units={"alt": "mm"})]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            ex = LogExtractor(str(f), convert_units=False)
            row = ex.extract_all()["GLOBAL_POSITION_INT"].iloc[0]

        assert row["alt"] == 584_090
        assert ex.units["GLOBAL_POSITION_INT"]["alt"] == "mm"


class TestCorruptData:
    """Regression (#8): one bad packet must not truncate the rest of the log."""

    def test_reader_exception_mid_log_does_not_truncate(self, tmp_path, caplog):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("ATTITUDE", {"roll": 0.1}, ts_usec=1_000_000),
            ValueError("corrupt packet"),
            _make_msg("ATTITUDE", {"roll": 0.2}, ts_usec=2_000_000),
            _make_msg("ATTITUDE", {"roll": 0.3}, ts_usec=3_000_000),
        ]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            ex = LogExtractor(str(f))
            frames = ex.extract_all()

        assert frames["ATTITUDE"]["roll"].tolist() == [0.1, 0.2, 0.3]
        assert ex.stats["errors"] == 1
        assert "1 read error" in caplog.text

    def test_schema_only_also_survives_errors(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [
            _make_msg("HEARTBEAT", {"type": 6}),
            RuntimeError("bad"),
            _make_msg("VFR_HUD", {"alt": 1.0}),
        ]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            schema = LogExtractor(str(f)).schema_only()

        assert set(schema) == {"HEARTBEAT", "VFR_HUD"}

    def test_bad_data_counted(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        bad = MagicMock()
        bad.get_type.return_value = "BAD_DATA"
        msgs = [bad, bad, _make_msg("HEARTBEAT", {"type": 6}, ts_usec=1)]
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            ex = LogExtractor(str(f))
            ex.extract_all()

        assert ex.stats == {"messages": 1, "bad_data": 2, "errors": 0}

    def test_gives_up_after_many_consecutive_errors(self, tmp_path, caplog, monkeypatch):
        import mavpose.log_extractor as le
        monkeypatch.setattr(le, "MAX_CONSECUTIVE_ERRORS", 5)
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        conn = MagicMock()
        conn.recv_match.side_effect = ValueError("stuck")   # never advances
        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=conn):
            ex = LogExtractor(str(f))
            frames = ex.extract_all()

        assert frames == {}
        assert ex.stats["errors"] == 5
        assert "consecutive read errors" in caplog.text


class TestExportParquet:
    def test_parquet_file_created(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")
        out = str(tmp_path / "out.parquet")

        msgs = [
            _make_msg("ATTITUDE", {"roll": 0.1, "pitch": 0.05}, ts_usec=1_000_000),
            _make_msg("ATTITUDE", {"roll": 0.2, "pitch": 0.10}, ts_usec=2_000_000),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            extractor = LogExtractor(str(f))
            extractor.extract_all()
            extractor.export_parquet(["ATTITUDE"], out)

        assert os.path.exists(out)
        df = pd.read_parquet(out)
        assert "time_s" in df.columns
        assert "msg_type" in df.columns
        assert set(df["msg_type"].unique()) == {"ATTITUDE"}

    def test_summary_contains_dtype_and_range(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")
        out = str(tmp_path / "out.parquet")

        msgs = [
            _make_msg("ATTITUDE", {"roll": 0.1, "pitch": -0.05}, ts_usec=1_000_000),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            extractor = LogExtractor(str(f))
            extractor.extract_all()
            summary = extractor.export_parquet(["ATTITUDE"], out)

        assert "ATTITUDE" in summary
        assert "rows" in summary["ATTITUDE"]
        cols = summary["ATTITUDE"]["columns"]
        assert "roll" in cols
        assert "dtype" in cols["roll"]
        assert "min" in cols["roll"]
        assert "max" in cols["roll"]
        assert cols["time_s"]["unit"] == "s"

    def test_raises_when_extract_not_called(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")
        extractor = LogExtractor(str(f))
        with pytest.raises(RuntimeError, match="extract_all"):
            extractor.export_parquet(["ATTITUDE"], str(tmp_path / "out.parquet"))

    def test_raises_when_no_types_found(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        msgs = [_make_msg("HEARTBEAT", {"type": 6}, ts_usec=1_000_000)]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            extractor = LogExtractor(str(f))
            extractor.extract_all()
            with pytest.raises(ValueError, match="None of the requested"):
                extractor.export_parquet(["NONEXISTENT_TYPE"], str(tmp_path / "out.parquet"))

    def test_skips_unknown_types_gracefully(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")
        out = str(tmp_path / "out.parquet")

        msgs = [_make_msg("HEARTBEAT", {"type": 6}, ts_usec=1_000_000)]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            extractor = LogExtractor(str(f))
            extractor.extract_all()
            # "GHOST_TYPE" is unknown but "HEARTBEAT" is real
            summary = extractor.export_parquet(["HEARTBEAT", "GHOST_TYPE"], out)

        assert "HEARTBEAT" in summary
        assert "GHOST_TYPE" not in summary
