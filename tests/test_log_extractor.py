"""
tests/test_log_extractor.py

Unit tests for mavpose/log_extractor.py.

Because we cannot ship a real MAVLink binary in the test suite, all
tests that exercise extraction logic use monkeypatching to inject a
fake mavutil.mavlink_connection that yields synthetic MAVLink messages.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mavpose.log_extractor import LogExtractor, _to_float, _timestamp_us


# ---------------------------------------------------------------------------
# Helpers — synthetic MAVLink message factory
# ---------------------------------------------------------------------------

def _make_msg(msg_type: str, fields: dict, ts_usec: int | None = None) -> MagicMock:
    """Return a MagicMock that quacks like a pymavlink MAVLink_message."""
    msg = MagicMock()
    msg.get_type.return_value = msg_type
    msg.get_fieldnames.return_value = list(fields.keys())
    for k, v in fields.items():
        setattr(msg, k, v)
    if ts_usec is not None:
        msg.time_usec = ts_usec
    else:
        # Ensure _timestamp_us returns None for this message
        msg.time_usec = 0
        msg.time_unix_usec = 0
        msg.time_us = 0
        msg.usec = 0
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


class TestTimestampUs:
    def test_time_usec(self):
        msg = MagicMock()
        msg.time_usec = 1_000_000
        msg.time_unix_usec = 0
        msg.time_us = 0
        msg.usec = 0
        assert _timestamp_us(msg) == 1_000_000

    def test_fallback_to_time_us(self):
        msg = MagicMock()
        msg.time_usec = 0
        msg.time_unix_usec = 0
        msg.time_us = 500_000
        msg.usec = 0
        assert _timestamp_us(msg) == 500_000

    def test_no_timestamp(self):
        msg = MagicMock(spec=[])  # no timestamp attrs
        assert _timestamp_us(msg) is None


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

    def test_fallback_time_when_no_timestamp(self, tmp_path):
        f = tmp_path / "flight.tlog"
        f.write_bytes(b"\x00")

        # Messages with no timestamp fields
        msgs = [
            _make_msg("NAMED_VALUE_FLOAT", {"value": 1.0}),
            _make_msg("NAMED_VALUE_FLOAT", {"value": 2.0}),
        ]

        with patch("mavpose.log_extractor.mavutil.mavlink_connection",
                   return_value=_fake_connection(msgs)):
            frames = LogExtractor(str(f)).extract_all()

        # time_s should be non-negative and increasing
        times = sorted(frames["NAMED_VALUE_FLOAT"]["time_s"].tolist())
        assert all(t >= 0 for t in times)


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
