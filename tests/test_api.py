"""
tests/test_api.py

The stable public API: mavpose.load() -> FlightLog.  Runs on genuine
.tlog, DataFlash .bin and PX4 .ulg files from tests/log_builders.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mavpose
from mavpose import FileValidationError, FlightLog
from tests.log_builders import (
    DURATION_S,
    UNKNOWN_SAMPLES,
    build_dataflash_bin,
    build_tlog,
    build_ulog,
)

BUILDERS = {
    "mavlink": (build_tlog, "flight.tlog", "GLOBAL_POSITION_INT", "alt", "m"),
    "dataflash": (build_dataflash_bin, "flight.bin", "BAT", "Volt", "V"),
    "ulog": (build_ulog, "flight.ulg", "battery_status", "voltage_v", "V"),
}


@pytest.fixture(params=list(BUILDERS))
def case(request, tmp_path):
    builder, name, msg, field, unit = BUILDERS[request.param]
    path = builder(tmp_path / name)
    return {"format": request.param, "path": path, "msg": msg, "field": field, "unit": unit}


@pytest.fixture
def tlog(tmp_path):
    return mavpose.load(str(build_tlog(tmp_path / "flight.tlog")))


class TestPublicSurface:
    """Guards the documented API: changing these is a breaking change."""

    def test_top_level_names(self):
        assert set(mavpose.__all__) == {
            "load", "FlightLog", "LogExtractor", "validate_mavlink_file",
            "FileValidationError", "__version__",
        }
        for name in mavpose.__all__:
            assert hasattr(mavpose, name)

    def test_version_is_semver(self):
        parts = mavpose.__version__.split("+")[0].split(".")
        assert len(parts) >= 3 and all(p.isdigit() for p in parts[:3])

    def test_flightlog_methods(self):
        for name in ("messages", "fields", "frame", "series", "unit", "summary",
                     "to_parquet", "units", "unknown_counts", "stats", "duration_s"):
            assert hasattr(FlightLog, name), name

    def test_star_import_does_not_need_chat_extra(self):
        namespace: dict = {}
        exec("from mavpose import *", namespace)
        assert "load" in namespace and "PlotCreator" not in namespace


class TestLoadEveryFormat:

    def test_load_and_describe(self, case):
        log = mavpose.load(str(case["path"]))
        assert isinstance(log, FlightLog)
        assert log.format == case["format"]
        assert case["msg"] in log.messages()
        assert log.messages() == sorted(log.messages())
        assert case["msg"] in log and len(log) == len(log.messages())
        assert log.duration_s == pytest.approx(DURATION_S, abs=0.6)
        assert repr(log).startswith(f"<FlightLog '{case['path'].name}': {case['format']}")

    def test_series_has_time_index_and_unit(self, case):
        log = mavpose.load(str(case["path"]))
        s = log.series(case["msg"], case["field"])
        assert s.index.name == "time_s"
        assert s.name == case["field"]
        assert s.attrs == {"unit": case["unit"], "msg_type": case["msg"]}
        assert s.index.is_monotonic_increasing
        assert log.unit(case["msg"], case["field"]) == case["unit"]

    def test_to_parquet_round_trip(self, case, tmp_path):
        log = mavpose.load(str(case["path"]))
        out = tmp_path / "out.parquet"
        summary = log.to_parquet(str(out), [case["msg"]])
        df = pd.read_parquet(out)
        assert set(df["msg_type"]) == {case["msg"]}
        assert summary[case["msg"]]["columns"][case["field"]]["unit"] == case["unit"]


class TestFlightLogBehaviour:

    def test_frame_is_a_copy(self, tlog):
        df = tlog.frame("SYS_STATUS")
        df["voltage_battery"] = 0.0
        assert tlog.frame("SYS_STATUS")["voltage_battery"].iloc[0] == pytest.approx(12.6)
        assert tlog["SYS_STATUS"].equals(tlog.frame("SYS_STATUS"))

    def test_fields_excludes_bookkeeping_columns(self, tlog):
        fields = tlog.fields("GLOBAL_POSITION_INT")
        assert "alt" in fields and "time_s" not in fields and "msg_type" not in fields

    def test_to_parquet_defaults_to_everything(self, tlog, tmp_path):
        tlog.to_parquet(str(tmp_path / "all.parquet"))
        df = pd.read_parquet(tmp_path / "all.parquet")
        assert set(df["msg_type"]) == set(tlog.messages())

    def test_unknown_counts_and_stats_exposed(self, tlog):
        assert tlog.unknown_counts["SYS_STATUS"]["current_battery"] == UNKNOWN_SAMPLES
        assert tlog.stats["unknown_values"] > 0
        tlog.stats["messages"] = -1                      # returned copies
        assert tlog.stats["messages"] > 0

    def test_load_options_pass_through(self, tmp_path):
        path = str(build_tlog(tmp_path / "f.tlog"))
        raw = mavpose.load(path, convert_units=False, filter_unknown=False)
        assert raw.series("GLOBAL_POSITION_INT", "alt").attrs["unit"] == "mm"
        assert raw.series("SYS_STATUS", "current_battery").iloc[0] == -1   # raw marker kept


class TestHelpfulErrors:

    def test_unknown_message_suggests_close_match(self, tlog):
        with pytest.raises(KeyError, match="Did you mean: GLOBAL_POSITION_INT"):
            tlog.frame("GLOBAL_POSITION")

    def test_unknown_field_suggests_close_match(self, tlog):
        with pytest.raises(KeyError, match="in GLOBAL_POSITION_INT.*Did you mean: alt"):
            tlog.series("GLOBAL_POSITION_INT", "altt")

    def test_bookkeeping_columns_are_not_fields(self, tlog):
        with pytest.raises(KeyError):
            tlog.series("GLOBAL_POSITION_INT", "time_s")

    def test_to_parquet_rejects_unknown_types(self, tlog, tmp_path):
        with pytest.raises(KeyError, match="NOPE"):
            tlog.to_parquet(str(tmp_path / "x.parquet"), ["NOPE"])

    def test_load_validates_the_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mavpose.load(str(tmp_path / "missing.tlog"))
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with pytest.raises(FileValidationError):
            mavpose.load(str(bad))
