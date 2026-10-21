"""
tests/test_mcp_server.py

The MAVPose MCP server (roadmap R5), exercised through a real MCP client:
in-process for most tests, and over stdio as a separate `mavpose-mcp`
process to prove the real transport works.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="needs the [mcp] extra")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from mavpose import mcp_server  # noqa: E402
from mavpose.mcp_server import build_server  # noqa: E402
from tests.log_builders import (  # noqa: E402
    DURATION_S,
    RECEIVE_EPOCH_S,
    UNKNOWN_SAMPLES,
    build_dataflash_bin,
    build_tlog,
    build_ulog,
)

EXPECTED_TOOLS = {"list_logs", "summarize_flight", "list_messages", "describe_message",
                  "get_series", "get_log_messages", "export_parquet"}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "px4").mkdir()
    build_tlog(tmp_path / "flight.tlog")
    build_dataflash_bin(tmp_path / "flight.bin")
    build_ulog(tmp_path / "px4" / "flight.ulg")
    (tmp_path / "notes.txt").write_text("not a log")
    return tmp_path


def _call(server, name, **arguments):
    """Call one tool; return (is_error, parsed JSON or error text)."""
    async def go():
        async with Client(server) as client:
            result = await client.call_tool(name, arguments)
            text = result.content[0].text
            return result.is_error, (text if result.is_error else json.loads(text))
    return anyio.run(go)


def _ok(server, name, **arguments):
    is_error, value = _call(server, name, **arguments)
    assert not is_error, value
    return value


def _err(server, name, **arguments):
    is_error, value = _call(server, name, **arguments)
    assert is_error, f"expected an error, got {value}"
    return value


def _tlog_with_statustext(path: Path) -> Path:
    """A small .tlog with STATUSTEXT messages at known times and severities."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink

    class Sink:
        def write(self, b):
            pass

    mav = mavlink.MAVLink(Sink(), srcSystem=1, srcComponent=1)
    out = bytearray()
    for t, severity, text in [(0.0, 6, b"Arming motors"), (3.0, 4, b"Low battery"),
                              (5.0, 2, b"Crash: Disarming")]:
        stamp = int((RECEIVE_EPOCH_S + t) * 1e6)
        out += struct.pack(">Q", stamp) + mavlink.MAVLink_statustext_message(severity, text).pack(mav)
    path.write_bytes(bytes(out))
    return path


def _tlog_with_spike(path: Path, n: int = 51, spike_index: int = 7) -> Path:
    """GLOBAL_POSITION_INT at 5 Hz, flat at 2 m except one 50 m spike."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink

    class Sink:
        def write(self, b):
            pass

    mav = mavlink.MAVLink(Sink(), srcSystem=1, srcComponent=1)
    out = bytearray()
    for i in range(n):
        rel_mm = 50_000 if i == spike_index else 2_000
        msg = mavlink.MAVLink_global_position_int_message(i * 200, 0, 0, rel_mm, rel_mm, 0, 0, 0, 0)
        out += struct.pack(">Q", int((RECEIVE_EPOCH_S + i * 0.2) * 1e6)) + msg.pack(mav)
    path.write_bytes(bytes(out))
    return path


class TestToolList:

    def test_tools_and_annotations(self, root):
        async def go():
            async with Client(build_server([root])) as client:
                return (await client.list_tools()).tools
        tools = {t.name: t for t in anyio.run(go)}
        assert set(tools) == EXPECTED_TOOLS
        for name, tool in tools.items():
            assert tool.description and len(tool.description) > 30, name
            assert tool.annotations.read_only_hint is (name != "export_parquet"), name
            assert tool.annotations.destructive_hint is False
        assert "fields" in tools["get_series"].input_schema["required"]


class TestDiscovery:

    def test_list_logs_finds_every_format(self, root):
        out = _ok(build_server([root]), "list_logs")
        found = {log["path"]: log["format"] for log in out["logs"]}
        assert found == {"flight.tlog": "mavlink", "flight.bin": "dataflash",
                         os.path.join("px4", "flight.ulg"): "ulog"}

    def test_list_logs_in_subfolder(self, root):
        out = _ok(build_server([root]), "list_logs", folder="px4")
        assert out["count"] == 1

    def test_list_messages(self, root):
        out = _ok(build_server([root]), "list_messages", path="flight.tlog")
        names = [m["name"] for m in out["messages"]]
        assert names == sorted(names) and "GLOBAL_POSITION_INT" in names
        gpi = next(m for m in out["messages"] if m["name"] == "GLOBAL_POSITION_INT")
        assert gpi["start_s"] == pytest.approx(0.0) and gpi["end_s"] == pytest.approx(DURATION_S)
        assert out["format"] == "mavlink"

    def test_describe_message_units_and_unknowns(self, root):
        out = _ok(build_server([root]), "describe_message", path="flight.tlog", msg_type="SYS_STATUS")
        assert out["fields"]["voltage_battery"]["unit"] == "V"
        assert out["fields"]["current_battery"]["unknown"] == UNKNOWN_SAMPLES
        assert "time_s" not in out["fields"]


class TestSummarizeFlight:

    def test_tlog_highlights(self, root):
        out = _ok(build_server([root]), "summarize_flight", path="flight.tlog")
        h = out["highlights"]
        assert out["duration_s"] == pytest.approx(DURATION_S)
        assert h["altitude_relative"] == {"min": 0.0, "max": 10.0, "unit": "m",
                                          "source": "GLOBAL_POSITION_INT.relative_alt"}
        assert h["battery_current"]["min"] == pytest.approx(10.5)   # -1 markers excluded
        assert out["unknown_values"]["SYS_STATUS"]["current_battery"] == UNKNOWN_SAMPLES

    def test_ulog_highlights_use_px4_topics(self, root):
        out = _ok(build_server([root]), "summarize_flight", path="px4/flight.ulg")
        assert out["format"] == "ulog"
        assert out["highlights"]["battery_voltage"]["source"] == "battery_status.voltage_v"
        assert out["highlights"]["altitude_msl"]["unit"] == "m"
        assert out["text_messages"]["total"] == 2
        assert out["text_messages"]["warnings_or_worse"] == 1     # "Low battery"

    def test_dataflash_highlights(self, root):
        out = _ok(build_server([root]), "summarize_flight", path="flight.bin")
        assert out["highlights"]["battery_voltage"] == {
            "min": 12.6, "max": 12.6, "unit": "V", "source": "BAT.Volt"}


class TestGetSeries:

    def test_values_units_and_nulls(self, root):
        out = _ok(build_server([root]), "get_series", path="flight.tlog",
                  msg_type="SYS_STATUS", fields=["voltage_battery", "current_battery"])
        assert out["fields"]["voltage_battery"]["unit"] == "V"
        assert out["values"]["current_battery"][:UNKNOWN_SAMPLES] == [None] * UNKNOWN_SAMPLES
        assert out["fields"]["current_battery"]["null_count"] == UNKNOWN_SAMPLES
        assert out["downsampled"] is False
        assert len(out["time_s"]) == len(out["values"]["voltage_battery"]) == out["returned"]

    def test_time_window(self, root):
        out = _ok(build_server([root]), "get_series", path="flight.tlog",
                  msg_type="GLOBAL_POSITION_INT", fields=["relative_alt"], start_s=2, end_s=4)
        assert out["time_s"][0] >= 2 and out["time_s"][-1] <= 4
        assert out["rows_in_window"] == 11                         # 5 Hz, inclusive

    def test_downsampling_keeps_true_extremes(self, root):
        out = _ok(build_server([root]), "get_series", path="flight.tlog",
                  msg_type="GLOBAL_POSITION_INT", fields=["relative_alt"], max_points=7)
        assert out["downsampled"] is True and out["returned"] <= 7
        assert out["fields"]["relative_alt"]["max"] == pytest.approx(10.0)
        assert out["time_s"][0] == pytest.approx(0.0)              # endpoints kept
        assert out["time_s"][-1] == pytest.approx(DURATION_S)

    def test_downsampling_does_not_hide_a_spike(self, root):
        """A brief spike between the returned samples must still show in max."""
        _tlog_with_spike(root / "spike.tlog")
        out = _ok(build_server([root]), "get_series", path="spike.tlog",
                  msg_type="GLOBAL_POSITION_INT", fields=["relative_alt"], max_points=5)
        assert out["downsampled"] is True
        assert max(out["values"]["relative_alt"]) == pytest.approx(2.0)   # spike not sampled
        assert out["fields"]["relative_alt"]["max"] == pytest.approx(50.0)  # but reported

    def test_unknown_field_suggestion(self, root):
        msg = _err(build_server([root]), "get_series", path="flight.tlog",
                   msg_type="GLOBAL_POSITION_INT", fields=["altt"])
        assert "Did you mean: alt" in msg


class TestLogMessages:

    def test_mavlink_statustext_with_severity(self, root):
        _tlog_with_statustext(root / "events.tlog")
        out = _ok(build_server([root]), "get_log_messages", path="events.tlog")
        assert [(m["level"], m["text"]) for m in out["messages"]] == [
            ("INFO", "Arming motors"), ("WARNING", "Low battery"), ("CRITICAL", "Crash: Disarming")]
        assert [m["severity"] for m in out["messages"]] == [6, 4, 2]
        assert [m["time_s"] for m in out["messages"]] == pytest.approx([0.0, 3.0, 5.0])

    def test_min_level_filter(self, root):
        _tlog_with_statustext(root / "events.tlog")
        out = _ok(build_server([root]), "get_log_messages", path="events.tlog", min_level="warning")
        assert [m["text"] for m in out["messages"]] == ["Low battery", "Crash: Disarming"]
        assert "must be one of" in _err(build_server([root]), "get_log_messages",
                                        path="events.tlog", min_level="LOUD")

    def test_ulog_messages(self, root):
        out = _ok(build_server([root]), "get_log_messages", path="px4/flight.ulg")
        assert [m["text"] for m in out["messages"]] == [
            "[commander] Takeoff detected", "[commander] Low battery"]

    def test_log_without_text(self, root):
        out = _ok(build_server([root]), "get_log_messages", path="flight.bin")
        assert out["count"] == 0 and "no text messages" in out["note"]


class TestSafety:

    def test_paths_outside_roots_refused(self, root, tmp_path_factory):
        other = tmp_path_factory.mktemp("elsewhere")
        build_tlog(other / "secret.tlog")
        server = build_server([root])
        assert "outside the allowed folders" in _err(server, "list_messages", path=str(other / "secret.tlog"))
        assert "outside the allowed folders" in _err(server, "list_messages", path="../" + other.name + "/secret.tlog")
        assert "outside the allowed folders" in _err(server, "list_logs", folder=str(other))

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
    def test_symlinked_log_refused(self, root):
        os.symlink(root / "flight.tlog", root / "link.tlog")
        assert "Symlinks are not allowed" in _err(build_server([root]), "list_messages", path="link.tlog")

    def test_non_log_and_missing_files(self, root):
        server = build_server([root])
        assert "Invalid file type" in _err(server, "list_messages", path="notes.txt")
        assert "File not found" in _err(server, "list_messages", path="nope.tlog")

    def test_export_writes_inside_root_only(self, root, tmp_path_factory):
        (root / "out").mkdir()
        server = build_server([root])
        out = _ok(server, "export_parquet", path="flight.tlog", output_path="out/t.parquet",
                  msg_types=["SYS_STATUS"])
        assert out["written"] == os.path.join("out", "t.parquet") and out["message_types"] == ["SYS_STATUS"]
        assert (root / "out" / "t.parquet").exists()
        assert "already exists" in _err(server, "export_parquet", path="flight.tlog",
                                        output_path="out/t.parquet")
        _ok(server, "export_parquet", path="flight.tlog", output_path="out/t.parquet", overwrite=True)
        other = tmp_path_factory.mktemp("elsewhere")
        assert "outside the allowed folders" in _err(server, "export_parquet", path="flight.tlog",
                                                     output_path=str(other / "x.parquet"))
        assert "must end in .parquet" in _err(server, "export_parquet", path="flight.tlog",
                                              output_path="out/evil.py")
        assert not (other / "x.parquet").exists() and not (root / "out" / "evil.py").exists()


class TestCaching:

    def test_parsed_once_and_reparsed_when_file_changes(self, root, monkeypatch):
        import mavpose

        calls = []
        real_load = mavpose.load
        monkeypatch.setattr(mavpose, "load", lambda p, **k: calls.append(p) or real_load(p, **k))
        access = mcp_server.LogAccess([root])
        access.load("flight.tlog")
        access.load("flight.tlog")
        assert len(calls) == 1
        build_tlog(root / "flight.tlog", junk_after_index=5)   # same path, new contents
        access.load("flight.tlog")
        assert len(calls) == 2


class TestCommandLine:

    def test_stdio_end_to_end(self, root):
        """Launch `python -m mavpose.mcp_server` as a client would, over stdio."""
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mavpose.mcp_server", "--root", str(root)],
            cwd=str(Path(__file__).resolve().parent.parent),
        )

        async def go():
            async with Client(params) as client:
                tools = {t.name for t in (await client.list_tools()).tools}
                result = await client.call_tool("summarize_flight", {"path": "flight.tlog"})
                return tools, json.loads(result.content[0].text)

        tools, summary = anyio.run(go)
        assert tools == EXPECTED_TOOLS
        assert summary["highlights"]["altitude_relative"]["max"] == pytest.approx(10.0)

    def test_roots_from_environment(self, root, monkeypatch):
        monkeypatch.setenv("MAVPOSE_MCP_ROOTS", str(root))
        captured = {}

        class FakeServer:
            def run(self, transport):
                captured["transport"] = transport

        def fake_build(roots):
            captured["roots"] = roots
            return FakeServer()

        monkeypatch.setattr(mcp_server, "build_server", fake_build)
        mcp_server.main([])
        assert captured == {"roots": [str(root)], "transport": "stdio"}

    def test_missing_root_folder_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            mcp_server.main(["--root", str(tmp_path / "missing")])
        assert exit_info.value.code == 2
        assert "Root folder does not exist" in capsys.readouterr().err

    def test_missing_extra_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(mcp_server.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(SystemExit) as exit_info:
            mcp_server.main([])
        assert exit_info.value.code == 1
        assert "pip install 'mavpose[mcp]'" in capsys.readouterr().err
