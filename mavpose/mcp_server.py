"""
mavpose/mcp_server.py

An MCP (Model Context Protocol) server that lets AI agents read drone flight
logs through MAVPose: one time axis, real units, "unknown" values removed.

    pip install "mavpose[mcp]"
    mavpose-mcp --root ~/flight-logs

Design rules:

* **Read-only by default.** Every tool only reads logs, except
  ``export_parquet``, which writes one new .parquet file and never overwrites
  unless asked.  No tool runs code.
* **Folder allowlist.** Paths must resolve inside the ``--root`` folders
  (the current directory if none are given).  Symlinked log files are refused.
* **Context-friendly.** ``get_series`` downsamples to ``max_points`` but
  reports min/max/mean over *all* samples in the window, so an agent never
  misses a spike because of downsampling.
* **Helpful errors.** Mistakes come back as plain messages with suggestions
  (e.g. "Did you mean: GLOBAL_POSITION_INT?").
"""


import argparse
import importlib.util
import math
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOG_EXTENSIONS = (".tlog", ".bin", ".log", ".ulg")
MAX_POINTS_LIMIT = 5000
INSTALL_HINT = "pip install 'mavpose[mcp]'"

SERVER_INSTRUCTIONS = """\
MAVPose reads drone flight logs: MAVLink .tlog, ArduPilot DataFlash .bin/.log
and PX4 ULog .ulg.

Guarantees for every log:
- time_s is seconds from log start, on ONE clock shared by all message types,
  so values from different messages at the same time_s happened together.
- Values are in natural units (m, deg, V, A, m/s); every field reports its unit.
- "Value not provided" markers (e.g. battery current -1) are null, not data.

Suggested workflow: list_logs -> summarize_flight -> list_messages ->
describe_message -> get_series (use start_s/end_s to zoom in) and
get_log_messages for warnings and errors reported by the vehicle.
Message names: MAVLink names for .tlog (GLOBAL_POSITION_INT), ArduPilot names
for .bin (BAT, GPS, ATT), PX4 topic names for .ulg (battery_status,
battery_status_1 for a second instance).
"""

# MAVLink MAV_SEVERITY / ULog log levels, most severe first.
SEVERITIES = ["EMERGENCY", "ALERT", "CRITICAL", "ERROR", "WARNING", "NOTICE", "INFO", "DEBUG"]

# Highlights for summarize_flight: metric -> candidate (msg_type, field, scale).
# The first candidate present in the log is used; its source is reported.
_HIGHLIGHTS: Dict[str, List[tuple]] = {
    "altitude_relative": [
        ("GLOBAL_POSITION_INT", "relative_alt", 1.0),
        ("CTUN", "Alt", 1.0),
        ("vehicle_local_position", "z", -1.0),      # NED: z is down
    ],
    "altitude_msl": [
        ("GLOBAL_POSITION_INT", "alt", 1.0),
        ("GPS", "Alt", 1.0),
        ("vehicle_global_position", "alt", 1.0),
        ("sensor_gps", "altitude_msl_m", 1.0),
        ("vehicle_gps_position", "alt", 1.0),
    ],
    "groundspeed": [
        ("VFR_HUD", "groundspeed", 1.0),
        ("GPS", "Spd", 1.0),
        ("sensor_gps", "vel_m_s", 1.0),
        ("vehicle_gps_position", "vel_m_s", 1.0),
    ],
    "battery_voltage": [
        ("SYS_STATUS", "voltage_battery", 1.0),
        ("BAT", "Volt", 1.0),
        ("battery_status", "voltage_v", 1.0),
    ],
    "battery_current": [
        ("SYS_STATUS", "current_battery", 1.0),
        ("BAT", "Curr", 1.0),
        ("battery_status", "current_a", 1.0),
    ],
    "gps_satellites": [
        ("GPS_RAW_INT", "satellites_visible", 1.0),
        ("GPS", "NSats", 1.0),
        ("sensor_gps", "satellites_used", 1.0),
        ("vehicle_gps_position", "satellites_used", 1.0),
    ],
}


# ---------------------------------------------------------------------------
# Helpers (no MCP dependency, so they are easy to test)
# ---------------------------------------------------------------------------

def _num(value) -> Optional[float]:
    """JSON-safe float: NaN/inf become None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):            # numpy scalar
        value = value.item()
        if isinstance(value, (bool, int, str)):
            return value
    return _num(value)


class ToolProblem(Exception):
    """A user-facing error; the server turns it into an MCP tool error."""


class LogAccess:
    """
    Resolves paths inside the allowed roots and caches parsed logs.

    Parsed logs are cached by (path, mtime, size), so repeated tool calls on
    the same log are fast and an edited file is re-read.
    """

    def __init__(self, roots: Sequence[os.PathLike], cache_size: int = 4) -> None:
        resolved = [Path(r).expanduser().resolve() for r in roots] or [Path.cwd().resolve()]
        for root in resolved:
            if not root.is_dir():
                raise ValueError(f"Root folder does not exist: {root}")
        self.roots: List[Path] = resolved
        self._cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def resolve(self, path: str, *, must_exist: bool = True) -> Path:
        raw = Path(os.path.expanduser(path))
        candidate = raw if raw.is_absolute() else self.roots[0] / raw
        if must_exist and candidate.is_symlink():
            raise ToolProblem(f"Symlinks are not allowed: {path}")
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.roots):
            allowed = ", ".join(str(r) for r in self.roots)
            raise ToolProblem(f"{path} is outside the allowed folders ({allowed}).")
        if must_exist and not resolved.exists():
            raise ToolProblem(f"File not found: {path}")
        return resolved

    def load(self, path: str):
        import mavpose

        resolved = self.resolve(path)
        stat = resolved.stat()
        key = (str(resolved), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        try:
            log = mavpose.load(str(resolved))
        except (FileNotFoundError, ValueError) as exc:     # FileValidationError is a ValueError
            raise ToolProblem(str(exc)) from exc
        with self._lock:
            self._cache[key] = log
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return log


def _key_error(exc: KeyError) -> ToolProblem:
    return ToolProblem(exc.args[0] if exc.args else str(exc))


def _rel(access: LogAccess, path: Path) -> str:
    for root in access.roots:
        if root in path.parents:
            return str(path.relative_to(root)) if len(access.roots) == 1 else str(path)
    return str(path)


# ---------------------------------------------------------------------------
# Tool implementations (plain functions; registered on the server below)
# ---------------------------------------------------------------------------

def list_logs(access: LogAccess, folder: Optional[str] = None, limit: int = 200) -> dict:
    bases = [access.resolve(folder)] if folder else access.roots
    found = []
    for base in bases:
        if not base.is_dir():
            raise ToolProblem(f"Not a folder: {folder}")
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in LOG_EXTENSIONS and p.is_file() and not p.is_symlink():
                found.append(p)
                if len(found) >= limit:
                    break
    logs = [{
        "path": _rel(access, p),
        "format": {".tlog": "mavlink", ".bin": "dataflash", ".log": "dataflash",
                   ".ulg": "ulog"}[p.suffix.lower()],
        "size_mb": round(p.stat().st_size / 1e6, 3),
    } for p in found]
    return {"roots": [str(r) for r in access.roots], "count": len(logs),
            "truncated": len(found) >= limit, "logs": logs}


def list_messages(access: LogAccess, path: str) -> dict:
    log = access.load(path)
    messages = []
    for mt, info in log.summary().items():
        time_col = info["columns"].get("time_s", {})
        messages.append({
            "name": mt,
            "rows": int(info["rows"]),
            "fields": len(info["columns"]) - 2,
            "start_s": _num(time_col.get("min")),
            "end_s": _num(time_col.get("max")),
        })
    messages.sort(key=lambda m: m["name"])
    return {"path": path, "format": log.format, "duration_s": _num(log.duration_s),
            "message_types": len(messages), "messages": messages}


def describe_message(access: LogAccess, path: str, msg_type: str) -> dict:
    log = access.load(path)
    try:
        info = log.summary([msg_type])[msg_type]
    except KeyError as exc:
        raise _key_error(exc) from exc
    fields = {name: _jsonable(meta) for name, meta in info["columns"].items()
              if name not in ("time_s", "msg_type")}
    return {"msg_type": msg_type, "rows": info["rows"], "fields": fields}


def get_series(
    access: LogAccess,
    path: str,
    msg_type: str,
    fields: List[str],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    max_points: int = 500,
) -> dict:
    import numpy as np

    if not fields:
        raise ToolProblem("Give at least one field name.")
    if not 1 <= max_points <= MAX_POINTS_LIMIT:
        raise ToolProblem(f"max_points must be between 1 and {MAX_POINTS_LIMIT}.")
    log = access.load(path)
    try:
        series = {f: log.series(msg_type, f) for f in fields}
    except KeyError as exc:
        raise _key_error(exc) from exc

    time = next(iter(series.values())).index.to_numpy(dtype=float)
    mask = np.ones(len(time), dtype=bool)
    if start_s is not None:
        mask &= time >= start_s
    if end_s is not None:
        mask &= time <= end_s
    rows = int(mask.sum())
    pick = np.flatnonzero(mask)
    downsampled = rows > max_points
    if downsampled:
        pick = pick[np.unique(np.linspace(0, rows - 1, max_points).round().astype(int))]

    out_values, out_fields = {}, {}
    for name, s in series.items():
        values = s.to_numpy()
        window = values[mask]
        numeric = np.issubdtype(window.dtype, np.number)
        finite = window[np.isfinite(window)] if numeric else window
        meta: Dict[str, Any] = {"unit": s.attrs.get("unit")}
        if numeric:
            meta.update({
                "min": _num(finite.min()) if len(finite) else None,
                "max": _num(finite.max()) if len(finite) else None,
                "mean": _num(finite.mean()) if len(finite) else None,
                "null_count": int(len(window) - len(finite)),
            })
        out_fields[name] = meta
        out_values[name] = [_jsonable(v) for v in values[pick]]

    return {
        "msg_type": msg_type,
        "window": {"start_s": start_s, "end_s": end_s},
        "rows_in_window": rows,
        "returned": int(len(pick)),
        "downsampled": bool(downsampled),
        "note": ("Evenly spaced samples; fields[*].min/max/mean cover every row in "
                 "the window.") if downsampled else None,
        "fields": out_fields,
        "time_s": [_num(t) for t in time[pick]],
        "values": out_values,
    }


def _text_events(log) -> List[dict]:
    """Vehicle text messages from any format, oldest first."""
    events = []
    if "STATUSTEXT" in log:                                   # MAVLink
        df = log.frame("STATUSTEXT")
        for _, row in df.iterrows():
            sev = _num(row.get("severity"))
            level = SEVERITIES[int(sev)] if sev is not None and 0 <= int(sev) < 8 else None
            events.append({"time_s": _num(row["time_s"]), "level": level,
                           "severity": None if sev is None else int(sev),
                           "text": str(row.get("text", "")).strip()})
    if "MSG" in log:                                          # ArduPilot DataFlash
        df = log.frame("MSG")
        for _, row in df.iterrows():
            events.append({"time_s": _num(row["time_s"]), "level": None,
                           "text": str(row.get("Message", "")).strip()})
    if "logged_messages" in log:                              # PX4 ULog
        df = log.frame("logged_messages")
        for _, row in df.iterrows():
            events.append({"time_s": _num(row["time_s"]), "level": row.get("level"),
                           "text": str(row.get("message", "")).strip()})
    events.sort(key=lambda e: (e["time_s"] is None, e["time_s"]))
    return events


def get_log_messages(
    access: LogAccess,
    path: str,
    min_level: Optional[str] = None,
    limit: int = 200,
) -> dict:
    log = access.load(path)
    events = _text_events(log)
    if min_level is not None:
        level = min_level.upper()
        if level not in SEVERITIES:
            raise ToolProblem(f"min_level must be one of: {', '.join(SEVERITIES)}.")
        cutoff = SEVERITIES.index(level)
        events = [e for e in events if e["level"] in SEVERITIES
                  and SEVERITIES.index(e["level"]) <= cutoff]
    return {"count": len(events), "truncated": len(events) > limit,
            "messages": events[:limit],
            "note": None if events or "STATUSTEXT" in log or "MSG" in log
            or "logged_messages" in log else "This log contains no text messages."}


def summarize_flight(access: LogAccess, path: str) -> dict:
    import numpy as np

    log = access.load(path)
    highlights = {}
    for metric, candidates in _HIGHLIGHTS.items():
        for msg_type, field, scale in candidates:
            if msg_type in log and field in log.fields(msg_type):
                values = log.series(msg_type, field).to_numpy(dtype=float) * scale
                finite = values[np.isfinite(values)]
                if not len(finite):
                    continue
                unit = log.unit(msg_type, field)
                highlights[metric] = {
                    "min": _num(finite.min()), "max": _num(finite.max()), "unit": unit,
                    "source": f"{msg_type}.{field}" + (" (negated: NED z is down)"
                                                       if scale < 0 else ""),
                }
                break

    events = _text_events(log)
    serious = [e for e in events if e["level"] in SEVERITIES[:5]]   # WARNING or worse
    return {
        "path": path,
        "format": log.format,
        "duration_s": _num(log.duration_s),
        "message_types": len(log),
        "stats": _jsonable(log.stats),
        "highlights": highlights,
        "unknown_values": _jsonable(log.unknown_counts),
        "text_messages": {"total": len(events), "warnings_or_worse": len(serious),
                          "first_warnings": serious[:10]},
    }


def export_parquet(
    access: LogAccess,
    path: str,
    output_path: str,
    msg_types: Optional[List[str]] = None,
    overwrite: bool = False,
) -> dict:
    if not output_path.lower().endswith(".parquet"):
        raise ToolProblem("output_path must end in .parquet")
    target = access.resolve(output_path, must_exist=False)
    if target.exists() and not overwrite:
        raise ToolProblem(f"{output_path} already exists; pass overwrite=true to replace it.")
    if not target.parent.is_dir():
        raise ToolProblem(f"Folder does not exist: {target.parent}")
    log = access.load(path)
    try:
        summary = log.to_parquet(str(target), msg_types)
    except KeyError as exc:
        raise _key_error(exc) from exc
    return {"written": _rel(access, target), "size_mb": round(target.stat().st_size / 1e6, 3),
            "message_types": sorted(summary),
            "rows": int(sum(info["rows"] for info in summary.values()))}


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def build_server(roots: Sequence[os.PathLike] = ()):
    """Create the MCP server, serving logs inside *roots* (default: cwd)."""
    from typing import Annotated

    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from pydantic import Field

    import mavpose

    access = LogAccess(roots)
    server = MCPServer(
        "mavpose",
        title="MAVPose flight logs",
        version=mavpose.__version__,
        instructions=SERVER_INSTRUCTIONS,
        log_level="WARNING",
    )
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                idempotent_hint=True, open_world_hint=False)

    def run(fn, *args, **kwargs):
        try:
            return fn(access, *args, **kwargs)
        except ToolProblem as exc:
            raise ToolError(str(exc)) from exc

    Path_ = Annotated[str, Field(description="Log file path, absolute or relative to the first allowed folder")]
    Msg = Annotated[str, Field(description="Message type, e.g. GLOBAL_POSITION_INT, BAT or battery_status")]

    @server.tool(name="list_logs", annotations=read_only)
    def list_logs_tool(
        folder: Annotated[Optional[str], Field(description="Sub-folder to search (default: all allowed folders)")] = None,
        limit: Annotated[int, Field(ge=1, le=1000, description="Maximum files to return")] = 200,
    ) -> dict:
        """Find flight logs (.tlog, .bin, .log, .ulg) in the allowed folders."""
        return run(list_logs, folder, limit)

    @server.tool(name="summarize_flight", annotations=read_only)
    def summarize_flight_tool(path: Path_) -> dict:
        """Overview of a flight: duration, altitude, speed, battery and GPS ranges
        (each with its source field and unit), parse stats, unknown values, and
        warnings/errors the vehicle reported. Start here."""
        return run(summarize_flight, path)

    @server.tool(name="list_messages", annotations=read_only)
    def list_messages_tool(path: Path_) -> dict:
        """List the message types in a log with row counts, field counts and time range."""
        return run(list_messages, path)

    @server.tool(name="describe_message", annotations=read_only)
    def describe_message_tool(path: Path_, msg_type: Msg) -> dict:
        """Fields of one message type: dtype, unit, min, max and count of unknown (null) values."""
        return run(describe_message, path, msg_type)

    @server.tool(name="get_series", annotations=read_only)
    def get_series_tool(
        path: Path_,
        msg_type: Msg,
        fields: Annotated[List[str], Field(min_length=1, description="Field names, e.g. ['alt', 'relative_alt']")],
        start_s: Annotated[Optional[float], Field(description="Window start, seconds from log start")] = None,
        end_s: Annotated[Optional[float], Field(description="Window end, seconds from log start")] = None,
        max_points: Annotated[int, Field(ge=1, le=MAX_POINTS_LIMIT, description="Maximum samples to return")] = 500,
    ) -> dict:
        """Time series of one or more fields of a message type, on the shared time_s
        axis, with units. Long windows are downsampled to max_points; min/max/mean
        always cover every sample in the window. Unknown values are null."""
        return run(get_series, path, msg_type, fields, start_s, end_s, max_points)

    @server.tool(name="get_log_messages", annotations=read_only)
    def get_log_messages_tool(
        path: Path_,
        min_level: Annotated[Optional[str], Field(description="Only this severity or worse: EMERGENCY, ALERT, CRITICAL, ERROR, WARNING, NOTICE, INFO, DEBUG")] = None,
        limit: Annotated[int, Field(ge=1, le=5000, description="Maximum messages to return")] = 200,
    ) -> dict:
        """Text messages the vehicle logged (MAVLink STATUSTEXT, ArduPilot MSG, PX4
        log messages), oldest first, with time_s and severity where known.
        Note: some older ArduPilot firmware used non-standard STATUSTEXT
        severity numbers, so levels in old .tlog files may be unreliable; the
        raw 'severity' number is included."""
        return run(get_log_messages, path, min_level, limit)

    @server.tool(name="export_parquet", annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                             idempotent_hint=True, open_world_hint=False))
    def export_parquet_tool(
        path: Path_,
        output_path: Annotated[str, Field(description="Where to write, inside an allowed folder; must end in .parquet")],
        msg_types: Annotated[Optional[List[str]], Field(description="Message types to export (default: all)")] = None,
        overwrite: Annotated[bool, Field(description="Replace output_path if it exists")] = False,
    ) -> dict:
        """Write message types to one Parquet file (msg_type column distinguishes them).
        The only tool that writes; it never overwrites unless overwrite is true."""
        return run(export_parquet, path, output_path, msg_types, overwrite)

    return server


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mavpose-mcp",
        description="MCP server exposing MAVPose flight-log tools over stdio.",
    )
    parser.add_argument(
        "--root", action="append", default=[], metavar="DIR",
        help="Folder the server may read logs from (repeatable). "
             "Default: $MAVPOSE_MCP_ROOTS (os.pathsep-separated) or the current directory.",
    )
    args = parser.parse_args(argv)

    if importlib.util.find_spec("mcp") is None:
        print("The MAVPose MCP server needs the 'mcp' package. Install it with:\n\n"
              f"    {INSTALL_HINT}\n", file=sys.stderr)
        sys.exit(1)

    roots = args.root or [r for r in os.environ.get("MAVPOSE_MCP_ROOTS", "").split(os.pathsep) if r]
    try:
        server = build_server(roots)
    except ValueError as exc:
        print(f"mavpose-mcp: {exc}", file=sys.stderr)
        sys.exit(2)
    server.run("stdio")


if __name__ == "__main__":
    main()
