"""
mavpose/ulog_reader.py

PX4 ULog (.ulg) support for LogExtractor, built on PX4's own pyulog.

ULog stores each uORB topic as columns (numpy arrays), so frames are built
directly from those arrays rather than message-by-message.

Conventions, matching the MAVLink / DataFlash paths:

* **One clock.**  Every topic's ``timestamp`` is µs since boot on the same
  clock.  ``time_s`` is measured from the ULog logging-start timestamp in the
  file header.  A few topics carry their last value from just *before*
  logging started; those rows keep honest negative ``time_s`` values.  Rows
  whose timestamp is 0 were never stamped and are dropped (and counted).
* **Topic names.**  ``battery_status`` for instance 0, ``battery_status_1``
  for instance 1, and so on.  Array fields keep pyulog's names, e.g. ``q[0]``.
* **Units.**  ULog files carry no unit metadata, so units come from, in
  order: PX4's own message annotations (``# [A] [@invalid -1]``, generated
  into ``data/px4_fields.json`` by tools/gen_px4_field_info.py); a small
  table for older unannotated fields; and PX4's unit-suffix naming
  convention (``voltage_v``, ``accelerometer_m_s2``, ``gyro_rad``).  Older
  logs store GPS position as integers in degE7 and mm; those are converted
  to degrees and metres.
* **Unknown values.**  Fields PX4 annotates with a numeric ``@invalid``
  marker (e.g. ``battery_status.current_a = -1``) become NaN and are counted,
  as for MAVLink.  PX4 already uses NaN for most unknown floats.
* **Log messages.**  PX4's text log (``INFO [commander] Takeoff detected``)
  becomes a ``logged_messages`` frame with ``level`` and ``message`` columns.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOGGED_MESSAGES = "logged_messages"

# PX4 unit suffixes, longest first so "_m_s2" wins over "_m_s" and "_s".
_SUFFIX_UNITS: Tuple[Tuple[str, str], ...] = (
    ("_m_s2", "m/s^2"),
    ("_rad_s", "rad/s"),
    ("_deg_s", "deg/s"),
    ("_m_s", "m/s"),
    ("_mah", "mAh"),
    ("_rad", "rad"),
    ("_deg", "deg"),
    ("_dbm", "dBm"),
    ("_hpa", "hPa"),
    ("_pa", "Pa"),
    ("_hz", "Hz"),
    ("_ga", "gauss"),
    ("_us", "us"),
    ("_ms", "ms"),
    ("_wh", "Wh"),
    ("_m", "m"),
    ("_v", "V"),
    ("_a", "A"),
    ("_c", "degC"),
    ("_s", "s"),
)

# Fields that predate the suffix convention: (topic, field) -> raw unit.
# Integer GPS fields are degE7 / mm and get converted by resolve_unit().
_KNOWN_FIELDS: Dict[str, Dict[str, str]] = {
    "vehicle_global_position": {
        "lat": "deg", "lon": "deg", "alt": "m", "alt_ellipsoid": "m",
        "eph": "m", "epv": "m", "terrain_alt": "m",
    },
    "vehicle_local_position": {
        "x": "m", "y": "m", "z": "m", "vx": "m/s", "vy": "m/s", "vz": "m/s",
        "ax": "m/s^2", "ay": "m/s^2", "az": "m/s^2", "heading": "rad",
        "ref_lat": "deg", "ref_lon": "deg", "ref_alt": "m",
        "dist_bottom": "m", "eph": "m", "epv": "m", "evh": "m/s", "evv": "m/s",
    },
    "vehicle_attitude": {
        "rollspeed": "rad/s", "pitchspeed": "rad/s", "yawspeed": "rad/s",
    },
    "vehicle_gps_position": {
        "lat": "degE7", "lon": "degE7", "alt": "mm", "alt_ellipsoid": "mm",
        "eph": "m", "epv": "m", "hdop": "", "vdop": "",
    },
    "sensor_gps": {
        "lat": "degE7", "lon": "degE7", "alt": "mm", "alt_ellipsoid": "mm",
        "eph": "m", "epv": "m",
    },
}

# Fields that are always in these units, whatever the topic.
_ANY_TOPIC_FIELDS = {"timestamp": "us", "timestamp_sample": "us", "temperature": "degC"}

_ARRAY_INDEX = re.compile(r"\[\d+\]$")


@lru_cache(maxsize=1)
def px4_field_info() -> Dict[str, Dict[str, dict]]:
    """PX4 spec annotations: topic -> {field: {"unit"?: str, "invalid"?: number}}."""
    text = resources.files("mavpose").joinpath("data/px4_fields.json").read_text(encoding="utf-8")
    return json.loads(text)["topics"]


def topic_name(name: str, multi_id: int) -> str:
    """msg_type for a ULog topic instance."""
    return name if multi_id == 0 else f"{name}_{multi_id}"


def raw_ulog_unit(topic: str, field: str, dtype: np.dtype) -> Optional[str]:
    """Best-known raw unit for a ULog field, or None."""
    base = _ARRAY_INDEX.sub("", field)
    spec = px4_field_info().get(topic, {}).get(base, {})
    if "unit" in spec and not (base in _KNOWN_FIELDS.get(topic, {})
                               and np.issubdtype(dtype, np.integer)):
        return spec["unit"]
    if base in _ANY_TOPIC_FIELDS:
        return _ANY_TOPIC_FIELDS[base]
    table = _KNOWN_FIELDS.get(topic, {})
    if base in table:
        unit = table[base]
        # Integer degE7 / mm only applies to old integer encodings
        if unit in ("degE7", "mm") and not np.issubdtype(dtype, np.integer):
            return {"degE7": "deg", "mm": "m"}[unit]
        return unit or None
    lower = base.lower()
    for suffix, unit in _SUFFIX_UNITS:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            return unit
    return None


def read_ulog(path: str):
    """Parse *path* with pyulog; raise ValueError with a clear message on failure."""
    from pyulog import ULog

    try:
        return ULog(path, disable_str_exceptions=True)
    except Exception as exc:  # pyulog raises a variety of errors on bad input
        raise ValueError(f"Could not read PX4 ULog file {path}: {exc}") from exc


def ulog_schema(ulog) -> Dict[str, dict]:
    """Schema in the same shape as LogExtractor.schema_only()."""
    from mavpose.log_extractor import resolve_unit

    schema: Dict[str, dict] = {}
    for d in ulog.data_list:
        mt = topic_name(d.name, d.multi_id)
        fields, units = {}, {}
        for field, values in d.data.items():
            fields[field] = str(values.dtype)
            raw = raw_ulog_unit(d.name, field, values.dtype)
            natural = resolve_unit(raw)[1]
            if natural:
                units[field] = natural
        schema[mt] = {"count": int(len(d.data["timestamp"])), "fields": fields, "units": units}
    if ulog.logged_messages:
        schema[LOGGED_MESSAGES] = {
            "count": len(ulog.logged_messages),
            "fields": {"level": "str", "message": "str"},
            "units": {},
        }
    return schema


def unknown_marker(topic: str, field: str):
    """PX4 numeric "unknown" marker for a field, or None."""
    base = _ARRAY_INDEX.sub("", field)
    return px4_field_info().get(topic, {}).get(base, {}).get("invalid")


def ulog_frames(ulog, convert_units: bool = True, filter_unknown: bool = True):
    """
    Build per-topic DataFrames from a parsed ULog.

    Returns (frames, units, raw_units, unknown_counts, stats).
    """
    from mavpose.log_extractor import resolve_unit

    start = int(ulog.start_timestamp)
    frames: Dict[str, pd.DataFrame] = {}
    units: Dict[str, Dict[str, str]] = {}
    raw_units: Dict[str, Dict[str, str]] = {}
    unknown_counts: Dict[str, Dict[str, int]] = {}
    unstamped = 0
    kept = 0

    for d in ulog.data_list:
        mt = topic_name(d.name, d.multi_id)
        ts = d.data["timestamp"].astype(np.int64)
        stamped = ts != 0
        unstamped += int((~stamped).sum())
        if not stamped.any():
            continue

        columns = {
            "time_s": (ts[stamped] - start) / 1e6,
            "msg_type": np.full(int(stamped.sum()), mt, dtype=object),
        }
        topic_units: Dict[str, str] = {"time_s": "s"}
        topic_raw: Dict[str, str] = {}
        for field, values in d.data.items():
            values = values[stamped]
            raw = raw_ulog_unit(d.name, field, values.dtype)
            factor, natural = resolve_unit(raw)
            if not convert_units:
                factor, natural = 1.0, raw
            if np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_:
                col = values.astype(np.float64)
                marker = unknown_marker(d.name, field) if filter_unknown else None
                if marker is not None:
                    hits = col == marker
                    if hits.any():
                        col[hits] = np.nan
                        counts = unknown_counts.setdefault(mt, {})
                        counts[field] = counts.get(field, 0) + int(hits.sum())
                if factor != 1.0:
                    col = col * factor
            else:
                col = values
            columns[field] = col
            if raw:
                topic_raw[field] = raw
            if natural:
                topic_units[field] = natural

        df = pd.DataFrame(columns)
        df.sort_values("time_s", inplace=True, kind="stable")
        df.reset_index(drop=True, inplace=True)
        frames[mt] = df
        units[mt] = topic_units
        raw_units[mt] = topic_raw
        kept += len(df)

    messages = [m for m in ulog.logged_messages if int(m.timestamp) != 0]
    if messages:
        frames[LOGGED_MESSAGES] = pd.DataFrame({
            "time_s": [(int(m.timestamp) - start) / 1e6 for m in messages],
            "msg_type": LOGGED_MESSAGES,
            "level": [m.log_level_str() for m in messages],
            "message": [m.message for m in messages],
        }).sort_values("time_s", kind="stable").reset_index(drop=True)
        units[LOGGED_MESSAGES] = {"time_s": "s"}
        raw_units[LOGGED_MESSAGES] = {}
        kept += len(messages)

    stats = {
        "messages": kept, "bad_data": 0, "errors": 0,
        "unknown_values": sum(n for f in unknown_counts.values() for n in f.values()),
    }
    if getattr(ulog, "file_corruption", False):
        stats["bad_data"] = 1
        logger.warning("ULog reports file corruption; data after the damaged "
                       "section was recovered where possible.")
    if unstamped:
        logger.warning("Dropped %d ULog row(s) with no timestamp (never stamped).", unstamped)
    return frames, units, raw_units, unknown_counts, stats
