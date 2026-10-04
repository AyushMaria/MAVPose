"""
mavpose/log_extractor.py

Headless data layer for drone flight logs.

Responsibilities
----------------
1. Parse a MAVLink telemetry log (.tlog) or an ArduPilot DataFlash log
   (.bin binary, .log text) into a dict of per-message-type pandas
   DataFrames, each with a ``time_s`` column (seconds from log start).
2. Provide a lightweight schema-only pass that returns message-type
   metadata without materialising full DataFrames (used for embeddings).
3. Export a subset of message types to a single Parquet file so the
   LLM only sees a clean, time-aligned, typed tabular structure —
   never raw binary data.

Design notes
------------
* **One clock.**  ``time_s`` is derived from pymavlink's per-message
  ``_timestamp``: the recorded receive time for .tlog files, and
  ``TimeUS`` (anchored to GPS time when available) for DataFlash logs.
  Every message type therefore shares the same time axis.  Native time
  fields (``time_boot_ms``, ``time_usec``, ``TimeUS`` …) are kept as
  ordinary columns.
* **Real units.**  MAVLink fields are converted from scaled integer
  units to natural ones using pymavlink's own unit metadata (e.g.
  ``alt`` mm → m, ``lat`` degE7 → deg, ``voltage_battery`` mV → V).
  DataFlash units come from the log's FMTU/UNIT/MULT records.  The
  resulting unit of every column is available via :attr:`units` and is
  included in the schema summary.  Time-valued fields are not rescaled.
* **Unknown markers removed.**  MAVLink fields that use a sentinel for
  "not provided" (e.g. ``current_battery = -1``, ``eph = UINT16_MAX``) are
  set to NaN, using the ``invalid`` markers from the official MAVLink
  definitions (``mavpose/data/mavlink_unknown_markers.json``).  Counts are
  reported per column (see :attr:`unknown_counts`).
* **Robust parsing.**  Corrupt packets are skipped and counted rather
  than ending the parse; a warning reports the totals (see :attr:`stats`).
* Non-scalar fields (arrays) are dropped; numeric columns are float64.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from functools import lru_cache
from importlib import resources
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from pymavlink import mavutil

logger = logging.getLogger(__name__)

# Columns injected by the extractor — not sourced from the log message.
_RESERVED_COLS = {"time_s", "msg_type"}

# DataFlash metadata records that describe the log rather than the flight.
_META_TYPES = {"FMT", "FMTU", "UNIT", "MULT"}

# Stop parsing only after this many *consecutive* reader exceptions, which
# indicates the reader can no longer make progress through the file.
MAX_CONSECUTIVE_ERRORS = 1000

# Scaled units → (factor, natural unit).  Units not listed here (and all
# time units such as ms / us) are left unchanged.
_UNIT_CONVERSIONS: Dict[str, Tuple[float, str]] = {
    # length
    "mm": (1e-3, "m"),
    "cm": (1e-2, "m"),
    "dm": (1e-1, "m"),
    "dam": (10.0, "m"),
    # angle / position
    "degE7": (1e-7, "deg"),
    "degE5": (1e-5, "deg"),
    "cdeg": (1e-2, "deg"),
    "ddeg": (1e-1, "deg"),
    # speed / rate
    "mm/s": (1e-3, "m/s"),
    "cm/s": (1e-2, "m/s"),
    "dm/s": (1e-1, "m/s"),
    "mrad/s": (1e-3, "rad/s"),
    "cdeg/s": (1e-2, "deg/s"),
    "ddeg/s": (1e-1, "deg/s"),
    # electrical
    "mV": (1e-3, "V"),
    "cV": (1e-2, "V"),
    "mA": (1e-3, "A"),
    "cA": (1e-2, "A"),
    # ratios / temperature / magnetic
    "d%": (1e-1, "%"),
    "c%": (1e-2, "%"),
    "cdegC": (1e-2, "degC"),
    "mgauss": (1e-3, "gauss"),
    "mGauss": (1e-3, "Gauss"),
}

# DataFlash units with an odd multiplier are rendered by pymavlink as
# "<factor> <unit>", e.g. "1e-07 deg".
_FACTOR_UNIT_RE = re.compile(r"^\s*([-+0-9.eE]+)\s+(\S+)\s*$")

# Units that measure time; these are never rescaled.
_TIME_UNITS = {"s", "ms", "us", "µs", "ns", "ds", "cs"}


@lru_cache(maxsize=1)
def unknown_markers() -> Dict[str, Dict[str, float]]:
    """
    MAVLink "unknown value" sentinels: msg_type -> {field: marker}.

    Generated from the MAVLink XML ``invalid`` attributes by
    tools/gen_unknown_markers.py.  A marker of NaN means NaN is the sentinel.
    """
    text = resources.files("mavpose").joinpath(
        "data/mavlink_unknown_markers.json"
    ).read_text(encoding="utf-8")
    raw = json.loads(text)["markers"]
    return {
        msg: {f: (math.nan if v == "NaN" else v) for f, v in fields.items()}
        for msg, fields in raw.items()
    }


def _is_unknown(value, marker) -> bool:
    """True if *value* equals the field's unknown marker."""
    if isinstance(marker, float) and math.isnan(marker):
        return isinstance(value, float) and math.isnan(value)
    return value == marker


def _to_float(value) -> Optional[float]:
    """Convert a scalar to float64; return None if not convertible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_unit(raw_unit: Optional[str]) -> Tuple[float, Optional[str]]:
    """
    Map a raw log unit to ``(factor, natural_unit)``.

    ``value_in_natural_unit = raw_value * factor``.  Unknown units and time
    units are returned unchanged with a factor of 1.
    """
    if not raw_unit:
        return 1.0, None
    unit = raw_unit.strip()
    factor = 1.0
    match = _FACTOR_UNIT_RE.match(unit)
    if match:
        try:
            factor = float(match.group(1))
            unit = match.group(2)
        except ValueError:
            return 1.0, raw_unit
    if unit in _TIME_UNITS:
        return 1.0, unit
    if unit in _UNIT_CONVERSIONS:
        scale, natural = _UNIT_CONVERSIONS[unit]
        return factor * scale, natural
    return factor, unit


def _raw_field_units(msg) -> Dict[str, str]:
    """Return ``{field: raw_unit}`` for a MAVLink or DataFlash message."""
    # MAVLink messages: unit metadata generated from the MAVLink XML.
    units = getattr(msg, "fieldunits_by_name", None)
    if isinstance(units, dict):
        return {k: v for k, v in units.items() if isinstance(v, str) and v}

    # DataFlash messages: units from FMTU/UNIT/MULT records, if present.
    fmt = getattr(msg, "fmt", None)
    get_unit = getattr(fmt, "get_unit", None)
    columns = getattr(fmt, "columns", None)
    if callable(get_unit) and isinstance(columns, (list, tuple)):
        found = {}
        for col in columns:
            try:
                unit = get_unit(col)
            except Exception:
                unit = ""
            if isinstance(unit, str) and unit:
                found[col] = unit
        return found
    return {}


def _message_timestamp(msg) -> Optional[float]:
    """Return pymavlink's per-message timestamp in seconds, if valid."""
    ts = getattr(msg, "_timestamp", None)
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    return float(ts)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to coerce a Series to numeric (float64).

    If the result is all-NaN the original string Series is returned
    unchanged so we don't silently destroy text columns.
    """
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().all():
        return series  # preserve string / mixed columns as-is
    return converted


class LogExtractor:
    """
    Parse a flight log file into structured DataFrames.

    Parameters
    ----------
    log_path:
        Path to a .tlog, .bin or .log file.
    convert_units:
        Convert scaled fields to natural units (mm → m, degE7 → deg,
        mV → V, …).  Defaults to True.
    filter_unknown:
        Replace MAVLink "unknown" sentinels (e.g. ``current_battery = -1``)
        with NaN.  Defaults to True.
    """

    def __init__(
        self,
        log_path: str,
        convert_units: bool = True,
        filter_unknown: bool = True,
    ) -> None:
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        self.log_path = log_path
        self.convert_units = convert_units
        self.filter_unknown = filter_unknown
        # Populated after extract_all()
        self._frames: Dict[str, pd.DataFrame] = {}
        self._schema: Dict[str, dict] = {}   # msg_type -> {count, fields, units}
        self._units: Dict[str, Dict[str, str]] = {}   # msg_type -> {col: unit}
        self._raw_units: Dict[str, Dict[str, str]] = {}
        self._unknown_counts: Dict[str, Dict[str, int]] = {}
        self._extracted = False
        self.stats: Dict[str, int] = {"messages": 0, "bad_data": 0, "errors": 0, "unknown_values": 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_messages(self) -> Iterator[object]:
        """
        Yield flight messages from the log, skipping corrupt data.

        Corrupt packets (BAD_DATA) and reader exceptions are counted in
        :attr:`stats` instead of stopping the parse.  Parsing only gives up
        after ``MAX_CONSECUTIVE_ERRORS`` exceptions in a row.
        """
        stats = {"messages": 0, "bad_data": 0, "errors": 0, "unknown_values": 0}
        self.stats = stats
        mav = mavutil.mavlink_connection(self.log_path)
        consecutive_errors = 0
        try:
            while True:
                try:
                    msg = mav.recv_match(blocking=False)
                except Exception as exc:
                    stats["errors"] += 1
                    consecutive_errors += 1
                    logger.debug("Skipping unreadable data in %s: %s", self.log_path, exc)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.warning(
                            "Stopped reading %s after %d consecutive read errors; "
                            "the rest of the file may be unreadable.",
                            self.log_path, consecutive_errors,
                        )
                        break
                    continue

                if msg is None:
                    break
                consecutive_errors = 0

                msg_type = msg.get_type()
                if msg_type == "BAD_DATA":
                    stats["bad_data"] += 1
                    continue
                if msg_type in _META_TYPES:
                    continue

                stats["messages"] += 1
                yield msg
        finally:
            close = getattr(mav, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        if stats["bad_data"] or stats["errors"]:
            logger.warning(
                "Skipped %d corrupt packet(s) and %d read error(s) in %s; "
                "%d valid messages were kept.",
                stats["bad_data"], stats["errors"], self.log_path, stats["messages"],
            )

    def _record_units(self, msg_type: str, msg) -> Dict[str, float]:
        """Record units for *msg_type* on first sight; return field factors."""
        raw = _raw_field_units(msg)
        units: Dict[str, str] = {}
        factors: Dict[str, float] = {}
        for field, raw_unit in raw.items():
            factor, natural = resolve_unit(raw_unit)
            if not self.convert_units:
                factor, natural = 1.0, raw_unit
            if natural:
                units[field] = natural
            if factor != 1.0:
                factors[field] = factor
        self._raw_units[msg_type] = raw
        self._units[msg_type] = units
        return factors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schema_only(self) -> Dict[str, dict]:
        """
        Fast single-pass scan that collects message-type metadata
        (field names, Python types, units, message count) without
        storing rows.

        Returns
        -------
        dict mapping msg_type -> {"count": int,
                                  "fields": {name: type_str},
                                  "units": {name: unit}}
        """
        schema: Dict[str, dict] = {}
        for msg in self._iter_messages():
            mt = msg.get_type()
            if mt not in schema:
                if mt not in self._units:
                    self._record_units(mt, msg)
                schema[mt] = {
                    "count": 1,
                    "fields": {
                        f: type(getattr(msg, f, None)).__name__
                        for f in msg.get_fieldnames()
                    },
                    "units": dict(self._units.get(mt, {})),
                }
            else:
                schema[mt]["count"] += 1
        self._schema = schema
        return schema

    def extract_all(self) -> Dict[str, pd.DataFrame]:
        """
        Full extraction into one DataFrame per message type.

        Returns
        -------
        dict mapping msg_type -> pd.DataFrame with columns:
            time_s (float64), msg_type (str), <field_0>, <field_1>, ...
        """
        raw: Dict[str, list] = {}  # msg_type -> list of row-dicts
        factors_by_type: Dict[str, Dict[str, float]] = {}
        markers_by_type: Dict[str, Dict[str, float]] = {}
        unknown_counts: Dict[str, Dict[str, int]] = {}
        all_markers = unknown_markers() if self.filter_unknown else {}
        t0: Optional[float] = None
        untimed = 0

        for msg in self._iter_messages():
            mt = msg.get_type()

            ts = _message_timestamp(msg)
            if ts is None:
                untimed += 1
                continue
            if t0 is None:
                t0 = ts

            if mt not in factors_by_type:
                factors_by_type[mt] = self._record_units(mt, msg)
                # Sentinels are defined for MAVLink messages only
                is_mavlink = isinstance(getattr(msg, "fieldunits_by_name", None), dict)
                markers_by_type[mt] = all_markers.get(mt, {}) if is_mavlink else {}
            factors = factors_by_type[mt]
            markers = markers_by_type[mt]

            row: dict = {"time_s": ts - t0, "msg_type": mt}
            for field in msg.get_fieldnames():
                val = getattr(msg, field, None)
                # Keep scalars only — drop lists / nested objects
                if field in markers and _is_unknown(val, markers[field]):
                    row[field] = None
                    counts = unknown_counts.setdefault(mt, {})
                    counts[field] = counts.get(field, 0) + 1
                elif isinstance(val, bool):
                    row[field] = val
                elif isinstance(val, (int, float)):
                    factor = factors.get(field)
                    row[field] = val * factor if factor is not None else val
                elif isinstance(val, (str, bytes)):
                    row[field] = (
                        val.decode("utf-8", errors="replace").rstrip("\x00")
                        if isinstance(val, bytes) else val
                    )
            raw.setdefault(mt, []).append(row)

        if untimed:
            logger.warning(
                "Dropped %d message(s) without a timestamp from %s.",
                untimed, self.log_path,
            )
        self._unknown_counts = unknown_counts
        self.stats["unknown_values"] = sum(
            n for fields in unknown_counts.values() for n in fields.values()
        )
        if self.stats["unknown_values"]:
            logger.info(
                "Replaced %d MAVLink 'unknown' marker value(s) with NaN in %s.",
                self.stats["unknown_values"], self.log_path,
            )

        # Build DataFrames
        frames: Dict[str, pd.DataFrame] = {}
        for mt, rows in raw.items():
            df = pd.DataFrame(rows)
            for col in df.columns:
                if col not in _RESERVED_COLS:
                    df[col] = _coerce_numeric(df[col])
                    if df[col].dtype == object and df[col].isna().all():
                        df[col] = df[col].astype("float64")   # all unknown
            df.sort_values("time_s", inplace=True, kind="stable")
            df.reset_index(drop=True, inplace=True)
            frames[mt] = df
            self._units.setdefault(mt, {})["time_s"] = "s"

        self._frames = frames
        self._extracted = True
        logger.info("Extracted %d message types from %s", len(frames), self.log_path)
        return frames

    def summary(self, msg_types: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Schema summary for extracted message types, suitable for an LLM
        prompt: msg_type -> {"rows": int,
                             "columns": {col: {dtype, unit?, min?, max?,
                                               unknown?}}}.

        ``unknown`` is the number of rows where the field held MAVLink's
        "not provided" marker (now NaN).
        """
        if not self._extracted:
            raise RuntimeError("Call extract_all() before summary().")
        types = list(self._frames) if msg_types is None else msg_types
        summary: Dict[str, dict] = {}
        for mt in types:
            if mt not in self._frames:
                continue
            df = self._frames[mt]
            units = self._units.get(mt, {})
            unknown = self._unknown_counts.get(mt, {})
            cols = {}
            for col in df.columns:
                meta: dict = {"dtype": str(df[col].dtype)}
                if col in units:
                    meta["unit"] = units[col]
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                    meta["min"] = round(float(df[col].min()), 6)
                    meta["max"] = round(float(df[col].max()), 6)
                if unknown.get(col):
                    meta["unknown"] = unknown[col]
                cols[col] = meta
            summary[mt] = {"rows": len(df), "columns": cols}
        return summary

    def export_parquet(
        self,
        msg_types: List[str],
        output_path: str,
    ) -> Dict[str, dict]:
        """
        Export the requested message types to a single Parquet file.

        Message types are distinguished by the ``msg_type`` column.  Only
        types actually present in the log are included (unknown types are
        skipped with a warning).

        Returns
        -------
        Schema summary dict (see :meth:`summary`) for the exported types.
        """
        if not self._extracted:
            raise RuntimeError("Call extract_all() before export_parquet().")

        selected = [self._frames[mt] for mt in msg_types if mt in self._frames]
        skipped = [mt for mt in msg_types if mt not in self._frames]
        if skipped:
            logger.warning("Requested types not found in log: %s", skipped)

        if not selected:
            raise ValueError(
                f"None of the requested message types were found in the log. "
                f"Requested: {msg_types}"
            )

        combined = pd.concat(selected, ignore_index=True)
        combined.sort_values("time_s", inplace=True, kind="stable")
        combined.reset_index(drop=True, inplace=True)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Exported %d rows to %s", len(combined), output_path)

        return self.summary(msg_types)

    @property
    def frames(self) -> Dict[str, pd.DataFrame]:
        """The extracted frames dict. Empty until extract_all() is called."""
        return self._frames

    @property
    def units(self) -> Dict[str, Dict[str, str]]:
        """Unit of each known column: msg_type -> {column: unit}."""
        return self._units

    @property
    def raw_units(self) -> Dict[str, Dict[str, str]]:
        """Units as stored in the log, before conversion."""
        return self._raw_units

    @property
    def unknown_counts(self) -> Dict[str, Dict[str, int]]:
        """Rows set to NaN per field: msg_type -> {field: count}."""
        return self._unknown_counts
