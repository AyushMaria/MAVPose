"""
mavpose/flightlog.py

The stable, high-level API: ``mavpose.load(path) -> FlightLog``.

    import mavpose

    log = mavpose.load("flight.tlog")          # or .bin, .log, .ulg
    log.messages()                             # ['ATTITUDE', 'GLOBAL_POSITION_INT', ...]
    alt = log.series("GLOBAL_POSITION_INT", "alt")
    alt.attrs["unit"]                          # 'm'
    log.to_parquet("telemetry.parquet")

Everything in this module is covered by MAVPose's API stability policy
(see docs/api-stability.md).
"""

from __future__ import annotations

import difflib
import os
from typing import Dict, List, Optional

import pandas as pd

from mavpose.file_validator import validate_mavlink_file
from mavpose.log_extractor import LogExtractor

__all__ = ["FlightLog", "load"]

_FORMATS = {
    ".tlog": "mavlink",
    ".bin": "dataflash",
    ".log": "dataflash",
    ".ulg": "ulog",
}


def _not_found(kind: str, name: str, options: List[str], where: str = "") -> KeyError:
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return KeyError(f"No {kind} {name!r}{where}.{hint}")


class FlightLog:
    """
    A parsed flight log: one time-aligned DataFrame per message type, with units.

    Create one with :func:`mavpose.load`.  All frames share a single
    ``time_s`` axis (seconds from log start), scaled fields are in natural
    units, and "unknown" markers are NaN.

    Attributes
    ----------
    path : str
        Path of the log file.
    format : str
        ``"mavlink"`` (.tlog), ``"dataflash"`` (.bin/.log) or ``"ulog"`` (.ulg).
    stats : dict
        Parse counters: ``messages``, ``bad_data``, ``errors``, ``unknown_values``.
    """

    def __init__(self, extractor: LogExtractor, frames: Dict[str, pd.DataFrame]) -> None:
        self._extractor = extractor
        self._frames = frames
        self.path: str = extractor.log_path
        self.format: str = _FORMATS.get(os.path.splitext(self.path)[1].lower(), "unknown")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def messages(self) -> List[str]:
        """Sorted names of the message types (topics) in the log."""
        return sorted(self._frames)

    def fields(self, msg_type: str) -> List[str]:
        """Data fields of *msg_type*, excluding ``time_s`` and ``msg_type``."""
        return [c for c in self.frame(msg_type).columns if c not in ("time_s", "msg_type")]

    def __contains__(self, msg_type: object) -> bool:
        return msg_type in self._frames

    def __iter__(self):
        return iter(self.messages())

    def __len__(self) -> int:
        return len(self._frames)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def frame(self, msg_type: str) -> pd.DataFrame:
        """
        All rows of *msg_type* as a DataFrame with ``time_s``, ``msg_type`` and
        one column per field.  Returns a copy, so it is safe to modify.

        Raises
        ------
        KeyError
            If the log has no such message type (suggests close matches).
        """
        if msg_type not in self._frames:
            raise _not_found("message type", msg_type, list(self._frames))
        return self._frames[msg_type].copy()

    __getitem__ = frame

    def series(self, msg_type: str, field: str) -> pd.Series:
        """
        One field as a Series indexed by ``time_s``.

        The unit is in ``series.attrs["unit"]`` (``None`` if unknown) and the
        source in ``series.attrs["msg_type"]``.

        Raises
        ------
        KeyError
            If the message type or field does not exist (suggests close matches).
        """
        df = self.frame(msg_type)
        if field not in df.columns or field in ("time_s", "msg_type"):
            raise _not_found("field", field, self.fields(msg_type), f" in {msg_type}")
        s = pd.Series(df[field].to_numpy(), index=pd.Index(df["time_s"].to_numpy(), name="time_s"),
                      name=field)
        s.attrs["unit"] = self.unit(msg_type, field)
        s.attrs["msg_type"] = msg_type
        return s

    def unit(self, msg_type: str, field: str) -> Optional[str]:
        """Unit of a field after conversion (e.g. ``"m"``), or ``None`` if unknown."""
        return self._extractor.units.get(msg_type, {}).get(field)

    @property
    def units(self) -> Dict[str, Dict[str, str]]:
        """All known units: ``{msg_type: {field: unit}}``."""
        return {mt: dict(u) for mt, u in self._extractor.units.items() if mt in self._frames}

    @property
    def unknown_counts(self) -> Dict[str, Dict[str, int]]:
        """Values replaced by NaN because they were "unknown" markers."""
        return {mt: dict(c) for mt, c in self._extractor.unknown_counts.items()}

    @property
    def stats(self) -> Dict[str, int]:
        """
        Parse counters: ``messages`` kept, ``bad_data`` (corrupt packets
        skipped), ``errors`` (reader errors recovered from) and
        ``unknown_values`` (markers replaced by NaN).
        """
        return dict(self._extractor.stats)

    @property
    def duration_s(self) -> float:
        """Seconds between the first and last sample, across all message types."""
        if not self._frames:
            return 0.0
        start = min(df["time_s"].min() for df in self._frames.values())
        end = max(df["time_s"].max() for df in self._frames.values())
        return float(end - start)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def summary(self, msg_types: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Per-column dtype, unit, min/max and unknown count, for the given message
        types (all by default).  This is the schema MAVPose gives to LLMs.
        """
        if msg_types is not None:
            for mt in msg_types:
                if mt not in self._frames:
                    raise _not_found("message type", mt, list(self._frames))
        return self._extractor.summary(msg_types)

    def to_parquet(self, path: str, msg_types: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Write message types (all by default) to one Parquet file, distinguished
        by the ``msg_type`` column.  Returns :meth:`summary` for what was written.
        """
        types = self.messages() if msg_types is None else list(msg_types)
        for mt in types:
            if mt not in self._frames:
                raise _not_found("message type", mt, list(self._frames))
        return self._extractor.export_parquet(types, path)

    def __repr__(self) -> str:
        return (f"<FlightLog {os.path.basename(self.path)!r}: {self.format}, "
                f"{len(self._frames)} message types, {self.duration_s:.1f} s>")


def load(
    path: str,
    *,
    convert_units: bool = True,
    filter_unknown: bool = True,
) -> FlightLog:
    """
    Read a flight log into a :class:`FlightLog`.

    Parameters
    ----------
    path : str
        A MAVLink ``.tlog``, ArduPilot DataFlash ``.bin``/``.log`` or PX4
        ULog ``.ulg`` file.
    convert_units : bool, default True
        Convert scaled fields to natural units (mm → m, degE7 → deg, mV → V…).
    filter_unknown : bool, default True
        Replace "value not provided" markers (e.g. ``current_battery = -1``)
        with NaN.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    mavpose.FileValidationError
        Unsupported extension, empty, too large, or a symlink.
    ValueError
        The file could not be parsed as its format.
    """
    validate_mavlink_file(path)
    extractor = LogExtractor(path, convert_units=convert_units, filter_unknown=filter_unknown)
    frames = extractor.extract_all()
    return FlightLog(extractor, frames)
