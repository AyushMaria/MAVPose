"""
tests/log_builders.py

Build small but realistic flight logs for end-to-end tests:

* ``build_tlog``  — a MAVLink telemetry log (.tlog) written with pymavlink's
  own encoder: 8-byte big-endian receive timestamp (µs) + packet, per message.
* ``build_dataflash_bin`` — an ArduPilot DataFlash binary log (.bin) with
  FMT, UNIT, MULT and FMTU records, as written by modern ArduPilot.

Both describe the same 10-second flight so tests can check timing and
units against known values.
"""

from __future__ import annotations

import struct
from pathlib import Path

from pymavlink.dialects.v20 import ardupilotmega as mavlink

DURATION_S = 10.0
RATE_HZ = 5
RECEIVE_EPOCH_S = 1_700_000_000.0        # tlog receive-time base (Unix seconds)
BOOT_OFFSET_S = 42.0                      # vehicle had been powered on 42 s
GPS_WEEK = 2290
GPS_MS_START = 345_600_000


class _Buf:
    """Minimal file-like sink for the pymavlink encoder."""

    def __init__(self):
        self.data = bytearray()

    def write(self, b):
        self.data += b


def tlog_samples():
    """Ground-truth values for each sample time (seconds from start)."""
    n = int(DURATION_S * RATE_HZ) + 1
    for i in range(n):
        t = i / RATE_HZ
        yield {
            "t": t,
            "alt_m": 584.0 + t,               # MSL altitude, metres
            "rel_alt_m": t,
            "lat_deg": -35.3632621,
            "lon_deg": 149.1652374,
            "vx_mps": 1.5,
            "hdg_deg": 90.0,
            "volt_v": 12.6 - 0.01 * t,
            "curr_a": 10.5,
        }


def build_tlog(path: Path, junk_after_index: int | None = None) -> Path:
    """
    Write a .tlog.  If *junk_after_index* is given, 64 bytes of garbage are
    inserted after that sample to simulate a corrupt section.
    """
    buf = _Buf()
    mav = mavlink.MAVLink(buf, srcSystem=1, srcComponent=1)
    out = bytearray()

    def emit(msg, t):
        packed = msg.pack(mav)
        stamp = int((RECEIVE_EPOCH_S + t) * 1e6)
        out.extend(struct.pack(">Q", stamp) + packed)

    for i, s in enumerate(tlog_samples()):
        t = s["t"]
        boot_ms = int((BOOT_OFFSET_S + t) * 1000)
        if i % RATE_HZ == 0:
            emit(mavlink.MAVLink_heartbeat_message(2, 3, 81, 4, 4, 3), t)
            emit(mavlink.MAVLink_system_time_message(
                int((RECEIVE_EPOCH_S + t) * 1e6), boot_ms), t)
        emit(mavlink.MAVLink_global_position_int_message(
            boot_ms,
            int(round(s["lat_deg"] * 1e7)), int(round(s["lon_deg"] * 1e7)),
            int(round(s["alt_m"] * 1000)), int(round(s["rel_alt_m"] * 1000)),
            int(round(s["vx_mps"] * 100)), 0, 0,
            int(round(s["hdg_deg"] * 100)),
        ), t)
        emit(mavlink.MAVLink_sys_status_message(
            0, 0, 0, 500,
            int(round(s["volt_v"] * 1000)), int(round(s["curr_a"] * 100)), 80,
            0, 0, 0, 0, 0, 0,
        ), t)
        # GPS_RAW_INT.time_usec is boot-based here: a trap for naive timing
        emit(mavlink.MAVLink_gps_raw_int_message(
            int((BOOT_OFFSET_S + t) * 1e6), 3,
            int(round(s["lat_deg"] * 1e7)), int(round(s["lon_deg"] * 1e7)),
            int(round(s["alt_m"] * 1000)), 100, 100, 150, 9000, 10,
        ), t)
        if junk_after_index is not None and i == junk_after_index:
            out.extend(bytes(range(64)))

    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
# DataFlash (.bin)
# ---------------------------------------------------------------------------

_HEAD = b"\xa3\x95"
_FMT_ID = 128

# name -> (type id, format chars, columns)
_DF_FORMATS = {
    "FMT": (_FMT_ID, "BBnNZ", "Type,Length,Name,Format,Columns"),
    "UNIT": (177, "QbZ", "TimeUS,Id,Label"),
    "MULT": (178, "Qbd", "TimeUS,Id,Mult"),
    "FMTU": (179, "QBNN", "TimeUS,FmtType,UnitIds,MultIds"),
    "GPS": (130, "QBIHLLe", "TimeUS,Status,GMS,GWk,Lat,Lng,Alt"),
    "ATT": (131, "Qccc", "TimeUS,Roll,Pitch,Yaw"),
    "BARO": (132, "Qff", "TimeUS,Alt,Press"),
    "BAT": (133, "QHh", "TimeUS,Volt,Curr"),   # raw mV and cA integers
}

_STRUCT = {"B": "B", "b": "b", "H": "H", "h": "h", "I": "I", "i": "i",
           "Q": "Q", "f": "f", "d": "d", "n": "4s", "N": "16s", "Z": "64s",
           "c": "h", "e": "i", "L": "i"}


def _df_struct(fmt: str) -> str:
    return "<" + "".join(_STRUCT[c] for c in fmt)


def _df_msg(name: str, *values) -> bytes:
    type_id, fmt, _ = _DF_FORMATS[name]
    return _HEAD + bytes([type_id]) + struct.pack(_df_struct(fmt), *values)


def _df_fmt_record(name: str) -> bytes:
    type_id, fmt, cols = _DF_FORMATS[name]
    length = 3 + struct.calcsize(_df_struct(fmt))
    return _df_msg("FMT", type_id, length, name.encode(), fmt.encode(), cols.encode())


def dataflash_samples():
    n = int(DURATION_S * RATE_HZ) + 1
    for i in range(n):
        t = i / RATE_HZ
        yield {
            "t": t,
            "time_us": int((BOOT_OFFSET_S + t) * 1e6),
            "roll_deg": 12.34,
            "baro_alt_m": t,
            "press_pa": 101325.0 - 12.0 * t,
            "volt_v": 12.6,
            "curr_a": 10.5,
        }


def build_dataflash_bin(path: Path) -> Path:
    """Write a modern-style ArduPilot DataFlash .bin log."""
    out = bytearray()
    for name in _DF_FORMATS:
        out += _df_fmt_record(name)

    # Unit / multiplier definitions (ArduPilot's standard identifiers)
    for uid, label in [("s", "s"), ("m", "m"), ("P", "Pa"), ("d", "deg"),
                       ("v", "V"), ("A", "A"), ("-", "")]:
        out += _df_msg("UNIT", 0, ord(uid), label.encode())
    for mid, mult in [("-", 0.0), ("0", 1.0), ("B", 1e-2), ("C", 1e-3), ("F", 1e-6)]:
        out += _df_msg("MULT", 0, ord(mid), mult)
    # Units per field: BAT raw ints are mV (C) and cA (B)
    out += _df_msg("FMTU", 0, _DF_FORMATS["BARO"][0], b"smP", b"F00")
    out += _df_msg("FMTU", 0, _DF_FORMATS["ATT"][0], b"sddd", b"F000")
    out += _df_msg("FMTU", 0, _DF_FORMATS["BAT"][0], b"svA", b"FCB")

    for i, s in enumerate(dataflash_samples()):
        us = s["time_us"]
        if i % RATE_HZ == 0:
            out += _df_msg("GPS", us, 3, GPS_MS_START + int(s["t"] * 1000), GPS_WEEK,
                           -353632621, 1491652374, 58409)
        out += _df_msg("ATT", us + 100, int(round(s["roll_deg"] * 100)), 0, 9000)
        out += _df_msg("BARO", us + 200, s["baro_alt_m"], s["press_pa"])
        out += _df_msg("BAT", us + 300, int(round(s["volt_v"] * 1000)),
                       int(round(s["curr_a"] * 100)))

    path.write_bytes(bytes(out))
    return path
