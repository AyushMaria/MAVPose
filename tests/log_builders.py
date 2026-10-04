"""
tests/log_builders.py

Build small but realistic flight logs for end-to-end tests:

* ``build_tlog``  — a MAVLink telemetry log (.tlog) written with pymavlink's
  own encoder: 8-byte big-endian receive timestamp (µs) + packet, per message.
* ``build_dataflash_bin`` — an ArduPilot DataFlash binary log (.bin) with
  FMT, UNIT, MULT and FMTU records, as written by modern ArduPilot.

Both describe the same 10-second flight so tests can check timing and
units against known values.  The .tlog also contains MAVLink "unknown"
markers, as real autopilots send them: for the first ``UNKNOWN_SAMPLES``
samples the current sensor and GPS fix are not ready, and GPS vertical
accuracy (epv) is never provided.
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
UNKNOWN_SAMPLES = 3                       # leading samples with "unknown" markers
UINT16_MAX = 65535


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
        warming_up = i < UNKNOWN_SAMPLES
        emit(mavlink.MAVLink_sys_status_message(
            0, 0, 0, 500,
            int(round(s["volt_v"] * 1000)),
            -1 if warming_up else int(round(s["curr_a"] * 100)),   # -1 = unknown
            -1 if warming_up else 80,                               # -1 = unknown
            0, 0, 0, 0, 0, 0,
        ), t)
        # GPS_RAW_INT.time_usec is boot-based here: a trap for naive timing
        emit(mavlink.MAVLink_gps_raw_int_message(
            int((BOOT_OFFSET_S + t) * 1e6), 3,
            int(round(s["lat_deg"] * 1e7)), int(round(s["lon_deg"] * 1e7)),
            int(round(s["alt_m"] * 1000)),
            UINT16_MAX if warming_up else 100,   # eph: unknown until fix
            UINT16_MAX,                          # epv: never provided
            150, 9000, 10,
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


# ---------------------------------------------------------------------------
# PX4 ULog (.ulg)
# ---------------------------------------------------------------------------
#
# Written directly from the ULog file-format spec
# (https://docs.px4.io/main/en/dev_log/ulog_file_format.html):
# 16-byte header, then messages of [uint16 size][uint8 type][payload].

ULOG_START_US = 50_000_000        # logging started 50 s after boot
ULOG_PRESTART_S = 0.5             # vehicle_status sample published before logging

_ULOG_TYPES = {"uint64_t": "Q", "int32_t": "i", "uint8_t": "B", "float": "f", "double": "d"}

# topic -> list of (c_type, field name, array length or None)
_ULOG_FORMATS = {
    # Old PX4: integer degE7 / mm GPS position
    "vehicle_gps_position": [("uint64_t", "timestamp", None), ("int32_t", "lat", None),
                             ("int32_t", "lon", None), ("int32_t", "alt", None),
                             ("float", "vel_m_s", None), ("float", "eph", None)],
    # New PX4: floating-point degrees and metres
    "sensor_gps": [("uint64_t", "timestamp", None), ("double", "latitude_deg", None),
                   ("double", "longitude_deg", None), ("double", "altitude_msl_m", None)],
    "battery_status": [("uint64_t", "timestamp", None), ("float", "voltage_v", None),
                       ("float", "current_a", None), ("float", "remaining", None),
                       ("float", "temperature", None), ("float", "voltage_cell_v", 4)],
    "vehicle_attitude": [("uint64_t", "timestamp", None), ("float", "q", 4),
                         ("float", "rollspeed", None)],
    "vehicle_status": [("uint64_t", "timestamp", None), ("uint8_t", "nav_state", None)],
    # A topic that was never stamped (timestamp 0), as seen in real PX4 logs
    "commander_state": [("uint64_t", "timestamp", None), ("uint8_t", "main_state", None)],
}


def ulog_samples():
    n = int(DURATION_S * RATE_HZ) + 1
    for i in range(n):
        t = i / RATE_HZ
        yield {
            "t": t,
            "us": ULOG_START_US + int(round(t * 1e6)),
            "lat_deg": 47.3977419,
            "lon_deg": 8.5455938,
            "alt_m": 488.0 + t,
            "volt_v": 16.2 - 0.01 * t,
            "curr_a": 12.5,
            "volt_v_1": 25.2,           # second battery (multi_id 1)
        }


def _ulog_msg(msg_type: str, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), ord(msg_type)) + payload


def _ulog_struct(topic: str) -> str:
    out = "<"
    for ctype, _name, n in _ULOG_FORMATS[topic]:
        out += (str(n) if n else "") + _ULOG_TYPES[ctype]
    return out


def build_ulog(path: Path, truncate_to: int | None = None) -> Path:
    """Write a PX4 ULog describing the same 10-second flight."""
    out = bytearray(b"ULog\x01\x12\x35" + bytes([1]) + struct.pack("<Q", ULOG_START_US))
    # Flag bits (required first message for ULog v1+)
    out += _ulog_msg("B", bytes(16) + struct.pack("<3Q", 0, 0, 0))
    key = b"char[3] sys_name"
    out += _ulog_msg("I", bytes([len(key)]) + key + b"PX4")
    for topic, fields in _ULOG_FORMATS.items():
        spec = ";".join(f"{c}{f'[{n}]' if n else ''} {name}" for c, name, n in fields) + ";"
        out += _ulog_msg("F", f"{topic}:{spec}".encode())

    # Subscriptions: (msg_id, multi_id, topic)
    subs = [(0, 0, "vehicle_gps_position"), (1, 0, "sensor_gps"), (2, 0, "battery_status"),
            (3, 1, "battery_status"), (4, 0, "vehicle_attitude"), (5, 0, "vehicle_status"),
            (6, 0, "commander_state")]
    for msg_id, multi_id, topic in subs:
        out += _ulog_msg("A", struct.pack("<BH", multi_id, msg_id) + topic.encode())

    def data(msg_id, topic, *values):
        return _ulog_msg("D", struct.pack("<H", msg_id) + struct.pack(_ulog_struct(topic), *values))

    # Published 0.5 s before logging started, and a never-stamped topic
    out += data(5, "vehicle_status", ULOG_START_US - int(ULOG_PRESTART_S * 1e6), 0)
    for _ in range(3):
        out += data(6, "commander_state", 0, 1)

    for i, s in enumerate(ulog_samples()):
        us = s["us"]
        warming_up = i < UNKNOWN_SAMPLES
        if i % RATE_HZ == 0:
            out += data(0, "vehicle_gps_position", us,
                        int(round(s["lat_deg"] * 1e7)), int(round(s["lon_deg"] * 1e7)),
                        int(round(s["alt_m"] * 1000)), 1.5, 0.8)
            out += data(1, "sensor_gps", us, s["lat_deg"], s["lon_deg"], s["alt_m"])
            out += data(5, "vehicle_status", us, 2)
        # 4-cell slots, 3 cells fitted: slot 3 reads 0 (PX4's "unknown")
        out += data(2, "battery_status", us + 10, s["volt_v"],
                    -1.0 if warming_up else s["curr_a"],         # -1 = unknown
                    0.8, 31.5, 4.05, 4.04, 4.06, 0.0)
        out += data(3, "battery_status", us + 20, s["volt_v_1"], 3.0, 0.9, 30.0,
                    4.2, 4.2, 4.2, 4.2)
        out += data(4, "vehicle_attitude", us + 30, 1.0, 0.0, 0.0, 0.0, 0.01)
        if i == 10:
            out += _ulog_msg("L", b"6" + struct.pack("<Q", us) + b"[commander] Takeoff detected")
        if i == 40:
            out += _ulog_msg("L", b"4" + struct.pack("<Q", us) + b"[commander] Low battery")

    if truncate_to is not None:
        out = out[:truncate_to]
    path.write_bytes(bytes(out))
    return path
