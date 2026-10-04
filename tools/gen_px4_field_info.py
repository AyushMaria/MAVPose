#!/usr/bin/env python3
"""
tools/gen_px4_field_info.py

Generate mavpose/data/px4_fields.json from PX4's uORB message definitions.

PX4 annotates message fields with their unit and "unknown" marker, e.g.

    float32 current_a   # [A] [@invalid -1] Battery current

ULog files do not store this metadata, so we extract it once from
PX4-Autopilot/msg and ship the result as package data.

Usage:
    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/PX4/PX4-Autopilot /tmp/px4
    git -C /tmp/px4 sparse-checkout set msg
    python tools/gen_px4_field_info.py /tmp/px4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "mavpose" / "data" / "px4_fields.json"

# Fields marked [@invalid 0] where 0 is also a normal, real reading.
EXCLUDE_INVALID = {
    ("cellular_status", "link_tx_rate"),
    ("cellular_status", "link_rx_rate"),
}

# Annotations that are not units.
_NOT_UNITS = {"-", "bool", "boolean", "norm", ""}
_UNIT_ALIASES = {"microseconds": "us", "°C": "degC"}

_INT_LIMITS = {}
for bits in (8, 16, 32, 64):
    _INT_LIMITS[f"INT{bits}_MAX"] = 2 ** (bits - 1) - 1
    _INT_LIMITS[f"INT{bits}_MIN"] = -(2 ** (bits - 1))
    _INT_LIMITS[f"UINT{bits}_MAX"] = 2 ** bits - 1

_FIELD = re.compile(r"^\s*([A-Za-z]\w*)(\[\d+\])?\s+([a-z]\w*)\s*(#.*)?$")
_UNIT = re.compile(r"\[([^\]@][^\]]*)\]")
_INVALID = re.compile(r"\[@invalid\s+([^\]]+)\]")
_NUMERIC_TYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                  "uint64", "float32", "float64", "bool"}


def snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def parse_unit(comment: str):
    match = _UNIT.search(comment)
    if not match:
        return None
    unit = match.group(1).strip()
    unit = _UNIT_ALIASES.get(unit, unit)
    if unit in _NOT_UNITS or "," in unit:   # "[-1, 1]" is a range, not a unit
        return None
    return unit


def parse_invalid(comment: str):
    """First token of [@invalid X], as a number; None for NaN / non-numeric."""
    match = _INVALID.search(comment)
    if not match:
        return None
    token = match.group(1).split()[0]
    if token in _INT_LIMITS:
        return _INT_LIMITS[token]
    try:
        return int(token, 0)
    except ValueError:
        pass
    try:
        value = float(token)
    except ValueError:
        return None
    return None if value != value else value   # NaN needs no filtering


def parse_msg(path: Path):
    """Return (topics, {field: {"unit": ..., "invalid": ...}})."""
    topics = [snake_case(path.stem)]
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# TOPICS"):
            topics = stripped[len("# TOPICS"):].split()
            continue
        if "=" in line.split("#")[0]:
            continue   # constant definition
        match = _FIELD.match(line)
        if not match:
            continue
        ftype, _array, name, comment = match.groups()
        comment = comment or ""
        info = {}
        unit = parse_unit(comment)
        if unit:
            info["unit"] = unit
        if ftype in _NUMERIC_TYPES:
            invalid = parse_invalid(comment)
            if invalid is not None:
                info["invalid"] = invalid
        if info:
            fields[name] = info
    return topics, fields


def generate(px4_repo: Path) -> dict:
    msg_dir = px4_repo / "msg"
    files = sorted(msg_dir.glob("*.msg")) + sorted(msg_dir.glob("versioned/*.msg"))
    if not files:
        sys.exit(f"No .msg files found under {msg_dir}")

    out: dict = {}
    for path in files:
        topics, fields = parse_msg(path)
        if not fields:
            continue
        for topic in topics:
            topic_fields = out.setdefault(topic, {})
            for name, info in fields.items():
                info = dict(info)
                if (topic, name) in EXCLUDE_INVALID:
                    info.pop("invalid", None)
                if info:
                    topic_fields[name] = info

    try:
        commit = subprocess.run(
            ["git", "-C", str(px4_repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    return {
        "_source": f"https://github.com/PX4/PX4-Autopilot @ {commit} "
                   "(msg/*.msg field annotations [unit] [@invalid X])",
        "_generator": "tools/gen_px4_field_info.py",
        "_excluded_invalid": sorted(f"{t}.{f}" for t, f in EXCLUDE_INVALID),
        "topics": {t: dict(sorted(f.items())) for t, f in sorted(out.items()) if f},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("px4_repo", type=Path, help="path to a PX4-Autopilot checkout (msg/ is enough)")
    args = parser.parse_args()
    data = generate(args.px4_repo)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    topics = data["topics"]
    n_units = sum("unit" in v for f in topics.values() for v in f.values())
    n_invalid = sum("invalid" in v for f in topics.values() for v in f.values())
    print(f"Wrote {n_units} units and {n_invalid} unknown markers for "
          f"{len(topics)} topics to {OUTPUT}")


if __name__ == "__main__":
    main()
