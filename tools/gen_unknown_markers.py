#!/usr/bin/env python3
"""
tools/gen_unknown_markers.py

Generate mavpose/data/mavlink_unknown_markers.json from the official MAVLink
message definitions.

MAVLink marks "value unknown / not provided" sentinels on fields with an
``invalid`` attribute, e.g.

    <field type="int16_t" name="current_battery" units="cA" invalid="-1">

pymavlink's generated Python does not carry this attribute, so we extract it
from the XML once and ship the result as package data.

Usage:
    git clone --depth 1 https://github.com/mavlink/mavlink /tmp/mavlink
    python tools/gen_unknown_markers.py /tmp/mavlink
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "mavpose" / "data" / "mavlink_unknown_markers.json"

# Fields the spec marks invalid="0" where 0 is also a normal, real reading.
# Treating those as unknown would hide genuine data.
EXCLUDE = {
    ("CURRENT_MODE", "intended_custom_mode"),   # 0 is a real mode (e.g. Copter STABILIZE)
    ("CELLULAR_STATUS", "link_tx_rate"),        # 0 KiB/s is a real rate
    ("CELLULAR_STATUS", "link_rx_rate"),
}

_INT_LIMITS = {}
for bits in (8, 16, 32, 64):
    _INT_LIMITS[f"INT{bits}_MAX"] = 2 ** (bits - 1) - 1
    _INT_LIMITS[f"INT{bits}_MIN"] = -(2 ** (bits - 1))
    _INT_LIMITS[f"UINT{bits}_MAX"] = 2 ** bits - 1
    _INT_LIMITS[f"UINT{bits}_MIN"] = 0


def _enum_values(roots) -> dict:
    values = {}
    for root in roots:
        for enum in root.iter("enum"):
            for entry in enum.iter("entry"):
                name, value = entry.get("name"), entry.get("value")
                if name and value is not None:
                    try:
                        values[name] = int(value, 0)
                    except ValueError:
                        pass
    return values


def resolve(token: str, enums: dict):
    """Resolve an ``invalid`` attribute to a number, "NaN", or None."""
    token = token.strip()
    if token.upper() == "NAN":
        return "NaN"
    if token in _INT_LIMITS:
        return _INT_LIMITS[token]
    if token in enums:
        return enums[token]
    try:
        return int(token, 0)
    except ValueError:
        pass
    try:
        value = float(token)
        return "NaN" if math.isnan(value) else value
    except ValueError:
        return None


def generate(mavlink_repo: Path) -> dict:
    xml_dir = mavlink_repo / "message_definitions" / "v1.0"
    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        sys.exit(f"No MAVLink XML found under {xml_dir}")
    roots = [ET.parse(f).getroot() for f in files]
    enums = _enum_values(roots)

    markers: dict = {}
    skipped = []
    for root in roots:
        for msg in root.iter("message"):
            msg_name = msg.get("name")
            for field in msg.iter("field"):
                token = field.get("invalid")
                if token is None:
                    continue
                name, ftype = field.get("name"), field.get("type", "")
                if "[" in ftype or token.startswith("["):
                    continue  # array fields: the extractor drops arrays
                if (msg_name, name) in EXCLUDE:
                    continue
                value = resolve(token, enums)
                if value is None:
                    skipped.append(f"{msg_name}.{name}={token}")
                    continue
                markers.setdefault(msg_name, {})[name] = value

    try:
        commit = subprocess.run(
            ["git", "-C", str(mavlink_repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    if skipped:
        print("Unresolved markers (skipped):", ", ".join(skipped), file=sys.stderr)

    return {
        "_source": f"https://github.com/mavlink/mavlink @ {commit} "
                   "(message_definitions/v1.0, field 'invalid' attributes)",
        "_generator": "tools/gen_unknown_markers.py",
        "_excluded": sorted(f"{m}.{f}" for m, f in EXCLUDE),
        "markers": {k: dict(sorted(v.items())) for k, v in sorted(markers.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("mavlink_repo", type=Path, help="path to a mavlink/mavlink checkout")
    args = parser.parse_args()
    data = generate(args.mavlink_repo)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    n = sum(len(v) for v in data["markers"].values())
    print(f"Wrote {n} markers for {len(data['markers'])} messages to {OUTPUT}")


if __name__ == "__main__":
    main()
