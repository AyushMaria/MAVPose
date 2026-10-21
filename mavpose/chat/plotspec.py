"""
mavpose/chat/plotspec.py

Plot specs: a small JSON description of a plot that MAVPose draws itself.

The plot assistant asks the model for a spec instead of Python code, so no
generated code runs.  A spec is plain data, so it can also be saved,
shared, and redrawn on any log with the same message types::

    {
      "title": "Altitude and battery",
      "start_s": 60, "end_s": 300,                 # optional time window
      "panels": [
        {"title": "Altitude",
         "series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "relative_alt",
                     "label": "Altitude"}]},
        {"kind": "line",
         "series": [{"msg_type": "SYS_STATUS", "field": "voltage_battery"},
                    {"msg_type": "SYS_STATUS", "field": "current_battery",
                     "axis": "right", "color": "tab:red"}]}
      ]
    }

Series are plotted against ``time_s`` unless they set ``x_field`` (another
field of the same message type, e.g. ``lon`` for a ``lat`` track).  Axis
labels get their units from the log, so they are right by construction.

Use :func:`validate_spec` to check a spec against a log and
:func:`render_spec` to draw it.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

__all__ = ["PlotSpecError", "parse_spec", "validate_spec", "render_spec",
           "MAX_PANELS", "MAX_SERIES"]

MAX_PANELS = 6
MAX_SERIES = 8
KINDS = ("line", "scatter", "step")
AXES = ("left", "right")


class PlotSpecError(ValueError):
    """A plot spec is malformed or refers to data the log doesn't have."""


def parse_spec(text: str) -> Dict[str, Any]:
    """
    Read a spec from a model reply or a file's text.

    Accepts bare JSON or JSON inside a markdown code fence, with or without
    surrounding prose.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise PlotSpecError("No JSON object found. Reply with the plot spec as JSON.")
        candidate = text[start:end + 1]
    try:
        spec = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PlotSpecError(f"The plot spec is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise PlotSpecError("The plot spec must be a JSON object.")
    return spec


def _match(name: Any, options: List[str], what: str, where: str = "") -> str:
    """Return the option equal to *name* (ignoring case), or explain the mistake."""
    if not isinstance(name, str) or not name.strip():
        raise PlotSpecError(f"Missing {what}{where}.")
    by_lower = {o.lower(): o for o in options}
    hit = by_lower.get(name.strip().lower())
    if hit:
        return hit
    close = difflib.get_close_matches(name, options, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    listing = "" if close else f" Available: {', '.join(sorted(options)[:40])}."
    raise PlotSpecError(f"No {what} {name!r}{where}.{hint}{listing}")


def _number(value: Any, what: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotSpecError(f"{what} must be a number of seconds, not {value!r}.")
    return float(value)


def validate_spec(spec: Mapping[str, Any], frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Check *spec* against the log's *frames* and return a normalised copy.

    Message type and field names are matched ignoring case and corrected to
    the log's spelling; defaults are filled in.  Unknown keys are ignored.

    Raises
    ------
    PlotSpecError
        With a message that says what to change, suitable for showing to
        the model so it can fix the spec.
    """
    if not isinstance(spec, Mapping):
        raise PlotSpecError("The plot spec must be a JSON object.")
    out: Dict[str, Any] = {
        "title": str(spec.get("title") or ""),
        "start_s": _number(spec.get("start_s"), "start_s"),
        "end_s": _number(spec.get("end_s"), "end_s"),
        "panels": [],
    }
    if (out["start_s"] is not None and out["end_s"] is not None
            and out["start_s"] >= out["end_s"]):
        raise PlotSpecError("start_s must be smaller than end_s.")

    panels = spec.get("panels")
    if not isinstance(panels, list) or not panels:
        raise PlotSpecError('The spec needs a non-empty "panels" list.')
    if len(panels) > MAX_PANELS:
        raise PlotSpecError(f"At most {MAX_PANELS} panels, got {len(panels)}.")

    types = list(frames)
    for p_i, panel in enumerate(panels, 1):
        if not isinstance(panel, Mapping):
            raise PlotSpecError(f"Panel {p_i} must be an object.")
        kind = str(panel.get("kind") or "line").lower()
        if kind not in KINDS:
            raise PlotSpecError(f"Panel {p_i}: kind must be one of {', '.join(KINDS)}, not {kind!r}.")
        series = panel.get("series")
        if not isinstance(series, list) or not series:
            raise PlotSpecError(f'Panel {p_i} needs a non-empty "series" list.')
        if len(series) > MAX_SERIES:
            raise PlotSpecError(f"Panel {p_i}: at most {MAX_SERIES} series, got {len(series)}.")
        norm_series = []
        for s_i, s in enumerate(series, 1):
            where = f" (panel {p_i}, series {s_i})"
            if not isinstance(s, Mapping):
                raise PlotSpecError(f"Series{where} must be an object.")
            mt = _match(s.get("msg_type"), types, "message type", where)
            columns = [c for c in frames[mt].columns if c not in ("time_s", "msg_type")]
            field = _match(s.get("field"), columns, "field", f" in {mt}{where}")
            if not pd.api.types.is_numeric_dtype(frames[mt][field]):
                raise PlotSpecError(f"{mt}.{field} is not numeric, so it can't be plotted{where}.")
            x_field = s.get("x_field")
            if x_field not in (None, "", "time_s"):
                x_field = _match(x_field, columns, "x_field", f" in {mt}{where}")
                if not pd.api.types.is_numeric_dtype(frames[mt][x_field]):
                    raise PlotSpecError(f"{mt}.{x_field} is not numeric{where}.")
            else:
                x_field = None
            axis = str(s.get("axis") or "left").lower()
            if axis not in AXES:
                raise PlotSpecError(f"axis must be 'left' or 'right'{where}.")
            color = s.get("color")
            if color is not None:
                from matplotlib.colors import is_color_like
                if not is_color_like(color):
                    raise PlotSpecError(f"{color!r} is not a colour matplotlib knows{where}.")
            norm_series.append({
                "msg_type": mt, "field": field, "x_field": x_field, "axis": axis,
                "label": str(s.get("label") or ""), "color": color,
            })
        xs = {bool(s["x_field"]) for s in norm_series}
        if len(xs) > 1:
            raise PlotSpecError(
                f"Panel {p_i} mixes time-based and x_field series; put them in separate panels.")
        out["panels"].append({
            "title": str(panel.get("title") or ""),
            "kind": kind,
            "y_label": str(panel.get("y_label") or ""),
            "series": norm_series,
        })
    return out


def _unit(units: Mapping[str, Mapping[str, str]], mt: str, field: str) -> Optional[str]:
    return (units.get(mt) or {}).get(field)


def _axis_label(series: List[dict], units, given: str) -> str:
    if given:
        return given
    found = {_unit(units, s["msg_type"], s["field"]) for s in series}
    if len(found) == 1 and None not in found:
        return found.pop()
    return ""


def render_spec(
    spec: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    units: Mapping[str, Mapping[str, str]],
    output_path: str,
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Validate *spec* against the log and draw it to *output_path*.

    Parameters
    ----------
    spec:
        The plot spec (a dict, e.g. from :func:`parse_spec`).
    frames:
        ``{msg_type: DataFrame}`` with a ``time_s`` column, e.g.
        ``LogExtractor.frames`` or ``{m: log[m] for m in log.messages()}``.
    units:
        ``{msg_type: {field: unit}}``, e.g. ``log.units``.
    output_path:
        Image file to write; the format follows the extension (``.png``,
        ``.svg``, ``.pdf``…).
    dpi:
        Resolution for raster formats.

    Returns
    -------
    The normalised spec that was drawn.

    Raises
    ------
    PlotSpecError
        The spec is invalid, or selects no data (e.g. an empty time window).
    """
    fig, drawn = _build_figure(spec, frames, units)
    fig.savefig(output_path, dpi=dpi)
    return drawn


def _build_figure(spec, frames, units):
    """Validate *spec* and draw it on a new Figure (no pyplot, no global state)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    spec = validate_spec(spec, frames)
    start, end = spec["start_s"], spec["end_s"]
    panels = spec["panels"]
    time_based = all(not s["x_field"] for p in panels for s in p["series"])

    fig = Figure(figsize=(10, 3.2 * len(panels) + (0.4 if spec["title"] else 0)),
                 layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(panels), 1, sharex=time_based, squeeze=False)[:, 0]
    if spec["title"]:
        fig.suptitle(spec["title"])

    for ax, panel in zip(axes, panels, strict=True):
        right_ax = None
        handles = []
        for s in panel["series"]:
            df = frames[s["msg_type"]]
            if start is not None:
                df = df[df["time_s"] >= start]
            if end is not None:
                df = df[df["time_s"] <= end]
            x = df[s["x_field"]] if s["x_field"] else df["time_s"]
            y = df[s["field"]]
            if y.notna().sum() == 0:
                window = "" if start is None and end is None else " in the chosen time window"
                raise PlotSpecError(f"{s['msg_type']}.{s['field']} has no data{window}.")
            unit = _unit(units, s["msg_type"], s["field"])
            label = s["label"] or f"{s['msg_type']}.{s['field']}"
            if unit and f"({unit})" not in label:
                label = f"{label} ({unit})"
            target = ax
            if s["axis"] == "right":
                right_ax = right_ax or ax.twinx()
                target = right_ax
            style = {"label": label}
            if s["color"]:
                style["color"] = s["color"]
            if panel["kind"] == "scatter":
                h = target.scatter(x, y, s=6, **style)
            elif panel["kind"] == "step":
                (h,) = target.step(x, y, where="post", **style)
            else:
                (h,) = target.plot(x, y, linewidth=1.2, **style)
            handles.append(h)

        left = [s for s in panel["series"] if s["axis"] == "left"]
        right = [s for s in panel["series"] if s["axis"] == "right"]
        ax.set_ylabel(_axis_label(left, units, panel["y_label"]))
        if right_ax is not None:
            right_ax.set_ylabel(_axis_label(right, units, ""))
        if panel["title"]:
            ax.set_title(panel["title"])
        ax.grid(True, alpha=0.3)
        if len(handles) > 1 or right_ax is not None:
            ax.legend(handles=handles, loc="best", fontsize="small")
        first = panel["series"][0]
        if first["x_field"]:
            xu = _unit(units, first["msg_type"], first["x_field"])
            ax.set_xlabel(f"{first['x_field']} ({xu})" if xu else first["x_field"])
            if {first["x_field"], first["field"]} in ({"lat", "lon"}, {"Lat", "Lng"}):
                ax.set_aspect("equal", adjustable="datalim")
    # Time panels share one x axis label: on the bottom panel when every
    # panel is time-based, otherwise on each time panel.
    time_axes = [ax for ax, p in zip(axes, panels, strict=True) if not p["series"][0]["x_field"]]
    for ax in (time_axes[-1:] if time_based else time_axes):
        ax.set_xlabel("time (s)")

    return fig, copy.deepcopy(spec)
