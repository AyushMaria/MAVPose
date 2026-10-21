"""
tests/test_plotspec.py

Plot specs (mavpose.chat.plotspec): parsing model replies, checking specs
against a real log, and drawing them.  No model is involved.
"""

import json

import pytest

pytest.importorskip("matplotlib", reason="needs the [chat] extra")

from mavpose import LogExtractor  # noqa: E402
from mavpose.chat.plotspec import (  # noqa: E402
    PlotSpecError,
    _build_figure,
    parse_spec,
    render_spec,
    validate_spec,
)
from tests.log_builders import build_dataflash_bin, build_tlog  # noqa: E402


@pytest.fixture(scope="module")
def tlog(tmp_path_factory):
    ex = LogExtractor(str(build_tlog(tmp_path_factory.mktemp("t") / "f.tlog")))
    ex.extract_all()
    return ex


def one(msg_type, field, **series):
    return {"panels": [{"series": [{"msg_type": msg_type, "field": field, **series}]}]}


class TestParseSpec:

    def test_bare_json(self):
        assert parse_spec('{"panels": []}') == {"panels": []}

    def test_fenced_json_with_prose(self):
        text = 'Here is the plot:\n```json\n{"title": "x", "panels": []}\n```\nEnjoy.'
        assert parse_spec(text)["title"] == "x"

    def test_json_inside_prose_without_fence(self):
        assert parse_spec('Sure! {"title": "t", "panels": []} Done.')["title"] == "t"

    def test_no_json(self):
        with pytest.raises(PlotSpecError, match="No JSON object"):
            parse_spec("I can't do that.")

    def test_invalid_json(self):
        with pytest.raises(PlotSpecError, match="not valid JSON"):
            parse_spec('{"panels": [,]}')


class TestValidateSpec:

    def test_names_are_corrected_to_the_logs_spelling(self, tlog):
        spec = validate_spec(one("global_position_int", "RELATIVE_ALT"), tlog.frames)
        s = spec["panels"][0]["series"][0]
        assert (s["msg_type"], s["field"]) == ("GLOBAL_POSITION_INT", "relative_alt")

    def test_defaults_are_filled_in(self, tlog):
        spec = validate_spec(one("SYS_STATUS", "voltage_battery"), tlog.frames)
        panel = spec["panels"][0]
        assert panel["kind"] == "line"
        assert panel["series"][0]["axis"] == "left"
        assert spec["start_s"] is None and spec["end_s"] is None

    def test_unknown_keys_are_ignored(self, tlog):
        spec = one("SYS_STATUS", "voltage_battery", linewidth=3)
        spec["theme"] = "dark"
        validate_spec(spec, tlog.frames)

    @pytest.mark.parametrize("spec,message", [
        ({}, 'non-empty "panels"'),
        ({"panels": [{"series": []}]}, 'non-empty "series"'),
        (one("GLOBAL_POS", "alt"), "Did you mean: GLOBAL_POSITION_INT"),
        (one("GLOBAL_POSITION_INT", "altt"), "Did you mean: alt"),
        (one("GLOBAL_POSITION_INT", "alt", axis="top"), "axis must be"),
        (one("GLOBAL_POSITION_INT", "alt", color="blurple"), "not a colour"),
        (one("GLOBAL_POSITION_INT", "alt", x_field="nope"), "No x_field"),
        ({"panels": [{"kind": "pie", "series": [{"msg_type": "SYS_STATUS", "field": "load"}]}]},
         "kind must be"),
        ({"start_s": 5, "end_s": 2, **one("SYS_STATUS", "load")}, "start_s must be smaller"),
        ({"start_s": "soon", **one("SYS_STATUS", "load")}, "number of seconds"),
        ({"panels": [{"series": [{"msg_type": "SYS_STATUS", "field": "load"}]}] * 7}, "At most 6"),
    ])
    def test_errors_say_what_to_change(self, tlog, spec, message):
        with pytest.raises(PlotSpecError, match=message):
            validate_spec(spec, tlog.frames)

    def test_mixing_time_and_xy_series_in_one_panel(self, tlog):
        spec = {"panels": [{"series": [
            {"msg_type": "GLOBAL_POSITION_INT", "field": "lat", "x_field": "lon"},
            {"msg_type": "GLOBAL_POSITION_INT", "field": "alt"}]}]}
        with pytest.raises(PlotSpecError, match="separate panels"):
            validate_spec(spec, tlog.frames)

    def test_normalised_spec_is_stable(self, tlog):
        once = validate_spec(one("sys_status", "voltage_battery"), tlog.frames)
        assert validate_spec(json.loads(json.dumps(once)), tlog.frames) == once


class TestRender:

    def test_axis_labels_carry_the_logs_units(self, tlog):
        spec = {"panels": [
            {"series": [{"msg_type": "GLOBAL_POSITION_INT", "field": "relative_alt"}]},
            {"series": [{"msg_type": "SYS_STATUS", "field": "voltage_battery"},
                        {"msg_type": "SYS_STATUS", "field": "current_battery", "axis": "right"}]},
        ]}
        fig, _ = _build_figure(spec, tlog.frames, tlog.units)
        alt_ax, batt_ax = fig.axes[0], fig.axes[1]
        assert alt_ax.get_ylabel() == "m"
        assert batt_ax.get_ylabel() == "V"
        right = [a for a in fig.axes if a not in (alt_ax, batt_ax)][0]
        assert right.get_ylabel() == "A"
        assert batt_ax.get_xlabel() == "time (s)"
        labels = [t.get_text() for t in batt_ax.get_legend().get_texts()]
        assert labels == ["SYS_STATUS.voltage_battery (V)", "SYS_STATUS.current_battery (A)"]

    def test_values_are_plotted_in_converted_units(self, tlog):
        fig, _ = _build_figure(one("GLOBAL_POSITION_INT", "relative_alt"), tlog.frames, tlog.units)
        y = fig.axes[0].get_lines()[0].get_ydata()
        assert max(y) == pytest.approx(10.0)          # 10 m climb, not 10000 mm

    def test_time_window(self, tlog):
        spec = {"start_s": 2.0, "end_s": 4.0, **one("SYS_STATUS", "voltage_battery")}
        fig, _ = _build_figure(spec, tlog.frames, tlog.units)
        x = fig.axes[0].get_lines()[0].get_xdata()
        assert min(x) >= 2.0 and max(x) <= 4.0 and len(x) > 0

    def test_empty_window_is_an_error(self, tlog):
        with pytest.raises(PlotSpecError, match="no data in the chosen time window"):
            _build_figure({"start_s": 500, **one("SYS_STATUS", "voltage_battery")},
                          tlog.frames, tlog.units)

    def test_track_uses_x_field(self, tlog):
        spec = {"panels": [{"kind": "scatter", "series": [
            {"msg_type": "GLOBAL_POSITION_INT", "field": "lat", "x_field": "lon"}]}]}
        fig, _ = _build_figure(spec, tlog.frames, tlog.units)
        assert fig.axes[0].get_xlabel() == "lon (deg)"

    def test_writes_png_and_svg(self, tlog, tmp_path):
        render_spec(one("SYS_STATUS", "load"), tlog.frames, tlog.units, str(tmp_path / "p.png"))
        assert (tmp_path / "p.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        render_spec(one("SYS_STATUS", "load"), tlog.frames, tlog.units, str(tmp_path / "p.svg"))
        assert b"<svg" in (tmp_path / "p.svg").read_bytes()[:500]

    def test_dataflash_log(self, tmp_path):
        ex = LogExtractor(str(build_dataflash_bin(tmp_path / "f.bin")))
        ex.extract_all()
        fig, _ = _build_figure(one("BAT", "Volt"), ex.frames, ex.units)
        assert fig.axes[0].get_ylabel() == "V"
