"""
tests/test_extract_columns.py

extract_all() builds one list per field instead of one dict per message
(#18).  These tests pin down the cases where messages of one type don't all
carry the same scalar fields, using fake messages so every case is exact.
"""

import math

import pandas as pd
import pytest

from mavpose.log_extractor import LogExtractor


class FakeMsg:
    def __init__(self, mt, ts, **fields):
        self._type, self._timestamp, self._fields = mt, ts, fields
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self):
        return self._type

    def get_fieldnames(self):
        return list(self._fields)


@pytest.fixture
def extract(tmp_path, monkeypatch):
    def run(messages):
        log = tmp_path / "fake.bin"
        log.write_bytes(b"x")
        ex = LogExtractor(str(log))
        monkeypatch.setattr(ex, "_iter_messages", lambda: iter(messages))
        return ex.extract_all()
    return run


def test_columns_follow_first_appearance(extract):
    frames = extract([FakeMsg("A", 10.0, x=1, y=2), FakeMsg("A", 11.0, x=3, y=4)])
    df = frames["A"]
    assert list(df.columns) == ["time_s", "msg_type", "x", "y"]
    assert df["time_s"].tolist() == [0.0, 1.0]
    assert (df["msg_type"] == "A").all()
    assert df["x"].tolist() == [1, 3]


def test_field_missing_from_a_message_is_nan(extract):
    # A list value is not a scalar, so it is skipped for that message only.
    frames = extract([FakeMsg("A", 1.0, x=1, y=2.5),
                      FakeMsg("A", 2.0, x=2, y=[1, 2]),
                      FakeMsg("A", 3.0, x=3, y=4.5)])
    y = frames["A"]["y"].tolist()
    assert y[0] == 2.5 and math.isnan(y[1]) and y[2] == 4.5


def test_field_first_seen_later_is_backfilled(extract):
    frames = extract([FakeMsg("A", 1.0, x=1),
                      FakeMsg("A", 2.0, x=2),
                      FakeMsg("A", 3.0, x=3, z=9.0)])
    df = frames["A"]
    assert list(df.columns) == ["time_s", "msg_type", "x", "z"]
    assert df["z"].isna().tolist() == [True, True, False]


def test_types_are_kept_apart_and_rows_sorted_by_time(extract):
    frames = extract([FakeMsg("A", 5.0, x=1), FakeMsg("B", 6.0, s="hi"),
                      FakeMsg("A", 4.0, x=0), FakeMsg("B", 7.0, s=b"yo\x00\x00")])
    assert frames["A"]["x"].tolist() == [0, 1]           # sorted by time_s
    assert frames["A"]["time_s"].tolist() == [-1.0, 0.0]  # t0 = first message seen
    assert frames["B"]["s"].tolist() == ["hi", "yo"]      # bytes decoded, NULs stripped


def test_untimed_messages_are_dropped(extract):
    frames = extract([FakeMsg("A", 1.0, x=1), FakeMsg("A", None, x=2), FakeMsg("A", 2.0, x=3)])
    assert frames["A"]["x"].tolist() == [1, 3]


def test_frames_match_a_row_dict_build(extract):
    """Same result as building the frame from one dict per message (the old way)."""
    msgs = [FakeMsg("A", 1.0 + i, x=i, y=(None if i % 3 else i * 0.5), flag=bool(i % 2))
            for i in range(20)]
    df = extract(msgs)["A"]
    rows = [{"time_s": float(i), "msg_type": "A", "x": i,
             "y": (None if i % 3 else i * 0.5), "flag": bool(i % 2)} for i in range(20)]
    expected = pd.DataFrame(rows)
    expected["y"] = pd.to_numeric(expected["y"], errors="coerce")
    pd.testing.assert_frame_equal(df[["time_s", "msg_type", "x", "y"]],
                                  expected[["time_s", "msg_type", "x", "y"]],
                                  check_dtype=False)
    assert df["flag"].astype(bool).tolist() == [bool(i % 2) for i in range(20)]
