"""Unit tests for P0-4: null group handling and typed group-key identity.

``make_group_key`` must key on each column's *typed* value, not a
pre-stringified one -- otherwise ``None``, ``""``, and the literal string
``"None"`` all stringify to ``"None"``/``""`` and collide into the same group.
``_slice_group_frame`` must filter a null group with ``is_null()`` rather than
comparing a Utf8-cast column to the string ``"None"``, which never matches a
real null (a null cast to Utf8 stays null).
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from sorethumb._pipeline import _slice_group_frame
from sorethumb.profiling.plan import FeaturePlan
from sorethumb.store.workspace import make_group_key


def _make_plan(*, chosen_time_column: str | None = None) -> FeaturePlan:
    return FeaturePlan(
        schema_fingerprint="test",
        n_rows=0,
        decisions=[],
        output_features=[],
        derived_to_original={},
        one_hot_categories={},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=chosen_time_column,
        time_derivatives=[],
    )


# ---------------------------------------------------------------------------
# make_group_key: typed identity
# ---------------------------------------------------------------------------


def test_null_empty_string_and_literal_none_produce_distinct_keys():
    keys = {
        make_group_key({"cat": None}),
        make_group_key({"cat": ""}),
        make_group_key({"cat": "None"}),
    }
    assert len(keys) == 3, "null, empty string, and literal 'None' must be distinct group identities"


def test_key_is_stable_regardless_of_value_insertion_order():
    a = make_group_key({"cat": "x", "region": "eu"})
    b = make_group_key({"region": "eu", "cat": "x"})
    assert a == b


def test_key_distinguishes_numeric_types_that_stringify_the_same():
    # 1 (int), 1.0 (float), "1" (str) must not collide just because str(x) == "1".
    keys = {
        make_group_key({"v": 1}),
        make_group_key({"v": 1.0}),
        make_group_key({"v": "1"}),
    }
    assert len(keys) == 3


def test_key_handles_non_json_native_types_like_date():
    # Must not raise, and must not collide with the equivalent ISO string.
    date_key = make_group_key({"d": datetime.date(2024, 1, 15)})
    str_key = make_group_key({"d": "2024-01-15"})
    assert date_key != str_key


def test_key_is_deterministic_across_calls():
    assert make_group_key({"cat": "A", "n": 3}) == make_group_key({"cat": "A", "n": 3})


# ---------------------------------------------------------------------------
# _slice_group_frame: typed comparison + null handling
# ---------------------------------------------------------------------------


def test_slice_group_frame_selects_only_the_null_group():
    df = pl.DataFrame(
        {
            "cat": [None, "", "None", None, "A"],
            "v": [1, 2, 3, 4, 5],
        }
    )
    plan = _make_plan()

    sliced = _slice_group_frame(df, ["cat"], {"cat": None}, plan)

    assert sorted(sliced["v"].to_list()) == [1, 4]


def test_slice_group_frame_distinguishes_empty_string_from_none():
    df = pl.DataFrame(
        {
            "cat": [None, "", "None", "A"],
            "v": [1, 2, 3, 4],
        }
    )
    plan = _make_plan()

    empty_group = _slice_group_frame(df, ["cat"], {"cat": ""}, plan)
    none_literal_group = _slice_group_frame(df, ["cat"], {"cat": "None"}, plan)

    assert empty_group["v"].to_list() == [2]
    assert none_literal_group["v"].to_list() == [3]


@pytest.mark.parametrize(
    ("dtype", "values", "target"),
    [
        (pl.Float64, [1.0, 1.5, 2.0], 1.5),
        (pl.Boolean, [True, False, True], False),
        (pl.Int64, [10, 20, 30], 20),
    ],
)
def test_slice_group_frame_matches_typed_non_string_values(dtype, values, target):
    df = pl.DataFrame({"cat": pl.Series(values, dtype=dtype), "v": list(range(len(values)))})
    plan = _make_plan()

    sliced = _slice_group_frame(df, ["cat"], {"cat": target}, plan)

    expected = [i for i, val in enumerate(values) if val == target]
    assert sliced["v"].to_list() == expected


def test_slice_group_frame_sorts_by_time_column_after_filtering():
    df = pl.DataFrame(
        {
            "cat": ["A", "A", "A"],
            "ts": [3, 1, 2],
            "v": ["third", "first", "second"],
        }
    )
    plan = _make_plan(chosen_time_column="ts")

    sliced = _slice_group_frame(df, ["cat"], {"cat": "A"}, plan)

    assert sliced["v"].to_list() == ["first", "second", "third"]
