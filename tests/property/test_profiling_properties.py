"""Hypothesis property tests for the profiling layer.

Properties are checked at threshold boundaries to catch off-by-one bugs.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sorethumb.config import ColumnsConfig, ProfilingConfig
from sorethumb.profiling.classify import ColumnClass, classify_column
from sorethumb.profiling.profile import profile_columns


def _classify_float_col(
    values: list[float | None],
    **profiling_overrides: object,
) -> ColumnClass:
    df = pl.DataFrame({"x": values})
    cfg = ProfilingConfig(**profiling_overrides)  # type: ignore[arg-type]
    p = profile_columns(df, cfg)[0]
    col_class, _ = classify_column(p, cfg, ColumnsConfig(), set())
    return col_class


@given(
    n_null=st.integers(min_value=1, max_value=50),
    n_total=st.integers(min_value=2, max_value=50),
    threshold=st.floats(min_value=0.01, max_value=0.99),
)
@settings(max_examples=200)
def test_high_null_iff_ratio_exceeds_threshold(n_null: int, n_total: int, threshold: float) -> None:
    if n_null > n_total:
        return  # invalid, skip
    vals: list[float | None] = [None] * n_null + [float(i) for i in range(n_total - n_null)]
    null_ratio = n_null / n_total
    col_class = _classify_float_col(vals, null_ratio_drop=threshold)

    # Classification priority means constant / near_constant / empty can win
    # before high_null. Our values are [None]*n_null + [0.0, 1.0, ...], so all
    # non-null values are distinct (n_unique_non_null == n_non_null).
    n_non_null = n_total - n_null
    near_constant_threshold = 3  # default ProfilingConfig.near_constant_distinct

    if n_non_null == 0:
        assert col_class == ColumnClass.empty
    elif n_non_null == 1:
        # Only one distinct non-null value → constant wins
        assert col_class == ColumnClass.constant
    elif n_non_null <= near_constant_threshold:
        # n_unique ≤ near_constant_distinct → near_constant wins before high_null
        assert col_class == ColumnClass.near_constant
    elif null_ratio > threshold:
        assert col_class == ColumnClass.high_null
    else:
        assert col_class != ColumnClass.high_null


@given(
    n_unique=st.integers(min_value=1, max_value=10),
    n_rows=st.integers(min_value=20, max_value=100),
    threshold=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=200)
def test_near_constant_iff_n_unique_at_or_below_threshold(n_unique: int, n_rows: int, threshold: int) -> None:
    if n_unique > n_rows:
        return  # invalid
    cats = [str(i) for i in range(n_unique)]
    values = [cats[i % n_unique] for i in range(n_rows)]
    df = pl.DataFrame({"s": values})
    cfg = ProfilingConfig(near_constant_distinct=threshold)
    p = profile_columns(df, cfg)[0]
    col_class, _ = classify_column(p, cfg, ColumnsConfig(), set())

    if n_unique == 1:
        assert col_class == ColumnClass.constant
    elif n_unique <= threshold:
        assert col_class == ColumnClass.near_constant
    else:
        assert col_class not in (ColumnClass.constant, ColumnClass.near_constant)


@given(st.integers(min_value=2, max_value=100))
@settings(max_examples=100)
def test_constant_never_classified_empty(n_rows: int) -> None:
    df = pl.DataFrame({"x": [42.0] * n_rows})
    p = profile_columns(df, ProfilingConfig())[0]
    assert p.is_constant
    assert not p.is_empty


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_all_null_always_empty_not_constant(n_rows: int) -> None:
    df = pl.DataFrame({"x": [None] * n_rows}, schema={"x": pl.Float64})
    p = profile_columns(df, ProfilingConfig())[0]
    assert p.is_empty
    assert not p.is_constant


@given(
    n_rows=st.integers(min_value=10, max_value=200),
    null_count=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=200)
def test_null_ratio_is_in_unit_interval(n_rows: int, null_count: int) -> None:
    if null_count > n_rows:
        return
    vals: list[float | None] = [None] * null_count + [1.0] * (n_rows - null_count)
    df = pl.DataFrame({"x": vals})
    p = profile_columns(df, ProfilingConfig())[0]
    assert 0.0 <= p.null_ratio <= 1.0
    assert not math.isnan(p.null_ratio)
