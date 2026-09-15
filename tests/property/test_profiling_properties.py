"""Hypothesis property tests for the profiling layer.

Properties are checked at threshold boundaries to catch off-by-one bugs.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sorethumb.config import ColumnsConfig, FeaturesConfig, ProfilingConfig
from sorethumb.profiling.classify import ColumnClass, classify_column, treatment_for
from sorethumb.profiling.profile import profile_columns
from tests.synth import make_frame

pytestmark = pytest.mark.property


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
    assume(n_null <= n_total)
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
    elif n_non_null > 2 and n_non_null <= near_constant_threshold:
        # n_unique in (3..threshold] → near_constant wins before high_null
        assert col_class == ColumnClass.near_constant
    elif null_ratio > threshold:
        # Binary columns (n_non_null == 2) no longer fire near_constant,
        # so high_null can win if the ratio exceeds the drop threshold
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
    assume(n_unique <= n_rows)
    cats = [str(i) for i in range(n_unique)]
    values = [cats[i % n_unique] for i in range(n_rows)]
    df = pl.DataFrame({"s": values})
    cfg = ProfilingConfig(near_constant_distinct=threshold)
    p = profile_columns(df, cfg)[0]
    col_class, _ = classify_column(p, cfg, ColumnsConfig(), set())

    if n_unique == 1:
        assert col_class == ColumnClass.constant
    elif n_unique == 2:
        # Binary columns are exempt from near_constant regardless of threshold
        assert col_class == ColumnClass.categorical
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
    assume(null_count <= n_rows)
    vals: list[float | None] = [None] * null_count + [1.0] * (n_rows - null_count)
    df = pl.DataFrame({"x": vals})
    p = profile_columns(df, ProfilingConfig())[0]
    assert 0.0 <= p.null_ratio <= 1.0
    assert not math.isnan(p.null_ratio)


# ---------------------------------------------------------------------------
# Profiling/treatment decisions are invariant to how the rows are ordered or
# duplicated -- neither should change what a column *is* or how it's treated.
# ---------------------------------------------------------------------------


def _classify_and_treat_every_column(
    df: pl.DataFrame, profiling_cfg: ProfilingConfig, features_cfg: FeaturesConfig
) -> list[tuple[ColumnClass, object]]:
    columns_cfg = ColumnsConfig()
    results = []
    for profile in profile_columns(df, profiling_cfg):
        col_class, _ = classify_column(profile, profiling_cfg, columns_cfg, set())
        results.append((col_class, treatment_for(col_class, profile, features_cfg)))
    return results


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_rows=st.integers(min_value=10, max_value=100),
    shuffle_seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50)
def test_classification_and_treatment_invariant_to_row_permutation(
    seed: int, n_rows: int, shuffle_seed: int
) -> None:
    df, _ = make_frame(
        n_rows=n_rows,
        seed=seed,
        null_ratio=0.1,
        with_low_cardinality_string=True,
        with_boolean=True,
        with_timestamp=True,
    )
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)

    profiling_cfg, features_cfg = ProfilingConfig(), FeaturesConfig()
    original = _classify_and_treat_every_column(df, profiling_cfg, features_cfg)
    permuted = _classify_and_treat_every_column(shuffled, profiling_cfg, features_cfg)

    for name, before, after in zip(df.columns, original, permuted, strict=True):
        assert before == after, f"{name}: (class, treatment) changed under row permutation"


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_rows=st.integers(min_value=10, max_value=100),
)
@settings(max_examples=50)
def test_classification_and_ratios_invariant_to_exact_row_duplication(seed: int, n_rows: int) -> None:
    """Duplicating every row (2x each) doubles the population without
    changing its *shape*: null_ratio is a ratio of two quantities that both
    double in lockstep, so it must be unchanged, and so must the resulting
    classification/treatment. cardinality_ratio is deliberately NOT checked
    here -- exact duplication adds no new distinct values while doubling the
    denominator, so unique/total is mathematically halved by construction,
    not preserved."""
    df, _ = make_frame(
        n_rows=n_rows,
        seed=seed,
        null_ratio=0.1,
        with_low_cardinality_string=True,
        with_boolean=True,
    )
    doubled = pl.concat([df, df])

    profiling_cfg = ProfilingConfig()
    profiles_before = profile_columns(df, profiling_cfg)
    profiles_after = profile_columns(doubled, profiling_cfg)

    features_cfg = FeaturesConfig()
    columns_cfg = ColumnsConfig()
    for name, pa, pb in zip(df.columns, profiles_before, profiles_after, strict=True):
        assert pa.null_ratio == pytest.approx(pb.null_ratio), f"{name}: null_ratio changed on duplication"
        class_a, _ = classify_column(pa, profiling_cfg, columns_cfg, set())
        class_b, _ = classify_column(pb, profiling_cfg, columns_cfg, set())
        assert class_a == class_b, f"{name}: classification changed on duplication"
        assert treatment_for(class_a, pa, features_cfg) == treatment_for(class_b, pb, features_cfg)
