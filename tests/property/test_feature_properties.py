"""Hypothesis property tests for feature transforms: fit_features always
preserves row count and produces a fully finite matrix, across the space of
column types/nulls/correlation make_frame can produce.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sorethumb.config import Config, RunConfig, SourceConfig
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.profiling.plan import build_feature_plan
from tests.factories.hypothesis_profiles import scaled_examples
from tests.synth import make_frame

pytestmark = pytest.mark.property

# build_feature_plan/fit_features never touch the filesystem -- workdir is
# just a string field on RunConfig, so a real directory isn't needed here.
_CFG = Config(source=SourceConfig(uri="dummy"), run=RunConfig(workdir="/unused"))


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_rows=st.integers(min_value=10, max_value=150),
    null_ratio=st.floats(min_value=0.0, max_value=0.5),
    with_low_card=st.booleans(),
    with_boolean=st.booleans(),
    with_timestamp=st.booleans(),
    with_correlated=st.booleans(),
)
@settings(max_examples=scaled_examples(100))
def test_fit_features_preserves_row_count_and_is_finite(
    seed: int,
    n_rows: int,
    null_ratio: float,
    with_low_card: bool,
    with_boolean: bool,
    with_timestamp: bool,
    with_correlated: bool,
) -> None:
    df, _ = make_frame(
        n_rows=n_rows,
        seed=seed,
        null_ratio=null_ratio,
        with_low_cardinality_string=with_low_card,
        with_boolean=with_boolean,
        with_timestamp=with_timestamp,
        with_correlated=with_correlated,
    )
    plan = build_feature_plan(df, _CFG)
    space = fit_features(df, plan, _CFG)

    assert space.matrix.shape[0] == len(df)
    assert np.isfinite(space.matrix).all(), "fit_features must never leak NaN/Inf into the matrix"


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_rows=st.integers(min_value=10, max_value=150),
    with_low_card=st.booleans(),
    with_boolean=st.booleans(),
)
@settings(max_examples=scaled_examples(50))
def test_apply_feature_plan_preserves_row_count_and_is_finite(
    seed: int, n_rows: int, with_low_card: bool, with_boolean: bool
) -> None:
    """Same property, through the score-forward path (fit once, apply to a
    fresh frame with the same schema) rather than fit_features."""
    df, _ = make_frame(
        n_rows=n_rows, seed=seed, with_low_cardinality_string=with_low_card, with_boolean=with_boolean
    )
    plan = build_feature_plan(df, _CFG)
    fit_features(df, plan, _CFG)

    fresh, _ = make_frame(
        n_rows=n_rows, seed=seed + 1, with_low_cardinality_string=with_low_card, with_boolean=with_boolean
    )
    space = apply_feature_plan(fresh, plan)

    assert space.matrix.shape[0] == len(fresh)
    assert np.isfinite(space.matrix).all(), "apply_feature_plan must never leak NaN/Inf into the matrix"
