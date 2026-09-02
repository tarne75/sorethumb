"""Unit tests for M2: feature construction (encode, scale, correlate, reduce, build)."""

import math
import warnings

import numpy as np
import polars as pl
import pytest

from sorethumb.config import (
    ColumnsConfig,
    Config,
    FeaturesConfig,
    ProfilingConfig,
    RunConfig,
    SourceConfig,
)
from sorethumb.errors import (
    FeatureWidthWarning,
    LowVarianceWarning,
    MemoryBudgetError,
    PlanError,
)
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.features.correlate import correlated_pairs, drop_correlated
from sorethumb.features.encode import (
    _array_derive_exprs,
    _frequency_expr,
    _missing_indicator_expr,
    _one_hot_exprs,
    _time_derivative_exprs,
    compute_demotions,
)
from sorethumb.features.reduce import apply_pca, fit_pca
from sorethumb.features.scale import apply_scaler, fit_scaler
from sorethumb.features.space import FeatureSpace
from sorethumb.profiling.plan import build_feature_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**feature_kwargs) -> Config:
    """Build a minimal Config with optional FeaturesConfig overrides."""
    return Config(
        source=SourceConfig(uri="file://dummy"),
        features=FeaturesConfig(**feature_kwargs),
        run=RunConfig(workdir="/tmp/test"),
    )


def _make_cat_df() -> pl.DataFrame:
    """DataFrame with a mix of categorical, numeric, boolean, and temporal columns."""
    import datetime

    n = 100
    return pl.DataFrame(
        {
            "cat_low": (["a", "b", "c", "d"] * 25),  # 4 categories → one_hot (≤20)
            "cat_high": [f"val_{i}" for i in range(n)],  # 100 categories → frequency
            "num": [float(i) for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
            "ts": [datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i) for i in range(n)],
            "with_nulls": [float(i) if i % 5 != 0 else None for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Encoding expression tests
# ---------------------------------------------------------------------------


def test_one_hot_exprs_basic():
    df = pl.DataFrame({"col": ["a", "b", "c", "a", None]})
    cats = ["a", "b"]
    exprs = _one_hot_exprs("col", cats)
    result = df.select(exprs)
    assert result["col__a"].to_list() == [1, 0, 0, 1, 0]
    assert result["col__b"].to_list() == [0, 1, 0, 0, 0]
    assert result["col____other"].to_list() == [0, 0, 1, 0, 0]


def test_one_hot_null_is_all_zeros():
    df = pl.DataFrame({"col": [None, "a"]})
    cats = ["a"]
    result = df.select(_one_hot_exprs("col", cats))
    # null row → all zeros (not routed to __other)
    assert result["col__a"].to_list() == [0, 1]
    assert result["col____other"].to_list() == [0, 0]


def test_frequency_expr_known_value():
    df = pl.DataFrame({"col": ["a", "b", "c", None]})
    freq_map = {"a": 0.5, "b": 0.3}
    result = df.select(_frequency_expr("col", freq_map))
    assert result["col"].to_list() == pytest.approx([0.5, 0.3, 0.0, 0.0])


def test_frequency_expr_unseen_maps_to_zero():
    df = pl.DataFrame({"col": ["x", "y"]})
    result = df.select(_frequency_expr("col", {"a": 0.9}))
    assert result["col"].to_list() == [0.0, 0.0]


def test_frequency_expr_empty_map():
    df = pl.DataFrame({"col": ["a", "b"]})
    result = df.select(_frequency_expr("col", {}))
    assert result["col"].to_list() == [0.0, 0.0]


def test_missing_indicator_expr():
    df = pl.DataFrame({"x": [1.0, None, 3.0]})
    result = df.select(_missing_indicator_expr("x"))
    assert result["x__is_missing"].to_list() == [0, 1, 0]


def test_time_derivative_exprs_datetime():
    import datetime

    df = pl.DataFrame({"ts": [datetime.datetime(2024, 3, 15, 10, 0, 0)]})
    exprs = _time_derivative_exprs(
        "ts", ["hour", "day", "month", "dayofweek", "year", "quarter"], "Datetime[μs]"
    )
    result = df.select(exprs)
    assert result["ts__hour"][0] == 10
    assert result["ts__day"][0] == 15
    assert result["ts__month"][0] == 3
    assert result["ts__year"][0] == 2024


def test_time_derivative_exprs_date_skips_hour():
    import datetime

    df = pl.DataFrame({"d": [datetime.date(2024, 6, 1)]})
    exprs = _time_derivative_exprs("d", ["hour", "day", "month"], "Date")
    names = [e.meta.output_name() for e in exprs]
    assert "d__hour" not in names
    assert "d__day" in names
    assert "d__month" in names


def test_array_derive_exprs_numeric():
    df = pl.DataFrame({"arr": [[1, 2, 3], [4, 5], None]}, schema={"arr": pl.List(pl.Int64)})
    exprs = _array_derive_exprs("arr", df.schema)
    result = df.select(exprs)
    assert result["arr__len"].to_list() == [3, 2, 0]
    assert result["arr__is_null"].to_list() == [0, 0, 1]
    assert result["arr__is_empty"].to_list() == [0, 0, 0]
    assert result["arr__mean"].to_list() == pytest.approx([2.0, 4.5, None], nan_ok=True)


def test_array_derive_exprs_string_no_stats():
    df = pl.DataFrame({"arr": [["x", "y"], ["z"]]}, schema={"arr": pl.List(pl.String)})
    exprs = _array_derive_exprs("arr", df.schema)
    names = [e.meta.output_name() for e in exprs]
    assert "arr__len" in names
    assert "arr__mean" not in names


# ---------------------------------------------------------------------------
# Width control demotion
# ---------------------------------------------------------------------------


def test_compute_demotions_no_demotion_needed():
    df = _make_cat_df()
    config = _make_config(one_hot_max_cardinality=20, max_feature_width=10_000)
    plan = build_feature_plan(df, config)
    demoted = compute_demotions(plan, 10_000)
    assert demoted == set()


def test_compute_demotions_triggers_demotion():
    df = pl.DataFrame({"cat": [f"c{i}" for i in range(50)] * 2})
    config = _make_config(one_hot_max_cardinality=100, max_feature_width=5)
    plan = build_feature_plan(df, config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        demoted = compute_demotions(plan, 5)
    assert "cat" in demoted
    assert any(isinstance(w.category, type) and issubclass(w.category, FeatureWidthWarning) for w in caught)


def test_demotion_is_deterministic():
    """Same input always produces the same demotion set."""
    df = pl.DataFrame(
        {
            "a": [f"a{i % 30}" for i in range(100)],
            "b": [f"b{i % 25}" for i in range(100)],
        }
    )
    config = _make_config(one_hot_max_cardinality=100, max_feature_width=10)
    plan = build_feature_plan(df, config)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        d1 = compute_demotions(plan, 10)
        d2 = compute_demotions(plan, 10)
    assert d1 == d2


# ---------------------------------------------------------------------------
# Scaler tests
# ---------------------------------------------------------------------------


def test_robust_scaler_center_and_scale():
    df = pl.DataFrame({"x": [float(i) for i in range(101)]})
    params = fit_scaler(df, ["x"], "robust")
    assert "x" in params
    # median of 0..100 is 50.0; IQR of 0..100 is 50.0
    assert params["x"]["center"] == pytest.approx(50.0, abs=1.0)
    assert params["x"]["scale"] == pytest.approx(50.0, abs=1.0)


def test_robust_scaler_zero_iqr_clamped_to_one():
    """Constant column → IQR=0 → clamped to 1.0, so (x - median) / 1.0 = 0."""
    df = pl.DataFrame({"c": [5.0] * 20})
    params = fit_scaler(df, ["c"], "robust")
    assert params["c"]["scale"] == 1.0
    scaled = apply_scaler(df, params, ["c"])
    assert scaled["c"].to_list() == pytest.approx([0.0] * 20)


def test_standard_scaler_zero_std_clamped():
    """Constant column → std=0 → clamped to 1.0."""
    df = pl.DataFrame({"c": [3.0] * 20})
    params = fit_scaler(df, ["c"], "standard")
    assert params["c"]["scale"] == 1.0


def test_apply_scaler_passthrough_unknown_col():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    params = fit_scaler(df.select("a"), ["a"], "robust")
    scaled = apply_scaler(df, params, ["a"])
    # b should be passed through unchanged
    assert "b" in scaled.columns


def test_robust_scaler_finite_after_apply():
    df = pl.DataFrame({"x": [float(i) for i in range(10)]})
    params = fit_scaler(df, ["x"], "robust")
    scaled = apply_scaler(df, params, ["x"])
    assert all(math.isfinite(v) for v in scaled["x"].to_list())


# ---------------------------------------------------------------------------
# Correlation reduction tests
# ---------------------------------------------------------------------------


def test_drop_correlated_removes_duplicate():
    n = 200
    x = np.linspace(0, 1, n)
    df = pl.DataFrame({"a": x, "b": x * 2.0 + 0.001})  # near-perfect correlation
    trimmed, dropped = drop_correlated(df, threshold=0.95)
    assert len(dropped) == 1
    assert "a" in trimmed.columns or "b" in trimmed.columns
    assert not ("a" in trimmed.columns and "b" in trimmed.columns)


def test_drop_correlated_keeps_first_by_order():
    n = 200
    x = np.linspace(0, 1, n)
    df = pl.DataFrame({"first": x, "second": x + 1e-6})
    trimmed, dropped = drop_correlated(df, threshold=0.95)
    assert "first" in trimmed.columns
    assert "second" in dropped


def test_drop_correlated_no_drop_below_threshold():
    rng = np.random.default_rng(42)
    df = pl.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    trimmed, dropped = drop_correlated(df, threshold=0.95)
    assert dropped == []
    assert len(trimmed.columns) == 2


def test_drop_correlated_single_column():
    df = pl.DataFrame({"only": [1.0, 2.0, 3.0]})
    trimmed, dropped = drop_correlated(df, threshold=0.95)
    assert dropped == []
    assert trimmed.columns == ["only"]


def test_correlated_pairs_returns_frame():
    n = 200
    x = np.linspace(0, 1, n)
    df = pl.DataFrame({"a": x, "b": x})
    pairs = correlated_pairs(df, threshold=0.95)
    assert "feature_a" in pairs.columns
    assert "feature_b" in pairs.columns
    assert "pearson_r" in pairs.columns
    assert len(pairs) >= 1


def test_correlated_pairs_empty_below_threshold():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    pairs = correlated_pairs(df, threshold=0.999)
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------


def test_fit_pca_shape():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 10)).astype(np.float64)
    config = FeaturesConfig(pca=True, pca_max_components=5, pca_min_explained_variance=0.0)
    components, mean, evr = fit_pca(matrix, config)
    assert components.shape == (5, 10)
    assert mean.shape == (10,)
    assert evr.shape == (5,)


def test_fit_pca_k_cap():
    """k is capped at n_features - 1."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(50, 4)).astype(np.float64)
    config = FeaturesConfig(pca=True, pca_max_components=100)
    components, _, _ = fit_pca(matrix, config)
    # k = max(1, min(100, 4-1)) = 3
    assert components.shape[0] == 3


def test_fit_pca_low_variance_warns():
    """PCA warns when explained variance is below threshold."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 20)).astype(np.float64)
    # Only 1 component of 20 — will explain far less than 80%
    config = FeaturesConfig(pca=True, pca_max_components=1, pca_min_explained_variance=0.80)
    with pytest.warns(LowVarianceWarning):
        fit_pca(matrix, config)


def test_apply_pca_shape_mismatch_raises():
    components = np.eye(3)  # (3, 3)
    mean = np.zeros(3)
    matrix = np.ones((10, 5))  # n_features=5 ≠ 3
    with pytest.raises(PlanError):
        apply_pca(matrix, components, mean, n_features=3, n_components=3)


def test_apply_pca_roundtrip():
    """apply_pca(fit_pca(X)) has same shape and is a valid projection."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(100, 10)).astype(np.float64)
    config = FeaturesConfig(pca=True, pca_max_components=5, pca_min_explained_variance=0.0)
    components, mean, _ = fit_pca(matrix, config)
    projected = apply_pca(matrix, components, mean, n_features=10, n_components=5)
    assert projected.shape == (100, 5)


# ---------------------------------------------------------------------------
# Full pipeline: fit_features and apply_feature_plan
# ---------------------------------------------------------------------------


def test_fit_features_returns_finite_matrix():
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert np.isfinite(space.matrix).all()


def test_fit_features_stores_scaler_params_in_plan():
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)
    assert len(plan.scaler_params) > 0


def test_fit_features_row_ids_monotonic():
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert list(space.row_ids) == list(range(len(df)))


def test_fit_features_feature_schema_hash_stable():
    df = _make_cat_df()
    config = _make_config()
    plan1 = build_feature_plan(df, config)
    plan2 = build_feature_plan(df, config)
    space1 = fit_features(df, plan1, config)
    space2 = fit_features(df, plan2, config)
    assert space1.feature_schema_hash == space2.feature_schema_hash


def test_apply_feature_plan_same_hash_as_fit():
    """Score-forward: apply_feature_plan produces the same feature_schema_hash as fit."""
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_space = fit_features(df, plan, config)

    # Apply the fitted plan to the same data
    apply_space = apply_feature_plan(df, plan)
    assert apply_space.feature_schema_hash == fit_space.feature_schema_hash


def test_apply_feature_plan_identical_scores_for_identical_record():
    """An identical record scored forward gets an identical feature vector."""
    df = _make_cat_df()
    config = _make_config(correlation_reduction=False, pca=False)
    plan = build_feature_plan(df, config)
    fit_space = fit_features(df, plan, config)
    apply_space = apply_feature_plan(df, plan)
    np.testing.assert_array_almost_equal(fit_space.matrix, apply_space.matrix, decimal=5)


def test_fit_features_with_pca():
    df = _make_cat_df()
    config = _make_config(pca=True, pca_max_components=3, correlation_reduction=False)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    # With PCA on, feature names are pc_0, pc_1, ...
    assert all(n.startswith("pc_") for n in space.feature_names)
    assert space.matrix.shape[1] <= 3


def test_apply_feature_plan_with_pca():
    df = _make_cat_df()
    config = _make_config(pca=True, pca_max_components=3, correlation_reduction=False)
    plan = build_feature_plan(df, config)
    fit_space = fit_features(df, plan, config)
    apply_space = apply_feature_plan(df, plan)
    assert apply_space.feature_schema_hash == fit_space.feature_schema_hash
    np.testing.assert_array_almost_equal(fit_space.matrix, apply_space.matrix, decimal=5)


def test_fit_features_correlation_reduction_drops_correlated():
    n = 100
    x = np.linspace(0.0, 1.0, n)
    df = pl.DataFrame({"a": x, "b": x * 2.0, "c": np.random.default_rng(0).normal(size=n)})
    config = _make_config(correlation_reduction=True, correlation_threshold=0.90)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert len(plan.correlation_drop_list) >= 1
    assert space.matrix.shape[1] < 3  # at least one was dropped


def test_fit_features_scaler_refits_after_corr_drop():
    """When correlation reduction drops a column, scaler is re-fitted on the trimmed set."""
    n = 100
    x = np.linspace(0.0, 1.0, n)
    df = pl.DataFrame({"a": x, "b": x})
    config = _make_config(correlation_reduction=True, correlation_threshold=0.90)
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)
    # After re-fit, scaler_params should only cover kept columns
    kept = [c for c in plan.scaler_params if c not in plan.correlation_drop_list]
    assert len(plan.scaler_params) == len(kept)


def test_fit_features_dtype_float32():
    df = _make_cat_df()
    config = _make_config(dtype="float32")
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert space.matrix.dtype == np.float32


def test_fit_features_dtype_float64():
    df = _make_cat_df()
    config = _make_config(dtype="float64")
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert space.matrix.dtype == np.float64


def test_memory_budget_exceeded_raises():
    n = 100
    df = pl.DataFrame({f"col_{i}": [float(j) for j in range(n)] for i in range(5)})
    # Bypass pydantic ge=256 constraint to test the budget logic with a tiny limit
    run = RunConfig.model_construct(
        workdir="/tmp/test",
        max_memory_mb=0,
        seed=42,
        strict=False,
        max_rows=None,
        reuse_models=False,
        retention_days=90,
        log_level="INFO",
        slow_stage_seconds=300,
    )
    config = Config.model_construct(
        source=SourceConfig(uri="file://dummy"),
        columns=ColumnsConfig(),
        profiling=ProfilingConfig(),
        features=FeaturesConfig(),
        run=run,
        detectors=[],
        scoring=None,
        explain=None,
        history=None,
        report=None,
    )
    plan = build_feature_plan(df, config)
    with pytest.raises(MemoryBudgetError):
        fit_features(df, plan, config)


def test_missing_indicators_emitted():
    df = pl.DataFrame({"num": [1.0, None, None, None, None, 1.0]})
    config = _make_config(missing_indicators=True)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert "num__is_missing" in space.feature_names


def test_missing_indicators_suppressed():
    df = pl.DataFrame({"num": [1.0, None, None, None, None, 1.0]})
    config = _make_config(missing_indicators=False)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert "num__is_missing" not in space.feature_names


def test_one_hot_categories_in_output():
    df = pl.DataFrame({"cat": ["a", "b", "c", "d", "e"] * 20})  # 5 cats, > near_constant_distinct=3
    config = _make_config(one_hot_max_cardinality=20)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert any("cat__" in f for f in space.feature_names)
    assert any("cat____other" in f for f in space.feature_names)


def test_frequency_column_is_single_feature():
    df = pl.DataFrame({"cat": [f"v{i}" for i in range(30)] * 4})
    config = _make_config(one_hot_max_cardinality=5)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    # 30 categories > 5 max → frequency → single feature named 'cat'
    assert "cat" in space.feature_names
    # No one-hot dummies
    assert not any(f.startswith("cat__v") for f in space.feature_names)


def test_feature_space_hash_changes_with_different_features():
    """Different feature sets → different hash."""
    h1 = FeatureSpace.make_hash(["a", "b", "c"])
    h2 = FeatureSpace.make_hash(["a", "b"])
    h3 = FeatureSpace.make_hash(["a", "c", "b"])  # reordered
    assert h1 != h2
    assert h1 != h3


def test_plan_json_roundtrip_after_fit():
    """FeaturePlan round-trips through JSON after M2 fields are populated."""
    df = _make_cat_df()
    config = _make_config(pca=False)
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    json_str = plan.to_json()
    restored = type(plan).from_json(json_str)
    assert restored.scaler_type == plan.scaler_type
    assert restored.output_dtype == plan.output_dtype
    assert set(restored.scaler_params.keys()) == set(plan.scaler_params.keys())
    assert restored.correlation_drop_list == plan.correlation_drop_list
