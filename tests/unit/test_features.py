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
    NonFiniteWarning,
    PlanError,
)
from sorethumb.features.build import _sanitize, apply_feature_plan, fit_features
from sorethumb.features.correlate import correlated_pairs, drop_correlated
from sorethumb.features.encode import (
    _array_derive_exprs,
    _frequency_expr,
    _missing_indicator_expr,
    _one_hot_exprs,
    _time_derivative_exprs,
    build_encoding_exprs,
    compute_demotions,
)
from sorethumb.features.reduce import apply_pca, fit_pca
from sorethumb.features.scale import apply_scaler, fit_scaler
from sorethumb.features.space import FeatureSpace
from sorethumb.profiling.classify import ColumnClass, Treatment
from sorethumb.profiling.plan import ColumnDecision, FeaturePlan, build_feature_plan

pytestmark = pytest.mark.unit

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


def test_time_derivative_exprs_unknown_derivative():
    """Unknown derivative name triggers logger.warning and is skipped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = _time_derivative_exprs("ts_col", ["hour", "no_such_deriv", "month"], "Datetime")
    # "no_such_deriv" should be skipped — only hour and month produce exprs
    assert len(exprs) == 2


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


def test_compute_demotions_break_path():
    """Demotion stops (break) once width fits, without demoting remaining columns."""
    # 3 one-hot columns with 5, 3, 3 categories each -> width = (5+1) + (3+1) + (3+1) = 14
    # max_feature_width = 8 -> first demotion (5 cats) brings 14 - 5 = 9, still > 8
    # second demotion (3 cats) brings 9 - 3 = 6 <= 8 -> break triggered for the third column
    cats_a = ["a1", "a2", "a3", "a4", "a5"]
    cats_b = ["b1", "b2", "b3"]
    cats_c = ["c1", "c2", "c3"]

    decisions = [
        ColumnDecision(
            column="col_a",
            col_class=ColumnClass.categorical,
            reason="",
            treatment=Treatment.one_hot,
            emit_missing_indicator=False,
        ),
        ColumnDecision(
            column="col_b",
            col_class=ColumnClass.categorical,
            reason="",
            treatment=Treatment.one_hot,
            emit_missing_indicator=False,
        ),
        ColumnDecision(
            column="col_c",
            col_class=ColumnClass.categorical,
            reason="",
            treatment=Treatment.one_hot,
            emit_missing_indicator=False,
        ),
    ]

    output_features = (
        [f"col_a__{c}" for c in cats_a]
        + ["col_a____other"]
        + [f"col_b__{c}" for c in cats_b]
        + ["col_b____other"]
        + [f"col_c__{c}" for c in cats_c]
        + ["col_c____other"]
    )
    d2o = dict.fromkeys(output_features[:6], "col_a")
    d2o.update(dict.fromkeys(output_features[6:10], "col_b"))
    d2o.update(dict.fromkeys(output_features[10:], "col_c"))

    plan = FeaturePlan(
        schema_fingerprint="abc",
        n_rows=100,
        decisions=decisions,
        output_features=output_features,
        derived_to_original=d2o,
        one_hot_categories={"col_a": cats_a, "col_b": cats_b, "col_c": cats_c},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=None,
        time_derivatives=[],
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        demoted = compute_demotions(plan, max_feature_width=8)

    # col_a and col_b demoted (needed to get under 8); col_c kept (break triggers)
    assert "col_a" in demoted
    assert "col_b" in demoted
    assert "col_c" not in demoted


# ---------------------------------------------------------------------------
# Encoding: build_encoding_exprs by treatment
# ---------------------------------------------------------------------------


def _make_single_column_plan(
    column: str, col_class: ColumnClass, treatment: Treatment, *, emit_missing_indicator: bool
) -> FeaturePlan:
    output_feature = f"{column}__is_missing" if emit_missing_indicator else column
    return FeaturePlan(
        schema_fingerprint="x",
        n_rows=10,
        decisions=[
            ColumnDecision(
                column=column,
                col_class=col_class,
                reason="",
                treatment=treatment,
                emit_missing_indicator=emit_missing_indicator,
            ),
        ],
        output_features=[output_feature],
        derived_to_original={output_feature: column},
        one_hot_categories={},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=None,
        time_derivatives=[],
    )


def test_build_encoding_exprs_passthrough_treatment():
    """Treatment.passthrough emits a simple col alias expression."""
    plan = _make_single_column_plan(
        "val", ColumnClass.numeric, Treatment.passthrough, emit_missing_indicator=False
    )
    schema = pl.Schema({"val": pl.Float64})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1


def test_build_encoding_exprs_indicator_only_with_emit():
    """Treatment.indicator_only with emit_missing_indicator=True emits __is_missing."""
    plan = _make_single_column_plan(
        "sparse", ColumnClass.high_null, Treatment.indicator_only, emit_missing_indicator=True
    )
    schema = pl.Schema({"sparse": pl.Float64})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1
    assert "__is_missing" in str(exprs[0])


def test_build_encoding_exprs_drop_with_emit():
    """Treatment.drop with emit_missing_indicator=True emits __is_missing."""
    plan = _make_single_column_plan(
        "id_col", ColumnClass.identifier_like, Treatment.drop, emit_missing_indicator=True
    )
    schema = pl.Schema({"id_col": pl.String})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1
    assert "__is_missing" in str(exprs[0])


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


def test_robust_scaler_zero_iqr_uses_unit_scale():
    """Constant column → IQR=0 → scale 1.0, so (x - median) / 1.0 = 0."""
    df = pl.DataFrame({"c": [5.0] * 20})
    params = fit_scaler(df, ["c"], "robust")
    assert params["c"]["scale"] == 1.0
    scaled = apply_scaler(df, params, ["c"])
    assert scaled["c"].to_list() == pytest.approx([0.0] * 20)


def test_standard_scaler_zero_std_uses_unit_scale():
    """Constant column → std=0 → scale 1.0."""
    df = pl.DataFrame({"c": [3.0] * 20})
    params = fit_scaler(df, ["c"], "standard")
    assert params["c"]["scale"] == 1.0


def test_robust_scaler_small_nonzero_iqr_not_clamped_up():
    """A genuine sub-1.0 IQR is used as-is, not inflated to 1.0 (which would
    flatten a fine-grained column to near-zero variance)."""
    df = pl.DataFrame({"x": [i * 0.001 for i in range(101)]})  # range 0..0.1
    params = fit_scaler(df, ["x"], "robust")
    assert params["x"]["scale"] == pytest.approx(0.05, abs=5e-3)
    scaled = apply_scaler(df, params, ["x"])
    # Spread is preserved, not flattened. Under the old max(iqr, 1.0) clamp the
    # scaled std would be ~0.029 (divided by 1.0); dividing by the real IQR
    # restores it to order-1.
    assert scaled["x"].std() > 0.3


def test_binary_indicator_column_survives_scaling():
    """A sparse 0/1 indicator (IQR 0) keeps unit scale, not a blow-up."""
    df = pl.DataFrame({"flag": [0.0] * 95 + [1.0] * 5})
    params = fit_scaler(df, ["flag"], "robust")
    assert params["flag"]["scale"] == 1.0
    scaled = apply_scaler(df, params, ["flag"])
    assert max(scaled["flag"].to_list()) == pytest.approx(1.0)


def test_standard_scaler_fit_resists_extreme_rows():
    """Standard-mode mean/std are trimmed, so a few extreme anomalies do not
    inflate the centre or the spread of the normal bulk."""
    normal = [float(i % 10) for i in range(1000)]
    clean = fit_scaler(pl.DataFrame({"x": normal}), ["x"], "standard")
    contaminated = fit_scaler(pl.DataFrame({"x": [*normal, 1e6, 1e6, -1e6]}), ["x"], "standard")
    assert contaminated["x"]["center"] == pytest.approx(clean["x"]["center"], abs=0.5)
    assert contaminated["x"]["scale"] == pytest.approx(clean["x"]["scale"], rel=0.2)


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


def test_drop_correlated_empty_df():
    """An empty frame has no sampleable matrix -> drop_correlated returns it as-is."""
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)})
    _, dropped = drop_correlated(df, threshold=0.95)
    assert dropped == []


def test_correlated_pairs_empty_df():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)})
    result = correlated_pairs(df, threshold=0.95)
    assert len(result) == 0


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


def test_fit_pca_nan_input_raises_plan_error():
    """fit_pca wraps sklearn's ValueError (NaN input) as PlanError."""
    matrix = np.array([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    config = FeaturesConfig(pca=True, pca_max_components=1)
    with pytest.raises(PlanError, match="PCA failed"):
        fit_pca(matrix, config, seed=0)


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
# _sanitize
# ---------------------------------------------------------------------------


def test_sanitize_replaces_nan_with_zero():
    """_sanitize emits NonFiniteWarning and zeros out NaN/Inf values."""
    matrix = np.array([[1.0, float("nan"), float("inf")]], dtype=np.float32)
    with pytest.warns(NonFiniteWarning, match="non-finite"):
        result = _sanitize(matrix, "float32")
    assert np.all(np.isfinite(result))
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[0, 2] == pytest.approx(0.0)


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
    config = _make_config(
        pca=True, pca_max_components=3, pca_min_explained_variance=0.70, correlation_reduction=False
    )
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    # With PCA on, feature names are pc_0, pc_1, ...
    assert all(n.startswith("pc_") for n in space.feature_names)
    assert space.matrix.shape[1] <= 3


def test_apply_feature_plan_with_pca():
    df = _make_cat_df()
    config = _make_config(
        pca=True, pca_max_components=3, pca_min_explained_variance=0.70, correlation_reduction=False
    )
    plan = build_feature_plan(df, config)
    fit_space = fit_features(df, plan, config)
    apply_space = apply_feature_plan(df, plan)
    assert apply_space.feature_schema_hash == fit_space.feature_schema_hash
    np.testing.assert_array_almost_equal(fit_space.matrix, apply_space.matrix, decimal=5)


def test_fit_features_pre_pca_feature_names_no_pca_matches_final_names():
    """With PCA off, the pre-PCA snapshot IS the final feature space."""
    df = _make_cat_df()
    config = _make_config(pca=False)
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)
    assert plan.pre_pca_feature_names == list(space.feature_names)


def test_fit_features_pre_pca_feature_names_with_pca():
    """With PCA on, plan.pre_pca_feature_names is the width/identity PCA was
    actually fit on -- not plan.output_features (pre-demotion, pre-correlation
    -drop) and not the post-PCA pc_i names either."""
    df = _make_cat_df()
    config = _make_config(
        pca=True, pca_max_components=3, pca_min_explained_variance=0.70, correlation_reduction=False
    )
    plan = build_feature_plan(df, config)
    space = fit_features(df, plan, config)

    assert plan.pre_pca_feature_names is not None
    assert not any(n.startswith("pc_") for n in plan.pre_pca_feature_names)
    assert all(n.startswith("pc_") for n in space.feature_names)  # final space IS PCA space
    # This is exactly the width back_project_pca's loadings-shape check needs:
    assert plan.pca_components is not None
    assert len(plan.pre_pca_feature_names) == len(plan.pca_components[0])


def test_apply_feature_plan_with_correlated_columns_dropped():
    """Regression: a column plan.correlation_drop_list removes must not need a
    scaler param (it never reaches the matrix) — apply must not raise for it."""
    df = _make_cat_df()
    config = _make_config(correlation_reduction=True, correlation_threshold=0.5)
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)
    assert plan.correlation_drop_list, "fixture must actually exercise correlation dropping"
    apply_space = apply_feature_plan(df, plan)  # must not raise
    assert not (set(plan.correlation_drop_list) & set(apply_space.feature_names))


def test_fit_features_recomputes_output_features_after_demotion():
    """A demoted column's real output is one frequency feature, not the
    one-hot dummies build_feature_plan planned before demotion ran. Left
    uncorrected, plan.output_features / derived_to_original describe columns
    that no longer exist in the matrix and are missing the one that does."""
    df = pl.DataFrame({"cat": [f"c{i}" for i in range(50)] * 2})
    config = _make_config(one_hot_max_cardinality=100, max_feature_width=5, pca=False)
    plan = build_feature_plan(df, config)
    # Before fit: build_feature_plan only knows the pre-demotion plan (one-hot,
    # cardinality 50 <= one_hot_max_cardinality=100).
    assert any(f.startswith("cat__") for f in plan.output_features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FeatureWidthWarning)
        space = fit_features(df, plan, config)

    assert "cat" in plan.demoted_columns
    # After fit: output_features must describe the matrix fit_features actually
    # produced, not the pre-demotion plan.
    assert plan.output_features == list(space.feature_names)
    assert plan.output_features == ["cat"]
    assert plan.derived_to_original == {"cat": "cat"}


# ---------------------------------------------------------------------------
# apply_feature_plan: schema-drift guard
# ---------------------------------------------------------------------------


def test_apply_feature_plan_raises_on_dtype_drift():
    """A column that changed dtype since the plan was fitted must be rejected,
    not silently mis-encoded or scored unscaled."""
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    drifted = df.with_columns(pl.col("num").cast(pl.Int32))
    with pytest.raises(PlanError, match="schema fingerprint"):
        apply_feature_plan(drifted, plan)


def test_apply_feature_plan_raises_on_column_added():
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    drifted = df.with_columns(pl.lit(1.0).alias("new_col"))
    with pytest.raises(PlanError, match="schema fingerprint"):
        apply_feature_plan(drifted, plan)


def test_apply_feature_plan_raises_on_column_removed():
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    drifted = df.drop("num")
    with pytest.raises(PlanError, match="schema fingerprint"):
        apply_feature_plan(drifted, plan)


def test_apply_feature_plan_same_schema_different_values_is_fine():
    """The whole point of the fingerprint check: identical schema, different
    data (the real score-forward case), must NOT raise."""
    df = _make_cat_df()
    config = _make_config()
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    new_data = df.with_columns((pl.col("num") * 2.0 + 1.0).alias("num"))
    apply_feature_plan(new_data, plan)  # must not raise


# ---------------------------------------------------------------------------
# apply_scaler: missing-parameter guard
# ---------------------------------------------------------------------------


def test_apply_scaler_raises_on_missing_param_for_requested_column():
    """A column the caller asks to scale but has no fitted params for is drift,
    not a no-op: silently leaving it unscaled would let it dominate a distance
    matrix (e.g. a raw `amount` column next to features on a [-3, 3] range)."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
    params = fit_scaler(df.select("a"), ["a"], "robust")  # only "a" has params
    with pytest.raises(PlanError, match=r"\['b'\]"):
        apply_scaler(df, params, ["a", "b"])  # "b" requested but has no params


def test_apply_scaler_raises_when_params_empty_but_columns_requested():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="a"):
        apply_scaler(df, {}, ["a"])


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


def test_feature_space_make_hash_deterministic():
    names = ["amount", "country__US", "country__GB", "hour_of_day"]
    h1 = FeatureSpace.make_hash(names)
    h2 = FeatureSpace.make_hash(names)
    assert h1 == h2
    assert len(h1) == 32


def test_feature_space_make_hash_empty():
    h = FeatureSpace.make_hash([])
    assert isinstance(h, str)
    assert len(h) == 32


def test_feature_space_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(FeatureSpace)
    fields = {f.name for f in dataclasses.fields(FeatureSpace)}
    assert fields == {"matrix", "feature_names", "row_ids", "plan", "feature_schema_hash"}


def test_feature_space_instantiation():
    from unittest.mock import MagicMock

    matrix = np.zeros((10, 3), dtype=np.float32)
    row_ids = np.arange(10)
    plan = MagicMock()
    names = ["f0", "f1", "f2"]
    fs = FeatureSpace(
        matrix=matrix,
        feature_names=names,
        row_ids=row_ids,
        plan=plan,
        feature_schema_hash=FeatureSpace.make_hash(names),
    )
    assert fs.matrix.shape == (10, 3)
    assert fs.feature_names == names
    assert len(fs.feature_schema_hash) == 32


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
