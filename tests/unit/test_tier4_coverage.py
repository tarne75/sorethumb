"""Tier 4 coverage tests — import-time code and uncovered branch paths.

Targets:
  - features/build.py     (75%) reload + _sanitize non-finite path
  - features/correlate.py (76%) reload + _sample_matrix edge cases
  - features/encode.py    (71%) reload + unknown derivative, non-numeric inner,
                                         passthrough/derive_array/indicator_only paths
  - features/reduce.py    (69%) reload + PCA exception path
  - features/scale.py     (85%) reload
  - profiling/profile.py  (68%) reload
  - store/db.py           (66%) reload
  - explain/gradient.py   (61%) reload + kernel_shap_attributions
  - explain/project.py    (83%) reload + permutation_importance row cap / equal scores
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Reloads — cover import-time code across all target modules
# ---------------------------------------------------------------------------


def test_features_build_reexecuted_under_coverage() -> None:
    import sorethumb.features.build as m

    importlib.reload(m)


def test_features_correlate_reexecuted_under_coverage() -> None:
    import sorethumb.features.correlate as m

    importlib.reload(m)


def test_features_encode_reexecuted_under_coverage() -> None:
    import sorethumb.features.encode as m

    importlib.reload(m)


def test_features_reduce_reexecuted_under_coverage() -> None:
    import sorethumb.features.reduce as m

    importlib.reload(m)


def test_features_scale_reexecuted_under_coverage() -> None:
    import sorethumb.features.scale as m

    importlib.reload(m)


def test_profiling_profile_reexecuted_under_coverage() -> None:
    import sorethumb.profiling.profile as m

    importlib.reload(m)


def test_store_db_reexecuted_under_coverage() -> None:
    import sorethumb.store.db as m

    importlib.reload(m)


def test_explain_gradient_reexecuted_under_coverage() -> None:
    import sorethumb.explain.gradient as m

    importlib.reload(m)


def test_explain_project_reexecuted_under_coverage() -> None:
    import sorethumb.explain.project as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# features/build.py — _sanitize with non-finite values
# ---------------------------------------------------------------------------


def test_sanitize_replaces_nan_with_zero() -> None:
    """_sanitize emits NonFiniteWarning and zeros out NaN/Inf values."""
    from sorethumb.errors import NonFiniteWarning
    from sorethumb.features.build import _sanitize

    matrix = np.array([[1.0, float("nan"), float("inf")]], dtype=np.float32)
    with pytest.warns(NonFiniteWarning, match="non-finite"):
        result = _sanitize(matrix, "float32")
    assert np.all(np.isfinite(result))
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[0, 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# features/correlate.py — _sample_matrix edge cases
# ---------------------------------------------------------------------------


def test_drop_correlated_empty_df() -> None:
    """_sample_matrix returns None for empty frame → drop_correlated returns as-is."""
    from sorethumb.features.correlate import drop_correlated

    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)})
    result, dropped = drop_correlated(df, threshold=0.95)
    assert dropped == []


def test_correlated_pairs_empty_df() -> None:
    """correlated_pairs on empty frame returns empty result frame."""
    from sorethumb.features.correlate import correlated_pairs

    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64), "b": pl.Series([], dtype=pl.Float64)})
    result = correlated_pairs(df, threshold=0.95)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# features/encode.py — uncovered branch paths
# ---------------------------------------------------------------------------


def test_time_derivative_exprs_unknown_derivative() -> None:
    """Unknown derivative name triggers logger.warning and is skipped."""
    import logging

    from sorethumb.features.encode import _time_derivative_exprs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = _time_derivative_exprs("ts_col", ["hour", "no_such_deriv", "month"], "Datetime")
    # "no_such_deriv" should be skipped — only hour and month produce exprs
    assert len(exprs) == 2


def test_array_derive_exprs_string_inner_no_numeric_stats() -> None:
    """List(String) inner type → no mean/min/max exprs added (non-numeric path)."""
    from sorethumb.features.encode import _array_derive_exprs

    schema = pl.Schema({"tags": pl.List(pl.String)})
    exprs = _array_derive_exprs("tags", schema)
    expr_names = [str(e) for e in exprs]
    # Only __len, __is_null, __is_empty — no __mean/__min/__max
    assert len(exprs) == 3


def test_compute_demotions_break_path() -> None:
    """Demotion stops (break) once width fits without demoting remaining columns."""
    from sorethumb.profiling.classify import ColumnClass, Treatment
    from sorethumb.profiling.plan import ColumnDecision, FeaturePlan

    # 3 one-hot columns with 5, 3, 3 categories each → width = (5+1) + (3+1) + (3+1) = 14
    # Set max_feature_width = 8 → first demotion (5 cats) brings 14 - 5 = 9, still > 8
    # Second demotion (3 cats) brings 9 - 3 = 6 ≤ 8 → break triggered for third column

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        demoted = importlib.import_module("sorethumb.features.encode").compute_demotions(
            plan, max_feature_width=8
        )

    # col_a and col_b demoted (needed to get under 8), col_c should be kept (break triggers)
    assert "col_a" in demoted
    assert "col_b" in demoted
    assert "col_c" not in demoted


def test_build_encoding_exprs_passthrough_treatment() -> None:
    """Treatment.passthrough emits a simple col alias expression."""
    from sorethumb.features.encode import build_encoding_exprs
    from sorethumb.profiling.classify import ColumnClass, Treatment
    from sorethumb.profiling.plan import ColumnDecision, FeaturePlan

    decisions = [
        ColumnDecision(
            column="val",
            col_class=ColumnClass.numeric,
            reason="",
            treatment=Treatment.passthrough,
            emit_missing_indicator=False,
        ),
    ]
    plan = FeaturePlan(
        schema_fingerprint="x",
        n_rows=10,
        decisions=decisions,
        output_features=["val"],
        derived_to_original={"val": "val"},
        one_hot_categories={},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=None,
        time_derivatives=[],
    )
    schema = pl.Schema({"val": pl.Float64})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1


def test_build_encoding_exprs_indicator_only_with_emit() -> None:
    """Treatment.indicator_only with emit_missing_indicator=True emits __is_missing."""
    from sorethumb.features.encode import build_encoding_exprs
    from sorethumb.profiling.classify import ColumnClass, Treatment
    from sorethumb.profiling.plan import ColumnDecision, FeaturePlan

    decisions = [
        ColumnDecision(
            column="sparse",
            col_class=ColumnClass.high_null,
            reason="",
            treatment=Treatment.indicator_only,
            emit_missing_indicator=True,
        ),
    ]
    plan = FeaturePlan(
        schema_fingerprint="x",
        n_rows=10,
        decisions=decisions,
        output_features=["sparse__is_missing"],
        derived_to_original={"sparse__is_missing": "sparse"},
        one_hot_categories={},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=None,
        time_derivatives=[],
    )
    schema = pl.Schema({"sparse": pl.Float64})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1
    assert "__is_missing" in str(exprs[0])


def test_build_encoding_exprs_drop_with_emit() -> None:
    """Treatment.drop with emit_missing_indicator=True emits __is_missing."""
    from sorethumb.features.encode import build_encoding_exprs
    from sorethumb.profiling.classify import ColumnClass, Treatment
    from sorethumb.profiling.plan import ColumnDecision, FeaturePlan

    decisions = [
        ColumnDecision(
            column="id_col",
            col_class=ColumnClass.identifier_like,
            reason="",
            treatment=Treatment.drop,
            emit_missing_indicator=True,
        ),
    ]
    plan = FeaturePlan(
        schema_fingerprint="x",
        n_rows=10,
        decisions=decisions,
        output_features=["id_col__is_missing"],
        derived_to_original={"id_col__is_missing": "id_col"},
        one_hot_categories={},
        frequency_maps={},
        imputation_medians={},
        chosen_time_column=None,
        time_derivatives=[],
    )
    schema = pl.Schema({"id_col": pl.String})
    exprs = build_encoding_exprs(schema, plan, set(), None)
    assert len(exprs) == 1
    assert "__is_missing" in str(exprs[0])


# ---------------------------------------------------------------------------
# features/reduce.py — PCA exception path
# ---------------------------------------------------------------------------


def test_fit_pca_nan_input_raises_plan_error() -> None:
    """fit_pca wraps sklearn ValueError (NaN input) as PlanError."""
    from sorethumb.config import FeaturesConfig
    from sorethumb.errors import PlanError
    from sorethumb.features.reduce import fit_pca

    X_with_nan = np.array([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    cfg = FeaturesConfig(pca=True, pca_max_components=1)
    with pytest.raises(PlanError, match="PCA failed"):
        fit_pca(X_with_nan, cfg, seed=0)


def test_apply_pca_shape_mismatch_raises_plan_error() -> None:
    """apply_pca raises PlanError when components shape doesn't match n_components/n_features."""
    from sorethumb.errors import PlanError
    from sorethumb.features.reduce import apply_pca

    matrix = np.ones((5, 4), dtype=np.float64)
    components = np.ones((3, 4), dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    with pytest.raises(PlanError, match="mismatch"):
        apply_pca(matrix, components, mean, n_features=4, n_components=2)


# ---------------------------------------------------------------------------
# explain/gradient.py — kernel_shap_attributions
# ---------------------------------------------------------------------------


def test_kernel_shap_attributions_shape() -> None:
    """kernel_shap_attributions returns (n_rows, n_features) with correct shape."""
    from sorethumb.explain.gradient import kernel_shap_attributions

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float64)

    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    det = IsolationForestDetector(n_estimators=20)
    det.fit(X, seed=0)

    attrs, tag = kernel_shap_attributions(det, X, background_k=5, max_rows=5000)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_kernel_shap_attributions_row_cap() -> None:
    """kernel_shap_attributions caps rows to max_rows when X is too large."""
    from sorethumb.explain.gradient import kernel_shap_attributions

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3)).astype(np.float64)

    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    det = IsolationForestDetector(n_estimators=10)
    det.fit(X, seed=0)

    attrs, _ = kernel_shap_attributions(det, X, background_k=3, max_rows=10)
    assert attrs.shape[0] == 10


# ---------------------------------------------------------------------------
# explain/project.py — uncovered branches
# ---------------------------------------------------------------------------


def test_permutation_importance_row_cap() -> None:
    """permutation_importance caps X to max_rows when input is larger."""
    from sorethumb.explain.project import permutation_importance

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float64)

    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    det = IsolationForestDetector(n_estimators=20)
    det.fit(X, seed=0)

    feature_names = [f"f{i}" for i in range(4)]
    d2o = {f: f for f in feature_names}

    result = permutation_importance(det, X, feature_names, d2o, n_repeats=1, max_rows=50, seed=0)
    assert isinstance(result, dict)
    assert len(result) == 4


def test_permutation_importance_equal_importance_returns_half() -> None:
    """When all features have equal importance, returns 0.5 for all (rng_v == 0 branch)."""
    from sorethumb.explain.project import permutation_importance

    # Create a detector whose score is constant regardless of input
    class ConstantDetector:
        def score_samples(self, X: np.ndarray) -> np.ndarray:
            return np.ones(len(X), dtype=np.float64)

    X = np.ones((20, 3), dtype=np.float64)
    feature_names = ["a", "b", "c"]
    d2o = {"a": "a", "b": "b", "c": "c"}
    result = permutation_importance(ConstantDetector(), X, feature_names, d2o, n_repeats=1, max_rows=1000)
    assert all(v == pytest.approx(0.5) for v in result.values())


def test_back_project_pca_mismatch_raises() -> None:
    """back_project_pca raises ExplainError when loadings shape is wrong."""
    from sorethumb.errors import ExplainError
    from sorethumb.explain.project import back_project_pca

    contrib = np.ones((5, 3))  # 5 rows, 3 components
    loadings = np.ones((2, 10))  # wrong shape
    with pytest.raises(ExplainError, match="mismatch"):
        back_project_pca(contrib, loadings, n_features=10)
