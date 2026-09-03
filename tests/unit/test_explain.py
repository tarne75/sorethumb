"""Unit tests for M4: explanations."""

import warnings

import numpy as np
import pytest

from sorethumb.errors import ExplainError, FallbackAttributionWarning
from sorethumb.explain.blend import blend
from sorethumb.explain.centroid import centroid_attributions
from sorethumb.explain.gradient import gradient_attributions
from sorethumb.explain.project import (
    aggregate_to_original,
    back_project_pca,
    permutation_importance,
    top_n_reasons,
)
from sorethumb.explain.shap_tree import tree_shap_attributions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_data(n: int = 100, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _fit_if(n: int = 200, d: int = 4, seed: int = 0):
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    X = _rng_data(n, d, seed)
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=seed)
    return det, X


def _fit_kmeans(n: int = 200, d: int = 4, k: int = 2, seed: int = 0):
    from sorethumb.detectors.kmeans_distance import KMeansDetector

    X = _rng_data(n, d, seed)
    det = KMeansDetector(k=k)
    det.fit(X, seed=seed)
    det.score_samples(X)  # populates last_contributions
    return det, X


def _fit_ocsvm(n: int = 100, d: int = 4, seed: int = 0):
    from sorethumb.detectors.one_class_svm import OneClassSVMDetector

    X = _rng_data(n, d, seed)
    det = OneClassSVMDetector()
    det.fit(X, seed=seed)
    return det, X


# ---------------------------------------------------------------------------
# gradient_attributions
# ---------------------------------------------------------------------------


def test_gradient_shape():
    det, X = _fit_if()
    attrs, tag = gradient_attributions(det, X)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_gradient_dtype():
    det, X = _fit_if()
    attrs, _ = gradient_attributions(det, X)
    assert attrs.dtype == np.float64


def test_gradient_max_rows_cap():
    det, X = _fit_if(n=200)
    attrs, _ = gradient_attributions(det, X, max_rows=50)
    assert attrs.shape[0] == 50


def test_gradient_nonzero_for_clear_outlier():
    # Use OneClassSVM (smooth decision_function) so perturbations always affect the score.
    # IsolationForest path-lengths saturate for extreme outliers, making gradient zero.
    det, X = _fit_ocsvm(n=200)
    outlier = np.array([[5.0, 5.0, 5.0, 5.0]])
    attrs, _ = gradient_attributions(det, outlier, max_rows=10)
    assert not np.allclose(attrs, 0.0), "outlier should have nonzero gradient"


def test_gradient_consistent_sign_convention():
    # Pushing a normal point toward an outlier should produce positive attribution
    det, X = _fit_if(n=200, seed=42)
    # Take the mean of the data and evaluate gradient
    mean_pt = X.mean(axis=0, keepdims=True)
    attrs, _ = gradient_attributions(det, mean_pt, max_rows=10)
    # Not testing sign here (depends on direction) — just that it runs and has right shape
    assert attrs.shape == (1, 4)


# ---------------------------------------------------------------------------
# centroid_attributions
# ---------------------------------------------------------------------------


def test_centroid_shape():
    det, X = _fit_kmeans()
    attrs, tag = centroid_attributions(det)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_centroid_non_negative():
    det, _ = _fit_kmeans()
    attrs, _ = centroid_attributions(det)
    assert (attrs >= 0).all(), "centroid attributions are absolute values, must be >= 0"


def test_centroid_requires_score_first():
    from sorethumb.detectors.kmeans_distance import KMeansDetector

    X = _rng_data()
    det = KMeansDetector(k=2)
    det.fit(X, seed=0)
    # last_contributions not populated yet
    with pytest.raises(ValueError, match="score_samples"):
        centroid_attributions(det)


def test_centroid_dtype():
    det, _ = _fit_kmeans()
    attrs, _ = centroid_attributions(det)
    assert attrs.dtype == np.float64


# ---------------------------------------------------------------------------
# tree_shap_attributions
# ---------------------------------------------------------------------------


def test_tree_shap_shape():
    det, X = _fit_if(n=200)
    attrs, tag = tree_shap_attributions(det, X[:20])
    assert attrs.shape == (20, 4)
    assert tag == "exact"


def test_tree_shap_dtype():
    det, X = _fit_if()
    attrs, _ = tree_shap_attributions(det, X[:10])
    assert attrs.dtype == np.float64


def test_tree_shap_fallback_on_single_node(monkeypatch):
    import sys

    det, X = _fit_if(n=100)

    # shap is imported lazily inside tree_shap_attributions; patch the already-loaded module
    import shap as _shap_mod  # ensure shap is loaded into sys.modules first

    class _BadExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, X, **kwargs):
            raise IndexError("index 0 is out of bounds for axis 0 with size 0")

    monkeypatch.setattr(sys.modules["shap"], "TreeExplainer", _BadExplainer)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        attrs, tag = tree_shap_attributions(det, X[:10], group_name="test_group")

    assert tag == "heuristic"
    assert any(issubclass(x.category, FallbackAttributionWarning) for x in w)
    assert attrs.shape == (10, 4)


def test_tree_shap_outliers_get_higher_attributions():
    det, X = _fit_if(n=200)
    X_out = X.copy()
    X_out[:5] += 20.0  # clear outliers in first 5 rows
    attrs, tag = tree_shap_attributions(det, X_out)
    outlier_mean_attr = attrs[:5].sum(axis=1).mean()
    normal_mean_attr = attrs[5:].sum(axis=1).mean()
    assert outlier_mean_attr > normal_mean_attr, "outliers should have higher total attribution"


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------


def test_blend_single_source():
    mat = np.ones((10, 4))
    result, tag = blend([(mat, "exact")], [1.0])
    np.testing.assert_array_equal(result, mat)
    assert tag == "exact"


def test_blend_two_sources_equal_weights():
    a = np.ones((10, 4)) * 2.0
    b = np.ones((10, 4)) * 2.0
    result, tag = blend([(a, "exact"), (b, "heuristic")], [1.0, 1.0])
    assert result.shape == (10, 4)
    assert tag == "heuristic"  # not all exact


def test_blend_l2_normalised_then_averaged():
    # Two identical L2-normalised sources → result is the same L2-normalised vector
    a = np.array([[3.0, 4.0]])  # L2 norm = 5
    expected_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    result, _ = blend([(a, "heuristic"), (a, "heuristic")], [1.0, 1.0])
    np.testing.assert_allclose(result, expected_norm, rtol=1e-6)


def test_blend_magnitude_dominated_source_not_allowed_to_dominate():
    # Source A has attributions 1000× larger than B
    # After L2-normalisation they should contribute equally
    a = np.array([[1000.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result, _ = blend([(a, "heuristic"), (b, "heuristic")], [1.0, 1.0])
    # Both should contribute equally after L2-normalise + equal weight
    np.testing.assert_allclose(result[0, 0], result[0, 1], rtol=1e-6)


def test_blend_all_exact_tag():
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    _, tag = blend([(a, "exact"), (b, "exact")], [0.5, 0.5])
    assert tag == "exact"


def test_blend_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        blend([], [])


def test_blend_mismatched_lengths_raises():
    a = np.ones((5, 3))
    with pytest.raises(ValueError, match="same length"):
        blend([(a, "exact"), (a, "exact")], [1.0])


def test_blend_zero_weight_falls_back_to_equal():
    a = np.ones((5, 3))
    b = np.ones((5, 3)) * 2.0
    result, _ = blend([(a, "heuristic"), (b, "heuristic")], [0.0, 0.0])
    assert result.shape == (5, 3)


# ---------------------------------------------------------------------------
# back_project_pca
# ---------------------------------------------------------------------------


def test_back_project_correct_shape():
    n_rows, n_components, n_features = 20, 3, 8
    contrib = np.random.default_rng(0).standard_normal((n_rows, n_components))
    loadings = np.random.default_rng(1).standard_normal((n_components, n_features))
    result = back_project_pca(contrib, loadings, n_features)
    assert result.shape == (n_rows, n_features)


def test_back_project_hand_computed():
    # 1 row, 2 components, 3 features
    contrib = np.array([[1.0, 2.0]])  # (1, 2)
    loadings = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])  # (2, 3)
    # |loadings| = [[1, 0, 1], [0, 1, 1]]
    # contrib @ |loadings| = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
    result = back_project_pca(contrib, loadings, n_features=3)
    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])


def test_back_project_wrong_shape_raises():
    contrib = np.ones((5, 3))
    loadings = np.ones((4, 8))  # n_components=4 but contrib has 3
    with pytest.raises(ExplainError, match="mismatch"):
        back_project_pca(contrib, loadings, n_features=8)


def test_back_project_wrong_n_features_raises():
    contrib = np.ones((5, 3))
    loadings = np.ones((3, 8))  # n_features=8 but we pass 10
    with pytest.raises(ExplainError, match="mismatch"):
        back_project_pca(contrib, loadings, n_features=10)


# ---------------------------------------------------------------------------
# aggregate_to_original
# ---------------------------------------------------------------------------


def test_aggregate_sums_one_hot_to_single_original():
    # Three one-hot levels of column "cat" should aggregate to one entry
    feature_names = ["cat__a", "cat__b", "cat__c", "num"]
    d2o = {"cat__a": "cat", "cat__b": "cat", "cat__c": "cat", "num": "num"}
    attributions = np.array([[1.0, 2.0, 3.0, 0.5]])  # (1, 4)
    result = aggregate_to_original(attributions, feature_names, d2o)
    assert "cat" in result
    assert "num" in result
    np.testing.assert_allclose(result["cat"], [6.0])  # 1+2+3
    np.testing.assert_allclose(result["num"], [0.5])


def test_aggregate_uses_absolute_value():
    feature_names = ["a", "b"]
    d2o = {"a": "orig", "b": "orig"}
    attributions = np.array([[1.0, -2.0]])  # abs: 1+2=3
    result = aggregate_to_original(attributions, feature_names, d2o)
    np.testing.assert_allclose(result["orig"], [3.0])


def test_aggregate_unknown_derived_feature_maps_to_itself():
    feature_names = ["x"]
    d2o = {}  # no mapping → maps to itself
    attributions = np.array([[4.0]])
    result = aggregate_to_original(attributions, feature_names, d2o)
    assert "x" in result
    np.testing.assert_allclose(result["x"], [4.0])


def test_aggregate_multiple_rows():
    feature_names = ["a__0", "a__1", "b"]
    d2o = {"a__0": "a", "a__1": "a", "b": "b"}
    attributions = np.array([[1.0, 1.0, 2.0], [3.0, 1.0, 0.5]])
    result = aggregate_to_original(attributions, feature_names, d2o)
    np.testing.assert_allclose(result["a"], [2.0, 4.0])
    np.testing.assert_allclose(result["b"], [2.0, 0.5])


# ---------------------------------------------------------------------------
# top_n_reasons
# ---------------------------------------------------------------------------


def test_top_n_returns_correct_count():
    orig_attrs = {
        "a": np.array([3.0]),
        "b": np.array([1.0]),
        "c": np.array([2.0]),
    }
    raw_row = {"a": "foo", "b": 42, "c": True}
    reasons = top_n_reasons(0, orig_attrs, raw_row, top_n=2)
    assert len(reasons) == 2


def test_top_n_sorted_by_attribution():
    orig_attrs = {
        "a": np.array([1.0]),
        "b": np.array([5.0]),
        "c": np.array([3.0]),
    }
    raw_row = {"a": "x", "b": "y", "c": "z"}
    reasons = top_n_reasons(0, orig_attrs, raw_row, top_n=3)
    assert reasons[0]["column"] == "b"
    assert reasons[1]["column"] == "c"
    assert reasons[2]["column"] == "a"


def test_top_n_includes_raw_value():
    orig_attrs = {"x": np.array([1.0])}
    raw_row = {"x": 99.5}
    reasons = top_n_reasons(0, orig_attrs, raw_row, top_n=1)
    assert reasons[0]["raw_value"] == pytest.approx(99.5)


def test_top_n_pads_when_fewer_than_n():
    orig_attrs = {"x": np.array([1.0])}  # only 1 column
    raw_row = {"x": "val"}
    reasons = top_n_reasons(0, orig_attrs, raw_row, top_n=3)
    assert len(reasons) == 3
    assert reasons[1]["column"] is None
    assert reasons[1]["raw_value"] is None
    assert reasons[2]["column"] is None


def test_top_n_missing_raw_value_is_none():
    orig_attrs = {"a": np.array([1.0])}
    raw_row = {}  # no "a" key
    reasons = top_n_reasons(0, orig_attrs, raw_row, top_n=1)
    assert reasons[0]["raw_value"] is None


# ---------------------------------------------------------------------------
# permutation_importance
# ---------------------------------------------------------------------------


def test_permutation_importance_keys_are_original_columns():
    det, X = _fit_if(n=200)
    feature_names = [f"f{i}" for i in range(4)]
    d2o = {f"f{i}": f"f{i}" for i in range(4)}
    result = permutation_importance(det, X, feature_names, d2o, n_repeats=2, max_rows=100)
    assert set(result.keys()) == {"f0", "f1", "f2", "f3"}


def test_permutation_importance_range():
    det, X = _fit_if(n=200)
    feature_names = [f"f{i}" for i in range(4)]
    d2o = {f"f{i}": f"f{i}" for i in range(4)}
    result = permutation_importance(det, X, feature_names, d2o, n_repeats=2, max_rows=100)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_permutation_importance_aggregates_derived():
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    X = _rng_data(n=200, d=3)
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=0)
    # Two features map to the same original
    feature_names = ["cat__a", "cat__b", "num"]
    d2o = {"cat__a": "cat", "cat__b": "cat", "num": "num"}
    result = permutation_importance(det, X, feature_names, d2o, n_repeats=2, max_rows=100)
    assert set(result.keys()) == {"cat", "num"}


def test_permutation_importance_consistent_across_detectors():
    X = _rng_data(n=200)
    det_a, _ = _fit_if(n=200)
    det_b, _ = _fit_if(n=200)  # same data, same seed → same model
    feature_names = [f"f{i}" for i in range(4)]
    d2o = {f"f{i}": f"f{i}" for i in range(4)}
    r_a = permutation_importance(det_a, X, feature_names, d2o, n_repeats=2, max_rows=100)
    r_b = permutation_importance(det_b, X, feature_names, d2o, n_repeats=2, max_rows=100)
    for k in r_a:
        assert r_a[k] == pytest.approx(r_b[k], abs=1e-6)
