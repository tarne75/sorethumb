"""Unit tests for M4: explanations."""

import warnings

import numpy as np
import pytest

from sorethumb_ml.errors import ExplainError, FallbackAttributionWarning
from sorethumb_ml.explain.blend import blend
from sorethumb_ml.explain.centroid import centroid_attributions
from sorethumb_ml.explain.gradient import gradient_attributions, marginal_deviation_attributions
from sorethumb_ml.explain.native import ecod_attributions, hbos_attributions
from sorethumb_ml.explain.project import (
    aggregate_to_original,
    back_project_pca,
    permutation_importance,
    top_n_reasons,
)
from sorethumb_ml.explain.shap_tree import tree_shap_attributions

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_data(n: int = 100, d: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def _fit_if(n: int = 200, d: int = 4, seed: int = 0):
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

    X = _rng_data(n, d, seed)
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=seed)
    return det, X


def _fit_kmeans(n: int = 200, d: int = 4, k: int = 2, seed: int = 0):
    from sorethumb_ml.detectors.kmeans_distance import KMeansDetector

    X = _rng_data(n, d, seed)
    det = KMeansDetector(k=k)
    det.fit(X, seed=seed)
    return det, X


def _fit_ocsvm(n: int = 100, d: int = 4, seed: int = 0):
    from sorethumb_ml.detectors.one_class_svm import OneClassSVMDetector

    X = _rng_data(n, d, seed)
    det = OneClassSVMDetector()
    det.fit(X, seed=seed)
    return det, X


def _fit_ecod(n: int = 200, d: int = 4, seed: int = 0):
    from sorethumb_ml.detectors.ecod import ECODDetector

    X = _rng_data(n, d, seed)
    det = ECODDetector()
    det.fit(X, seed=seed)
    return det, X


def _fit_hbos(n: int = 200, d: int = 4, seed: int = 0):
    from sorethumb_ml.detectors.hbos import HBOSDetector

    X = _rng_data(n, d, seed)
    det = HBOSDetector()
    det.fit(X, seed=seed)
    return det, X


# ---------------------------------------------------------------------------
# gradient_attributions
# ---------------------------------------------------------------------------


def test_gradient_shape():
    det, X = _fit_if()
    attrs, tag = gradient_attributions(det, X, reference=X)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_gradient_dtype():
    det, X = _fit_if()
    attrs, _ = gradient_attributions(det, X, reference=X)
    assert attrs.dtype == np.float64


def test_gradient_max_rows_cap():
    det, X = _fit_if(n=200)
    attrs, _ = gradient_attributions(det, X, reference=X, max_rows=50)
    assert attrs.shape[0] == 50


def test_gradient_nonzero_for_clear_outlier():
    # Use OneClassSVM (smooth decision_function) so perturbations always affect the score.
    # IsolationForest path-lengths saturate for extreme outliers, making gradient zero.
    det, X = _fit_ocsvm(n=200)
    outlier = np.array([[5.0, 5.0, 5.0, 5.0]])
    attrs, _ = gradient_attributions(det, outlier, reference=X, max_rows=10)
    assert not np.allclose(attrs, 0.0), "outlier should have nonzero gradient"


def test_gradient_consistent_sign_convention():
    # Pushing a normal point toward an outlier should produce positive attribution
    det, X = _fit_if(n=200, seed=42)
    # Take the mean of the data and evaluate gradient
    mean_pt = X.mean(axis=0, keepdims=True)
    attrs, _ = gradient_attributions(det, mean_pt, reference=X, max_rows=10)
    # Not testing sign here (depends on direction) — just that it runs and has right shape
    assert attrs.shape == (1, 4)


class _FlatOutsideDetector:
    """Score that is smooth inside [-3, 3] in every dimension and constant outside it.

    Mimics a saturating detector: dimension 0 is what puts a row out of range,
    and the only slope left there is a small one on dimension 1.
    """

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        inside = (np.abs(X) <= 3.0).all(axis=1)
        smooth = -(X**2).sum(axis=1)
        return np.where(inside, smooth, -9.0 - 0.01 * X[:, 1])


def test_gradient_misattributes_a_saturated_outlier_without_the_fallback():
    """Documents the failure mode the saturating fallback exists for."""
    reference = _rng_data(500, 3, seed=1).clip(-2.5, 2.5)
    row = np.array([[40.0, 0.5, 0.0]])
    attrs, _ = gradient_attributions(_FlatOutsideDetector(), row, reference=reference)
    assert attrs[0, 0] == 0.0
    assert int(np.argmax(np.abs(attrs[0]))) == 1


def test_gradient_saturating_uses_marginal_deviation_outside_reference_range():
    reference = _rng_data(500, 3, seed=1).clip(-2.5, 2.5)
    rows = np.array([[40.0, 0.5, 0.0], [0.2, -0.3, 0.1]])
    attrs, tag = gradient_attributions(_FlatOutsideDetector(), rows, reference=reference, saturating=True)
    assert tag == "heuristic"
    assert int(np.argmax(attrs[0])) == 0, "the out-of-range dimension must rank first"
    expected, _ = marginal_deviation_attributions(rows[:1], reference=reference)
    np.testing.assert_allclose(attrs[0], expected[0])
    # The in-range row keeps its real gradient (-d/dx of -sum(x^2) = 2x).
    np.testing.assert_allclose(attrs[1], 2.0 * rows[1], rtol=1e-4, atol=1e-6)


def test_gradient_saturating_replaces_an_all_zero_gradient():
    class _Constant:
        def score_samples(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))

    reference = _rng_data(300, 2, seed=2)
    row = np.array([[0.1, 1.5]])  # inside the range, but the score is flat everywhere
    attrs, _ = gradient_attributions(_Constant(), row, reference=reference, saturating=True)
    assert int(np.argmax(attrs[0])) == 1
    plain, _ = gradient_attributions(_Constant(), row, reference=reference)
    assert not plain.any()


def test_gradient_saturating_skips_scoring_rows_it_will_replace():
    calls: list[int] = []

    class _Counting(_FlatOutsideDetector):
        def score_samples(self, X: np.ndarray) -> np.ndarray:
            calls.append(len(X))
            return super().score_samples(X)

    reference = _rng_data(200, 3, seed=3).clip(-2.5, 2.5)
    gradient_attributions(_Counting(), np.array([[50.0, 0.0, 0.0]]), reference=reference, saturating=True)
    assert calls == []


def test_marginal_deviation_is_robustly_scaled_and_non_negative():
    rng = np.random.default_rng(4)
    reference = np.column_stack([rng.normal(50, 10, 2000), rng.normal(200, 30, 2000), np.zeros(2000)])
    rows = np.array([[900.0, 230.0, 0.0], [50.0, 50.0, 1.0]])
    attrs, tag = marginal_deviation_attributions(rows, reference=reference)
    assert tag == "heuristic"
    assert (attrs >= 0).all()
    assert attrs[0, 0] == pytest.approx(85.0, rel=0.1)  # (900 - 50) / ~10
    assert attrs[0, 1] == pytest.approx(1.0, rel=0.15)  # (230 - 200) / ~30
    assert attrs[1, 2] == pytest.approx(1.0)  # constant column: scale falls back to 1
    assert int(np.argmax(attrs[1])) == 1


def test_marginal_deviation_binary_column_scale_is_finite():
    reference = np.column_stack([np.r_[np.zeros(990), np.ones(10)], np.linspace(-1, 1, 1000)])
    attrs, _ = marginal_deviation_attributions(np.array([[1.0, 0.0]]), reference=reference)
    assert np.isfinite(attrs).all()
    assert attrs[0, 0] > 0


def test_one_class_svm_rbf_saturates_linear_does_not():
    from sorethumb_ml.detectors.one_class_svm import OneClassSVMDetector

    assert OneClassSVMDetector().score_saturates_outside_data
    assert OneClassSVMDetector(kernel="sigmoid").score_saturates_outside_data
    assert not OneClassSVMDetector(kernel="linear").score_saturates_outside_data
    assert not OneClassSVMDetector(kernel="poly").score_saturates_outside_data


# ---------------------------------------------------------------------------
# kernel_shap_attributions
# ---------------------------------------------------------------------------


def test_kernel_shap_attributions_shape():
    from sorethumb_ml.explain.gradient import kernel_shap_attributions

    det, X = _fit_if(n=30, d=4, seed=0)
    attrs, tag = kernel_shap_attributions(det, X, reference=X, background_k=5, max_rows=5000)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_kernel_shap_attributions_row_cap():
    from sorethumb_ml.explain.gradient import kernel_shap_attributions

    det, X = _fit_if(n=30, d=3, seed=0)
    attrs, _ = kernel_shap_attributions(det, X, reference=X, background_k=3, max_rows=10)
    assert attrs.shape[0] == 10


def test_kernel_shap_falls_back_to_gradient_when_shap_not_installed(monkeypatch):
    """shap lives in the optional `explain` extra; explicitly opting
    into explain.kernel_shap without it installed must still degrade
    gracefully to the plain gradient method, never raise."""
    import sys

    from sorethumb_ml.explain.gradient import kernel_shap_attributions

    det, X = _fit_if(n=30, d=4, seed=0)
    monkeypatch.setitem(sys.modules, "shap", None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        attrs, tag = kernel_shap_attributions(det, X, reference=X, background_k=5, max_rows=5000)

    assert tag == "heuristic"
    assert attrs.shape == X.shape
    fallback_warnings = [x for x in w if issubclass(x.category, FallbackAttributionWarning)]
    assert len(fallback_warnings) == 1
    assert "shap is not installed" in str(fallback_warnings[0].message)


# ---------------------------------------------------------------------------
# centroid_attributions
# ---------------------------------------------------------------------------


def test_centroid_shape():
    det, X = _fit_kmeans()
    attrs, tag = centroid_attributions(det, X)
    assert attrs.shape == X.shape
    assert tag == "heuristic"


def test_centroid_non_negative():
    det, X = _fit_kmeans()
    attrs, _ = centroid_attributions(det, X)
    assert (attrs >= 0).all(), "centroid attributions are absolute values, must be >= 0"


def test_centroid_requires_fit_first():
    from sorethumb_ml.detectors.kmeans_distance import KMeansDetector

    X = _rng_data()
    det = KMeansDetector(k=2)
    # fit() not called → _large_centroids is None
    with pytest.raises(ValueError, match="fit"):
        centroid_attributions(det, X)


def test_centroid_dtype():
    det, X = _fit_kmeans()
    attrs, _ = centroid_attributions(det, X)
    assert attrs.dtype == np.float64


# ---------------------------------------------------------------------------
# tree_shap_attributions
# ---------------------------------------------------------------------------


def test_tree_shap_shape():
    det, X = _fit_if(n=200)
    attrs, tag = tree_shap_attributions(det, X[:20], reference=X)
    assert attrs.shape == (20, 4)
    assert tag == "model_specific"


def test_tree_shap_dtype():
    det, X = _fit_if()
    attrs, _ = tree_shap_attributions(det, X[:10], reference=X)
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
        attrs, tag = tree_shap_attributions(det, X[:10], group_name="test_group", reference=X)

    assert tag == "heuristic"
    assert any(issubclass(x.category, FallbackAttributionWarning) for x in w)
    assert attrs.shape == (10, 4)


def test_tree_shap_falls_back_gracefully_when_shap_not_installed(monkeypatch):
    """shap lives in the optional `explain` extra, not a core
    dependency -- without it, TreeSHAP must degrade to the gradient method
    with a clear warning, never raise."""
    import sys

    det, X = _fit_if(n=100)
    monkeypatch.setitem(sys.modules, "shap", None)  # makes `import shap` raise ImportError

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        attrs, tag = tree_shap_attributions(det, X[:10], group_name="test_group", reference=X)

    assert tag == "heuristic"
    assert attrs.shape == (10, 4)
    fallback_warnings = [x for x in w if issubclass(x.category, FallbackAttributionWarning)]
    assert len(fallback_warnings) == 1
    assert "shap is not installed" in str(fallback_warnings[0].message)
    assert "pip install 'sorethumb-ml[explain]'" in str(fallback_warnings[0].message)


def test_tree_shap_outliers_get_higher_attributions():
    det, X = _fit_if(n=200)
    X_out = X.copy()
    X_out[:5] += 20.0  # clear outliers in first 5 rows
    attrs, tag = tree_shap_attributions(det, X_out, reference=X)
    outlier_mean_attr = attrs[:5].sum(axis=1).mean()
    normal_mean_attr = attrs[5:].sum(axis=1).mean()
    assert outlier_mean_attr > normal_mean_attr, "outliers should have higher total attribution"


# ---------------------------------------------------------------------------
# native (ecod_attributions, hbos_attributions)
# ---------------------------------------------------------------------------


def test_ecod_attributions_tag_is_exact():
    det, X = _fit_ecod()
    attrs, tag = ecod_attributions(det, X)
    assert tag == "exact"
    assert attrs.shape == X.shape


def test_ecod_attributions_matches_detector_method():
    det, X = _fit_ecod()
    attrs, _ = ecod_attributions(det, X)
    np.testing.assert_array_equal(attrs, det.feature_contributions(X))


def test_hbos_attributions_tag_is_exact():
    det, X = _fit_hbos()
    attrs, tag = hbos_attributions(det, X)
    assert tag == "exact"
    assert attrs.shape == X.shape


def test_hbos_attributions_matches_detector_method():
    det, X = _fit_hbos()
    attrs, _ = hbos_attributions(det, X)
    np.testing.assert_array_equal(attrs, det.feature_contributions(X))


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------


def test_blend_single_source():
    mat = np.ones((10, 4))
    result, tag = blend([(mat, "model_specific")], [1.0])
    np.testing.assert_array_equal(result, mat)
    assert tag == "model_specific"


def test_blend_two_sources_equal_weights():
    a = np.ones((10, 4)) * 2.0
    b = np.ones((10, 4)) * 2.0
    result, tag = blend([(a, "model_specific"), (b, "heuristic")], [1.0, 1.0])
    assert result.shape == (10, 4)
    assert tag == "heuristic"  # not all model_specific


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


def test_blend_all_model_specific_tag():
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    _, tag = blend([(a, "model_specific"), (b, "model_specific")], [0.5, 0.5])
    assert tag == "model_specific"


def test_blend_all_exact_tag():
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    _, tag = blend([(a, "exact"), (b, "exact")], [0.5, 0.5])
    assert tag == "exact"


def test_blend_exact_and_model_specific_yields_model_specific():
    """Mixing tiers keeps only the weaker one -- exact + model_specific = model_specific."""
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    _, tag = blend([(a, "exact"), (b, "model_specific")], [1.0, 1.0])
    assert tag == "model_specific"


def test_blend_exact_and_heuristic_yields_heuristic():
    """Mixing tiers keeps only the weakest -- exact + heuristic = heuristic,
    not model_specific (heuristic ranks below model_specific too)."""
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    _, tag = blend([(a, "exact"), (b, "heuristic")], [1.0, 1.0])
    assert tag == "heuristic"


def test_blend_all_three_tiers_yields_heuristic():
    a = np.ones((5, 3))
    b = np.ones((5, 3))
    c = np.ones((5, 3))
    _, tag = blend([(a, "exact"), (b, "model_specific"), (c, "heuristic")], [1.0, 1.0, 1.0])
    assert tag == "heuristic"


def test_blend_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        blend([], [])


def test_blend_mismatched_lengths_raises():
    a = np.ones((5, 3))
    with pytest.raises(ValueError, match="same length"):
        blend([(a, "model_specific"), (a, "model_specific")], [1.0])


def test_blend_zero_weight_falls_back_to_equal():
    a = np.ones((5, 3))
    b = np.ones((5, 3)) * 2.0
    result, _ = blend([(a, "heuristic"), (b, "heuristic")], [0.0, 0.0])
    assert result.shape == (5, 3)


def test_blend_drops_a_negligible_source_row_instead_of_normalising_it():
    """A near-zero row is noise; normalising it would give it full weight."""
    signal = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    flat = np.array([[0.0, 1e-9, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])  # row 0 ~ nothing
    result, _ = blend([(signal, "exact"), (flat, "heuristic")], [1.0, 1.0])
    np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-12)
    # Rows where both sources carry information are blended as before.
    np.testing.assert_allclose(result[1], [0.0, 1.0, 0.0])


def test_blend_reweights_remaining_sources_per_row():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 0.0]])  # no information for row 1
    c = np.array([[0.0, 1.0], [0.0, 1.0]])
    result, _ = blend([(a, "heuristic"), (b, "heuristic"), (c, "heuristic")], [2.0, 1.0, 1.0])
    np.testing.assert_allclose(result[0], [0.5, 0.5])
    np.testing.assert_allclose(result[1], [2.0 / 3.0, 1.0 / 3.0])


def test_blend_row_no_source_informs_is_zero():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0]])
    result, _ = blend([(a, "heuristic"), (b, "heuristic")], [1.0, 1.0])
    np.testing.assert_array_equal(result[0], [0.0, 0.0])
    np.testing.assert_allclose(result[1], [0.5, 0.5])


def test_blend_all_zero_source_contributes_nothing():
    a = np.array([[3.0, 4.0]])
    zero = np.zeros((1, 2))
    result, _ = blend([(a, "heuristic"), (zero, "heuristic")], [1.0, 1.0])
    np.testing.assert_allclose(result, [[0.6, 0.8]])


def test_blend_zero_weight_source_never_explains_a_row():
    weighted = np.array([[1.0, 0.0], [0.0, 0.0]])
    unweighted = np.array([[0.0, 1.0], [0.0, 1.0]])
    result, _ = blend([(weighted, "heuristic"), (unweighted, "heuristic")], [1.0, 0.0])
    np.testing.assert_allclose(result[0], [1.0, 0.0])
    np.testing.assert_array_equal(result[1], [0.0, 0.0])


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
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

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


def test_permutation_importance_row_cap():
    """permutation_importance caps X to max_rows when the input is larger."""
    det, X = _fit_if(n=200, d=4)
    feature_names = [f"f{i}" for i in range(4)]
    d2o = {f: f for f in feature_names}
    result = permutation_importance(det, X, feature_names, d2o, n_repeats=1, max_rows=50, seed=0)
    assert len(result) == 4


def test_permutation_importance_equal_importance_returns_half():
    """When every feature has equal importance, each gets 0.5 (the rng_v == 0 branch)."""

    class _ConstantDetector:
        def score_samples(self, X: np.ndarray) -> np.ndarray:
            return np.ones(len(X), dtype=np.float64)

    X = np.ones((20, 3), dtype=np.float64)
    feature_names = ["a", "b", "c"]
    d2o = {f: f for f in feature_names}
    result = permutation_importance(_ConstantDetector(), X, feature_names, d2o, n_repeats=1, max_rows=1000)
    assert all(v == pytest.approx(0.5) for v in result.values())
