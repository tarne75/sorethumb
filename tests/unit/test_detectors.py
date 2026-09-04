"""Unit tests for M3: detectors."""

import warnings

import numpy as np
import pytest

from sorethumb.detectors import registry
from sorethumb.detectors._protocol import check_protocol
from sorethumb.detectors.ecod import ECODDetector
from sorethumb.detectors.hbos import HBOSDetector, _auto_bins
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector, _elbow_index
from sorethumb.detectors.lof import LOFDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.errors import DetectorError, SlowStageWarning

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normal_data(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 4)).astype(np.float64)


def _data_with_outliers(n: int = 200, n_outliers: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float64)
    X[:n_outliers] += 10.0
    return X


# ---------------------------------------------------------------------------
# Protocol: check_protocol
# ---------------------------------------------------------------------------


class _GoodDetector:
    name = "good"
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


class _MissingName:
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


class _MissingMethod:
    name = "bad"
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    # natural_flag deliberately missing

    def get_params(self):
        return {}


def test_check_protocol_passes_good_detector():
    check_protocol(_GoodDetector)


def test_check_protocol_raises_missing_name():
    with pytest.raises(DetectorError, match="name"):
        check_protocol(_MissingName)


def test_check_protocol_raises_missing_method():
    with pytest.raises(DetectorError, match="natural_flag"):
        check_protocol(_MissingMethod)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_builtins():
    assert "isolation_forest" in registry
    assert "kmeans_distance" in registry
    assert "one_class_svm" in registry


def test_registry_register_valid():
    from sorethumb.detectors import register

    register(_GoodDetector)
    assert "good" in registry


def test_registry_register_invalid_raises():
    from sorethumb.detectors import register

    with pytest.raises(DetectorError):
        register(_MissingName)


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------


def test_isolation_forest_fit_score_shape():
    X = _normal_data()
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=42)
    scores = det.score_samples(X)
    assert scores.shape == (len(X),)


def test_isolation_forest_scores_are_floats():
    X = _normal_data()
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=42)
    scores = det.score_samples(X)
    assert scores.dtype.kind == "f"


def test_isolation_forest_higher_more_normal():
    X = _data_with_outliers()
    det = IsolationForestDetector(n_estimators=100)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    outlier_mean = scores[:10].mean()
    normal_mean = scores[10:].mean()
    assert normal_mean > outlier_mean, "outliers should score lower than normals"


def test_isolation_forest_natural_flag_shape():
    X = _normal_data()
    det = IsolationForestDetector(n_estimators=50)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (len(X),)
    assert flags.dtype == bool


def test_isolation_forest_natural_flag_flags_outliers():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = IsolationForestDetector(n_estimators=100)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    # Outliers (first 10) should be flagged more than normals
    assert flags[:10].mean() > flags[10:].mean()


def test_isolation_forest_get_params():
    det = IsolationForestDetector(n_estimators=77, max_samples=512)
    params = det.get_params()
    assert params["n_estimators"] == 77
    assert params["max_samples"] == 512


def test_isolation_forest_class_vars():
    assert IsolationForestDetector.name == "isolation_forest"
    assert IsolationForestDetector.supports_tree_shap is True
    assert IsolationForestDetector.default_train_row_cap == 250_000


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


def test_kmeans_fit_score_shape():
    X = _normal_data(n=300)
    det = KMeansDetector(k=3)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (300,)


def test_kmeans_scores_negative_distance():
    X = _normal_data(n=100)
    det = KMeansDetector(k=2)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    # All scores should be <= 0 (negative distance)
    assert (scores <= 0).all()


def test_kmeans_last_labels_populated():
    X = _normal_data(n=100)
    det = KMeansDetector(k=3)
    det.fit(X, seed=0)
    det.score_samples(X)
    assert det.last_labels is not None
    assert det.last_labels.shape == (100,)


def test_kmeans_last_contributions_populated():
    X = _normal_data(n=100)
    det = KMeansDetector(k=3)
    det.fit(X, seed=0)
    det.score_samples(X)
    assert det.last_contributions is not None
    assert det.last_contributions.shape == (100, 4)


def test_kmeans_natural_flag_shape_and_dtype():
    X = _normal_data(n=100)
    det = KMeansDetector(k=2)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (100,)
    assert flags.dtype == bool


def test_kmeans_natural_flag_tukey_outliers():
    # Use k=1 so all points share one centroid: clear outliers have large distances
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 4))
    X[:5] += 20.0  # clear outliers
    det = KMeansDetector(k=1)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    # Outlier flag rate must exceed normal flag rate
    assert flags[:5].mean() > flags[5:].mean(), "clear outliers should be flagged more than normals"


def test_kmeans_auto_k_selection():
    # Should not crash and should select k in [k_min, k_max]
    X = _normal_data(n=300)
    det = KMeansDetector(k_min=2, k_max=5)
    det.fit(X, seed=0)
    assert det._chosen_k is not None
    assert 2 <= det._chosen_k <= 5


def test_kmeans_get_params():
    det = KMeansDetector(k=4, k_min=2, k_max=8, n_init=5)
    params = det.get_params()
    assert params["k_fixed"] == 4
    assert params["k_min"] == 2
    assert params["k_max"] == 8
    assert params["n_init"] == 5


def test_kmeans_class_vars():
    assert KMeansDetector.name == "kmeans_distance"
    assert KMeansDetector.supports_tree_shap is False
    assert KMeansDetector.default_train_row_cap == 200_000


def test_elbow_index_monotone():
    inertias = [100.0, 60.0, 40.0, 35.0, 32.0]
    idx = _elbow_index(inertias)
    # Elbow should be near the first large drop (index 1 or 2)
    assert 0 <= idx < len(inertias)


def test_elbow_index_constant_returns_zero():
    idx = _elbow_index([50.0, 50.0, 50.0])
    assert idx == 0


# ---------------------------------------------------------------------------
# OneClassSVM
# ---------------------------------------------------------------------------


def test_ocsvm_fit_score_shape():
    X = _normal_data(n=200)
    det = OneClassSVMDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (200,)


def test_ocsvm_decision_function_sign():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = OneClassSVMDetector(nu=0.1)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    # Decision function: positive = inlier, negative = outlier
    # Outliers (first 10) should have lower (more negative) scores
    outlier_mean = scores[:10].mean()
    normal_mean = scores[10:].mean()
    assert normal_mean > outlier_mean


def test_ocsvm_natural_flag_shape():
    X = _normal_data(n=100)
    det = OneClassSVMDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (100,)
    assert flags.dtype == bool


def test_ocsvm_natural_flag_below_zero():
    # Manually craft scores to check the threshold
    scores = np.array([-0.5, 0.0, 0.5, -1.0, 1.0])
    det = OneClassSVMDetector()
    flags = det.natural_flag(scores)
    expected = np.array([True, False, False, True, False])
    np.testing.assert_array_equal(flags, expected)


def test_ocsvm_auto_nu_defaults_to_01():
    X = _normal_data(n=100)
    det = OneClassSVMDetector(nu="auto")
    det.fit(X, seed=0)
    assert det._resolved_nu == pytest.approx(0.1)


def test_ocsvm_slow_stage_warning(monkeypatch):
    import time

    import sorethumb.detectors.one_class_svm as _mod

    call_count = 0
    original_monotonic = time.monotonic

    def fast_start():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return 0.0
        return 9999.0  # fake huge elapsed time

    monkeypatch.setattr(_mod.time, "monotonic", fast_start)
    X = _normal_data(n=100)
    det = OneClassSVMDetector(slow_stage_seconds=0.001)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        det.fit(X, seed=0)
    assert any(issubclass(x.category, SlowStageWarning) for x in w)


def test_ocsvm_get_params():
    det = OneClassSVMDetector(nu=0.05, kernel="linear", gamma="auto")
    det.fit(_normal_data(n=100), seed=0)
    params = det.get_params()
    assert params["nu"] == 0.05
    assert params["kernel"] == "linear"
    assert params["gamma"] == "auto"
    assert params["resolved_nu"] == pytest.approx(0.05)


def test_ocsvm_class_vars():
    assert OneClassSVMDetector.name == "one_class_svm"
    assert OneClassSVMDetector.supports_tree_shap is False
    assert OneClassSVMDetector.default_train_row_cap == 25_000


# ---------------------------------------------------------------------------
# ECOD
# ---------------------------------------------------------------------------


def test_ecod_fit_score_shape():
    X = _normal_data(n=200)
    det = ECODDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (200,)


def test_ecod_scores_are_floats():
    X = _normal_data(n=200)
    det = ECODDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.dtype.kind == "f"


def test_ecod_higher_more_normal():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = ECODDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores[10:].mean() > scores[:10].mean(), "outliers should score lower than normals"


def test_ecod_natural_flag_shape_and_dtype():
    X = _normal_data(n=200)
    det = ECODDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (200,)
    assert flags.dtype == bool


def test_ecod_natural_flag_flags_outliers():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = ECODDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags[:10].mean() > flags[10:].mean()


def test_ecod_get_params():
    det = ECODDetector()
    assert det.get_params() == {}


def test_ecod_class_vars():
    assert ECODDetector.name == "ecod"
    assert ECODDetector.supports_tree_shap is False
    assert ECODDetector.default_train_row_cap == 500_000


def test_ecod_score_on_unseen_data():
    rng = np.random.default_rng(7)
    X_train = rng.standard_normal((200, 4))
    X_test = rng.standard_normal((50, 4))
    det = ECODDetector()
    det.fit(X_train, seed=0)
    scores = det.score_samples(X_test)
    assert scores.shape == (50,)


# ---------------------------------------------------------------------------
# LOF
# ---------------------------------------------------------------------------


def test_lof_fit_score_shape():
    X = _normal_data(n=200)
    det = LOFDetector(n_neighbors=10)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (200,)


def test_lof_scores_are_floats():
    X = _normal_data(n=200)
    det = LOFDetector(n_neighbors=10)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.dtype.kind == "f"


def test_lof_higher_more_normal():
    # LOF novelty mode scores *new* points — test on held-out data
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 4))
    det = LOFDetector(n_neighbors=10)
    det.fit(X_train, seed=0)
    X_normal = rng.standard_normal((20, 4))
    X_outlier = rng.standard_normal((20, 4)) + 10.0  # far from training distribution
    assert det.score_samples(X_normal).mean() > det.score_samples(X_outlier).mean()


def test_lof_natural_flag_shape_and_dtype():
    X = _normal_data(n=200)
    det = LOFDetector(n_neighbors=10)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (200,)
    assert flags.dtype == bool


def test_lof_natural_flag_flags_outliers():
    # Score held-out outliers against a model trained on normal data
    rng = np.random.default_rng(5)
    X_train = rng.standard_normal((200, 4))
    det = LOFDetector(n_neighbors=10)
    det.fit(X_train, seed=0)
    X_normal = rng.standard_normal((20, 4))
    X_outlier = rng.standard_normal((20, 4)) + 10.0
    normal_flags = det.natural_flag(det.score_samples(X_normal))
    outlier_flags = det.natural_flag(det.score_samples(X_outlier))
    assert outlier_flags.mean() > normal_flags.mean()


def test_lof_get_params():
    det = LOFDetector(n_neighbors=15)
    assert det.get_params() == {"n_neighbors": 15}


def test_lof_class_vars():
    assert LOFDetector.name == "lof"
    assert LOFDetector.supports_tree_shap is False
    assert LOFDetector.default_train_row_cap == 50_000


# ---------------------------------------------------------------------------
# HBOS
# ---------------------------------------------------------------------------


def test_hbos_fit_score_shape():
    X = _normal_data(n=200)
    det = HBOSDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (200,)


def test_hbos_scores_are_floats():
    X = _normal_data(n=200)
    det = HBOSDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.dtype.kind == "f"


def test_hbos_higher_more_normal():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = HBOSDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores[10:].mean() > scores[:10].mean(), "outliers should score lower than normals"


def test_hbos_natural_flag_shape_and_dtype():
    X = _normal_data(n=200)
    det = HBOSDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags.shape == (200,)
    assert flags.dtype == bool


def test_hbos_natural_flag_flags_outliers():
    X = _data_with_outliers(n=200, n_outliers=10)
    det = HBOSDetector()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    flags = det.natural_flag(scores)
    assert flags[:10].mean() > flags[10:].mean()


def test_hbos_fixed_bins():
    X = _normal_data(n=200)
    det = HBOSDetector(n_bins=20)
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores.shape == (200,)
    assert det.get_params() == {"n_bins": 20}


def test_hbos_get_params_auto():
    det = HBOSDetector()
    assert det.get_params() == {"n_bins": "auto"}


def test_hbos_class_vars():
    assert HBOSDetector.name == "hbos"
    assert HBOSDetector.supports_tree_shap is False
    assert HBOSDetector.default_train_row_cap == 500_000


def test_hbos_score_on_unseen_data():
    rng = np.random.default_rng(99)
    X_train = rng.standard_normal((200, 4))
    X_test = rng.standard_normal((50, 4))
    det = HBOSDetector()
    det.fit(X_train, seed=0)
    scores = det.score_samples(X_test)
    assert scores.shape == (50,)


def test_auto_bins_normal_data():
    rng = np.random.default_rng(0)
    col = rng.standard_normal(1000)
    b = _auto_bins(col)
    assert 10 <= b <= 256


def test_auto_bins_constant_column():
    col = np.ones(500)
    b = _auto_bins(col)
    assert b >= 10


def test_auto_bins_tiny():
    b = _auto_bins(np.array([1.0, 2.0]))
    assert b >= 10


# ---------------------------------------------------------------------------
# Registry: new detectors present
# ---------------------------------------------------------------------------


def test_registry_contains_new_detectors():
    assert "ecod" in registry
    assert "lof" in registry
    assert "hbos" in registry


def test_new_detectors_pass_protocol():
    check_protocol(ECODDetector)
    check_protocol(LOFDetector)
    check_protocol(HBOSDetector)
