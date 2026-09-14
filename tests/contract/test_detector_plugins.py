"""Detector plugin contract: the ``Detector`` protocol every detector class
must satisfy, the registry every built-in detector is exported through, and
the ``available_extra_params()`` enumeration used by docs and `sorethumb
init`. Per-detector fit/score/natural_flag *behaviour* lives in
tests/unit/detectors/test_detectors.py; this file is about the shape every
plugin -- built-in or third-party -- must have.
"""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb.detectors import registry
from sorethumb.detectors._protocol import check_protocol
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.lof import LOFDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.errors import DetectorError

pytestmark = pytest.mark.contract

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


class _NonCallableFit:
    name = "bad"
    supports_tree_shap = False
    default_train_row_cap = 1000
    fit = "not_a_function"  # non-callable

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


def test_check_protocol_raises_non_callable_method():
    """A method *present* but overridden with a non-callable is a distinct
    failure from a missing method entirely (test_check_protocol_raises_missing_method)."""
    with pytest.raises(DetectorError, match="fit"):
        check_protocol(_NonCallableFit)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_builtins():
    assert "isolation_forest" in registry
    assert "kmeans_distance" in registry
    assert "one_class_svm" in registry


def test_registry_values_are_detector_classes():
    assert registry["isolation_forest"] is IsolationForestDetector
    assert registry["kmeans_distance"] is KMeansDetector
    assert registry["one_class_svm"] is OneClassSVMDetector


def test_registry_register_valid():
    from sorethumb.detectors import register

    register(_GoodDetector)
    assert "good" in registry


def test_registry_register_invalid_raises():
    from sorethumb.detectors import register

    with pytest.raises(DetectorError):
        register(_MissingName)


def test_detectors_all_list():
    """__all__ is the plugin surface: adding a detector means exporting it here."""
    import sorethumb.detectors as det

    expected = {
        "Detector",
        "ECODDetector",
        "HBOSDetector",
        "IsolationForestDetector",
        "KMeansDetector",
        "LOFDetector",
        "OneClassSVMDetector",
        "register",
        "registry",
    }
    assert set(det.__all__) == expected


# ---------------------------------------------------------------------------
# available_extra_params() — enumeration for docs / `sorethumb init`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "curated"),
    [
        (IsolationForestDetector, {"n_estimators", "max_samples"}),
        (LOFDetector, {"n_neighbors"}),
        (KMeansDetector, {"k", "k_min", "k_max", "n_init", "large_cluster_coverage"}),
        (OneClassSVMDetector, {"nu", "kernel", "gamma"}),
    ],
)
def test_available_extra_params_excludes_managed_and_curated(cls, curated):
    keys = set(cls.available_extra_params())
    assert keys, f"{cls.name} should expose at least one extra_params key"
    # Nothing sorethumb manages, and nothing already a wrapper argument.
    assert not (keys & {"random_state", "seed", "contamination", "novelty", "n_clusters"})
    assert not (keys & curated)


@pytest.mark.parametrize(
    "cls",
    [IsolationForestDetector, LOFDetector, KMeansDetector, OneClassSVMDetector],
)
def test_available_extra_params_all_accept_when_passed_back(cls):
    """Every advertised key must actually be accepted as an extra_param."""
    advertised = cls.available_extra_params()
    # Passing the whole set (at its sklearn default) must not raise.
    cls(extra_params=dict(advertised))


def test_available_extra_params_is_sorted():
    keys = list(IsolationForestDetector.available_extra_params())
    assert keys == sorted(keys)
