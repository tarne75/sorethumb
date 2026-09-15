"""Detector plugin contract: the ``Detector`` protocol every detector class
must satisfy, the registry every built-in detector is exported through, the
``available_extra_params()`` enumeration used by docs and `sorethumb init`,
and the per-detector contract table below (fit/score shape, seed-sensitivity,
score orientation, natural-boundary mechanism, training cap, extra-params
support, attribution kind, and a save_model/load_model round-trip). Detector-
*specific* nuance (CBLOF, elbow-k selection, ECOD/HBOS feature contributions,
LOF's small-dataset clamp, ...) lives in tests/unit/detectors/test_detectors.py;
this file is about the shape every plugin -- built-in or third-party -- must
have in common.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sorethumb.detectors import registry
from sorethumb.detectors._protocol import check_protocol
from sorethumb.detectors.ecod import ECODDetector
from sorethumb.detectors.hbos import HBOSDetector
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.lof import LOFDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.errors import DetectorError
from sorethumb.store.models import load_model, plan_digest, save_model
from sorethumb.store.workspace import make_group_key
from tests.factories.workspaces import open_workspace

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


# ---------------------------------------------------------------------------
# The detector contract table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorSpec:
    """One row of the detector plugin contract.

    ``min_rows`` is the smallest sample count every detector is guaranteed to
    fit and score on without raising (verified here, not just claimed);
    ``stochastic`` records whether ``seed`` actually changes the fit (checked
    against the *_hyperparams reserved-key list, not sklearn source-diving);
    ``natural_boundary`` names the mechanism ``natural_flag`` uses;
    ``training_cap`` must match the class's ``default_train_row_cap``;
    ``attribution_kind`` is the tag the explain pipeline routes this detector
    to (see src/sorethumb/explain/{native,shap_tree,centroid,gradient}.py).
    """

    id: str
    cls: type
    min_rows: int
    stochastic: bool
    natural_boundary: str
    training_cap: int
    supports_extra_params: bool
    attribution_kind: str


DETECTOR_SPECS = [
    DetectorSpec(
        id="isolation_forest",
        cls=IsolationForestDetector,
        min_rows=10,
        stochastic=True,
        natural_boundary="fitted_offset",
        training_cap=250_000,
        supports_extra_params=True,
        attribution_kind="model_specific",
    ),
    DetectorSpec(
        id="kmeans_distance",
        cls=KMeansDetector,
        min_rows=10,
        stochastic=True,
        natural_boundary="tukey_fence",
        training_cap=200_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="one_class_svm",
        cls=OneClassSVMDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="zero_hyperplane",
        training_cap=25_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="lof",
        cls=LOFDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="fitted_offset",
        training_cap=50_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="ecod",
        cls=ECODDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="percentile_95",
        training_cap=500_000,
        supports_extra_params=False,
        attribution_kind="exact",
    ),
    DetectorSpec(
        id="hbos",
        cls=HBOSDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="percentile_95",
        training_cap=500_000,
        supports_extra_params=False,
        attribution_kind="exact",
    ),
]

_SPEC_IDS = [s.id for s in DETECTOR_SPECS]


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_class_vars_match_contract_table(spec: DetectorSpec) -> None:
    assert spec.cls.name == spec.id
    assert spec.cls.default_train_row_cap == spec.training_cap
    # ECOD/HBOS don't advertise available_extra_params() at all -- they have no
    # underlying estimator to forward keys to, so its absence *is* "no support".
    advertised = getattr(spec.cls, "available_extra_params", None)
    assert bool(advertised and advertised()) == spec.supports_extra_params


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_fits_and_scores_at_min_rows(spec: DetectorSpec) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((spec.min_rows, 4)).astype(np.float64)
    det = spec.cls()
    det.fit(X, seed=0)  # must not raise at the documented minimum

    scores = det.score_samples(X)
    assert scores.shape == (spec.min_rows,)
    assert scores.dtype.kind == "f"

    flags = det.natural_flag(scores)
    assert flags.shape == (spec.min_rows,)
    assert flags.dtype == bool

    params = det.get_params()
    assert isinstance(params, dict)


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_higher_score_means_more_normal(spec: DetectorSpec) -> None:
    """The shared protocol invariant: score_samples() ranks outliers lower."""
    rng = np.random.default_rng(1)
    n_normal, n_outlier = 40, 10
    X_normal = rng.standard_normal((n_normal, 4))
    X_outlier = rng.standard_normal((n_outlier, 4)) + 10.0
    X = np.vstack([X_normal, X_outlier])

    det = spec.cls()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores[:n_normal].mean() > scores[n_normal:].mean(), (
        f"{spec.id}: outliers must score lower (more anomalous) than normals"
    )


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_seed_sensitivity_matches_contract_table(spec: DetectorSpec) -> None:
    """Same seed -> identical scores always; a different seed changes the fit
    only for detectors the table marks stochastic."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 4)).astype(np.float64)

    det_a = spec.cls()
    det_a.fit(X, seed=0)
    det_b = spec.cls()
    det_b.fit(X, seed=0)
    np.testing.assert_array_equal(det_a.score_samples(X), det_b.score_samples(X))

    det_c = spec.cls()
    det_c.fit(X, seed=123)
    changed = not np.array_equal(det_a.score_samples(X), det_c.score_samples(X))
    assert changed == spec.stochastic, (
        f"{spec.id}: stochastic={spec.stochastic} in the table, but changing the "
        f"seed {'changed nothing' if spec.stochastic else 'changed the fit'}"
    )


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_save_load_model_roundtrip(spec: DetectorSpec, tmp_path) -> None:
    """Every detector -- not just the shipped default three -- must round-trip
    through save_model/load_model with an identical calibrated score."""
    from sorethumb.scoring.calibrate import Calibrator

    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 4)).astype(np.float64)

    det = spec.cls()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    cal = Calibrator()
    cal.fit(scores)

    with open_workspace(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", len(X), 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", "hash_abc", len(X), 0)

        det2, cal2, manifest = load_model(ws, "run1", gk, spec.id, expected_plan_digest=plan_digest("{}"))

    assert det2.name == spec.id
    np.testing.assert_array_equal(det2.score_samples(X), scores)
    np.testing.assert_allclose(cal2.transform(scores), cal.transform(scores))
    assert manifest["detector_name"] == spec.id
