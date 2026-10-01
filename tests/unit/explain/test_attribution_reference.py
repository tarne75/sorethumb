"""Explainers take *target* rows separately from a *reference* population.

Perturbation scales (finite-difference gradients) and the KernelSHAP background
must come from the reference -- the full or normal population the detector was
fitted on -- never from the few flagged rows being explained. Those rows' own
spread is tiny, contaminated by the anomalies, and changes with whichever other
rows happen to be flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb_ml._pipeline import _MIN_NORMAL_REFERENCE_ROWS, _attribution_reference
from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector
from sorethumb_ml.detectors.lof import LOFDetector
from sorethumb_ml.detectors.one_class_svm import OneClassSVMDetector
from sorethumb_ml.errors import ExplainError
from sorethumb_ml.explain import gradient as gradient_mod
from sorethumb_ml.explain.gradient import gradient_attributions, kernel_shap_attributions
from sorethumb_ml.explain.shap_tree import tree_shap_attributions

pytestmark = pytest.mark.unit

_D = 6
_PLANTED_FEATURE = 3


def _population(seed: int, n: int = 400, shift: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normal rows, planted anomaly row, population = both). The anomaly differs ONLY in one feature."""
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((n, _D))
    anomaly = rng.standard_normal((1, _D)) * 0.2  # near the centre everywhere ...
    anomaly[0, _PLANTED_FEATURE] = shift  # ... except the planted feature
    return normal, anomaly, np.vstack([normal, anomaly])


class _RecordingDetector:
    """Linear-ish stub that records every perturbation so the step sizes can be read back."""

    def __init__(self) -> None:
        self.steps: list[np.ndarray] = []

    def score_samples(self, batch: np.ndarray) -> np.ndarray:
        self.steps.append(batch[0] - batch[1])  # = 2h along exactly one dimension
        return batch.sum(axis=1)


# ---------------------------------------------------------------------------
# Planted-feature attribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("detector_cls", [OneClassSVMDetector, LOFDetector])
def test_gradient_attributes_a_planted_feature_when_scaled_by_the_normal_reference(detector_cls, seed):
    normal, anomaly, population = _population(seed)
    det = detector_cls()
    det.fit(population, seed=seed)

    attrs, tag = gradient_attributions(det, anomaly, reference=normal)

    assert tag == "heuristic"
    assert int(np.argmax(np.abs(attrs[0]))) == _PLANTED_FEATURE, attrs[0]
    assert attrs[0, _PLANTED_FEATURE] > 0, "the planted feature must push the row toward anomalous"


@pytest.mark.parametrize("seed", [0, 1])
def test_kernel_shap_attributes_a_planted_feature_against_a_normal_background(seed):
    pytest.importorskip("shap")
    normal, anomaly, population = _population(seed, n=300, shift=8.0)
    det = IsolationForestDetector(n_estimators=100)
    det.fit(population, seed=seed)

    attrs, tag = kernel_shap_attributions(det, anomaly, reference=normal, background_k=10)

    assert tag == "heuristic"
    assert int(np.argmax(np.abs(attrs[0]))) == _PLANTED_FEATURE, attrs[0]
    assert attrs[0, _PLANTED_FEATURE] > 0


# ---------------------------------------------------------------------------
# Targets never define the scale
# ---------------------------------------------------------------------------


def test_perturbation_steps_come_from_the_reference_not_the_targets():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((500, 4)) * np.array([0.5, 1.0, 2.0, 4.0])
    targets = np.full((1, 4), 999.0)  # a lone extreme row: its own std is 0, its own |mean| is huge
    det = _RecordingDetector()

    gradient_attributions(det, targets, reference=reference, step_factor=0.01)

    expected_h = 0.01 * reference.std(axis=0)
    seen_h = np.zeros(4)
    for diff in det.steps:
        d = int(np.flatnonzero(diff)[0])
        seen_h[d] = abs(diff[d]) / 2.0
    np.testing.assert_allclose(seen_h, expected_h)


def test_a_rows_attribution_does_not_depend_on_which_other_rows_are_explained():
    normal, anomaly, population = _population(7)
    det = OneClassSVMDetector()
    det.fit(population, seed=7)
    others = normal[:5] + 3.0  # a batch whose spread would have changed the old, target-derived scale

    alone, _ = gradient_attributions(det, anomaly, reference=normal)
    batched, _ = gradient_attributions(det, np.vstack([anomaly, others]), reference=normal)

    np.testing.assert_array_equal(alone[0], batched[0])


def test_zero_variance_reference_dimension_falls_back_to_a_magnitude_relative_step():
    reference = np.column_stack([np.full(50, 4.0), np.zeros(50), np.linspace(-1, 1, 50)])
    det = _RecordingDetector()
    gradient_attributions(det, np.array([[10.0, 10.0, 10.0]]), reference=reference, step_factor=0.01)
    h = {int(np.flatnonzero(d)[0]): abs(d[np.flatnonzero(d)[0]]) / 2.0 for d in det.steps}
    assert h[0] == pytest.approx(0.01 * 4.0)  # constant non-zero feature: relative to its own magnitude
    assert h[1] == pytest.approx(1e-3)  # all-zero feature: absolute floor
    assert h[2] == pytest.approx(0.01 * reference[:, 2].std())


def test_kernel_shap_background_is_summarised_from_the_reference_not_the_targets(monkeypatch):
    shap = pytest.importorskip("shap")
    normal, anomaly, population = _population(0, n=120)
    det = IsolationForestDetector(n_estimators=20)
    det.fit(population, seed=0)
    seen: dict[str, object] = {}
    real_kmeans = shap.kmeans

    def _spy(data, k, *a, **kw):
        seen["data"], seen["k"] = np.asarray(data), k
        return real_kmeans(data, k, *a, **kw)

    monkeypatch.setattr(shap, "kmeans", _spy)

    kernel_shap_attributions(det, anomaly, reference=normal, background_k=500)

    np.testing.assert_array_equal(seen["data"], normal)  # the reference, whole, and not the target row
    assert seen["k"] == len(normal)  # background_k=500 clamped to the reference's size


def test_kernel_shap_subsamples_a_huge_reference_deterministically(monkeypatch):
    shap = pytest.importorskip("shap")
    cap = gradient_mod._REFERENCE_SAMPLE_CAP
    big = np.random.default_rng(1).standard_normal((cap + 500, 3))
    det = IsolationForestDetector(n_estimators=10)
    det.fit(big[:200], seed=0)
    samples: list[np.ndarray] = []
    real_kmeans = shap.kmeans

    def _spy(data, _k, *a, **kw):
        samples.append(np.asarray(data))
        return real_kmeans(data, 2, *a, **kw)

    monkeypatch.setattr(shap, "kmeans", _spy)
    target = big[:1]
    kernel_shap_attributions(det, target, reference=big, background_k=2)
    kernel_shap_attributions(det, target, reference=big, background_k=2)

    assert samples[0].shape == (cap, 3)
    np.testing.assert_array_equal(samples[0], samples[1])


# ---------------------------------------------------------------------------
# TreeSHAP fallback and API strictness
# ---------------------------------------------------------------------------


def test_tree_shap_gradient_fallback_scales_from_the_reference(monkeypatch):
    import sys

    normal, anomaly, population = _population(0, n=120)
    det = IsolationForestDetector(n_estimators=20)
    det.fit(population, seed=0)
    monkeypatch.setitem(sys.modules, "shap", None)  # shap "not installed" -> gradient fallback
    seen: dict[str, np.ndarray] = {}
    real = gradient_mod.gradient_attributions

    def _spy(detector, X, *, reference, **kw):
        seen["reference"] = reference
        return real(detector, X, reference=reference, **kw)

    monkeypatch.setattr(gradient_mod, "gradient_attributions", _spy)

    with pytest.warns(UserWarning, match="shap is not installed"):
        _, tag = tree_shap_attributions(det, anomaly, reference=normal)

    assert tag == "heuristic"
    np.testing.assert_array_equal(seen["reference"], normal)


def test_reference_is_required_and_validated():
    det = _RecordingDetector()
    x = np.zeros((2, 3))
    with pytest.raises(TypeError):
        gradient_attributions(det, x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        kernel_shap_attributions(det, x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        tree_shap_attributions(det, x)  # type: ignore[call-arg,arg-type]
    with pytest.raises(ExplainError, match="reference has 4 feature"):
        gradient_attributions(det, x, reference=np.zeros((5, 4)))
    with pytest.raises(ExplainError, match="no rows"):
        gradient_attributions(det, x, reference=np.zeros((0, 3)))
    with pytest.raises(ExplainError, match="2-D"):
        gradient_attributions(det, x, reference=np.zeros(3))


# ---------------------------------------------------------------------------
# The pipeline's choice of reference
# ---------------------------------------------------------------------------


def test_attribution_reference_is_the_unflagged_rows():
    X = np.arange(40, dtype=np.float64).reshape(20, 2)
    flagged = np.array([3, 11])
    ref = _attribution_reference(X, flagged)
    assert ref.shape == (18, 2)
    assert not any((ref == X[i]).all(axis=1).any() for i in flagged)


def test_attribution_reference_falls_back_to_the_full_population_when_too_few_rows_are_normal():
    n = _MIN_NORMAL_REFERENCE_ROWS + 3
    X = np.random.default_rng(0).standard_normal((n, 2))
    flagged = np.arange(n - (_MIN_NORMAL_REFERENCE_ROWS - 1))  # leaves 9 unflagged
    assert _attribution_reference(X, flagged) is X
