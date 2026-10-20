"""KMeans fits are reproducible: the same data and seed give the same centroids every run.

scikit-learn's KMeans reduces partial sums across OpenMP threads in an order that
varies between runs, so unguarded fits on this data differ by a rounding step (on an
8-core machine, every repeat). On data with many duplicate rows that moved whole tied
blocks across the KMeans flag fence: a KDD Cup 1999 backtest flagged up to 90 rows
more or fewer between identical runs. See sorethumb_ml._determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb_ml.detectors.kmeans_distance import KMeansDetector, _select_k
from sorethumb_ml.explain.reference import _background

pytestmark = pytest.mark.unit


def _duplicated_float32(n: int = 2000, d: int = 10) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n // 4, d)).astype(np.float32) * 50
    return np.vstack([base, base, base + 0.001, base * 1.5])


def test_repeated_detector_fits_give_identical_centroids_and_fence() -> None:
    X = _duplicated_float32()
    fits = []
    for _ in range(5):
        det = KMeansDetector(k=6)
        det.fit(X, seed=0)
        fits.append(det)
    first = fits[0]
    for det in fits[1:]:
        assert np.array_equal(det.large_centroids, first.large_centroids)
        assert det._natural_threshold == first._natural_threshold
        assert np.array_equal(det.score_samples(X), first.score_samples(X))


def test_k_selection_is_reproducible() -> None:
    X = _duplicated_float32()
    assert len({_select_k(X, 2, 8, seed=0) for _ in range(4)}) == 1


def test_the_explanation_reference_centres_are_reproducible() -> None:
    X = _duplicated_float32().astype(np.float64)
    runs = [_background(X, 8, seed=0) for _ in range(4)]
    for centres, weights in runs[1:]:
        assert np.array_equal(centres, runs[0][0])
        assert np.array_equal(weights, runs[0][1])
