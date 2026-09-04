"""HBOS — Histogram-Based Outlier Score.

Score per row = −(1/d) * Σ_j log(density_j(x_j))

where density_j is the empirical bin density (count / (n × bin_width)) for
feature j evaluated at value x_j.

HBOS is extremely fast — O(n·d) fit and score — and makes no distributional
assumption beyond that outliers fall in low-density histogram bins. Its key
weakness is the independence assumption: it evaluates each feature in
isolation and cannot detect anomalies that only appear in the joint
distribution (e.g. unusual combinations of otherwise normal values).

Use HBOS when speed matters more than detection precision, or as a
complementary signal alongside detectors that model joint structure.

score_samples() returns the negated HBOS score (higher = more normal).
natural_flag() flags rows whose training-set HBOS score exceeds the 95th
percentile of the training distribution.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


def _auto_bins(col: np.ndarray) -> int:
    """Freedman-Diaconis bin count estimate, clamped to [10, 256]."""
    n = len(col)
    if n < 4:
        return 10
    q25, q75 = np.percentile(col, [25, 75])
    iqr = float(q75 - q25)
    data_range = float(col.max() - col.min())
    if iqr == 0 or data_range == 0:
        return max(10, min(256, int(np.ceil(np.sqrt(n)))))
    bin_width = 2.0 * iqr * n ** (-1.0 / 3.0)
    return max(10, min(256, int(np.ceil(data_range / bin_width))))


class HBOSDetector:
    """Histogram-Based Outlier Score — fast, parameter-free, independence-assumed."""

    name: ClassVar[str] = "hbos"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 500_000

    def __init__(self, n_bins: int | str = "auto") -> None:
        """Initialise with a fixed bin count or 'auto' for Freedman-Diaconis selection."""
        self._n_bins = n_bins
        self._edges: list[np.ndarray] = []
        self._log_densities: list[np.ndarray] = []
        self._score_threshold: float = 0.0

    def fit(self, X: np.ndarray, *, seed: int) -> None:  # noqa: ARG002
        """Build per-feature histograms and establish the natural flag threshold."""
        n, d = X.shape
        self._edges = []
        self._log_densities = []

        for j in range(d):
            col = X[:, j]
            bins = _auto_bins(col) if self._n_bins == "auto" else int(self._n_bins)
            counts, edges = np.histogram(col, bins=bins)
            widths = np.diff(edges)
            widths = np.where(widths < 1e-10, 1e-10, widths)
            densities = counts / (n * widths)
            self._edges.append(edges)
            self._log_densities.append(np.log(np.maximum(densities, 1e-10)))

        logger.info("HBOS: fit on %d rows x %d features.", n, d)
        train_outlier_scores = self._hbos_score(X)
        self._score_threshold = float(np.percentile(train_outlier_scores, 95))

    def _hbos_score(self, X: np.ndarray) -> np.ndarray:
        """Per-row HBOS outlier score. Higher = more anomalous."""
        n_test = X.shape[0]
        d = len(self._edges)
        scores = np.zeros(n_test, dtype=np.float64)

        for j, (edges, log_dens) in enumerate(zip(self._edges, self._log_densities, strict=True)):
            col = X[:, j]
            # Map each value to its bin index, clipped to valid range
            bin_idx = np.clip(
                np.searchsorted(edges[1:-1], col),
                0,
                len(log_dens) - 1,
            )
            scores += -log_dens[bin_idx]

        return scores / d

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more normal (protocol convention)."""
        return -self._hbos_score(X)

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows whose HBOS outlier score exceeds the training 95th percentile."""
        return scores < -self._score_threshold

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        return {"n_bins": self._n_bins}
