"""ECOD — Empirical Cumulative distribution functions Outlier Detection.

Score per row = −(1/d) * Σ_j log(min(P_L(x_j), P_R(x_j)))

where P_L(x_j) = P(X_j ≤ x) and P_R(x_j) = P(X_j ≥ x) are the empirical
left- and right-tail probabilities estimated from the training column.

The two-tailed minimum captures outliers in either direction without any
distributional assumption. The log-sum aggregates evidence across features
while staying near-zero for inliers and growing for points that are extreme
in many features simultaneously.

score_samples() returns the negated ECOD score so that higher = more normal,
matching the Detector protocol convention.

natural_flag() uses the 95th-percentile of training scores as the boundary:
approximately 5 % of the training population is naturally flagged, giving
contamination=auto a principled starting estimate.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


class ECODDetector:
    """Empirical-CDF outlier detector — parameter-free and near-linear."""

    name: ClassVar[str] = "ecod"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 500_000

    def __init__(self) -> None:
        """Initialise ECOD — no hyper-parameters required."""
        self._sorted_cols: list[np.ndarray] = []
        self._n_train: int = 0
        self._score_threshold: float = 0.0  # 95th-pct ECOD outlier score on training data

    def fit(self, X: np.ndarray, *, seed: int) -> None:  # noqa: ARG002
        """Store sorted columns for O(log n) ECDF lookup at score time."""
        n, d = X.shape
        self._n_train = n
        self._sorted_cols = [np.sort(X[:, j]) for j in range(d)]
        logger.info("ECOD: fit on %d rows x %d features.", n, d)

        train_outlier_scores = self._ecod_score(X)
        self._score_threshold = float(np.percentile(train_outlier_scores, 95))

    def _ecod_score(self, X: np.ndarray) -> np.ndarray:
        """Per-row ECOD outlier score. Higher = more anomalous."""
        n_test = X.shape[0]
        n = self._n_train
        log_probs = np.zeros(n_test, dtype=np.float64)

        for j, sorted_col in enumerate(self._sorted_cols):
            # Left-tail:  P(X_j ≤ x) — searchsorted 'right' counts values ≤ x
            p_left = np.searchsorted(sorted_col, X[:, j], side="right") / n
            # Right-tail: P(X_j ≥ x) — complement of strictly-less count
            p_right = (n - np.searchsorted(sorted_col, X[:, j], side="left")) / n

            tail = np.minimum(p_left, p_right)
            tail = np.clip(tail, 1e-10, 1.0)
            log_probs += -np.log(tail)

        return log_probs / len(self._sorted_cols)  # normalise by feature count

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more normal (protocol convention)."""
        return -self._ecod_score(X)

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows whose ECOD outlier score exceeds the training 95th percentile."""
        return scores < -self._score_threshold

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters (none for ECOD)."""
        return {}
