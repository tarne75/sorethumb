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

from sorethumb.detectors._hyperparams import validate_extra_params

logger = logging.getLogger(__name__)


class ECODDetector:
    """Empirical-CDF outlier detector — parameter-free and near-linear."""

    name: ClassVar[str] = "ecod"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 500_000

    def __init__(self, extra_params: dict[str, Any] | None = None) -> None:
        """Initialise ECOD — no hyper-parameters required.

        extra_params is accepted for API uniformity but must be empty: ECOD has
        no underlying estimator to forward kwargs to.
        """
        validate_extra_params(self.name, None, extra_params, curated=frozenset())
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

    def _per_feature_terms(self, X: np.ndarray) -> np.ndarray:
        """Per-(row, feature) term of the ECOD score, before averaging.

        ``_ecod_score`` and ``feature_contributions`` are both thin wrappers
        over this: the score is ``terms.sum(axis=1) / d`` by definition, so
        this one array is the exact, sole source of truth for both.
        """
        n_test = X.shape[0]
        n = self._n_train
        d = len(self._sorted_cols)
        terms = np.zeros((n_test, d), dtype=np.float64)

        for j, sorted_col in enumerate(self._sorted_cols):
            # Left-tail:  P(X_j ≤ x) — searchsorted 'right' counts values ≤ x
            p_left = np.searchsorted(sorted_col, X[:, j], side="right") / n
            # Right-tail: P(X_j ≥ x) — complement of strictly-less count
            p_right = (n - np.searchsorted(sorted_col, X[:, j], side="left")) / n

            tail = np.minimum(p_left, p_right)
            tail = np.clip(tail, 1e-10, 1.0)
            terms[:, j] = -np.log(tail)

        return terms

    def _ecod_score(self, X: np.ndarray) -> np.ndarray:
        """Per-row ECOD outlier score. Higher = more anomalous."""
        return self._per_feature_terms(X).sum(axis=1) / len(self._sorted_cols)  # normalise by feature count

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Exact per-feature decomposition of the ECOD outlier score.

        Positive = pushes the row toward being anomalous (same convention as
        every other attribution source). Summing across features recovers
        ``_ecod_score(X)`` with zero error — this *is* the score's own
        definition (an average of independent per-feature tail terms), not an
        approximation of it, so "exact" here carries no caveat.
        """
        return self._per_feature_terms(X) / len(self._sorted_cols)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more normal (protocol convention)."""
        return -self._ecod_score(X)

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows whose ECOD outlier score exceeds the training 95th percentile."""
        return scores < -self._score_threshold

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters (none for ECOD)."""
        return {"extra_params": {}}
