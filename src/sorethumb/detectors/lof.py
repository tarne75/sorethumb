"""LOF — Local Outlier Factor detector.

sklearn's LocalOutlierFactor in novelty mode. score_samples() returns the
negative LOF score so that higher values correspond to inliers (matching
sklearn's own convention and sorethumb's higher=more-normal contract).

natural_flag() uses sklearn's fitted offset_: the threshold below which a
point is considered anomalous by the model's own local-density criterion.
The default offset_ of approximately −1.5 means a point must have a local
density at least 1.5× lower than its neighbours to be flagged.

LOF is a local method — it flags points whose neighbourhood is far denser
than the point itself, catching anomalies that global methods (IsolationForest,
ECOD) miss when the data has clusters of very different densities. The trade-off
is that it stores the full training kNN graph in memory.

train_row_cap defaults to 50 000. sklearn's BallTree construction is
O(n log²n) and scoring is O(n_test × k × log n_train); beyond 50 k rows the
fit becomes slow without a cap.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


class LOFDetector:
    """sklearn LocalOutlierFactor (novelty=True) wrapped to satisfy the Detector protocol."""

    name: ClassVar[str] = "lof"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 50_000

    def __init__(self, n_neighbors: int = 20) -> None:
        """Initialise with the number of neighbours for local density estimation."""
        self._n_neighbors = n_neighbors
        self._model: Any = None

    def fit(self, X: np.ndarray, *, seed: int) -> None:  # noqa: ARG002
        """Fit LOF on X. seed is accepted for API uniformity but unused."""
        from sklearn.neighbors import LocalOutlierFactor  # noqa: PLC0415

        logger.info(
            "LOF: fitting n_neighbors=%d on %d rows x %d features.",
            self._n_neighbors,
            X.shape[0],
            X.shape[1],
        )
        self._model = LocalOutlierFactor(n_neighbors=self._n_neighbors, novelty=True)
        self._model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return LOF scores. Higher = more normal (sklearn convention, no flip needed)."""
        return self._model.score_samples(X)  # type: ignore[no-any-return]

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows below sklearn's fitted offset_ (default ≈ -1.5 in novelty mode)."""
        return scores < float(self._model.offset_)

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        return {"n_neighbors": self._n_neighbors}
