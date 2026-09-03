"""KMeans distance detector.

Score = negative Euclidean distance to the assigned centroid, so higher (closer to
zero) means more normal (closer to the cluster centre). A single global threshold
is applied after scoring — never per-chunk or per-partition, because a per-partition
threshold flags each partition's own outliers and manufactures false positives when
group sizes differ.

k is selected automatically by combining the elbow criterion (normalised inertia
curve) with the best silhouette score, sampled to 20 000 rows for silhouette because
it is O(n²). See docs/approximations.md.

After scoring, ``last_labels`` and ``last_contributions`` are populated for the
explanation layer (§8.3 centroid attribution).
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)

_SILHOUETTE_SAMPLE = 20_000


class KMeansDetector:
    """sklearn KMeans wrapped to satisfy the Detector protocol."""

    name: ClassVar[str] = "kmeans_distance"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 200_000

    def __init__(
        self,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 10,
        n_init: int = 10,
    ) -> None:
        """Initialise with optional fixed k or auto-selection bounds."""
        self._k_fixed = k
        self._k_min = k_min
        self._k_max = k_max
        self._n_init = n_init
        self._model: Any = None
        self._chosen_k: int | None = None
        self.last_labels: np.ndarray | None = None
        self.last_contributions: np.ndarray | None = None

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        """Select k (if not fixed) and fit KMeans on X."""
        from sklearn.cluster import KMeans  # noqa: PLC0415

        k = self._k_fixed if self._k_fixed is not None else _select_k(X, self._k_min, self._k_max, seed)
        self._chosen_k = k
        logger.info("KMeans: fitting k=%d on %d rows x %d features.", k, X.shape[0], X.shape[1])
        self._model = KMeans(n_clusters=k, n_init=self._n_init, random_state=seed)
        self._model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negative Euclidean distance to assigned centroid. Higher = more normal."""
        labels: np.ndarray = self._model.predict(X)
        centres = self._model.cluster_centers_
        diffs = X - centres[labels]
        distances = np.linalg.norm(diffs, axis=1)
        self.last_labels = labels
        self.last_contributions = diffs  # signed per-dimension contribution to distance
        return -distances

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows whose distance is an outlier by Tukey's method (1.5 × IQR fence)."""
        distances = -scores
        q25 = float(np.percentile(distances, 25))
        q75 = float(np.percentile(distances, 75))
        upper_fence = q75 + 1.5 * (q75 - q25)
        return distances > upper_fence

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        return {
            "k_fixed": self._k_fixed,
            "k_min": self._k_min,
            "k_max": self._k_max,
            "chosen_k": self._chosen_k,
            "n_init": self._n_init,
        }


# ---------------------------------------------------------------------------
# k-selection
# ---------------------------------------------------------------------------


def _select_k(X: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    """Choose k by combining normalised-elbow and silhouette criteria."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    ks = list(range(k_min, min(k_max, len(X) - 1) + 1))
    if len(ks) == 1:
        return ks[0]

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed)
        km.fit(X)
        inertias.append(float(km.inertia_))

        sample = (
            X
            if len(X) <= _SILHOUETTE_SAMPLE
            else X[np.random.default_rng(seed).choice(len(X), _SILHOUETTE_SAMPLE, replace=False)]
        )
        labels = km.predict(sample)
        silhouettes.append(float(silhouette_score(sample, labels)))
        logger.debug("k=%d inertia=%.2f silhouette=%.4f", k, inertias[-1], silhouettes[-1])

    elbow_k = ks[_elbow_index(inertias)]
    sil_k = ks[int(np.argmax(silhouettes))]

    chosen = max(2, round((elbow_k + sil_k) / 2))
    logger.info("k-selection: elbow=%d silhouette=%d → chosen=%d", elbow_k, sil_k, chosen)
    return chosen


def _elbow_index(inertias: list[float]) -> int:
    """Index of the elbow point by maximum perpendicular distance (normalised axes)."""
    n = len(inertias)
    if n <= 2:
        return 0

    x = np.linspace(0.0, 1.0, n)  # normalised k axis
    y_raw = np.array(inertias, dtype=float)

    y_range = y_raw[0] - y_raw[-1]
    if y_range == 0.0:
        return 0
    # Normalise so the curve goes from ~(0,1) to (1,0); larger y = more inertia = worse
    y = (y_raw - y_raw[-1]) / y_range

    # Line from (0, y[0]/y[0]=1) to (1, 0): y + x = 1 → distance = |x + y - 1| / sqrt(2)
    dists = np.abs(x + y - 1.0) / math.sqrt(2.0)
    return int(np.argmax(dists))
