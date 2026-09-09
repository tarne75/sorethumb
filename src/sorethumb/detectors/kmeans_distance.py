"""KMeans distance detector.

Score = negative Euclidean distance to the nearest *large-cluster* centroid
(CBLOF-style), so higher (closer to zero) means more normal.

After fitting, clusters are ranked by size (descending). The subset whose
cumulative population covers at least ``large_cluster_coverage`` (default 0.90)
of the training data are called *large* clusters; the remainder are *small*.
Every point — including those that were assigned to a small cluster during
training — is scored against the nearest large-cluster centroid. Points that
belong to a small anomaly sub-group are far from every large centroid and
therefore score very low (most anomalous). This prevents the classic KMeans
failure mode where a tight anomaly cluster captures its own centroid and is
incorrectly ranked as normal.

A single global threshold is applied after scoring — never per-chunk or
per-partition, because a per-partition threshold flags each partition's own
outliers and manufactures false positives when group sizes differ.

k is selected automatically by combining the elbow criterion (normalised inertia
curve) with the best silhouette score, sampled to 20 000 rows for silhouette
because it is O(n²). See docs/approximations.md.

After scoring, ``last_labels`` and ``last_contributions`` are populated for the
explanation layer (§8.3 centroid attribution).
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import numpy as np

from sorethumb.detectors._hyperparams import validate_extra_params

logger = logging.getLogger(__name__)

_SILHOUETTE_SAMPLE = 20_000
_DEFAULT_LARGE_COVERAGE = 0.90
_CURATED = frozenset({"k", "k_min", "k_max", "n_init", "large_cluster_coverage"})


class KMeansDetector:
    """sklearn KMeans wrapped to satisfy the Detector protocol.

    Uses CBLOF-style scoring: every point is measured against the nearest
    large-cluster centroid rather than its assigned centroid.  This prevents
    anomaly clusters from capturing their own centroid and being ranked normal.
    """

    name: ClassVar[str] = "kmeans_distance"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 200_000

    def __init__(
        self,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 8,
        n_init: int = 10,
        large_cluster_coverage: float = _DEFAULT_LARGE_COVERAGE,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialise with optional fixed k or auto-selection bounds.

        Parameters
        ----------
        k:
            Fixed number of clusters. When ``None``, k is chosen automatically
            by combining the elbow and silhouette criteria.
        k_min:
            Lower bound for auto-selected k.
        k_max:
            Upper bound for auto-selected k. Kept intentionally small (default 8)
            to reduce the chance that a minority anomaly cluster captures its own
            centroid even before CBLOF scoring is applied.
        n_init:
            Number of KMeans initialisations.
        large_cluster_coverage:
            Fraction of training points that must be covered by the "large"
            cluster subset (sorted by size descending). Clusters covering this
            cumulative fraction are used as the reference centroids for scoring.
            Default 0.90 means the largest clusters covering 90% of the data
            are used; small minority clusters (including anomaly sub-groups) are
            excluded from the reference set.
        extra_params:
            Additional kwargs forwarded verbatim to sklearn's KMeans constructor
            (tol, max_iter, algorithm, init, …). Validated now; ``n_clusters``
            and ``random_state`` are reserved and ``n_init`` must be set via the
            wrapper argument.

        """
        from sklearn.cluster import KMeans  # noqa: PLC0415

        self._k_fixed = k
        self._k_min = k_min
        self._k_max = k_max
        self._n_init = n_init
        self._large_coverage = large_cluster_coverage
        self._extra = validate_extra_params(self.name, KMeans, extra_params, curated=_CURATED)
        self._model: Any = None
        self._chosen_k: int | None = None
        self._large_centroids: np.ndarray | None = None  # subset of centroids used for scoring
        self.last_labels: np.ndarray | None = None
        self.last_contributions: np.ndarray | None = None

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        """Select k (if not fixed), fit KMeans, and identify large clusters."""
        from sklearn.cluster import KMeans  # noqa: PLC0415

        k = self._k_fixed if self._k_fixed is not None else _select_k(X, self._k_min, self._k_max, seed)
        self._chosen_k = k
        logger.info("KMeans: fitting k=%d on %d rows x %d features.", k, X.shape[0], X.shape[1])
        self._model = KMeans(n_clusters=k, n_init=self._n_init, random_state=seed, **self._extra)
        self._model.fit(X)

        # Identify large clusters (CBLOF: clusters covering >= large_coverage of data)
        labels: np.ndarray = self._model.labels_
        centres = self._model.cluster_centers_
        cluster_sizes = np.bincount(labels, minlength=k)
        # Sort clusters by size descending
        sorted_idx = np.argsort(cluster_sizes)[::-1]
        cumulative = np.cumsum(cluster_sizes[sorted_idx]) / len(X)
        # Include clusters until coverage threshold is met (at least one always included)
        n_large = int(np.searchsorted(cumulative, self._large_coverage)) + 1
        n_large = max(1, min(n_large, k))
        large_idx = sorted_idx[:n_large]
        self._large_centroids = centres[large_idx]
        logger.info(
            "KMeans: %d/%d large clusters cover %.1f%% of %d training rows.",
            n_large,
            k,
            100.0 * cumulative[n_large - 1],
            len(X),
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return negative distance to nearest large-cluster centroid.

        Higher (closer to zero) means more normal.  Points near any large
        cluster score well; points far from all large clusters (e.g. members
        of small anomaly sub-groups) score very low.
        """
        assert self._large_centroids is not None, "fit() must be called before score_samples()"
        centres = self._large_centroids

        # Distance from each point to each large centroid: shape (n, n_large)
        # Using broadcasting: X[:, None, :] - centres[None, :, :]
        diffs_all = X[:, np.newaxis, :] - centres[np.newaxis, :, :]  # (n, n_large, d)
        dist_all = np.linalg.norm(diffs_all, axis=2)  # (n, n_large)

        nearest_idx = np.argmin(dist_all, axis=1)  # (n,)
        distances = dist_all[np.arange(len(X)), nearest_idx]  # (n,)

        # Store for explanation layer: contributions against the nearest large centroid
        self.last_labels = nearest_idx
        self.last_contributions = X - centres[nearest_idx]  # signed per-dimension

        return -distances

    @property
    def large_centroids(self) -> np.ndarray | None:
        """Large-cluster centroids used for scoring; None until fit() is called."""
        return self._large_centroids

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows whose distance is an outlier by Tukey's method (1.5 × IQR fence)."""
        distances = -scores
        q25 = float(np.percentile(distances, 25))
        q75 = float(np.percentile(distances, 75))
        upper_fence = q75 + 1.5 * (q75 - q25)
        return distances > upper_fence

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        n_large = len(self._large_centroids) if self._large_centroids is not None else None
        return {
            "k_fixed": self._k_fixed,
            "k_min": self._k_min,
            "k_max": self._k_max,
            "chosen_k": self._chosen_k,
            "n_init": self._n_init,
            "large_cluster_coverage": self._large_coverage,
            "n_large_clusters": n_large,
            "extra_params": dict(self._extra),
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
