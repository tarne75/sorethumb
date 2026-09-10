"""KMeans centroid attributions.

The per-dimension signed contribution to the distance from the nearest large-cluster
centroid is x - c_nearest. This is a direct decomposition of the L2 distance:
||x - c||² = Σ_d (x_d - c_d)².

We take the absolute value of each component so the magnitude reflects how far each
dimension is from the centroid, regardless of sign. This is comparable with
attributions from other detectors after blending.

Contributions are recomputed from X and the detector's stored large-cluster centroids
rather than read from mutable scorer state, making attribution deterministic and
independent of call ordering.

Tag is "heuristic" because this decomposition is exact for the distance metric
but does not generalise outside KMeans (e.g. it ignores the relative importance
of dimensions from the model's perspective).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sorethumb.detectors.kmeans_distance import KMeansDetector

logger = logging.getLogger(__name__)


def centroid_attributions(
    detector: KMeansDetector,
    X: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return per-row attributions derived from centroid distance components.

    Parameters
    ----------
    detector:
        A fitted KMeansDetector (fit() must have been called).
    X:
        Feature matrix, shape (n_rows, n_features).

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Values are unsigned (absolute) per-dimension
        differences from the nearest large-cluster centroid.
    tag:
        Always "heuristic".

    """
    if detector.large_centroids is None:
        msg = "centroid_attributions requires fit() to have been called first."
        raise ValueError(msg)

    from sorethumb.detectors.kmeans_distance import _nearest_large_centroid  # noqa: PLC0415

    centres = detector.large_centroids  # (n_large, d)
    # nearest large centroid per row, without the (n, n_large, d) difference
    # tensor the old broadcast built (~d x larger; OOM on wide/large inputs).
    nearest, _ = _nearest_large_centroid(X, centres)
    contributions = X - centres[nearest]  # signed (x - c_nearest), (n, d)
    return np.abs(contributions).astype(np.float64), "heuristic"
