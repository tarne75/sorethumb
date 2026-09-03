"""KMeans centroid attributions.

The per-dimension signed contribution to the distance from the assigned centroid
is simply x - c (where c is the centroid for the assigned cluster). This is a
direct decomposition of the L2 distance: ||x - c||² = Σ_d (x_d - c_d)².

We take the absolute value of each component and normalise to unit L2 norm so
the magnitude is comparable with attributions from other detectors after blending.

The raw signed differences are stored in detector.last_contributions after
score_samples() is called — this module just retrieves and normalises them.

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
) -> tuple[np.ndarray, str]:
    """Return per-row attributions derived from centroid distance components.

    Requires that detector.score_samples() has been called first so that
    last_contributions is populated.

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes row away from centroid
        (more anomalous). Values are unsigned (absolute) differences.
    tag:
        Always "heuristic".

    """
    if detector.last_contributions is None:
        msg = "centroid_attributions requires score_samples() to have been called first."
        raise ValueError(msg)

    contributions = detector.last_contributions  # signed (x - c), shape (n_rows, n_features)
    # Use absolute value: direction doesn't matter for anomaly attribution, magnitude does
    attributions = np.abs(contributions).astype(np.float64)
    return attributions, "heuristic"
