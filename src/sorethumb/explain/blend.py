"""Blend attribution vectors from multiple detectors.

When more than one detector flagged a record, L2-normalise each source vector
first so a source with a larger native magnitude cannot dominate, then take
the weighted component-wise mean using the scoring weights.

A single source is returned as-is (no normalisation needed).

Blended tag is ``exact`` only if every contributing source is ``exact``.
In practice this means only a pure IsolationForest run without PCA produces
an ``exact`` tag; any mixture or PCA back-projection yields ``heuristic``.
"""

from __future__ import annotations

import numpy as np


def blend(
    sources: list[tuple[np.ndarray, str]],
    weights: list[float],
) -> tuple[np.ndarray, str]:
    """L2-normalise each source matrix, then compute a weighted mean.

    Parameters
    ----------
    sources:
        List of (attribution_matrix, tag) pairs. Each matrix has shape
        (n_rows, n_features).
    weights:
        Per-source weights. Must have the same length as sources. Need not sum
        to 1 — they are normalised internally.

    Returns
    -------
    blended:
        Shape (n_rows, n_features). Combined attribution vector.
    tag:
        "exact" iff all source tags are "exact", else "heuristic".

    """
    if not sources:
        msg = "blend() requires at least one source."
        raise ValueError(msg)

    if len(sources) != len(weights):
        msg = f"sources ({len(sources)}) and weights ({len(weights)}) must have the same length."
        raise ValueError(msg)

    if len(sources) == 1:
        return sources[0]

    total_weight = sum(weights)
    if total_weight == 0.0:
        normed_weights = [1.0 / len(weights)] * len(weights)
    else:
        normed_weights = [w / total_weight for w in weights]

    blended = np.zeros_like(sources[0][0], dtype=np.float64)
    all_exact = True

    for (mat, tag), w in zip(sources, normed_weights, strict=True):
        if tag != "exact":
            all_exact = False
        # L2-normalise each row independently to prevent magnitude domination
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        blended += w * normed

    final_tag = "exact" if all_exact else "heuristic"
    return blended, final_tag
