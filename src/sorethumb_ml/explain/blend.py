"""Blend attribution vectors from multiple detectors.

When more than one detector flagged a record, L2-normalise each source vector
first so a source with a larger native magnitude cannot dominate, then take
the weighted component-wise mean using the scoring weights.

A single source is returned as-is (no normalisation needed).

Blended tag is the *weakest* of the contributing tags, ranked ``exact`` >
``model_specific`` > ``heuristic``: averaging in a weaker source doesn't
un-corrupt it, so the blend can only be as trustworthy as its least
trustworthy input. In practice this means a pure ECOD/HBOS run keeps
``exact``, a pure IsolationForest run keeps ``model_specific``, and any
mixture (or a source that fell back to a heuristic) drops to ``heuristic``.
"""

from __future__ import annotations

import numpy as np

# Best to worst. A tag not listed here (shouldn't happen — every producer in
# this package uses one of these three) ranks as worse than "heuristic" so an
# unrecognised source can never make a blend look more trustworthy than it is.
_TAG_RANK: dict[str, int] = {"exact": 0, "model_specific": 1, "heuristic": 2}


def _weakest_tag(tags: list[str]) -> str:
    """Return the lowest-ranked (least trustworthy) tag among *tags*."""
    return max(tags, key=lambda t: _TAG_RANK.get(t, len(_TAG_RANK)))


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
        The weakest tag among the contributing sources (see module docstring).

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

    for (mat, _tag), w in zip(sources, normed_weights, strict=True):
        # L2-normalise each row independently to prevent magnitude domination
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        blended += w * normed

    final_tag = _weakest_tag([tag for _, tag in sources])
    return blended, final_tag
