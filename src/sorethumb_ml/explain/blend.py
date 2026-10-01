"""Blend attribution vectors from multiple detectors.

When more than one detector flagged a record, L2-normalise each source vector
first so a source with a larger native magnitude cannot dominate, then take
the weighted component-wise mean using the scoring weights.

A source row whose norm is negligible next to that source's typical row norm
(below ``_NEGLIGIBLE_RTOL`` times the median of its non-zero row norms, or
exactly zero) carries no information about that row: a flat gradient, or a
row beyond the source's ``explain.max_rows`` cap. Normalising it would blow
numerical noise up to the same unit length as a real signal, so it is left out
of that row's blend and the remaining sources' weights are rescaled to sum to 1
for that row. A row no (positively weighted) source informs blends to all zeros.

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

# A source row is "no information" when its L2 norm is at most this fraction of
# the median non-zero row norm of the same source.
_NEGLIGIBLE_RTOL = 1e-3


def _weakest_tag(tags: list[str]) -> str:
    """Return the lowest-ranked (least trustworthy) tag among *tags*."""
    return max(tags, key=lambda t: _TAG_RANK.get(t, len(_TAG_RANK)))


def _informative_rows(norms: np.ndarray) -> np.ndarray:
    """Mask of rows whose norm is not negligible relative to the source's own scale."""
    nonzero = norms[norms > 0.0]
    if nonzero.size == 0:
        return np.zeros(norms.shape, dtype=bool)
    return np.asarray(norms > _NEGLIGIBLE_RTOL * float(np.median(nonzero)))


def blend(
    sources: list[tuple[np.ndarray, str]],
    weights: list[float],
) -> tuple[np.ndarray, str]:
    """L2-normalise each source row, then compute a per-row weighted mean of the informative ones.

    Parameters
    ----------
    sources:
        List of (attribution_matrix, tag) pairs. Each matrix has shape
        (n_rows, n_features).
    weights:
        Per-source weights. Must have the same length as sources. Need not sum
        to 1 — they are normalised internally, per row, over the sources that
        carry information for that row (see the module docstring).

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

    n_rows = sources[0][0].shape[0]
    blended = np.zeros_like(sources[0][0], dtype=np.float64)
    row_weight = np.zeros(n_rows, dtype=np.float64)  # weight of the sources informing each row

    for (mat, _tag), w in zip(sources, normed_weights, strict=True):
        # L2-normalise each row independently to prevent magnitude domination,
        # but only rows that carry information for this source.
        norms = np.linalg.norm(mat, axis=1)
        informative = _informative_rows(norms)
        normed = np.zeros_like(blended)
        normed[informative] = mat[informative] / norms[informative, None]
        blended += w * normed
        row_weight += w * informative

    # Rescale each row by the weight of the sources that informed it. A row
    # informed only by zero-weight sources stays zero, as it always did.
    has_weight = row_weight > 0.0
    blended[has_weight] /= row_weight[has_weight, None]

    final_tag = _weakest_tag([tag for _, tag in sources])
    return blended, final_tag
