"""Score calibration via a tie-aware percentile rank.

Calibration converts raw detector scores (higher = more normal) into a
[0, 1] anomaly score: 0.0 = perfectly normal, 1.0 = certain anomaly. The sign
flip (lower raw score -> higher calibrated score) is applied here, after
collecting from detectors that return higher = more normal.

The reference distribution is whatever scores :meth:`Calibrator.fit` is given.
In the pipeline that is a group's own training scores ("self" calibration), so a
single run's calibrated scores are only meaningful *within* that run -- by
construction they sit close to uniform on [0, 1]. Scores comparable *across*
runs come from reusing an already-fitted calibrator via
``sorethumb score --from-run`` (see :func:`sorethumb.store.models.score_with_existing`),
not from a second calibration mode.

Mapping
-------
For a query score ``s`` the calibrated anomaly score is ``1 - F(s)`` where ``F``
is the *mid-rank* (average-rank) empirical CDF of the reference::

    F(s) = ( #{ref < s} + 0.5 * #{ref == s} ) / n

The mid-rank term is what makes ties well defined: a value that occurs many
times in the reference maps to the midpoint of the band it occupies rather than
to an arbitrary end of it.

Reference representation
-------------------------
``fit`` stores the reference as sorted *unique* values with per-value weights
(occurrence counts), which is enough to compute ``F`` exactly via a weighted
mid-rank -- no interpolation, so ``F`` is exact at and between every reference
value, ties included. A reference with at most ``_EXACT_MAX_UNIQUE`` distinct
values is kept in full (``max_calibration_error == 0.0``: this *is* the
documented empirical mid-rank CDF, not an approximation of it).

A reference with more distinct values than that is compressed to at most
``_COMPRESSED_N_POINTS`` weighted points by merging runs of adjacent values
into one representative (always a real value from the reference, never an
interpolated one) carrying the sum of the merged weights. ``max_calibration_error``
records the largest single point's weight fraction after compression -- an
honest upper bound on how far a query landing inside that point's merged span
could read from the true (uncompressed) ``F``.

Constant-reference guard
------------------------
When every reference score is identical the detector carries no ranking
information and every row calibrates to 0.5.

Persistence
-----------
``to_dict`` / ``from_dict`` round-trip through the (possibly compressed)
weighted reference, tagged with ``schema_version: 2``. A dict from before
this schema (no ``schema_version`` key, carrying a flat ``quantile_values``
array -- 10 000 ``np.quantile`` interpolation points, and possibly an even
older ``"mode"`` key, both ignored) is accepted on read and migrated
in-memory into the new representation; ``max_calibration_error`` for a
migrated legacy calibrator is left unset (``None``) since the true bound
from before this schema existed cannot be reconstructed after the fact.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

# A reference with at most this many distinct values is kept exactly (every
# unique value + its occurrence count); F is then exact, not approximated.
_EXACT_MAX_UNIQUE = 10_000

# A reference with more distinct values than that is compressed to at most
# this many weighted points.
_COMPRESSED_N_POINTS = 2_000


def _compress_reference(
    values: np.ndarray, counts: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``(values, counts)`` into at most ``n_points`` weighted points.

    ``values``/``counts`` must be sorted ascending and unique. Each output
    value is one of the original values -- never a synthetic,
    interpolated one -- and each output weight is the sum of the original
    counts merged into it, so total weight is preserved exactly. Boundaries
    are placed at (roughly) equal cumulative weight, so no single output
    point absorbs much more than ``total / n_points`` of the mass unless one
    original value already carried that much weight on its own.
    """
    total = float(counts.sum())
    cum = np.cumsum(counts)
    boundaries = np.linspace(total / n_points, total, n_points)
    idx = np.searchsorted(cum, boundaries, side="left")
    idx = np.clip(idx, 0, len(values) - 1)
    idx = np.unique(idx)  # multiple boundaries can land on the same source index

    out_values = values[idx]
    kept_cum = cum[idx]
    prev_cum = np.concatenate([[0.0], kept_cum[:-1]])
    out_weights = kept_cum - prev_cum
    return out_values, out_weights


class Calibrator:
    """Tie-aware percentile-rank calibrator.

    Fit on a reference distribution of raw scores (higher = more normal), then
    :meth:`transform` maps scores onto ``1 - mid_rank_cdf(score)`` in [0, 1].
    """

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self._ref_values: np.ndarray | None = None  # sorted ascending, unique
        self._ref_weights: np.ndarray | None = None  # occurrence count per ref_values entry
        self._n_total: int = 0
        self._max_calibration_error: float | None = None

    def fit(self, train_scores: np.ndarray) -> None:
        """Store the reference distribution (raw scores; higher = more normal)."""
        ref = np.asarray(train_scores, dtype=np.float64)
        if len(ref) == 0:
            msg = "Cannot fit Calibrator on empty scores array."
            raise ValueError(msg)

        values, counts = np.unique(ref, return_counts=True)
        self._n_total = len(ref)
        if len(values) <= _EXACT_MAX_UNIQUE:
            self._ref_values = values
            self._ref_weights = counts.astype(np.float64)
            self._max_calibration_error = 0.0
        else:
            self._ref_values, self._ref_weights = _compress_reference(values, counts, _COMPRESSED_N_POINTS)
            self._max_calibration_error = float(self._ref_weights.max()) / self._n_total
        logger.debug(
            "Calibrator fitted on %d reference scores (%d distinct%s).",
            self._n_total,
            len(values),
            "" if self._max_calibration_error == 0.0 else ", compressed",
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to calibrated anomaly scores in [0, 1].

        A raw score at the p-th percentile of the reference becomes ``1 - p``:
        low-scoring (more anomalous) rows get a value near 1. Ties in the
        reference are resolved with a mid-rank empirical CDF, so a repeated
        reference value maps to the middle of its band.
        """
        if self._ref_values is None or self._ref_weights is None:
            msg = "Calibrator.fit() must be called before transform()."
            raise RuntimeError(msg)

        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return np.empty(0, dtype=np.float64)

        values = self._ref_values  # sorted ascending, unique
        if float(values[-1] - values[0]) == 0.0:
            logger.debug("Calibrator: constant reference distribution; returning 0.5 for all rows.")
            return np.full(len(scores), 0.5, dtype=np.float64)

        # Weighted mid-rank empirical CDF. `prefix[i]` = total weight of the
        # first i reference values (prefix[0] == 0), so prefix[idx_left] and
        # prefix[idx_right] give #{ref < s} and #{ref <= s} respectively for
        # any query s, exact ties included.
        prefix = np.concatenate([[0.0], np.cumsum(self._ref_weights)])
        idx_left = np.searchsorted(values, scores, side="left")
        idx_right = np.searchsorted(values, scores, side="right")
        count_below = prefix[idx_left]
        count_at_or_below = prefix[idx_right]
        mid_rank = (count_below + count_at_or_below) / (2.0 * self._n_total)
        # Flip: higher raw score (more normal) -> lower anomaly score.
        return 1.0 - mid_rank

    def fit_transform(self, train_scores: np.ndarray) -> np.ndarray:
        """Fit on ``train_scores`` then transform them in one call."""
        self.fit(train_scores)
        return self.transform(train_scores)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "ref_values": self._ref_values.tolist() if self._ref_values is not None else None,
            "ref_weights": self._ref_weights.tolist() if self._ref_weights is not None else None,
            "n_total": self._n_total,
            "max_calibration_error": self._max_calibration_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Calibrator:
        """Reconstruct from a dict produced by ``to_dict``.

        Accepts a pre-schema-versioning legacy dict (a flat ``quantile_values``
        array, an even older ``"mode"`` key ignored) and migrates it in memory
        into the current representation.
        """
        obj = cls()
        if data.get("schema_version") == _SCHEMA_VERSION:
            rv = data.get("ref_values")
            rw = data.get("ref_weights")
            if rv is not None and rw is not None:
                obj._ref_values = np.asarray(rv, dtype=np.float64)
                obj._ref_weights = np.asarray(rw, dtype=np.float64)
                obj._n_total = int(cast("float", data.get("n_total") or 0))
                err = data.get("max_calibration_error")
                obj._max_calibration_error = float(cast("float", err)) if err is not None else None
            return obj

        # Legacy (pre-schema_version) dict: a flat array of quantile points.
        # Each entry counts as its own reference value; np.unique folds any
        # repeats those quantile points happened to land on into a weight.
        # This is a best-effort migration -- the true pre-migration bound
        # can't be reconstructed, so max_calibration_error is left unset.
        legacy_values = data.get("quantile_values")
        if legacy_values is not None:
            legacy = np.asarray(legacy_values, dtype=np.float64)
            values, counts = np.unique(legacy, return_counts=True)
            obj._ref_values = values
            obj._ref_weights = counts.astype(np.float64)
            obj._n_total = len(legacy)
            obj._max_calibration_error = None
        return obj
