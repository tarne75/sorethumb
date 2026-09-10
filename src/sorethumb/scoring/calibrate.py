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
to an arbitrary end of it. Plain interpolation (``np.interp``) over a reference
with repeated values is undefined on the flat segments and can place tied
scores anywhere in the band.

Constant-reference guard
------------------------
When every reference score is identical the detector carries no ranking
information and every row calibrates to 0.5.

Persistence
-----------
``to_dict`` / ``from_dict`` roundtrip through 10 000 sorted quantile points -- a
fixed-size summary of the reference -- suitable for JSON and workspace storage.
Dicts written by older versions carry a ``"mode"`` key; it is ignored on read.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_N_QUANTILE_POINTS = 10_000


class Calibrator:
    """Tie-aware percentile-rank calibrator.

    Fit on a reference distribution of raw scores (higher = more normal), then
    :meth:`transform` maps scores onto ``1 - mid_rank_cdf(score)`` in [0, 1].
    """

    def __init__(self) -> None:
        """Create an unfitted calibrator."""
        self._quantile_values: np.ndarray | None = None  # sorted low -> high raw scores
        self._quantile_probs: np.ndarray = np.linspace(0.0, 1.0, _N_QUANTILE_POINTS)

    def fit(self, train_scores: np.ndarray) -> None:
        """Store the reference distribution (raw scores; higher = more normal)."""
        ref = np.asarray(train_scores, dtype=np.float64)
        if len(ref) == 0:
            msg = "Cannot fit Calibrator on empty scores array."
            raise ValueError(msg)

        self._quantile_values = np.quantile(ref, self._quantile_probs)
        logger.debug("Calibrator fitted on %d reference scores.", len(ref))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to calibrated anomaly scores in [0, 1].

        A raw score at the p-th percentile of the reference becomes ``1 - p``:
        low-scoring (more anomalous) rows get a value near 1. Ties in the
        reference are resolved with a mid-rank empirical CDF, so a repeated
        reference value maps to the middle of its band.
        """
        if self._quantile_values is None:
            msg = "Calibrator.fit() must be called before transform()."
            raise RuntimeError(msg)

        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return np.empty(0, dtype=np.float64)

        ref = self._quantile_values  # sorted ascending
        if float(ref[-1] - ref[0]) == 0.0:
            logger.debug("Calibrator: constant reference distribution; returning 0.5 for all rows.")
            return np.full(len(scores), 0.5, dtype=np.float64)

        # Mid-rank empirical CDF: average of the strict-below and at-or-below
        # counts, so tied reference values land at the midpoint of their band.
        below = np.searchsorted(ref, scores, side="left")  # #{ref < s}
        at_or_below = np.searchsorted(ref, scores, side="right")  # #{ref <= s}
        mid_rank = (below + at_or_below) / (2.0 * len(ref))
        # Flip: higher raw score (more normal) -> lower anomaly score.
        return 1.0 - mid_rank

    def fit_transform(self, train_scores: np.ndarray) -> np.ndarray:
        """Fit on ``train_scores`` then transform them in one call."""
        self.fit(train_scores)
        return self.transform(train_scores)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "quantile_values": self._quantile_values.tolist()
            if self._quantile_values is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Calibrator:
        """Reconstruct from a dict produced by ``to_dict``.

        A ``"mode"`` key written by an older version is ignored.
        """
        obj = cls()
        qv = data.get("quantile_values")
        if qv is not None:
            obj._quantile_values = np.asarray(qv, dtype=np.float64)
        return obj
