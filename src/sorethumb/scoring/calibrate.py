"""Score calibration via percentile rank.

Calibration converts raw detector scores (higher = more normal) into a
[0, 1] anomaly probability: 0.0 = perfectly normal, 1.0 = certain anomaly.

The mapping is: percentile_rank_in_reference / 100. For a row scoring at the
5th percentile of the reference distribution its calibrated score is 0.95 —
it is more anomalous than 95% of the reference. The sign flip (lower raw
score → higher calibrated score) is applied here, after collecting from
detectors that return higher = more normal.

Two modes
---------
self:
    Reference = training scores. Calibrated scores are relative to the
    training population. Use when no external reference is available.
reference:
    Reference = explicitly supplied scores (e.g. from a held-out baseline
    window). Use for drift-detection contexts where a stable anchor is needed.

Constant-score guard
--------------------
When all reference scores are identical (std = 0), ``np.interp`` would
produce all-0.5 — this is the correct fallback for a useless detector.

Persistence
-----------
``to_dict`` / ``from_dict`` roundtrip through 10 000 quantile points stored
as Python lists, suitable for JSON and M5 workspace storage.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_N_QUANTILE_POINTS = 10_000


class Calibrator:
    """Percentile-rank calibrator.

    Parameters
    ----------
    mode:
        "self" or "reference". Controls which scores become the reference
        distribution when ``fit`` is called.

    """

    def __init__(self, mode: str = "self") -> None:
        """Initialise with 'self' or 'reference' calibration mode."""
        if mode not in {"self", "reference"}:
            msg = f"mode must be 'self' or 'reference'; got {mode!r}"
            raise ValueError(msg)
        self._mode = mode
        self._quantile_values: np.ndarray | None = None  # sorted low→high raw scores
        self._quantile_probs: np.ndarray = np.linspace(0.0, 1.0, _N_QUANTILE_POINTS)

    @property
    def mode(self) -> str:
        """Return the calibration mode ('self' or 'reference')."""
        return self._mode

    def fit(self, train_scores: np.ndarray, reference_scores: np.ndarray | None = None) -> None:
        """Store the reference distribution.

        Parameters
        ----------
        train_scores:
            Raw scores from the training window (higher = more normal).
        reference_scores:
            Used only when mode="reference". If None and mode="reference", falls
            back to train_scores with a warning.

        """
        if self._mode == "reference":
            if reference_scores is None:
                logger.warning(
                    "Calibrator mode='reference' but no reference_scores supplied; "
                    "falling back to train_scores."
                )
                ref = train_scores
            else:
                ref = reference_scores
        else:
            ref = train_scores

        if len(ref) == 0:
            msg = "Cannot fit Calibrator on empty scores array."
            raise ValueError(msg)

        self._quantile_values = np.quantile(ref, self._quantile_probs)
        logger.debug("Calibrator fitted on %d reference scores (mode=%s).", len(ref), self._mode)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to calibrated anomaly probabilities in [0, 1].

        A raw score at the p-th percentile of the reference distribution
        becomes (1 - p): low-scoring (more anomalous) rows get a score near 1.
        """
        if self._quantile_values is None:
            msg = "Calibrator.fit() must be called before transform()."
            raise RuntimeError(msg)

        if len(scores) == 0:
            return np.empty(0, dtype=np.float64)

        # Clamp identical-distribution guard: std=0 → all 0.5
        if float(np.std(self._quantile_values)) == 0.0:
            logger.debug("Calibrator: constant reference distribution; returning 0.5 for all rows.")
            return np.full(len(scores), 0.5, dtype=np.float64)

        # Percentile rank: fraction of reference below each score
        percentile_rank = np.interp(scores, self._quantile_values, self._quantile_probs)
        # Flip: higher raw score (more normal) → lower anomaly probability
        return 1.0 - percentile_rank

    def fit_transform(
        self,
        train_scores: np.ndarray,
        reference_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit then transform ``train_scores`` in one call."""
        self.fit(train_scores, reference_scores)
        return self.transform(train_scores)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "mode": self._mode,
            "quantile_values": self._quantile_values.tolist() if self._quantile_values is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Calibrator:
        """Reconstruct from a plain dict produced by ``to_dict``."""
        obj = cls(mode=str(data["mode"]))
        qv = data.get("quantile_values")
        if qv is not None:
            obj._quantile_values = np.asarray(qv, dtype=np.float64)
        return obj
