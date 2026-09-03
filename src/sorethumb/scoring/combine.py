"""Composite score combination and anomaly thresholding.

ScoreEnsemble takes multiple per-detector calibrated scores (each in [0, 1],
higher = more anomalous) and combines them into a single final score plus a
binary anomaly flag.

Weighting strategies
--------------------
equal:
    All detectors receive weight 1 / n. Default, no configuration needed.
manual:
    User supplies a dict mapping detector name → weight. Weights are
    normalised to sum to 1.0 before use.
agreement:
    Weight each detector by the fraction of all detectors that agree with its
    natural_flag on each row, averaged over the dataset. Agreement weights
    reward detectors whose flags are consistent with the ensemble, penalising
    outlier detectors that fire on rows the others consider normal.

Combination strategies
----------------------
composite:
    Weighted average of calibrated scores. Smooth, suitable for ranking.
intersection:
    Score = min of calibrated scores. A row is only high-scoring when ALL
    detectors agree it is anomalous. Conservative.
union:
    Score = max of calibrated scores. A row is high-scoring if ANY detector
    flags it. Permissive.

Thresholding
------------
The binary flag is set where composite score >= threshold.

``contamination="auto"``:
    Threshold is set at the (1 − median_natural_flag_rate) quantile of the
    combined score distribution. The natural flag rate is the fraction of rows
    that each detector's natural_flag method marks as anomalous; the median
    across detectors is logged as a heuristic estimate. This is deliberately
    conservative: auto contamination is always labelled as a heuristic in logs
    and in the returned metadata dict.

``contamination=float``:
    Threshold is set at the (1 − contamination) quantile, e.g. 0.05 → 95th
    percentile threshold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ScoreEnsemble:
    """Combine calibrated scores from multiple detectors into a final anomaly decision.

    Parameters
    ----------
    weighting:
        "equal", "manual", or "agreement".
    combination:
        "composite", "intersection", or "union".
    contamination:
        "auto" or a float in (0, 1). Controls the anomaly threshold.
    manual_weights:
        Required when weighting="manual". Dict from detector name → weight.

    """

    def __init__(
        self,
        weighting: str = "equal",
        combination: str = "composite",
        contamination: str | float = "auto",
        manual_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialise ensemble with weighting, combination strategy, and contamination."""
        if weighting not in {"equal", "manual", "agreement"}:
            msg = f"weighting must be 'equal', 'manual', or 'agreement'; got {weighting!r}"
            raise ValueError(msg)
        if combination not in {"composite", "intersection", "union"}:
            msg = f"combination must be 'composite', 'intersection', or 'union'; got {combination!r}"
            raise ValueError(msg)
        if weighting == "manual" and not manual_weights:
            msg = "manual_weights must be provided when weighting='manual'."
            raise ValueError(msg)
        if isinstance(contamination, float) and not (0.0 < contamination < 1.0):
            msg = f"contamination float must be in (0, 1); got {contamination}"
            raise ValueError(msg)

        self._weighting = weighting
        self._combination = combination
        self._contamination = contamination
        self._manual_weights = manual_weights

    def combine(
        self,
        scores: dict[str, np.ndarray],
        natural_flags: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Combine per-detector scores into a final score and flag array.

        Parameters
        ----------
        scores:
            Mapping from detector name → calibrated score array (higher = more anomalous).
        natural_flags:
            Mapping from detector name → boolean flag array (True = anomalous).

        Returns
        -------
        dict with keys:
            "combined_score": np.ndarray, shape (n,)
            "anomaly_flag": np.ndarray[bool], shape (n,)
            "threshold": float
            "contamination_used": float (the resolved contamination rate)
            "weights": dict[str, float]
            "is_auto_contamination": bool

        """
        names = list(scores.keys())
        if not names:
            msg = "scores dict is empty; need at least one detector."
            raise ValueError(msg)

        n = len(next(iter(scores.values())))
        score_matrix = np.column_stack([scores[d] for d in names])  # shape (n, k)
        flag_matrix = np.column_stack([natural_flags[d].astype(float) for d in names])  # (n, k)

        weights = self._resolve_weights(names, flag_matrix)

        combined = self._combine(score_matrix, weights)

        contamination, is_auto = self._resolve_contamination(flag_matrix)
        threshold = float(np.quantile(combined, 1.0 - contamination))

        anomaly_flag = combined >= threshold

        logger.info(
            "ScoreEnsemble: weighting=%s combination=%s contamination=%.4f%s threshold=%.4f "
            "flagged %d/%d (%.1f%%).",
            self._weighting,
            self._combination,
            contamination,
            " [auto/heuristic]" if is_auto else "",
            threshold,
            int(anomaly_flag.sum()),
            n,
            100.0 * anomaly_flag.mean(),
        )

        return {
            "combined_score": combined,
            "anomaly_flag": anomaly_flag,
            "threshold": threshold,
            "contamination_used": contamination,
            "weights": dict(zip(names, weights, strict=True)),
            "is_auto_contamination": is_auto,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, names: list[str], flag_matrix: np.ndarray) -> np.ndarray:
        k = len(names)
        if self._weighting == "equal":
            return np.ones(k) / k

        if self._weighting == "manual":
            assert self._manual_weights is not None  # validated in __init__ when weighting="manual"
            w = np.array([self._manual_weights.get(n, 0.0) for n in names], dtype=np.float64)
            total = w.sum()
            if total == 0:
                logger.warning("manual_weights sum to 0; falling back to equal weights.")
                return np.ones(k) / k
            return w / total

        # agreement: weight by average agreement fraction
        # For each detector d, agreement(d) = fraction of rows where flag_matrix[:,d]
        # matches the majority vote across all other detectors.
        if k == 1:
            return np.ones(1)

        majority = (flag_matrix.mean(axis=1) >= 0.5).astype(float)  # (n,)
        agreement_rates = np.array([float(np.mean(flag_matrix[:, i] == majority)) for i in range(k)])
        total = agreement_rates.sum()
        if total == 0:
            return np.ones(k) / k
        return agreement_rates / total

    def _combine(self, score_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self._combination == "composite":
            return score_matrix @ weights  # weighted average, shape (n,)

        if self._combination == "intersection":
            return score_matrix.min(axis=1)

        # union
        return score_matrix.max(axis=1)

    def _resolve_contamination(self, flag_matrix: np.ndarray) -> tuple[float, bool]:
        if self._contamination != "auto":
            return float(self._contamination), False

        # Auto: median natural_flag rate across detectors
        rates = flag_matrix.mean(axis=0)
        rate = float(np.median(rates))
        rate = max(0.001, min(0.5, rate))  # clamp to sane range
        logger.info(
            "Auto contamination: detector natural_flag rates %s → median=%.4f (heuristic).",
            [round(float(r), 4) for r in rates],
            rate,
        )
        return rate, True
