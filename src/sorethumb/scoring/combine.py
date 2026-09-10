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
    Weight each detector by how well its *ranking* agrees with the consensus:
    Spearman's rho between the detector's calibrated scores and the mean rank
    of every other detector (leave-one-out). Negative correlations and
    constant-score detectors get weight 0; if nothing correlates, weights fall
    back to equal. Only meaningful with ``combination="composite"`` — the
    set operations ignore weights.

Combination strategies
----------------------
composite:
    Weighted average of calibrated scores. A single global threshold is applied
    to the combined score. Smooth, suitable for ranking.
intersection:
    Each detector independently flags its top-contamination fraction of rows
    (or its natural boundary when contamination="auto"). A row is anomalous
    only when ALL detectors flag it. Conservative; use when you trust all
    detectors equally and want high precision.
union:
    Each detector independently flags its top-contamination fraction of rows
    (or its natural boundary when contamination="auto"). A row is anomalous
    when ANY detector flags it. Permissive; maximises recall.

Thresholding
------------
composite mode:
    Binary flag is set where combined score >= threshold, where threshold is
    the (1 − contamination) quantile of the combined score distribution.

intersection / union modes:
    Each detector's scores are thresholded independently at the
    (1 − contamination) quantile of that detector's scores, producing a
    per-detector boolean flag. Set intersection or union of those flags is the
    final result. No global score threshold is applied.

``contamination="auto"``:
    composite: threshold derived from median natural_flag rate across detectors.
    intersection/union: each detector uses its own natural_flag boundary
    (detector-specific internal threshold, e.g. zero-hyperplane for OCSVM,
    Tukey fence for KMeans). Each detector independently decides its anomalies;
    contamination is not assumed to be equal across detectors.

``contamination=float``:
    Each detector (or the combined score in composite mode) is thresholded so
    that exactly that fraction of rows are flagged as anomalous.
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
        if weighting != "equal" and combination != "composite":
            logger.warning(
                "weighting=%r has no effect with combination=%r — weights only "
                "apply to 'composite'; the set operation ignores them.",
                weighting,
                combination,
            )

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

        weights = self._resolve_weights(names, score_matrix)
        combined = self._combine(score_matrix, weights)

        if self._combination in ("intersection", "union"):
            # Per-detector thresholding then set operation.
            # Each detector independently flags its top-contamination fraction
            # (or its natural boundary when auto), then flags are AND/OR-ed.
            per_flags, contamination, is_auto = self._set_combine_flags(score_matrix, natural_flags, names)
            anomaly_flag = (
                per_flags.all(axis=1) if self._combination == "intersection" else per_flags.any(axis=1)
            )
            per_rates = [round(float(per_flags[:, i].mean() * 100), 1) for i in range(len(names))]
            logger.info(
                "ScoreEnsemble: weighting=%s combination=%s contamination=%s%s "
                "per-detector rates %s → %s: %d/%d (%.1f%%).",
                self._weighting,
                self._combination,
                f"{contamination:.4f}" if not is_auto else "auto",
                " [heuristic]" if is_auto else "",
                dict(zip(names, per_rates, strict=True)),
                self._combination,
                int(anomaly_flag.sum()),
                n,
                100.0 * anomaly_flag.mean(),
            )
            threshold = float("nan")
            contamination_used = float(self._contamination) if not is_auto else contamination
        else:
            # composite: single global threshold on combined score
            contamination, is_auto = self._resolve_contamination(flag_matrix)
            threshold = float(np.quantile(combined, 1.0 - contamination))
            anomaly_flag = combined >= threshold
            contamination_used = contamination
            logger.info(
                "ScoreEnsemble: weighting=%s combination=%s contamination=%.4f%s "
                "threshold=%.4f flagged %d/%d (%.1f%%).",
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
            "contamination_used": contamination_used,
            "weights": dict(zip(names, weights, strict=True)),
            "is_auto_contamination": is_auto,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, names: list[str], score_matrix: np.ndarray) -> np.ndarray:
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

        return self._agreement_weights(score_matrix)

    def _agreement_weights(self, score_matrix: np.ndarray) -> np.ndarray:
        """Weight each detector by how well its ranking agrees with the consensus.

        For detector *i*, take Spearman's rho between its calibrated-score ranking
        and the mean rank of every *other* detector (leave-one-out). Negative
        correlations and constant-score detectors get weight 0; if nothing
        correlates, weights fall back to equal.

        This replaces the old raw-flag-agreement rule: at realistic contamination
        almost no row is flagged, so every detector was compared against an
        all-"normal" majority and a detector that flagged *nothing* scored
        highest and got the largest weight.
        """
        from scipy.stats import rankdata  # noqa: PLC0415

        n, k = score_matrix.shape
        if k == 1:
            return np.ones(1)
        if n < 2:
            return np.ones(k) / k

        ranks = rankdata(score_matrix, axis=0)  # (n, k), tie-safe average ranks
        w = np.zeros(k)
        for i in range(k):
            own = ranks[:, i]
            others = np.delete(ranks, i, axis=1).mean(axis=1)  # consensus rank of the rest
            # A constant-score detector carries no ranking signal → rho 0.
            flat = own.std() == 0.0 or others.std() == 0.0
            rho = 0.0 if flat else float(np.corrcoef(own, others)[0, 1])  # Pearson on ranks = Spearman
            w[i] = max(rho, 0.0)

        total = w.sum()
        if total == 0.0:
            logger.warning(
                "agreement weighting: no detector's ranking correlates with the "
                "consensus; falling back to equal weights."
            )
            return np.ones(k) / k
        return w / total

    def _combine(self, score_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if self._combination == "composite":
            return score_matrix @ weights  # weighted average, shape (n,)

        if self._combination == "intersection":
            return score_matrix.min(axis=1)

        # union
        return score_matrix.max(axis=1)

    def _set_combine_flags(
        self,
        score_matrix: np.ndarray,
        natural_flags: dict[str, np.ndarray],
        names: list[str],
    ) -> tuple[np.ndarray, float, bool]:
        """Compute per-detector boolean flags for intersection/union modes.

        Returns (per_flags, contamination_rate, is_auto).
        per_flags shape: (n_rows, n_detectors), dtype bool.
        """
        k = len(names)
        flags = np.zeros((score_matrix.shape[0], k), dtype=bool)

        if self._contamination == "auto":
            for i, name in enumerate(names):
                flags[:, i] = natural_flags[name]
            rates = np.array([natural_flags[n].mean() for n in names])
            rate = float(np.median(rates))
            rate = max(0.001, min(0.5, rate))
            return flags, rate, True

        # Explicit contamination: threshold each detector at its own quantile
        c = float(self._contamination)
        for i in range(k):
            thr_i = float(np.quantile(score_matrix[:, i], 1.0 - c))
            flags[:, i] = score_matrix[:, i] >= thr_i
        return flags, c, False

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
