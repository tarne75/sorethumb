"""Composite score combination and anomaly thresholding.

ScoreEnsemble takes multiple per-detector calibrated scores (each in [0, 1],
higher = more anomalous) and combines them into a single final score plus a
binary anomaly flag.

Bad-member guard
----------------
Before weighting or combining, any detector whose score *ranking* is
anti-correlated with the ensemble median (Spearman's rho < ``_ANTICORR_DROP``)
is a candidate to drop from the combination — it is fighting the consensus
rather than adding a diverse-but-consistent view. Needs at least three members.

The guard only actually drops members for combination="composite", where
removing an outlier from a weighted average does not change the decision
rule. For combination="intersection"/"union" it does **not** drop anyone:
these modes are a vote, and silently excluding a detector silently changes
the vote count — a configured three-way intersection would quietly become a
two-way one with no visible change to `scoring.detectors`. Every configured
detector's vote is kept; an :class:`~sorethumb.errors.AntiCorrelatedMemberWarning`
is emitted instead (promoted to an error under ``run.strict``), so the signal
is surfaced without changing what the user asked for.

Dropped detectors (composite mode only) are listed in the result's
``dropped_members`` and appear in ``weights`` with 0.0; their per-detector
score columns are still recorded for inspection.

Weighting strategies
--------------------
equal:
    All detectors receive weight 1 / n. Default, no configuration needed.
manual:
    User supplies a dict mapping detector name → weight. Weights must be
    finite, non-negative, and sum to a positive total (validated at
    construction) -- normalised to sum to 1.0 before use.
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
composite mode, contamination="auto":
    Binary flag is set where combined score >= threshold, where threshold is
    the (1 − contamination) quantile of the combined score distribution
    (contamination itself a heuristic: the median natural_flag rate across
    detectors).

intersection / union modes, contamination="auto":
    Each detector uses its own natural_flag boundary (detector-specific
    internal threshold, e.g. zero-hyperplane for OCSVM, Tukey fence for
    KMeans). Each detector independently decides its anomalies;
    contamination is not assumed to be equal across detectors.

``contamination=float`` (composite, intersection, and union alike):
    Exactly ``k = round(n * contamination)`` rows are flagged -- the
    highest-scoring ones (per detector, for intersection/union; on the
    combined score, for composite) -- via ``_exact_k_flags``, never a
    quantile threshold. A quantile threshold flags every row *at* the
    boundary, which over- or under-shoots the requested fraction whenever
    there are ties there (routine for small groups and heavily-tied
    detectors like ECOD/HBOS). Ties are broken deterministically: highest
    score first, then earliest source row order for rows that tie exactly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from sorethumb.errors import AntiCorrelatedMemberWarning

logger = logging.getLogger(__name__)

# Bad-member guard: a detector whose score ranking has Spearman's rho below this
# against the per-row *median* rank of the other detectors is fighting the
# consensus (not merely diverse). Dropped from combination="composite"; kept
# (with a warning) for "intersection"/"union", where every configured vote is
# required by definition. Needs >= 3 members.
_ANTICORR_DROP = -0.15


def _spearman_ranks(score_matrix: np.ndarray) -> np.ndarray:
    """Tie-safe average ranks of each column of *score_matrix*, shape (n, k)."""
    from scipy.stats import rankdata  # noqa: PLC0415

    return rankdata(score_matrix, axis=0)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of *a* and *b*; 0.0 if either is constant."""
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _exact_k_flags(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Flag exactly ``k = round(n * contamination)`` rows -- the highest-scoring ones.

    Ties are broken deterministically by score, then by source row order
    (an earlier row wins a tie) -- a plain ``score >= quantile(1 - c)``
    threshold flags every tied row at the boundary, which can over- or
    under-shoot the requested fraction whenever the reference has repeated
    values (small groups and heavily-ties detectors like ECOD/HBOS hit this
    routinely). ``np.argsort(..., kind="stable")`` on ``-scores`` sorts
    descending by score while a stable sort's own guarantee -- equal keys
    keep their original relative order -- gives the row-order tie-break for
    free, with no extra key needed.
    """
    n = len(scores)
    k = round(n * contamination)
    k = max(0, min(k, n))
    flags = np.zeros(n, dtype=bool)
    if k == 0:
        return flags
    order = np.argsort(-scores, kind="stable")
    flags[order[:k]] = True
    return flags


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
        if weighting == "manual" and manual_weights:
            values = list(manual_weights.values())
            if not all(np.isfinite(v) for v in values):
                msg = f"manual_weights must be finite; got {manual_weights}"
                raise ValueError(msg)
            if any(v < 0 for v in values):
                msg = f"manual_weights must be non-negative; got {manual_weights}"
                raise ValueError(msg)
            if sum(values) <= 0:
                msg = f"manual_weights must sum to a positive total; got {manual_weights}"
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
            "contamination_used": float — the review-budget fraction actually
                applied; not an estimate of true anomaly prevalence
            "weights": dict[str, float] (dropped members appear with weight 0.0)
            "is_auto_contamination": bool
            "dropped_members": list[str] (detectors excluded by the bad-member guard;
                always empty for combination="intersection"/"union" -- see module docstring)
            "per_detector_rates": dict[str, float] — realised fraction each
                detector's own natural boundary flagged, all input detectors

        """
        names = list(scores.keys())
        if not names:
            msg = "scores dict is empty; need at least one detector."
            raise ValueError(msg)

        n = len(next(iter(scores.values())))

        # Realised rate of each detector's own heuristic boundary, before any
        # weighting, guard, or combination. Surfaced so nobody mistakes the
        # combined flag count (or contamination="auto", their median) for an
        # estimate of how many anomalies the data contains.
        per_detector_rates = {d: float(np.asarray(natural_flags[d], dtype=bool).mean()) for d in names}

        score_matrix = np.column_stack([scores[d] for d in names])  # shape (n, k)

        # ── Bad-member guard ──────────────────────────────────────────────
        # Detect any detector whose ranking is anti-correlated with the
        # ensemble median. Only actually dropped for combination="composite",
        # where excluding an outlier from a weighted average does not change
        # the decision rule. For "intersection"/"union" every configured vote
        # is required by definition -- dropping one silently changes the vote
        # count, so nothing is excluded there; a warning is raised instead
        # (see module docstring / AntiCorrelatedMemberWarning).
        keep_idx, candidate_drop = self._screen_members(names, score_matrix)
        dropped: list[str] = []
        if candidate_drop and self._combination == "composite":
            dropped = candidate_drop
            logger.warning(
                "ensemble guard: %s rank against the consensus median (Spearman rho < %.2f); "
                "excluded from the combination.",
                dropped,
                _ANTICORR_DROP,
            )
            names = [names[i] for i in keep_idx]
            score_matrix = score_matrix[:, keep_idx]
            natural_flags = {d: natural_flags[d] for d in names}
        elif candidate_drop:
            warnings.warn(
                f"ensemble guard: {candidate_drop} rank against the consensus median "
                f"(Spearman rho < {_ANTICORR_DROP:.2f}), but combination={self._combination!r} "
                "requires every configured detector's vote -- none excluded. Review "
                "scoring.detectors, or run under run.strict to fail loudly on this instead.",
                AntiCorrelatedMemberWarning,
                stacklevel=2,
            )

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
            if is_auto:
                # contamination="auto" is a heuristic estimate, not a hard
                # constraint -- left unchanged by P2-3, which only tightens
                # *explicit* numeric contamination (see module docstring).
                threshold = float(np.quantile(combined, 1.0 - contamination))
                anomaly_flag = combined >= threshold
            else:
                # Explicit contamination: exact-k select the top-scoring
                # rows rather than a quantile threshold, which over- or
                # under-flags whenever the combined score has ties at the
                # boundary (see _exact_k_flags).
                anomaly_flag = _exact_k_flags(combined, contamination)
                threshold = float(combined[anomaly_flag].min()) if anomaly_flag.any() else float("nan")
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

        weights_out = dict(zip(names, weights, strict=True))
        for d in dropped:
            weights_out[d] = 0.0

        return {
            "combined_score": combined,
            "anomaly_flag": anomaly_flag,
            "threshold": threshold,
            "contamination_used": contamination_used,
            "weights": weights_out,
            "is_auto_contamination": is_auto,
            "dropped_members": dropped,
            "per_detector_rates": per_detector_rates,
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

    @staticmethod
    def _screen_members(names: list[str], score_matrix: np.ndarray) -> tuple[list[int], list[str]]:
        """Return ``(kept_indices, dropped_names)`` for the bad-member guard.

        A member is dropped when Spearman's rho between its ranking and the
        per-row *median* rank of the other members is below ``_ANTICORR_DROP``.
        Needs >= 3 members (with 2 you cannot say which one is the outlier) and
        >= 3 rows; never drops more than ``k - 2`` (a run where most members
        look bad means the "consensus" itself is unreliable — keep everything
        and let the warning stand).
        """
        n, k = score_matrix.shape
        if k < 3 or n < 3:
            return list(range(k)), []

        ranks = _spearman_ranks(score_matrix)
        bad = [
            i
            for i in range(k)
            if _rho(ranks[:, i], np.median(np.delete(ranks, i, axis=1), axis=1)) < _ANTICORR_DROP
        ]
        if len(bad) > k - 2:
            return list(range(k)), []
        return [i for i in range(k) if i not in bad], [names[i] for i in bad]

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
        n, k = score_matrix.shape
        if k == 1:
            return np.ones(1)
        if n < 2:
            return np.ones(k) / k

        ranks = _spearman_ranks(score_matrix)  # (n, k), tie-safe average ranks
        w = np.array(
            [max(_rho(ranks[:, i], np.delete(ranks, i, axis=1).mean(axis=1)), 0.0) for i in range(k)]
        )

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

        # Explicit contamination: exact-k select each detector independently
        # (ties broken by score then source row order; see _exact_k_flags),
        # rather than a per-detector quantile threshold that over- or
        # under-flags whenever that detector's scores have ties at the
        # boundary.
        c = float(self._contamination)
        for i in range(k):
            flags[:, i] = _exact_k_flags(score_matrix[:, i], c)
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
