"""``sorethumb backtest``: compare ensemble policies on the user's own dataset.

The synthetic benchmark and its pre-registered backtests (``evaluate.backtest``)
say how policies behave on known anomaly types. Whether a different policy is
better on *your* data is a separate question, answered here with the same
statistics, on your rows and your configuration:

* **Replicates.** A run fits and flags the same rows, so a backtest does the
  same on ``replicates`` random subsamples (``subsample`` of the rows each,
  without replacement), plus one fit on all rows. Every policy is applied to
  the *same* fitted detectors in each replicate, so comparisons are paired.
* **Labels.** With ``columns.reference_column`` (0/1, never used to fit), each
  candidate's per-replicate flag F1 is compared with the default's by a paired
  bootstrap, Holm-corrected across every candidate tested in the call -- so
  testing many policies and keeping the best is accounted for.
* **Decision.** A candidate *outperforms* the default only if its flag F1 is
  higher (Holm-adjusted p <= ``alpha``), it does not spend more than the review
  budget on wrong rows where the default does not, and its flags are no less
  stable across replicates (Jaccard with the all-rows fit) by more than
  ``stability_tolerance``.
* **No labels.** Accuracy cannot be assessed: the backtest reports flag volume
  and stability only, and records the verdict ``not_assessable``.
* **Forward mode** (``mode="forward"``, ``--forward``). Instead of random
  subsamples, every ``history.period_granularity`` period is fitted and the
  next one scored with those fits and calibrators -- what ``score --from-run``
  does from one period to the next. Each scored period is one paired
  replicate; stability compares its flags with those of a refit on a
  subsample of the fitted period.

Each comparison is recorded in the workspace (``policy_evidence``), scoped to
the logical dataset and to the configuration apart from the policy
(``scoring.policies.policy_scope_hash``). A run with a non-default policy looks
for an ``outperforms`` record in that scope -- see ``docs/ensemble-policy.md``.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from sorethumb_ml.errors import ConfigError, SchemaError, SorethumbWarning
from sorethumb_ml.evaluate.backtest import holm_adjust, paired_bootstrap
from sorethumb_ml.scoring.policies import (
    DEFAULT_POLICY_ID,
    POLICIES,
    EnsemblePolicy,
    policy_by_id,
    policy_scope_hash,
)

if TYPE_CHECKING:
    from sorethumb_ml.config import Config

logger = logging.getLogger(__name__)

VERDICT_OUTPERFORMS = "outperforms"
VERDICT_DOES_NOT = "does_not_outperform"
VERDICT_NOT_ASSESSABLE = "not_assessable"
PRIMARY_METRIC = "flag_f1"


@dataclass(frozen=True)
class PolicySummary:
    """One policy's mean metrics over the replicates (NaN where undefined, e.g. without labels)."""

    policy_id: str
    flag_count: float
    over_flag_ratio: float
    flag_jaccard_stability: float
    flag_precision: float = math.nan
    flag_recall: float = math.nan
    flag_f1: float = math.nan
    average_precision: float = math.nan
    false_positive_budget_ratio: float = math.nan


@dataclass(frozen=True)
class CandidateVerdict:
    """One candidate's comparison with the default, as recorded in the workspace."""

    policy_id: str
    verdict: str
    mean_difference: float = math.nan  # candidate - default flag F1
    ci_low: float = math.nan
    ci_high: float = math.nan
    p_holm: float = math.nan
    reasons: tuple[str, ...] = ()
    evidence_id: str | None = None


@dataclass(frozen=True)
class DatasetBacktest:
    """Everything one ``sorethumb backtest`` call measured and decided."""

    dataset_fp: str
    snapshot_fp: str
    labelled: bool
    review_budget: float
    n_rows: int
    replicates: int
    subsample: float
    baseline_policy_id: str
    summaries: tuple[PolicySummary, ...]
    verdicts: tuple[CandidateVerdict, ...]
    scope_hash: str
    notes: tuple[str, ...] = field(default=())
    mode: str = "random"  # "random" subsamples, or "forward": fit period t, score t+1
    periods: tuple[str, ...] = ()  # forward mode: the scored period labels, in order
    # Per policy, one metrics dict per replicate (random) or scored period (forward),
    # in the same order for every policy. Not part of as_dict: it is the raw material
    # the verdicts rest on, recorded in the evidence's details instead.
    per_replicate: dict[str, list[dict[str, float]]] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (NaN becomes None), without the per-replicate values."""
        d = asdict(self)
        d.pop("per_replicate")
        return dict(_clean(d))


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Choosing what to compare
# ---------------------------------------------------------------------------


def candidate_policies(config: Config, requested: list[str] | None = None) -> list[EnsemblePolicy]:
    """Return the policies to test against the default.

    By default every registered policy except the default itself, and except
    ``composite-manual`` unless ``scoring.weights`` gives it weights to use.

    Raises:
        ConfigError: an unknown policy id, or the default named as a candidate.

    """
    if requested:
        chosen = []
        for policy_id in requested:
            policy = policy_by_id(policy_id)
            if policy is None:
                known = ", ".join(p.id for p in POLICIES)
                msg = f"Unknown ensemble policy {policy_id!r}; registered policies: {known}."
                raise ConfigError(msg)
            if policy.id == DEFAULT_POLICY_ID:
                msg = f"{DEFAULT_POLICY_ID} is the default: it is the baseline every candidate is compared with."
                raise ConfigError(msg)
            chosen.append(policy)
        return chosen
    return [
        p
        for p in POLICIES
        if p.id != DEFAULT_POLICY_ID and (p.weighting != "manual" or bool(config.scoring.weights))
    ]


# ---------------------------------------------------------------------------
# Fitting one frame the way a run does
# ---------------------------------------------------------------------------


@dataclass
class _Flags:
    """Per-policy scores and flags for one fitted frame, aligned to ``row_ids``."""

    row_ids: np.ndarray
    scores: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]


def _fit_frame(
    df: pl.DataFrame,
    config: Config,
    parsed_temporal: Any,
    policies: list[EnsemblePolicy],
    score_df: pl.DataFrame | None = None,
) -> _Flags:
    """Fit the configured pipeline on *df* (per group, as a run does) and apply every policy.

    Flags *df* itself, as a run does -- or, with *score_df*, scores those rows
    with the detectors and calibrators fitted on *df*, as ``score --from-run``
    does for a later period. A *score_df* group with no fitted counterpart, or
    one too small to fit, is skipped.
    """
    from sorethumb_ml._pipeline import (  # noqa: PLC0415
        _fit_detector,
        _group_label,
        _slice_group_frame,
    )
    from sorethumb_ml.detectors import registry  # noqa: PLC0415
    from sorethumb_ml.detectors._protocol import score_and_flag  # noqa: PLC0415
    from sorethumb_ml.features.build import apply_feature_plan, fit_features  # noqa: PLC0415
    from sorethumb_ml.profiling.plan import build_feature_plan  # noqa: PLC0415
    from sorethumb_ml.scoring.combine import ScoreEnsemble  # noqa: PLC0415

    plan = build_feature_plan(df, config, parsed_temporal)
    fit_features(df, plan, config)
    group_by = config.columns.group_by
    groups: list[dict[str, Any]] = (
        df.group_by(group_by).agg(pl.len().alias("__n__")).to_dicts() if group_by else [{}]
    )
    groups.sort(key=lambda g: _group_label(g, group_by) if group_by else "")
    enabled = [d.name for d in config.detectors if d.enabled and d.name in registry]

    row_ids: list[np.ndarray] = []
    scores: dict[str, list[np.ndarray]] = {p.id: [] for p in policies}
    flags: dict[str, list[np.ndarray]] = {p.id: [] for p in policies}
    for ginfo in groups:
        group_values = {col: ginfo[col] for col in group_by} if group_by else {}
        df_group = _slice_group_frame(df, group_by, group_values, plan)
        if len(df_group) < config.scoring.min_records:
            continue  # a run skips it too; its rows are never flagged or evaluated
        fit_space = apply_feature_plan(df_group, plan)
        space = fit_space
        if score_df is not None:
            target = _slice_group_frame(score_df, group_by, group_values, plan)
            if len(target) == 0:
                continue
            space = apply_feature_plan(target, plan)
        calibrated: dict[str, np.ndarray] = {}
        natural: dict[str, np.ndarray] = {}
        for name in enabled:
            det, raw, nat, cal, _ = _fit_detector(name, fit_space.matrix, config)
            if score_df is not None:
                raw, nat = score_and_flag(det, space.matrix)
            calibrated[name] = cal.transform(raw)
            natural[name] = nat
        ids = np.asarray(space.row_ids, dtype=np.int64)
        row_ids.append(ids)
        for policy in policies:
            ensemble = ScoreEnsemble(
                weighting=policy.weighting or "equal",
                combination=policy.combination,
                contamination=config.scoring.contamination,
                manual_weights=config.scoring.weights or None,
            )
            out = ensemble.combine(calibrated, natural, source_row=space.row_ids)
            scores[policy.id].append(np.asarray(out["combined_score"], dtype=np.float64))
            flags[policy.id].append(np.asarray(out["anomaly_flag"], dtype=bool))

    def _cat(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

    return _Flags(
        row_ids=_cat(row_ids, np.int64),
        scores={k: _cat(v, np.float64) for k, v in scores.items()},
        flags={k: _cat(v, bool) for k, v in flags.items()},
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _labels(df: pl.DataFrame, column: str) -> dict[int, int]:
    """Map source row id -> 0/1 label from the reference column.

    Raises:
        SchemaError: the column is missing, or holds anything but 0/1 (or booleans).

    """
    from sorethumb_ml.io.fingerprint import INTERNAL_ROW_ID_COLUMN  # noqa: PLC0415

    if column not in df.columns:
        msg = f"columns.reference_column {column!r} is not in the dataset."
        raise SchemaError(msg)
    series = df[column]
    if series.null_count():
        msg = f"columns.reference_column {column!r} has {series.null_count()} missing value(s); every row needs a 0/1 label."
        raise SchemaError(msg)
    values = series.cast(pl.Int64, strict=False) if series.dtype != pl.Boolean else series.cast(pl.Int64)
    if values.null_count() or not set(values.unique().to_list()) <= {0, 1}:
        msg = f"columns.reference_column {column!r} must hold only 0/1 (or true/false) labels."
        raise SchemaError(msg)
    return dict(zip(df[INTERNAL_ROW_ID_COLUMN].to_list(), values.to_list(), strict=True))


def _replicate_metrics(
    fitted: _Flags,
    policy_id: str,
    full: _Flags,
    labels: dict[int, int] | None,
    review_budget: float,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score  # noqa: PLC0415

    from sorethumb_ml.evaluate.metrics import evaluate_flags  # noqa: PLC0415

    flags = fitted.flags[policy_id]
    n = len(flags)
    budget_rows = round(review_budget * n)
    full_flagged = set(full.row_ids[full.flags[policy_id]].tolist())
    rep_flagged = set(fitted.row_ids[flags].tolist())
    shared = set(fitted.row_ids.tolist())
    full_on_shared = full_flagged & shared
    union = rep_flagged | full_on_shared
    out = {
        "flag_count": float(flags.sum()),
        "over_flag_ratio": float(flags.sum()) / (review_budget * n) if n else math.nan,
        "flag_jaccard_stability": len(rep_flagged & full_on_shared) / len(union) if union else 1.0,
    }
    if labels is None or n == 0:
        return out
    y = np.array([labels[int(i)] for i in fitted.row_ids], dtype=np.int64)
    fm = evaluate_flags(flags, y)
    false_positives = int(np.count_nonzero(flags & (y == 0)))
    ap = math.nan
    if 0 < y.sum() < len(y):
        ap = float(average_precision_score(y, fitted.scores[policy_id]))
    out |= {
        "flag_precision": fm.precision,
        "flag_recall": fm.recall,
        "flag_f1": fm.f1,
        "average_precision": ap,
        "false_positive_budget_ratio": false_positives / budget_rows if budget_rows else math.nan,
    }
    return out


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = np.array([r.get(key, math.nan) for r in rows], dtype=np.float64)
    return math.nan if values.size == 0 or np.all(np.isnan(values)) else float(np.nanmean(values))


def _summary(policy_id: str, rows: list[dict[str, float]]) -> PolicySummary:
    return PolicySummary(
        policy_id=policy_id,
        flag_count=_mean(rows, "flag_count"),
        over_flag_ratio=_mean(rows, "over_flag_ratio"),
        flag_jaccard_stability=_mean(rows, "flag_jaccard_stability"),
        flag_precision=_mean(rows, "flag_precision"),
        flag_recall=_mean(rows, "flag_recall"),
        flag_f1=_mean(rows, "flag_f1"),
        average_precision=_mean(rows, "average_precision"),
        false_positive_budget_ratio=_mean(rows, "false_positive_budget_ratio"),
    )


def decide_candidates(
    per_replicate: dict[str, list[dict[str, float]]],
    candidates: list[str],
    *,
    labelled: bool,
    alpha: float,
    stability_tolerance: float,
    n_resamples: int,
) -> list[CandidateVerdict]:
    """Apply the decision rule to every candidate against ``DEFAULT_POLICY_ID`` (pure; no fitting)."""
    base = per_replicate[DEFAULT_POLICY_ID]
    if not labelled:
        return [
            CandidateVerdict(
                policy_id=c,
                verdict=VERDICT_NOT_ASSESSABLE,
                reasons=("no columns.reference_column labels: accuracy cannot be assessed",),
            )
            for c in candidates
        ]
    estimates = []
    for c in candidates:
        diffs = np.array(
            [r[PRIMARY_METRIC] - b[PRIMARY_METRIC] for b, r in zip(base, per_replicate[c], strict=True)],
            dtype=np.float64,
        )
        estimates.append(paired_bootstrap(diffs, n_resamples=n_resamples, alpha=alpha))
    adjusted = holm_adjust([e.p_value for e in estimates])
    verdicts = []
    b_fp, b_stab = _mean(base, "false_positive_budget_ratio"), _mean(base, "flag_jaccard_stability")
    for c, est, p_holm in zip(candidates, estimates, adjusted, strict=True):
        reasons: list[str] = []
        if not (est.mean > 0.0 and p_holm <= alpha):
            reasons.append(
                f"flag F1 not significantly higher than {DEFAULT_POLICY_ID} "
                f"(mean difference {est.mean:+.4f}, Holm p {p_holm:.4f})"
            )
        c_fp, c_stab = (
            _mean(per_replicate[c], "false_positive_budget_ratio"),
            _mean(per_replicate[c], "flag_jaccard_stability"),
        )
        if not math.isnan(c_fp) and c_fp > 1.0 and not (b_fp > 1.0):
            reasons.append(
                f"spends more than the review budget on wrong rows (false-positive budget ratio "
                f"{c_fp:.2f} > 1.0; {DEFAULT_POLICY_ID}: {b_fp:.2f})"
            )
        if not (math.isnan(c_stab) or math.isnan(b_stab)) and c_stab < b_stab - stability_tolerance:
            reasons.append(
                f"flags less stable across replicates (Jaccard {c_stab:.3f} vs {b_stab:.3f}, "
                f"tolerance {stability_tolerance:g})"
            )
        verdicts.append(
            CandidateVerdict(
                policy_id=c,
                verdict=VERDICT_DOES_NOT if reasons else VERDICT_OUTPERFORMS,
                mean_difference=est.mean,
                ci_low=est.ci_low,
                ci_high=est.ci_high,
                p_holm=float(p_holm),
                reasons=tuple(reasons),
            )
        )
    return verdicts


# ---------------------------------------------------------------------------
# Forward (out-of-time) folds
# ---------------------------------------------------------------------------


def period_frames(df: pl.DataFrame, config: Config) -> list[tuple[str, pl.DataFrame]]:
    """Split *df* into its ``history.period_granularity`` periods, in time order.

    Each period's rows are exactly what ``sorethumb run --period LABEL`` (or a
    backfill of that label) would process. Periods with no rows are dropped.

    Raises:
        ConfigError: no ``columns.time_column`` is configured.
        SchemaError: the time column is missing or entirely null.

    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from sorethumb_ml._pipeline import _period_window_expr  # noqa: PLC0415
    from sorethumb_ml.history.periods import period_bounds, period_range, resolve_period  # noqa: PLC0415

    time_col = config.columns.time_column
    if not time_col:
        msg = "A forward backtest needs columns.time_column: it fits one period and scores the next."
        raise ConfigError(msg)
    if time_col not in df.columns:
        msg = f"columns.time_column={time_col!r} is not in the dataset."
        raise SchemaError(msg)
    observed = df[time_col].drop_nulls()
    if len(observed) == 0:
        msg = f"columns.time_column={time_col!r} has no values."
        raise SchemaError(msg)

    def _as_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return datetime(value.year, value.month, value.day, tzinfo=UTC)  # a date

    granularity = config.history.period_granularity
    roll = config.history.roll_non_business
    first = resolve_period(_as_datetime(observed.min()), granularity, roll)[2]
    last = resolve_period(_as_datetime(observed.max()), granularity, roll)[2]
    frames = []
    for label in period_range(first, last, granularity):
        lo, hi = period_bounds(label, granularity)
        frame = df.filter(_period_window_expr(time_col, df[time_col].dtype, lo, hi))
        if len(frame):
            frames.append((label, frame))
    return frames


def _forward_folds(
    periods: list[tuple[str, pl.DataFrame]],
    config: Config,
    parsed_temporal: Any,
    policies: list[EnsemblePolicy],
    labels: dict[int, int] | None,
    budget: float,
    subsample: float,
) -> tuple[list[str], dict[str, list[dict[str, float]]]]:
    """Fit each period, score the next; stability from a refit on a subsample of the fitted period."""
    per_fold: dict[str, list[dict[str, float]]] = {p.id: [] for p in policies}
    scored: list[str] = []
    for i, ((_, train), (label, test)) in enumerate(itertools.pairwise(periods)):
        if len(train) < config.scoring.min_records:
            continue
        main = _fit_frame(train, config, parsed_temporal, policies, score_df=test)
        if len(main.row_ids) == 0:
            continue
        rng = np.random.default_rng(config.run.seed + 7919 * (i + 1))
        n_sub = max(2, round(len(train) * subsample))
        idx = np.sort(rng.choice(len(train), size=n_sub, replace=False))
        refit = _fit_frame(train[idx.tolist()], config, parsed_temporal, policies, score_df=test)
        scored.append(label)
        for p in policies:
            per_fold[p.id].append(_replicate_metrics(main, p.id, refit, labels, budget))
    return scored, per_fold


def _measure(
    df: pl.DataFrame,
    config: Config,
    parsed_temporal: Any,
    all_policies: list[EnsemblePolicy],
    labels: dict[int, int] | None,
    budget: float,
    *,
    mode: str,
    folds: int | None,
    replicates: int,
    subsample: float,
) -> tuple[list[str], dict[str, list[dict[str, float]]], int]:
    """Fit and score every replicate (or period pair); return (scored periods, per-replicate metrics, count)."""
    scored_periods: list[str] = []
    with warnings.catch_warnings():
        # Per-fit advisories (an empty intersection, an anti-correlated member)
        # are expected across replicates and say nothing about the comparison.
        warnings.simplefilter("ignore", SorethumbWarning)
        if mode == "forward":
            periods = period_frames(df, config)
            if folds is not None:
                periods = periods[-(folds + 1) :]
            scored_periods, per_replicate = _forward_folds(
                periods, config, parsed_temporal, all_policies, labels, budget, subsample
            )
            if len(scored_periods) < 5:
                msg = (
                    f"A forward backtest needs at least 5 scorable period pairs at "
                    f"history.period_granularity={config.history.period_granularity!r}; the data "
                    f"has {len(scored_periods)}. Use a finer granularity or more history."
                )
                raise ConfigError(msg)
            replicates = len(scored_periods)
        else:
            full = _fit_frame(df, config, parsed_temporal, all_policies)
            per_replicate = {p.id: [] for p in all_policies}
            n_sub = max(2, round(len(df) * subsample))
            for r in range(replicates):
                rng = np.random.default_rng(config.run.seed + 7919 * (r + 1))
                idx = np.sort(rng.choice(len(df), size=n_sub, replace=False))
                fitted = _fit_frame(df[idx.tolist()], config, parsed_temporal, all_policies)
                for p in all_policies:
                    per_replicate[p.id].append(_replicate_metrics(fitted, p.id, full, labels, budget))
    return scored_periods, per_replicate, replicates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def backtest_dataset(
    config: Config,
    *,
    policies: list[str] | None = None,
    replicates: int = 20,
    subsample: float = 0.8,
    alpha: float = 0.05,
    stability_tolerance: float = 0.05,
    n_resamples: int = 10_000,
    record: bool = True,
    mode: str = "random",
    folds: int | None = None,
) -> DatasetBacktest:
    """Compare ensemble policies with the default on the configured dataset; record the evidence.

    See the module docstring for the method. *record* False measures without
    writing evidence to the workspace. *mode* ``"forward"`` replaces the random
    subsamples with time order: every ``history.period_granularity`` period is
    fitted and the next one scored with those fits (the last *folds* such pairs,
    default all, at least 5); stability compares those flags with a refit on
    *subsample* of the fitted period.

    Raises:
        ConfigError: ``scoring.contamination`` is ``"auto"`` (a backtest needs a
            fixed review budget), or an invalid argument or policy id.
        SchemaError: ``columns.reference_column`` is missing or not 0/1.

    """
    from sorethumb_ml.store.workspace import Workspace  # noqa: PLC0415

    budget = config.scoring.contamination
    if not isinstance(budget, float):
        msg = (
            "sorethumb backtest needs a fixed review budget: set scoring.contamination to the "
            "fraction of rows you can review (e.g. 0.05) instead of 'auto'. Evidence is only "
            "valid for runs at that same budget."
        )
        raise ConfigError(msg)
    if mode not in ("random", "forward"):
        msg = f"mode must be 'random' or 'forward'; got {mode!r}."
        raise ConfigError(msg)
    if mode == "random" and replicates < 5:
        msg = f"replicates must be at least 5 for a paired bootstrap to mean anything; got {replicates}."
        raise ConfigError(msg)
    if folds is not None and folds < 5:
        msg = f"folds must be at least 5 for a paired bootstrap to mean anything; got {folds}."
        raise ConfigError(msg)
    if not 0.2 <= subsample < 1.0:
        msg = f"subsample must be in [0.2, 1.0); got {subsample}."
        raise ConfigError(msg)

    baseline = policy_by_id(DEFAULT_POLICY_ID)
    assert baseline is not None
    candidates = candidate_policies(config, policies)
    if not candidates:
        msg = "No candidate policies to test."
        raise ConfigError(msg)
    all_policies = [baseline, *candidates]

    ws_path = Path(config.run.workdir)
    ws = Workspace.open(ws_path) if (ws_path / "sorethumb.db").exists() else Workspace.init(ws_path)
    with ws:
        df, parsed_temporal, ident = _load_frame(config, ws, filter_period=mode == "random")
        dataset_fp, snapshot_fp = ident["dataset_fp"], ident["snapshot_fp"]

        ref = config.columns.reference_column
        labels = _labels(df, ref) if ref else None
        notes: list[str] = []
        if labels is None:
            notes.append(
                "columns.reference_column is not set: flag volume and stability only; accuracy, "
                "and so whether a policy outperforms the default, cannot be assessed."
            )

        logger.info(
            "backtest: %d rows, %d replicates of %.0f%%, policies %s",
            len(df),
            replicates,
            subsample * 100,
            [p.id for p in all_policies],
        )
        scored_periods, per_replicate, replicates = _measure(
            df,
            config,
            parsed_temporal,
            all_policies,
            labels,
            budget,
            mode=mode,
            folds=folds,
            replicates=replicates,
            subsample=subsample,
        )

        verdicts = decide_candidates(
            per_replicate,
            [c.id for c in candidates],
            labelled=labels is not None,
            alpha=alpha,
            stability_tolerance=stability_tolerance,
            n_resamples=n_resamples,
        )
        scope = policy_scope_hash(config)
        if record:
            verdicts = _record_evidence(
                ws,
                config,
                verdicts,
                per_replicate,
                ident,
                labelled=labels is not None,
                scope=scope,
                settings={
                    "mode": mode,
                    "periods": scored_periods,
                    "replicates": replicates,
                    "subsample": subsample,
                    "alpha": alpha,
                    "stability_tolerance": stability_tolerance,
                },
            )

    return DatasetBacktest(
        dataset_fp=dataset_fp,
        snapshot_fp=snapshot_fp,
        labelled=labels is not None,
        review_budget=budget,
        n_rows=len(df),
        replicates=replicates,
        subsample=subsample,
        baseline_policy_id=DEFAULT_POLICY_ID,
        summaries=tuple(_summary(p.id, per_replicate[p.id]) for p in all_policies),
        verdicts=tuple(verdicts),
        scope_hash=scope,
        notes=tuple(notes),
        mode=mode,
        periods=tuple(scored_periods),
        per_replicate=per_replicate,
    )


def _nan_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def _load_frame(
    config: Config, ws: Any, *, filter_period: bool = True
) -> tuple[pl.DataFrame, Any, dict[str, Any]]:
    """Load and prepare the dataset exactly as a run does (before group fitting).

    *filter_period* False keeps every period (a forward backtest walks them all).
    """
    from sorethumb_ml._pipeline import (  # noqa: PLC0415
        _apply_row_cap,
        _resolve_and_filter_period,
        _stamp_source_row_id,
    )
    from sorethumb_ml.io.fingerprint import (  # noqa: PLC0415
        content_fingerprint,
        logical_dataset_id,
        schema_fingerprint,
        snapshot_fingerprint,
    )
    from sorethumb_ml.io.nested import unnest_all  # noqa: PLC0415
    from sorethumb_ml.io.readers import collect_frame, read_frame  # noqa: PLC0415
    from sorethumb_ml.io.source import resolve_source  # noqa: PLC0415
    from sorethumb_ml.profiling.temporal import parse_temporal_columns  # noqa: PLC0415

    local_path = resolve_source(config.source, ws.root / "cache" / "datasets")
    df = collect_frame(read_frame(local_path, config.source), local_path)
    if config.source.max_nesting_depth > 0:
        df = unnest_all(df, config.source.max_nesting_depth)
    content_fp, schema_fp = content_fingerprint(local_path), schema_fingerprint(df)
    raw_shape = df.shape  # what a run records on the dataset row: before stamping and filtering
    dataset_fp = logical_dataset_id(config.source.dataset_id, config.source.uri)
    snapshot_fp = snapshot_fingerprint(content_fp, schema_fp)
    df = _stamp_source_row_id(df)
    df, parsed_temporal = parse_temporal_columns(df, config.columns)
    if filter_period:
        df, _, _ = _resolve_and_filter_period(df, config, None)
    df = _apply_row_cap(df, config.run.max_rows)

    ident = {
        "dataset_fp": dataset_fp,
        "snapshot_fp": snapshot_fp,
        "content_fp": content_fp,
        "schema_fp": schema_fp,
        "n_rows": raw_shape[0],
        "n_cols": raw_shape[1],
    }
    return df, parsed_temporal, ident


def _record_evidence(
    ws: Any,
    config: Config,
    verdicts: list[CandidateVerdict],
    per_replicate: dict[str, list[dict[str, float]]],
    ident: dict[str, Any],
    *,
    labelled: bool,
    scope: str,
    settings: dict[str, Any],
) -> list[CandidateVerdict]:
    """Write one ``policy_evidence`` row per verdict; return the verdicts with their ids."""
    import sorethumb_ml  # noqa: PLC0415
    from sorethumb_ml.io.uri import display_source_uri, source_digest  # noqa: PLC0415

    ws.store.upsert_dataset(
        dataset_fp=ident["dataset_fp"],
        source_uri=display_source_uri(config.source.uri),
        source_digest=source_digest(config.source.uri),
        schema_fingerprint=ident["schema_fp"],
        content_fingerprint=ident["content_fp"],
        n_rows=ident["n_rows"],
        n_cols=ident["n_cols"],
        snapshot_fp=ident["snapshot_fp"],
    )
    recorded = []
    for v in verdicts:
        evidence_id = f"ev_{uuid.uuid4().hex[:16]}"
        details = {
            **settings,
            "reasons": list(v.reasons),
            "per_replicate": {
                DEFAULT_POLICY_ID: per_replicate[DEFAULT_POLICY_ID],
                v.policy_id: per_replicate[v.policy_id],
            },
        }
        ws.store.insert_policy_evidence(
            {
                "evidence_id": evidence_id,
                "dataset_fp": ident["dataset_fp"],
                "snapshot_fp": ident["snapshot_fp"],
                "policy_id": v.policy_id,
                "baseline_policy_id": DEFAULT_POLICY_ID,
                "scope_hash": scope,
                "verdict": v.verdict,
                "labelled": int(labelled),
                "primary_metric": PRIMARY_METRIC if labelled else None,
                "mean_difference": _nan_none(v.mean_difference),
                "ci_low": _nan_none(v.ci_low),
                "ci_high": _nan_none(v.ci_high),
                "p_holm": _nan_none(v.p_holm),
                "n_replicates": settings["replicates"],
                "review_budget": config.scoring.contamination,
                "library_version": sorethumb_ml.__version__,
                "details_json": json.dumps(_clean(details), sort_keys=True),
            }
        )
        recorded.append(CandidateVerdict(**{**asdict(v), "evidence_id": evidence_id}))
    return recorded
