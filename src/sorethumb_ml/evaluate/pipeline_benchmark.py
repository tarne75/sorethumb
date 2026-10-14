"""Full-pipeline synthetic benchmark: mixed numeric+categorical scenarios through sorethumb's real pipeline.

Fits each scenario's data through real feature encoding + scaling +
detectors + calibration + ensemble, compared against equivalent bare-sklearn
baselines with no feature pipeline at all.

Replaces the earlier bare-detector, in-sample-only synthetic benchmark rows in
``evaluate/benchmark.py`` (which is still used, unchanged, for its real-dataset
network-fetched smoke coverage -- KDDCup99/Covtype have no controllable
point/local/contextual/masking/swamping/varying-density taxonomy to offer, so
migrating them here would not add anything; see module docstring there).

Design principles
------------------
- Every scenario fits on one split and scores a disjoint held-out split
  (never trains and evaluates on the same rows) -- see ``evaluate/scenarios.py``.
- Every (scenario, ablation) cell is run over several seeds and reported as
  mean plus a 95% confidence interval (not just a bare mean, and not just a
  standard deviation with no distributional meaning attached).
- Every cell also runs in a fully isolated subprocess (``multiprocessing``,
  spawn context) so ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` -- a real
  OS-tracked high-water mark, not a before/after snapshot delta -- measures
  that cell's own peak memory, uncontaminated by every other cell's
  accumulated state in a shared long-lived process. This replaces the
  ``peak_rss_mb`` field removed earlier for being exactly
  that kind of unreliable before/after psutil delta.
- ``AblationSpec`` varies one pipeline knob at a time (PCA, scaler, detector
  set) against the same scenario/seeds, so a regression in one knob doesn't
  hide inside an aggregate number.
- ``run_pipeline_benchmark(..., include_baselines=True)`` (the default) also
  fits bare sklearn detectors (one-hot via pandas get_dummies + StandardScaler,
  no PCA, no calibration, no ensemble) on the exact same scenario data, so a
  claim like "the pipeline helps" has something concrete to compare against.
  PyOD is deliberately not used for this comparison (see
  docs/approximations.md) -- sklearn is already a dependency, PyOD is not.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sorethumb_ml.detectors._protocol import score_and_flag
from sorethumb_ml.evaluate.scenarios import (
    SCENARIOS,
    Scenario,
    scenario_by_name,
    swamping_at_risk_holdout,
    swamping_clean_reference,
    swamping_train_reference,
)

logger = logging.getLogger(__name__)

HOLDOUT_FRACTION = 0.3

_DEFAULT_DETECTORS = ("isolation_forest", "kmeans_distance", "one_class_svm")
_ALL_DETECTORS = (*_DEFAULT_DETECTORS, "ecod", "lof", "hbos")


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationSpec:
    """One pipeline configuration variant, applied on top of a scenario's data."""

    name: str
    pca: bool = False
    scaler: str = "robust"
    detector_names: tuple[str, ...] = _DEFAULT_DETECTORS
    combination: str = "intersection"
    weighting: str = "equal"
    # None: the shipped profiling.near_constant_distinct default.
    near_constant_distinct: int | None = None

    @property
    def policy_id(self) -> str:
        """The versioned ensemble policy this ablation runs (scoring.policies)."""
        from sorethumb_ml.scoring.policies import resolve_policy  # noqa: PLC0415

        return resolve_policy(self.combination, self.weighting).id


ABLATIONS: list[AblationSpec] = [
    AblationSpec("default"),
    AblationSpec("pca_on", pca=True),
    AblationSpec("scaler_standard", scaler="standard"),
    AblationSpec("isolation_forest_only", detector_names=("isolation_forest",)),
    AblationSpec("kmeans_distance_only", detector_names=("kmeans_distance",)),
    AblationSpec("one_class_svm_only", detector_names=("one_class_svm",)),
    AblationSpec("lof_only", detector_names=("lof",)),
    AblationSpec("all_detectors", detector_names=_ALL_DETECTORS),
    AblationSpec("combination_composite", combination="composite"),
    AblationSpec("combination_union", combination="union"),
    AblationSpec("combination_composite_agreement", combination="composite", weighting="agreement"),
    # Keep 3-value columns (the threshold before 2026-10-14's restore of 3) -- the
    # evidence for which profiling.near_constant_distinct default to ship.
    AblationSpec("near_constant_2", near_constant_distinct=2),
]

# One ablation per registered ensemble policy, everything else at the shipped
# default: the policy matrix compares policies, not other knobs.
POLICY_ABLATIONS: tuple[str, ...] = (
    "default",
    "combination_union",
    "combination_composite",
    "combination_composite_agreement",
)

DEFAULT_ABLATION = ABLATIONS[0]


def _swamping_generate_not_called(seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Never invoked; swamping's data comes from swamping_variant in the worker, not .generate."""
    msg = "swamping_scenario.generate is a placeholder and should never be called directly"
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------


@dataclass
class PipelineBenchmarkRow:
    """One (scenario, ablation-or-baseline) cell's aggregated result."""

    scenario: str
    kind: str
    ablation: str  # an AblationSpec.name, or "sklearn:<detector>" for a baseline row
    policy: str | None  # versioned ensemble policy id; None for a "sklearn:*" baseline row
    n_seeds: int
    n_train: int
    n_holdout: int
    review_budget: float

    roc_auc: float
    roc_auc_ci95: float
    average_precision: float
    average_precision_ci95: float
    precision_at_k: float
    recall_at_k: float
    f1_at_contamination: float

    fit_seconds: float
    score_seconds: float
    peak_memory_mb: float | None

    error: str | None = None

    # Metrics of the *actual* anomaly_flag this cell's ensemble decided on --
    # as opposed to roc_auc/average_precision/precision_at_k/recall_at_k/
    # f1_at_contamination above, which all describe the continuous ranking
    # (or a hypothetical top-k cut on it), never the real flag decision. None
    # for a "sklearn:*" baseline row (no ensemble/flag concept there).
    flag_precision: float | None = None
    flag_recall: float | None = None
    flag_f1: float | None = None
    flag_false_positive_rate: float | None = None
    flag_count: float | None = None
    # flag_count / (review_budget * n_holdout): 1.0 flags exactly the review
    # budget, above 1.0 over-flags it.
    over_flag_ratio: float | None = None
    # Mean overlap of the flagged rows, and Spearman correlation of the scores,
    # between the full fit and refits on training subsamples (NaN when the run
    # was configured with n_bootstrap=0). See _fit_score_one_seed_pipeline.
    flag_jaccard_stability: float | None = None
    rank_stability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Flatten to a JSON/CSV-serialisable dict."""
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "ablation": self.ablation,
            "policy": self.policy,
            "n_seeds": self.n_seeds,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "review_budget": self.review_budget,
            "roc_auc": self.roc_auc,
            "roc_auc_ci95": self.roc_auc_ci95,
            "average_precision": self.average_precision,
            "average_precision_ci95": self.average_precision_ci95,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_contamination": self.f1_at_contamination,
            "fit_seconds": self.fit_seconds,
            "score_seconds": self.score_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "error": self.error,
            "flag_precision": self.flag_precision,
            "flag_recall": self.flag_recall,
            "flag_f1": self.flag_f1,
            "flag_false_positive_rate": self.flag_false_positive_rate,
            "flag_count": self.flag_count,
            "over_flag_ratio": self.over_flag_ratio,
            "flag_jaccard_stability": self.flag_jaccard_stability,
            "rank_stability": self.rank_stability,
        }


def _ci95(values: list[float]) -> float:
    """Half-width of a 95% CI on the mean, via a normal approximation.

    NaN-safe: values undefined for a given seed (see evaluate_scores) are
    dropped before the CI is computed, matching how the mean itself is
    computed with nanmean.
    """
    arr = np.asarray([v for v in values if not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _peak_memory_mb() -> float | None:
    """Return this process's peak resident memory so far, in MB, or None where unavailable.

    POSIX: ``getrusage(RUSAGE_SELF).ru_maxrss``. Windows: the peak working set
    from ``GetProcessMemoryInfo`` (the Windows counterpart of peak RSS).
    """
    if sys.platform == "win32":
        return _windows_peak_working_set_mb()
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # POSIX ru_maxrss units are platform-defined: Linux reports KB, macOS bytes.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return max_rss / divisor


def _windows_peak_working_set_mb() -> float | None:
    """Return this process's PeakWorkingSetSize in MB via psapi, or None if the call fails."""
    if sys.platform != "win32":  # keeps mypy's Windows-only ctypes names out of POSIX checks
        return None
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = kernel32.K32GetProcessMemoryInfo
    get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------


def _split(
    df: pl.DataFrame, y: np.ndarray, seed: int, holdout_frac: float = HOLDOUT_FRACTION
) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray]:
    """Seeded random row split into (train, train_y, holdout, holdout_y)."""
    n = len(df)
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(n)
    n_holdout = max(1, round(n * holdout_frac))
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    return df[train_idx.tolist()], y[train_idx], df[holdout_idx.tolist()], y[holdout_idx]


# ---------------------------------------------------------------------------
# Pipeline worker (runs in an isolated subprocess)
# ---------------------------------------------------------------------------


def _build_pipeline_config(ablation: AblationSpec, seed: int) -> Any:
    """Build a Config for direct build_feature_plan/fit_features/apply_feature_plan calls.

    Never goes through run_detection/score_forward, so source.uri and
    run.workdir are never actually read from or written to -- data is
    generated in-memory and passed straight to the feature-pipeline
    functions (see _fit_score_one_seed_pipeline).
    """
    from sorethumb_ml.config import Config  # noqa: PLC0415

    detectors = [{"name": name, "enabled": True} for name in ablation.detector_names]
    profiling = (
        {}
        if ablation.near_constant_distinct is None
        else {"near_constant_distinct": ablation.near_constant_distinct}
    )
    return Config.model_validate(
        {
            "source": {"uri": "file:///dev/null"},
            "features": {"pca": ablation.pca, "scaler": ablation.scaler},
            "profiling": profiling,
            "scoring": {
                "contamination": "auto",
                "combination": ablation.combination,
                "weighting": ablation.weighting,
            },
            "detectors": detectors,
            "run": {"workdir": "/tmp/unused-pipeline-benchmark", "seed": seed},  # noqa: S108 -- never touched, see docstring
            "explain": {"enabled": False},
        }
    )


_STABILITY_SUBSAMPLE = 0.8


def _fit_and_score(
    train_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    ablation: AblationSpec,
    seed: int,
    review_budget: float,
) -> tuple[dict[str, Any], float, float]:
    """Fit the real pipeline on *train_df*, score *holdout_df*. Returns (combine result, fit s, score s).

    The combine result's per-row arrays are in *holdout_df*'s row order. The
    feature pipeline sorts rows by the plan's chosen time column, so every
    frame is stamped with the pipeline's own row-identity column first (as
    ``run_detection`` does) and the scores are put back in holdout order
    afterwards -- without that, a scenario with a date column would score
    against the wrong labels.
    """
    from sorethumb_ml.detectors import registry  # noqa: PLC0415
    from sorethumb_ml.features.build import apply_feature_plan, fit_features  # noqa: PLC0415
    from sorethumb_ml.io.fingerprint import INTERNAL_ROW_ID_COLUMN  # noqa: PLC0415
    from sorethumb_ml.profiling.plan import build_feature_plan  # noqa: PLC0415
    from sorethumb_ml.scoring.calibrate import Calibrator  # noqa: PLC0415
    from sorethumb_ml.scoring.combine import ScoreEnsemble  # noqa: PLC0415

    train_df = train_df.with_columns(
        pl.Series(INTERNAL_ROW_ID_COLUMN, np.arange(len(train_df), dtype=np.int64))
    )
    holdout_df = holdout_df.with_columns(
        pl.Series(INTERNAL_ROW_ID_COLUMN, np.arange(len(holdout_df), dtype=np.int64))
    )
    cfg = _build_pipeline_config(ablation, seed)
    plan = build_feature_plan(train_df, cfg)

    t0 = time.perf_counter()
    train_space = fit_features(train_df, plan, cfg)
    calibrators: dict[str, Calibrator] = {}
    detectors: dict[str, Any] = {}
    for name in ablation.detector_names:
        det = registry[name]()
        det.fit(train_space.matrix.astype(np.float64), seed=seed)
        raw, _ = score_and_flag(det, train_space.matrix.astype(np.float64))
        cal = Calibrator()
        cal.fit(raw)
        calibrators[name] = cal
        detectors[name] = det
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    holdout_space = apply_feature_plan(holdout_df, plan)
    calibrated_holdout: dict[str, np.ndarray] = {}
    natural_flags_holdout: dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        raw, natural_flags_holdout[name] = score_and_flag(det, holdout_space.matrix.astype(np.float64))
        calibrated_holdout[name] = calibrators[name].transform(raw)
    score_seconds = time.perf_counter() - t0

    ensemble = ScoreEnsemble(
        combination=ablation.combination, weighting=ablation.weighting, contamination=review_budget
    )
    combined = ensemble.combine(calibrated_holdout, natural_flags_holdout)
    order = np.empty(len(holdout_df), dtype=np.int64)
    order[np.asarray(holdout_space.row_ids, dtype=np.int64)] = np.arange(len(holdout_df))
    combined["combined_score"] = np.asarray(combined["combined_score"])[order]
    combined["anomaly_flag"] = np.asarray(combined["anomaly_flag"])[order]
    return combined, fit_seconds, score_seconds


def _flag_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two flagged-row sets; 1.0 when both are empty (they agree: nothing flagged)."""
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / union


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two score vectors over the same rows (NaN if either is constant)."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _fit_score_one_seed_pipeline(
    scenario_name: str,
    swamping_variant: str | None,
    ablation: AblationSpec,
    seed: int,
    n_bootstrap: int = 0,
) -> dict[str, Any]:
    """Fit on train, score holdout, through the real feature pipeline. One seed.

    *swamping_variant* is None for an ordinary scenario, or "clean"/
    "contaminated" for the matched swamping comparison -- see
    ``evaluate.scenarios`` module docstring. *scenario_name* still selects a
    real ``Scenario`` even when a swamping variant is set: only its
    ``review_budget`` is used (for calibration/thresholding), never its data.

    Stability (*n_bootstrap* > 0): the pipeline is refitted on that many random
    ``_STABILITY_SUBSAMPLE`` subsamples of the training rows (without
    replacement, so no row is duplicated into a detector's fit) and the same
    holdout is scored each time. ``flag_jaccard_stability`` is the mean overlap
    of each refit's flagged rows with the full fit's; ``rank_stability`` is the
    mean Spearman correlation of their scores. A policy whose flags move a lot
    between near-identical fits is not usable even if its average looks good.
    """
    from sorethumb_ml.evaluate.metrics import evaluate_flags, evaluate_scores  # noqa: PLC0415

    scenario = scenario_by_name(scenario_name)
    if scenario is None:
        msg = f"Unknown scenario: {scenario_name!r}"
        raise ValueError(msg)

    if swamping_variant is not None:
        train_df, _train_y = (
            swamping_clean_reference(seed) if swamping_variant == "clean" else swamping_train_reference(seed)
        )
        holdout_df, holdout_y = swamping_at_risk_holdout(seed)
    else:
        df, y = scenario.generate(seed)
        train_df, _train_y, holdout_df, holdout_y = _split(df, y, seed)

    combined, fit_seconds, score_seconds = _fit_and_score(
        train_df, holdout_df, ablation, seed, scenario.review_budget
    )
    scores = combined["combined_score"]
    flags = np.asarray(combined["anomaly_flag"], dtype=bool)

    jaccards: list[float] = []
    rank_corrs: list[float] = []
    rng = np.random.default_rng(seed + 2000)
    n_sub = max(2, round(len(train_df) * _STABILITY_SUBSAMPLE))
    for _ in range(n_bootstrap):
        idx = np.sort(rng.choice(len(train_df), size=n_sub, replace=False))
        refit, _, _ = _fit_and_score(
            train_df[idx.tolist()], holdout_df, ablation, seed, scenario.review_budget
        )
        jaccards.append(_flag_jaccard(flags, np.asarray(refit["anomaly_flag"], dtype=bool)))
        rank_corrs.append(_rank_correlation(np.asarray(scores), np.asarray(refit["combined_score"])))

    metrics = evaluate_scores(scores, holdout_y, contamination=scenario.review_budget)
    flag_metrics = evaluate_flags(flags, holdout_y)
    budget_rows = scenario.review_budget * len(holdout_df)
    return {
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "f1_at_contamination": metrics.f1_at_contamination,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "flag_precision": flag_metrics.precision,
        "flag_recall": flag_metrics.recall,
        "flag_f1": flag_metrics.f1,
        "flag_false_positive_rate": flag_metrics.false_positive_rate,
        "flag_count": float(flag_metrics.flag_count),
        "over_flag_ratio": float(flag_metrics.flag_count) / budget_rows if budget_rows > 0 else float("nan"),
        "flag_jaccard_stability": float(np.mean(jaccards)) if jaccards else float("nan"),
        "rank_stability": float(np.nanmean(rank_corrs)) if rank_corrs else float("nan"),
    }


def _pipeline_batch_worker(
    scenario_name: str,
    swamping_variant: str | None,
    ablation: AblationSpec,
    seeds: list[int],
    n_bootstrap: int,
    conn: Any,
) -> None:
    """Run every seed for one (scenario, ablation) cell; send results back via *conn*."""
    try:
        per_seed = [
            _fit_score_one_seed_pipeline(scenario_name, swamping_variant, ablation, s, n_bootstrap)
            for s in seeds
        ]
        conn.send({"per_seed": per_seed, "peak_memory_mb": _peak_memory_mb(), "error": None})
    except Exception as exc:  # noqa: BLE001 -- reported back to the parent, not raised in the child
        conn.send({"per_seed": [], "peak_memory_mb": _peak_memory_mb(), "error": str(exc)[:500]})
    finally:
        conn.close()


def _run_isolated(target: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    """Run *target(*args, conn)* in a spawned subprocess; return what it sent back."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child_conn))
    proc.start()
    child_conn.close()
    result: dict[str, Any]
    try:
        if parent_conn.poll(timeout=600):
            result = parent_conn.recv()
        else:
            result = {"per_seed": [], "peak_memory_mb": None, "error": "worker timed out after 600s"}
    except EOFError:
        result = {"per_seed": [], "peak_memory_mb": None, "error": "worker exited without a result"}
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
    return result


def _aggregate(
    scenario: Scenario,
    ablation_name: str,
    batch: dict[str, Any],
    policy: str | None = None,
) -> PipelineBenchmarkRow:
    per_seed = batch["per_seed"]
    if batch["error"] is not None or not per_seed:
        return PipelineBenchmarkRow(
            scenario=scenario.name,
            kind=scenario.kind,
            ablation=ablation_name,
            policy=policy,
            n_seeds=0,
            n_train=0,
            n_holdout=0,
            review_budget=scenario.review_budget,
            roc_auc=0.0,
            roc_auc_ci95=0.0,
            average_precision=0.0,
            average_precision_ci95=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            f1_at_contamination=0.0,
            fit_seconds=0.0,
            score_seconds=0.0,
            peak_memory_mb=batch.get("peak_memory_mb"),
            error=batch["error"] or "no seeds completed",
        )

    def _mean(key: str) -> float:
        values = [r[key] for r in per_seed]
        # A single-class holdout (the swamping scenario's at-risk-normals-
        # only population) makes evaluate_scores return NaN for every seed's
        # roc_auc/average_precision by design (see evaluate.metrics) -- that
        # is a real "undefined", not a numpy warning-worthy edge case.
        if all(isinstance(v, float) and np.isnan(v) for v in values):
            return float("nan")
        return float(np.nanmean(values))

    def _mean_or_none(key: str) -> float | None:
        # Absent for a "sklearn:*" baseline row -- no ensemble/flag concept
        # there (see _fit_score_one_seed_baseline) -- rather than a fake 0.0.
        if key not in per_seed[0]:
            return None
        return _mean(key)

    return PipelineBenchmarkRow(
        scenario=scenario.name,
        kind=scenario.kind,
        ablation=ablation_name,
        policy=policy,
        n_seeds=len(per_seed),
        n_train=per_seed[0]["n_train"],
        n_holdout=per_seed[0]["n_holdout"],
        review_budget=scenario.review_budget,
        roc_auc=_mean("roc_auc"),
        roc_auc_ci95=_ci95([r["roc_auc"] for r in per_seed]),
        average_precision=_mean("average_precision"),
        average_precision_ci95=_ci95([r["average_precision"] for r in per_seed]),
        precision_at_k=_mean("precision_at_k"),
        recall_at_k=_mean("recall_at_k"),
        f1_at_contamination=_mean("f1_at_contamination"),
        fit_seconds=_mean("fit_seconds"),
        score_seconds=_mean("score_seconds"),
        peak_memory_mb=batch.get("peak_memory_mb"),
        flag_precision=_mean_or_none("flag_precision"),
        flag_recall=_mean_or_none("flag_recall"),
        flag_f1=_mean_or_none("flag_f1"),
        flag_false_positive_rate=_mean_or_none("flag_false_positive_rate"),
        flag_count=_mean_or_none("flag_count"),
        over_flag_ratio=_mean_or_none("over_flag_ratio"),
        flag_jaccard_stability=_mean_or_none("flag_jaccard_stability"),
        rank_stability=_mean_or_none("rank_stability"),
    )


# ---------------------------------------------------------------------------
# sklearn baseline worker (no sorethumb feature pipeline at all)
# ---------------------------------------------------------------------------

_SKLEARN_BASELINE_DETECTORS = ("isolation_forest", "lof", "one_class_svm")


def _encode_bare(train_df: pl.DataFrame, holdout_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot the categorical columns + standard-scale everything -- no sorethumb pipeline."""
    import pandas as pd  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    train_pd = train_df.to_pandas()
    holdout_pd = holdout_df.to_pandas()
    # polars' to_pandas() renders string columns as pandas' StringDtype, not
    # the legacy `object` dtype -- `dtype == object` silently matches nothing
    # and one-hot-encodes zero columns. Select by "not numeric" instead.
    cat_cols = [c for c in train_pd.columns if not pd.api.types.is_numeric_dtype(train_pd[c])]
    combined = pd.concat([train_pd, holdout_pd], keys=["train", "holdout"])
    encoded = pd.get_dummies(combined, columns=cat_cols)
    train_enc = encoded.xs("train").to_numpy(dtype=np.float64)
    holdout_enc = encoded.xs("holdout").to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_enc)
    holdout_scaled = scaler.transform(holdout_enc)
    return train_scaled, holdout_scaled


def _fit_score_one_seed_baseline(scenario_name: str, detector_name: str, seed: int) -> dict[str, Any]:
    from sorethumb_ml.evaluate.metrics import evaluate_scores  # noqa: PLC0415

    scenario = scenario_by_name(scenario_name)
    if scenario is None:
        msg = f"Unknown scenario: {scenario_name!r}"
        raise ValueError(msg)
    df, y = scenario.generate(seed)
    train_df, _train_y, holdout_df, holdout_y = _split(df, y, seed)
    X_train, X_holdout = _encode_bare(train_df, holdout_df)

    t0 = time.perf_counter()
    if detector_name == "isolation_forest":
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415

        model = IsolationForest(n_estimators=200, random_state=seed)
        model.fit(X_train)
    elif detector_name == "lof":
        from sklearn.neighbors import LocalOutlierFactor  # noqa: PLC0415

        model = LocalOutlierFactor(novelty=True)
        model.fit(X_train)
    elif detector_name == "one_class_svm":
        from sklearn.svm import OneClassSVM  # noqa: PLC0415

        model = OneClassSVM(nu=scenario.review_budget)
        model.fit(X_train)
    else:
        msg = f"Unknown baseline detector: {detector_name!r}"
        raise ValueError(msg)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    # sklearn convention: higher decision_function = more normal, matching
    # sorethumb's own score_samples convention -- evaluate_scores wants
    # higher = more anomalous, so flip the sign, same as Calibrator does.
    scores = -model.decision_function(X_holdout)
    score_seconds = time.perf_counter() - t0

    metrics = evaluate_scores(scores, holdout_y, contamination=scenario.review_budget)
    return {
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "f1_at_contamination": metrics.f1_at_contamination,
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
    }


def _baseline_batch_worker(scenario_name: str, detector_name: str, seeds: list[int], conn: Any) -> None:
    try:
        per_seed = [_fit_score_one_seed_baseline(scenario_name, detector_name, s) for s in seeds]
        conn.send({"per_seed": per_seed, "peak_memory_mb": _peak_memory_mb(), "error": None})
    except Exception as exc:  # noqa: BLE001
        conn.send({"per_seed": [], "peak_memory_mb": _peak_memory_mb(), "error": str(exc)[:500]})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


@dataclass
class PipelineBenchmarkConfig:
    """Configuration for a full-pipeline benchmark run."""

    scenario_names: list[str] = field(default_factory=list)  # empty = all
    ablation_names: list[str] = field(default_factory=lambda: [DEFAULT_ABLATION.name])
    seed: int = 0
    n_seeds: int = 3
    include_baselines: bool = True
    include_swamping: bool = True
    # Refits per seed for the stability metrics; 0 skips them (they are then NaN).
    n_bootstrap: int = 0


def run_pipeline_benchmark(cfg: PipelineBenchmarkConfig | None = None) -> list[PipelineBenchmarkRow]:
    """Run the full scenario x ablation (+ sklearn baseline) matrix. Returns all rows.

    Each cell runs in its own isolated subprocess (see module docstring for
    why); this function is otherwise pure orchestration.
    """
    if cfg is None:
        cfg = PipelineBenchmarkConfig()

    selected_scenarios = [s for s in SCENARIOS if not cfg.scenario_names or s.name in cfg.scenario_names]
    selected_ablations = [a for a in ABLATIONS if a.name in cfg.ablation_names]
    seeds = [cfg.seed + i for i in range(max(1, cfg.n_seeds))]

    rows: list[PipelineBenchmarkRow] = []

    for scenario in selected_scenarios:
        for ablation in selected_ablations:
            logger.info("pipeline_benchmark: %s x %s", scenario.name, ablation.name)
            batch = _run_isolated(
                _pipeline_batch_worker, (scenario.name, None, ablation, seeds, cfg.n_bootstrap)
            )
            rows.append(_aggregate(scenario, ablation.name, batch, ablation.policy_id))

        if cfg.include_baselines:
            for det_name in _SKLEARN_BASELINE_DETECTORS:
                logger.info("pipeline_benchmark: %s x sklearn:%s", scenario.name, det_name)
                batch = _run_isolated(_baseline_batch_worker, (scenario.name, det_name, seeds))
                rows.append(_aggregate(scenario, f"sklearn:{det_name}", batch))

    if cfg.include_swamping:
        # Two placeholder Scenarios purely to carry (name, kind, review_budget)
        # through _aggregate -- both variants' actual data comes from
        # swamping_clean_reference/swamping_train_reference (train) +
        # swamping_at_risk_holdout (holdout), wired via swamping_variant in
        # _pipeline_batch_worker, not via .generate here. "point" is passed as
        # scenario_name purely for its review_budget (0.05); its data is
        # never touched when swamping_variant is set.
        for variant, description in (
            (
                "clean",
                "Matched swamping comparison: clean fit, at-risk normal holdout (see evaluate.scenarios).",
            ),
            (
                "contaminated",
                "Matched swamping comparison: contaminated fit, at-risk normal holdout (see evaluate.scenarios).",
            ),
        ):
            variant_scenario = Scenario(
                f"swamping_{variant}", "swamping", description, generate=_swamping_generate_not_called
            )
            for ablation in selected_ablations:
                logger.info("pipeline_benchmark: swamping_%s x %s", variant, ablation.name)
                batch = _run_isolated(
                    _pipeline_batch_worker, ("point", variant, ablation, seeds, cfg.n_bootstrap)
                )
                rows.append(_aggregate(variant_scenario, ablation.name, batch, ablation.policy_id))

    return rows


def expected_cells(cfg: PipelineBenchmarkConfig) -> set[tuple[str, str]]:
    """Return the complete set of (scenario, ablation) pairs *cfg* should produce.

    Mirrors ``run_pipeline_benchmark``'s own scenario/ablation/baseline/
    swamping selection exactly, so the two can never silently drift apart --
    used by :func:`assert_complete_and_error_free` to catch a *missing* cell
    (a worker that never even ran) as distinct from an *errored* one
    (attempted and failed), neither of which should ever ship silently as
    published benchmark evidence.
    """
    selected_scenarios = [s for s in SCENARIOS if not cfg.scenario_names or s.name in cfg.scenario_names]
    selected_ablation_names = [a.name for a in ABLATIONS if a.name in cfg.ablation_names]

    cells: set[tuple[str, str]] = set()
    for scenario in selected_scenarios:
        for ablation_name in selected_ablation_names:
            cells.add((scenario.name, ablation_name))
        if cfg.include_baselines:
            for det_name in _SKLEARN_BASELINE_DETECTORS:
                cells.add((scenario.name, f"sklearn:{det_name}"))
    if cfg.include_swamping:
        for variant in ("clean", "contaminated"):
            for ablation_name in selected_ablation_names:
                cells.add((f"swamping_{variant}", ablation_name))
    return cells


def assert_complete_and_error_free(rows: list[PipelineBenchmarkRow], expected: set[tuple[str, str]]) -> None:
    """Raise if *rows* doesn't cover exactly *expected* (scenario, ablation) pairs error-free.

    Intended for the publication path (``scripts/run_benchmark.py``,
    before injecting results into the README) -- a benchmark run
    that silently drops a cell, gains an unrequested one (a config/harness
    drift bug), or reports an errored cell must never ship as if it were
    complete, working evidence.
    """
    actual = {(r.scenario, r.ablation) for r in rows}
    missing = expected - actual
    unexpected = actual - expected
    errored = sorted((r.scenario, r.ablation, r.error) for r in rows if r.error is not None)

    problems = []
    if missing:
        problems.append(f"missing cell(s): {sorted(missing)}")
    if unexpected:
        problems.append(f"unexpected cell(s) not in the requested matrix: {sorted(unexpected)}")
    if errored:
        problems.append(f"errored cell(s): {errored}")
    if problems:
        msg = "Pipeline benchmark matrix is incomplete or contains errors -- " + "; ".join(problems)
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    "scenario",
    "kind",
    "ablation",
    "policy",
    "n_seeds",
    "n_train",
    "n_holdout",
    "roc_auc",
    "average_precision",
    "precision_at_k",
    "recall_at_k",
    "flag_precision",
    "flag_recall",
    "flag_f1",
    "flag_false_positive_rate",
    "flag_count",
    "over_flag_ratio",
    "flag_jaccard_stability",
    "rank_stability",
    "fit_seconds",
    "score_seconds",
    "peak_memory_mb",
]


def _fmt_row_for_table(row: PipelineBenchmarkRow) -> dict[str, str]:
    if row.error is not None:
        return dict.fromkeys(_TABLE_COLS, "") | {
            "scenario": row.scenario,
            "kind": row.kind,
            "ablation": row.ablation,
            "roc_auc": f"ERROR: {row.error[:60]}",
        }
    peak = f"{row.peak_memory_mb:.1f}" if row.peak_memory_mb is not None else "n/a"

    def _opt(v: float | None, fmt: str = ".4f") -> str:
        # None (no ensemble/flag concept for a baseline row) and NaN (a
        # single-class holdout -- e.g. swamping's at-risk-normals-only
        # population -- makes ROC-AUC/AP genuinely undefined, not a
        # fabricated number) are both rendered the same, honest way.
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:{fmt}}"

    def _opt_with_ci(v: float | None, ci: float) -> str:
        rendered = _opt(v)
        return f"{rendered} ± {ci:.4f}" if row.n_seeds > 1 and rendered != "n/a" else rendered

    roc = _opt_with_ci(row.roc_auc, row.roc_auc_ci95)
    ap = _opt_with_ci(row.average_precision, row.average_precision_ci95)

    return {
        "scenario": row.scenario,
        "kind": row.kind,
        "ablation": row.ablation,
        "policy": row.policy or "n/a",
        "n_seeds": str(row.n_seeds),
        "n_train": str(row.n_train),
        "n_holdout": str(row.n_holdout),
        "roc_auc": roc,
        "average_precision": ap,
        "precision_at_k": f"{row.precision_at_k:.4f}",
        "recall_at_k": f"{row.recall_at_k:.4f}",
        "flag_precision": _opt(row.flag_precision),
        "flag_recall": _opt(row.flag_recall),
        "flag_f1": _opt(row.flag_f1),
        "flag_false_positive_rate": _opt(row.flag_false_positive_rate),
        "flag_count": _opt(row.flag_count, ".1f"),
        "over_flag_ratio": _opt(row.over_flag_ratio, ".2f"),
        "flag_jaccard_stability": _opt(row.flag_jaccard_stability, ".3f"),
        "rank_stability": _opt(row.rank_stability, ".3f"),
        "fit_seconds": f"{row.fit_seconds:.3f}",
        "score_seconds": f"{row.score_seconds:.3f}",
        "peak_memory_mb": peak,
    }


def to_markdown(rows: list[PipelineBenchmarkRow]) -> str:
    """Render pipeline-benchmark rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return "_No pipeline-benchmark results._\n"
    header = "| " + " | ".join(_TABLE_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _TABLE_COLS) + " |"
    body = ["| " + " | ".join(_fmt_row_for_table(r).get(c, "") for c in _TABLE_COLS) + " |" for r in rows]
    return "\n".join([header, sep, *body]) + "\n"


def to_csv(rows: list[PipelineBenchmarkRow]) -> str:
    """Render pipeline-benchmark rows as CSV text (every field, not just the display columns)."""
    buf = io.StringIO()
    all_cols = list(PipelineBenchmarkRow.__dataclass_fields__.keys())
    writer = csv.DictWriter(buf, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return buf.getvalue()


def write_outputs(rows: list[PipelineBenchmarkRow], output_dir: Path) -> tuple[Path, Path]:
    """Write Markdown and CSV files to *output_dir*. Returns (md_path, csv_path)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "pipeline_benchmark_results.md"
    csv_path = output_dir / "pipeline_benchmark_results.csv"
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    csv_path.write_text(to_csv(rows), encoding="utf-8")
    logger.info("Pipeline benchmark results written to %s and %s", md_path, csv_path)
    return md_path, csv_path


_RESULTS_MARKER_START = "<!-- pipeline-benchmark-results-start -->"
_RESULTS_MARKER_END = "<!-- pipeline-benchmark-results-end -->"


def inject_into_readme(rows: list[PipelineBenchmarkRow], readme_path: Path) -> bool:
    """Inject the Markdown table into README.md between marker comments.

    Returns True if the file was modified, False if it was unchanged.
    Idempotent: repeated injection produces identical output.
    """
    readme_path = Path(readme_path)
    if not readme_path.exists():
        logger.warning("README not found at %s; skipping injection.", readme_path)
        return False

    original = readme_path.read_text(encoding="utf-8")
    if _RESULTS_MARKER_START not in original or _RESULTS_MARKER_END not in original:
        logger.warning(
            "README at %s does not contain pipeline-benchmark markers; skipping injection.\n"
            "Add these markers where you want the table:\n%s\n%s",
            readme_path,
            _RESULTS_MARKER_START,
            _RESULTS_MARKER_END,
        )
        return False

    auto_gen = "<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md` to regenerate. -->"
    table_md = f"{_RESULTS_MARKER_START}\n{auto_gen}\n\n{to_markdown(rows)}{_RESULTS_MARKER_END}"

    before = original[: original.index(_RESULTS_MARKER_START)]
    after = original[original.index(_RESULTS_MARKER_END) + len(_RESULTS_MARKER_END) :]
    updated = before + table_md + after

    if updated == original:
        return False

    readme_path.write_text(updated, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Policy matrix: every registered ensemble policy x every scenario
# ---------------------------------------------------------------------------

_POLICY_MATRIX_MARKER_START = "<!-- policy-matrix-results-start -->"
_POLICY_MATRIX_MARKER_END = "<!-- policy-matrix-results-end -->"

# Ablations the published policy matrix runs: one per policy, plus the
# near_constant_2 evidence row for the profiling.near_constant_distinct default.
POLICY_MATRIX_ABLATIONS: tuple[str, ...] = (*POLICY_ABLATIONS, "near_constant_2")


def policy_matrix_config(n_seeds: int = 10, n_bootstrap: int = 3) -> PipelineBenchmarkConfig:
    """Return the published policy matrix's configuration (every scenario and policy, stability on)."""
    return PipelineBenchmarkConfig(
        ablation_names=list(POLICY_MATRIX_ABLATIONS),
        n_seeds=n_seeds,
        n_bootstrap=n_bootstrap,
        include_baselines=False,
        include_swamping=True,
    )


def suite_version(cfg: PipelineBenchmarkConfig) -> str:
    """Fingerprint of everything that defines the benchmark suite a result came from.

    Covers the scenario generators, the metric definitions, this harness, the
    seeds, the number of stability refits and the holdout/subsample fractions.
    Results are only comparable within one suite version: a change to any of
    these moves it, even if no number changes.
    """
    from sorethumb_ml.evaluate import metrics, scenarios  # noqa: PLC0415

    digest = hashlib.sha256()
    for module_file in (scenarios.__file__, metrics.__file__, __file__):
        digest.update(Path(module_file).read_bytes())
    params = {
        "seed": cfg.seed,
        "n_seeds": cfg.n_seeds,
        "n_bootstrap": cfg.n_bootstrap,
        "holdout_fraction": HOLDOUT_FRACTION,
        "stability_subsample": _STABILITY_SUBSAMPLE,
    }
    digest.update(json.dumps(params, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def to_json(rows: list[PipelineBenchmarkRow], metadata: dict[str, Any]) -> str:
    """Serialise *rows* with their provenance *metadata* (suite version, policies, versions)."""

    def _clean(value: Any) -> Any:
        return None if isinstance(value, float) and np.isnan(value) else value

    payload = {
        "metadata": metadata,
        "rows": [{k: _clean(v) for k, v in row.as_dict().items()} for row in rows],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def inject_policy_matrix(rows: list[PipelineBenchmarkRow], doc_path: Path, metadata: dict[str, Any]) -> bool:
    """Inject the policy-matrix table and its provenance between the policy-matrix markers.

    Returns True if the file was modified.
    """
    original = Path(doc_path).read_text(encoding="utf-8")
    if _POLICY_MATRIX_MARKER_START not in original or _POLICY_MATRIX_MARKER_END not in original:
        logger.warning("%s has no policy-matrix markers; skipping injection.", doc_path)
        return False
    provenance = (
        f"Suite version `{metadata['suite_version']}`, budgets v{metadata['budgets_version']}, "
        f"{metadata['n_seeds']} seeds, {metadata['n_bootstrap']} stability refits per seed, "
        f"sorethumb {metadata['library_version']} at `{metadata['git_sha']}`. "
        "Full results with provenance: [`benchmark-results/policy-matrix.json`](benchmark-results/policy-matrix.json)."
    )
    auto_gen = (
        "<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir "
        "benchmark_results --no-legacy --no-pipeline --policy-matrix --readme docs/benchmarks.md` to regenerate. -->"
    )
    block = f"{_POLICY_MATRIX_MARKER_START}\n{auto_gen}\n\n{provenance}\n\n{to_markdown(rows)}{_POLICY_MATRIX_MARKER_END}"
    before = original[: original.index(_POLICY_MATRIX_MARKER_START)]
    after = original[original.index(_POLICY_MATRIX_MARKER_END) + len(_POLICY_MATRIX_MARKER_END) :]
    updated = before + block + after
    if updated == original:
        return False
    Path(doc_path).write_text(updated, encoding="utf-8")
    return True
