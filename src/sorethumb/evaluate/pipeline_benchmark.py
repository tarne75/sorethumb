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
  ``peak_rss_mb`` field removed in an earlier phase (P0-7) for being exactly
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
import io
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sorethumb.evaluate.scenarios import SCENARIOS, Scenario, scenario_by_name, swamping_train_reference

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
]

DEFAULT_ABLATION = ABLATIONS[0]


def _swamping_generate_not_called(seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """Never invoked; swamping's data comes from is_swamping=True in the worker, not .generate."""
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

    def as_dict(self) -> dict[str, Any]:
        """Flatten to a JSON/CSV-serialisable dict."""
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "ablation": self.ablation,
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
    """Return this process's peak RSS so far, in MB, or None where unavailable (Windows)."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # POSIX ru_maxrss units are platform-defined: Linux reports KB, macOS bytes.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return max_rss / divisor


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
    from sorethumb.config import Config  # noqa: PLC0415

    detectors = [{"name": name, "enabled": True} for name in ablation.detector_names]
    return Config.model_validate(
        {
            "source": {"uri": "file:///dev/null"},
            "features": {"pca": ablation.pca, "scaler": ablation.scaler},
            "scoring": {"contamination": "auto", "combination": ablation.combination},
            "detectors": detectors,
            "run": {"workdir": "/tmp/unused-pipeline-benchmark", "seed": seed},  # noqa: S108 -- never touched, see docstring
            "explain": {"enabled": False},
        }
    )


def _fit_score_one_seed_pipeline(
    scenario_name: str,
    is_swamping: bool,
    ablation: AblationSpec,
    seed: int,
) -> dict[str, Any]:
    """Fit on train, score holdout, through the real feature pipeline. One seed."""
    from sorethumb.detectors import registry  # noqa: PLC0415
    from sorethumb.evaluate.metrics import evaluate_scores  # noqa: PLC0415
    from sorethumb.features.build import apply_feature_plan, fit_features  # noqa: PLC0415
    from sorethumb.profiling.plan import build_feature_plan  # noqa: PLC0415
    from sorethumb.scoring.calibrate import Calibrator  # noqa: PLC0415
    from sorethumb.scoring.combine import ScoreEnsemble  # noqa: PLC0415

    scenario = scenario_by_name(scenario_name)
    if scenario is None:
        msg = f"Unknown scenario: {scenario_name!r}"
        raise ValueError(msg)

    if is_swamping:
        train_df, _train_y = swamping_train_reference(seed)
        holdout_df, holdout_y = scenario.generate(seed)
        _, _, holdout_df, holdout_y = _split(holdout_df, holdout_y, seed)
    else:
        df, y = scenario.generate(seed)
        train_df, _train_y, holdout_df, holdout_y = _split(df, y, seed)

    cfg = _build_pipeline_config(ablation, seed)
    plan = build_feature_plan(train_df, cfg)

    t0 = time.perf_counter()
    train_space = fit_features(train_df, plan, cfg)
    calibrated_train: dict[str, np.ndarray] = {}
    natural_flags_train: dict[str, np.ndarray] = {}
    calibrators: dict[str, Calibrator] = {}
    detectors: dict[str, Any] = {}
    for name in ablation.detector_names:
        det = registry[name]()
        det.fit(train_space.matrix.astype(np.float64), seed=seed)
        raw = det.score_samples(train_space.matrix.astype(np.float64))
        natural_flags_train[name] = det.natural_flag(raw)
        cal = Calibrator()
        cal.fit(raw)
        calibrated_train[name] = cal.transform(raw)
        calibrators[name] = cal
        detectors[name] = det
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    holdout_space = apply_feature_plan(holdout_df, plan)
    calibrated_holdout: dict[str, np.ndarray] = {}
    natural_flags_holdout: dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        raw = det.score_samples(holdout_space.matrix.astype(np.float64))
        natural_flags_holdout[name] = det.natural_flag(raw)
        calibrated_holdout[name] = calibrators[name].transform(raw)
    score_seconds = time.perf_counter() - t0

    ensemble = ScoreEnsemble(combination=ablation.combination, contamination=scenario.review_budget)
    combined = ensemble.combine(calibrated_holdout, natural_flags_holdout)
    scores = combined["combined_score"]

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


def _pipeline_batch_worker(
    scenario_name: str,
    is_swamping: bool,
    ablation: AblationSpec,
    seeds: list[int],
    conn: Any,
) -> None:
    """Run every seed for one (scenario, ablation) cell; send results back via *conn*."""
    try:
        per_seed = [_fit_score_one_seed_pipeline(scenario_name, is_swamping, ablation, s) for s in seeds]
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
) -> PipelineBenchmarkRow:
    per_seed = batch["per_seed"]
    if batch["error"] is not None or not per_seed:
        return PipelineBenchmarkRow(
            scenario=scenario.name,
            kind=scenario.kind,
            ablation=ablation_name,
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
        return float(np.nanmean([r[key] for r in per_seed]))

    return PipelineBenchmarkRow(
        scenario=scenario.name,
        kind=scenario.kind,
        ablation=ablation_name,
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
    from sorethumb.evaluate.metrics import evaluate_scores  # noqa: PLC0415

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
            batch = _run_isolated(_pipeline_batch_worker, (scenario.name, False, ablation, seeds))
            rows.append(_aggregate(scenario, ablation.name, batch))

        if cfg.include_baselines:
            for det_name in _SKLEARN_BASELINE_DETECTORS:
                logger.info("pipeline_benchmark: %s x sklearn:%s", scenario.name, det_name)
                batch = _run_isolated(_baseline_batch_worker, (scenario.name, det_name, seeds))
                rows.append(_aggregate(scenario, f"sklearn:{det_name}", batch))

    if cfg.include_swamping:
        # A placeholder Scenario purely to carry (name, kind, review_budget)
        # through _aggregate -- swamping's actual data comes from
        # swamping_train_reference (train) + "point" (holdout), wired via
        # is_swamping=True in _pipeline_batch_worker, not via .generate here.
        swamping_scenario = Scenario(
            "swamping",
            "swamping",
            "Contamination injected into training; compare against 'point' for the accuracy cost.",
            generate=_swamping_generate_not_called,
        )
        for ablation in selected_ablations:
            logger.info("pipeline_benchmark: swamping x %s", ablation.name)
            batch = _run_isolated(_pipeline_batch_worker, ("point", True, ablation, seeds))
            rows.append(_aggregate(swamping_scenario, ablation.name, batch))

    return rows


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    "scenario",
    "kind",
    "ablation",
    "n_seeds",
    "n_train",
    "n_holdout",
    "roc_auc",
    "average_precision",
    "precision_at_k",
    "recall_at_k",
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
    roc = f"{row.roc_auc:.4f} ± {row.roc_auc_ci95:.4f}" if row.n_seeds > 1 else f"{row.roc_auc:.4f}"
    ap = (
        f"{row.average_precision:.4f} ± {row.average_precision_ci95:.4f}"
        if row.n_seeds > 1
        else f"{row.average_precision:.4f}"
    )
    return {
        "scenario": row.scenario,
        "kind": row.kind,
        "ablation": row.ablation,
        "n_seeds": str(row.n_seeds),
        "n_train": str(row.n_train),
        "n_holdout": str(row.n_holdout),
        "roc_auc": roc,
        "average_precision": ap,
        "precision_at_k": f"{row.precision_at_k:.4f}",
        "recall_at_k": f"{row.recall_at_k:.4f}",
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

    auto_gen = "<!-- AUTO-GENERATED — do not edit manually; run `sorethumb benchmark` to regenerate. -->"
    table_md = f"{_RESULTS_MARKER_START}\n{auto_gen}\n\n{to_markdown(rows)}{_RESULTS_MARKER_END}"

    before = original[: original.index(_RESULTS_MARKER_START)]
    after = original[original.index(_RESULTS_MARKER_END) + len(_RESULTS_MARKER_END) :]
    updated = before + table_md + after

    if updated == original:
        return False

    readme_path.write_text(updated, encoding="utf-8")
    return True
