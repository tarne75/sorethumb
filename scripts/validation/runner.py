"""Executes one validation case: fit on a train split, score a held-out split, evaluate.

Unlike ``schema.py``/``data.py``/``planner.py``, everything here does real
I/O -- reading parquet files, running the full sorethumb pipeline, fitting
detectors -- and is exercised by a slow/integration-marked test rather than
the fast unit lane.
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from scripts.validation.data import (
    SPLIT_ROW_ID_COLUMN,
    label_to_anomaly_array,
    split_train_holdout,
    stamp_row_id,
)
from scripts.validation.schema import (
    CASE_SCHEMA_VERSION,
    CaseKey,
    CaseResult,
    ComboSpec,
    DatasetSpec,
    RunIdentity,
)

if TYPE_CHECKING:
    from sorethumb import Config
    from sorethumb._pipeline import RunResult

logger = logging.getLogger(__name__)

# Held out for evaluation, never trained on.
HOLDOUT_FRACTION = 0.3

# Fixed operating point for precision@k/recall@k/F1@k, and the contamination
# every detector in the combo is fit to target (see build_config) -- never
# derived from the case's own labels (that would leak the answer into the
# operating point; see sorethumb.evaluate.metrics.evaluate_scores).
REVIEW_BUDGET = 0.05


def build_config(
    dataset: DatasetSpec,
    pca: bool,
    combo: ComboSpec,
    seed: int,
    source_path: Path,
    workdir: Path,
    *,
    explain: bool = False,
) -> Config:
    """Build the resolved Config for one case, pointed at *source_path*."""
    from sorethumb import Config  # noqa: PLC0415

    detectors = [{"name": name, "enabled": True} for name in combo.detectors]
    return Config.model_validate(
        {
            "source": {"uri": str(source_path)},
            "columns": {"ignore": dataset.ignore, "id_column": SPLIT_ROW_ID_COLUMN},
            "features": {"pca": pca},
            # A fixed float contamination (not "auto") makes every detector in
            # the combo target the same review budget -- including wiring
            # OneClassSVM's nu to match it automatically (see
            # _pipeline.py's "Wire scoring contamination -> OCSVM nu").
            # This replaces the old approach of searching for the first nu
            # that emitted any anomalies at all: there is no search, and the
            # same fixed budget is used for every case and for the
            # precision@k/recall@k/F1@k operating point below.
            "scoring": {"contamination": REVIEW_BUDGET, "combination": "intersection"},
            "detectors": detectors,
            "run": {"workdir": str(workdir), "seed": seed},
            "explain": {"enabled": explain},
        }
    )


def build_identity(
    dataset_path: Path,
    config: Config,
    seed: int,
    code_rev: str,
    deps: dict[str, str],
) -> RunIdentity:
    """Compute a case's resume identity without running it.

    Uses *dataset_path* (the stable, canonical source file) and a *config*
    built against it -- never the ephemeral per-run temp split files, whose
    path changes every invocation and would make config_hash never match
    across runs, defeating resumability. *code_rev*/*deps* are taken as
    arguments (computed once by the caller via ``identity.code_revision``/
    ``identity.dependency_versions``) rather than recomputed per case, which
    would otherwise spawn a git subprocess for every case in the matrix.
    """
    from sorethumb.io.fingerprint import content_fingerprint  # noqa: PLC0415

    return RunIdentity(
        schema_version=CASE_SCHEMA_VERSION,
        code_revision=code_rev,
        data_fingerprint=content_fingerprint(dataset_path),
        config_hash=config.config_hash(),
        seed=seed,
        dependency_versions=deps,
    )


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else value


def _require_no_group_failures(n_failed: int, stage: str) -> None:
    if n_failed:
        msg = f"{n_failed} group(s) failed during {stage}"
        raise RuntimeError(msg)


def _fit_and_score(
    dataset: DatasetSpec,
    combo: ComboSpec,
    case: CaseKey,
    workdir: Path,
    train_df: pl.DataFrame,
    holdout_df: pl.DataFrame,
    warnings: list[str],
) -> tuple[RunResult, RunResult]:
    """Write the splits to temp parquet, fit on train, score-forward the holdout.

    Appends every warning issued to *warnings* (owned by the caller) as soon
    as it happens, not just on success -- so a group failure partway through
    still leaves the caller's warning list populated for the error result,
    instead of raising past an unpopulated one. Returns (train_result,
    score_result). Raises RuntimeError if any group failed during either
    stage -- the caller (``run_case``) turns that into a ``status="error"``
    CaseResult.
    """
    from sorethumb import run_detection, score_forward  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="sorethumb-validation-") as tmp_dir:
        train_path = Path(tmp_dir) / "train.parquet"
        holdout_path = Path(tmp_dir) / "holdout.parquet"
        train_df.write_parquet(train_path)
        holdout_df.write_parquet(holdout_path)

        train_cfg = build_config(dataset, case.pca, combo, case.seed, train_path, workdir)
        train_result = run_detection(train_cfg, no_report=True)
        warnings.extend(train_result.warnings_issued)
        _require_no_group_failures(train_result.n_failed, "training fit")

        score_cfg = build_config(dataset, case.pca, combo, case.seed, holdout_path, workdir)
        score_result = score_forward(score_cfg, train_result.run_id, no_report=True)
        warnings.extend(score_result.warnings_issued)
        _require_no_group_failures(score_result.n_failed, "scoring the holdout split")

    return train_result, score_result


def _full_population_scores(
    workdir: Path,
    train_result: RunResult,
    config: Config,
    detector_names: list[str],
    holdout_df: pl.DataFrame,
) -> np.ndarray | None:
    """Compute the ensemble-combined score for *every* holdout row.

    ``sorethumb.store.results.write_results`` only ever persists *flagged*
    rows (see its module docstring) -- score_forward's own results.parquet
    is unusable for ROC-AUC/AP, which need a ranking over the whole
    population, not just the top-k. This bypasses that persisted, filtered
    path and instead calls the same lower-level primitives
    ``_finalize_group`` in ``_pipeline.py`` uses internally
    (``score_with_existing`` -> ``ScoreEnsemble.combine``) directly on the
    full holdout matrix, reusing the *persisted, already-fitted* models
    (never re-fitting).

    Returns None (skip evaluation) if the plan reorders rows by a
    ``chosen_time_column`` -- this function relies on ``apply_feature_plan``
    preserving *holdout_df*'s row order 1:1 so the scores it returns can be
    zipped directly against labels read straight off *holdout_df*, with no
    join; none of the registered datasets have a real temporal column, so
    this is not expected to fire, but a silent misalignment would be worse
    than skipping.
    """
    from sorethumb.features.build import apply_feature_plan  # noqa: PLC0415
    from sorethumb.scoring.combine import ScoreEnsemble  # noqa: PLC0415
    from sorethumb.store.models import load_plan, plan_digest, score_with_existing  # noqa: PLC0415
    from sorethumb.store.workspace import Workspace  # noqa: PLC0415

    if len(train_result.groups) != 1:
        logger.warning(
            "Expected exactly one group (no group_by is configured for the "
            "validation matrix), got %d; skipping full-population evaluation.",
            len(train_result.groups),
        )
        return None

    ws = Workspace.open(workdir)
    plan = load_plan(ws, train_result.run_id)
    if plan.chosen_time_column is not None:
        logger.warning(
            "Plan sorts by chosen_time_column=%r; skipping full-population "
            "evaluation to avoid a silent score/label misalignment.",
            plan.chosen_time_column,
        )
        return None

    space = apply_feature_plan(holdout_df, plan)
    group_key = train_result.groups[0].group_key
    res = score_with_existing(
        ws,
        train_result.run_id,
        group_key,
        space.matrix.astype(np.float64),
        space.feature_schema_hash,
        detector_names,
        plan_digest(plan.to_json()),
        strict=False,
    )
    if not res["calibrated"]:
        logger.warning("No persisted models found for group %s; skipping evaluation.", group_key)
        return None

    ensemble = ScoreEnsemble(
        weighting=config.scoring.weighting,
        combination=config.scoring.combination,
        contamination=config.scoring.contamination,
        manual_weights=config.scoring.weights or None,
    )
    combined = ensemble.combine(res["calibrated"], res["natural_flags"])
    scores: np.ndarray = combined["combined_score"]
    return scores


def _evaluate_holdout(
    dataset: DatasetSpec,
    holdout_df: pl.DataFrame,
    scores: np.ndarray | None,
) -> tuple[int | None, float | None, float | None, float | None, float | None, float | None]:
    """Compute (n_true_anomalies, roc_auc, ap, precision_at_k, recall_at_k, f1) for a labelled dataset.

    Returns an all-None tuple when *dataset* has no label column, or *scores*
    is None (see ``_full_population_scores``).
    """
    from sorethumb.evaluate.metrics import evaluate_scores  # noqa: PLC0415

    if dataset.label_column is None or dataset.label_is_anomaly is None or scores is None:
        return None, None, None, None, None, None

    y_true = label_to_anomaly_array(holdout_df[dataset.label_column], dataset.label_is_anomaly)
    metrics = evaluate_scores(scores, y_true, contamination=REVIEW_BUDGET)
    return (
        int(y_true.sum()),
        _none_if_nan(metrics.roc_auc),
        _none_if_nan(metrics.average_precision),
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.f1_at_contamination,
    )


def run_case(
    case: CaseKey,
    dataset: DatasetSpec,
    combo: ComboSpec,
    workdir_base: Path,
    data_dir: Path,
    code_rev: str,
    deps: dict[str, str],
) -> CaseResult:
    """Fit on a train split, score-forward the held-out split, evaluate, return a CaseResult.

    Never raises: any failure -- including a missing/unreadable source file,
    a fit error, a scoring error, or a group failure -- is caught and
    returned as ``status="error"`` with the exception message, so one bad
    case can't abort the whole matrix. *code_rev*/*deps* come from the
    caller (computed once, not per case -- see ``build_identity``).
    """
    t0 = time.time()
    source_path = data_dir / dataset.file
    workdir = (
        workdir_base / dataset.name / ("pca_on" if case.pca else "pca_off") / case.combo / f"seed{case.seed}"
    )

    # A placeholder, overwritten by the real identity as the first step
    # inside the try block below. If even computing identity fails (e.g. the
    # source file is missing -- content_fingerprint does real file I/O), the
    # except handler still has *some* RunIdentity to attach to the error
    # result rather than raising past this function's "never raises"
    # contract.
    identity = RunIdentity(
        schema_version=CASE_SCHEMA_VERSION,
        code_revision=code_rev,
        data_fingerprint="unavailable",
        config_hash="unavailable",
        seed=case.seed,
        dependency_versions=deps,
    )
    # Declared before the try block (and mutated in place by _fit_and_score)
    # so the except handler below can always report whatever warnings fired
    # before the failure, even one raised partway through fit/score.
    warnings: list[str] = []
    try:
        # Identity is always computed against the canonical source file,
        # before any split/temp file is created -- see build_identity's
        # docstring.
        identity_cfg = build_config(dataset, case.pca, combo, case.seed, source_path, workdir)
        identity = build_identity(source_path, identity_cfg, case.seed, code_rev, deps)

        raw_df = pl.read_parquet(source_path)
        train_df, holdout_df = split_train_holdout(raw_df, HOLDOUT_FRACTION, case.seed)
        train_df = stamp_row_id(train_df)
        holdout_df = stamp_row_id(holdout_df)

        train_result, score_result = _fit_and_score(
            dataset, combo, case, workdir, train_df, holdout_df, warnings
        )

        n_holdout = len(holdout_df)

        # scoring.min_records (default 100) applies independently to each
        # split; a train or holdout split below it is a benign per-group
        # skip in the pipeline's own vocabulary ("too_few_records" is not
        # "failed"), not an exception _fit_and_score would raise. Left
        # undetected here, it reads as a real "0 anomalies found" result
        # instead of "nothing was actually fit or scored" -- distinguish it
        # explicitly rather than reporting a misleading zero.
        if train_result.n_succeeded == 0 or score_result.n_succeeded == 0:
            return CaseResult(
                dataset=dataset.name,
                pca=case.pca,
                combo=case.combo,
                detectors=list(combo.detectors),
                seed=case.seed,
                identity=identity,
                status="too_few_records",
                elapsed_seconds=round(time.time() - t0, 2),
                n_train=len(train_df),
                n_holdout=n_holdout,
                review_budget=REVIEW_BUDGET,
                run_id=train_result.run_id,
                score_run_id=score_result.run_id,
                warnings=warnings,
            )

        n_holdout_flagged = score_result.n_anomalies
        holdout_flag_rate = n_holdout_flagged / n_holdout if n_holdout else 0.0

        scores = _full_population_scores(
            workdir, train_result, identity_cfg, list(combo.detectors), holdout_df
        )
        (
            n_holdout_anomalies_true,
            roc_auc,
            average_precision,
            precision_at_k,
            recall_at_k,
            f1_at_contamination,
        ) = _evaluate_holdout(dataset, holdout_df, scores)

        return CaseResult(
            dataset=dataset.name,
            pca=case.pca,
            combo=case.combo,
            detectors=list(combo.detectors),
            seed=case.seed,
            identity=identity,
            status="success",
            elapsed_seconds=round(time.time() - t0, 2),
            n_train=len(train_df),
            n_holdout=n_holdout,
            n_holdout_anomalies_true=n_holdout_anomalies_true,
            n_holdout_flagged=n_holdout_flagged,
            holdout_flag_rate=round(holdout_flag_rate, 6),
            review_budget=REVIEW_BUDGET,
            roc_auc=roc_auc,
            average_precision=average_precision,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_contamination=f1_at_contamination,
            run_id=train_result.run_id,
            score_run_id=score_result.run_id,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001 -- one bad case must not abort the matrix
        logger.warning("Case %s failed: %s", case, exc)
        return CaseResult(
            dataset=dataset.name,
            pca=case.pca,
            combo=case.combo,
            detectors=list(combo.detectors),
            seed=case.seed,
            identity=identity,
            status="error",
            elapsed_seconds=round(time.time() - t0, 2),
            error=str(exc)[:500],
            warnings=warnings,
        )


def run_explain_check(
    dataset: DatasetSpec,
    combo: ComboSpec,
    seed: int,
    workdir_base: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Validate that explain=True runs end-to-end for *dataset*, on the full (unsplit) data.

    Kept as a small, separate matrix (one call per dataset, always the
    baseline combo, no PCA) rather than folded into the main accuracy
    matrix: explain is orthogonal to detector-combo/PCA accuracy and adds
    real per-row SHAP/gradient cost to every cell if run everywhere.
    """
    from sorethumb import run_detection  # noqa: PLC0415

    t0 = time.time()
    source_path = data_dir / dataset.file
    workdir = workdir_base / "_explain_check" / dataset.name
    cfg = build_config(dataset, False, combo, seed, source_path, workdir, explain=True)
    try:
        result = run_detection(cfg, no_report=True)
        status = "error" if result.n_failed else "success"
        return {
            "dataset": dataset.name,
            "status": status,
            "n_anomalies": result.n_anomalies,
            "elapsed_seconds": round(time.time() - t0, 2),
            "warnings": result.warnings_issued,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "dataset": dataset.name,
            "status": "error",
            "n_anomalies": 0,
            "elapsed_seconds": round(time.time() - t0, 2),
            "warnings": [],
            "error": str(exc)[:500],
        }
