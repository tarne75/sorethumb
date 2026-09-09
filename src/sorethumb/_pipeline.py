"""End-to-end pipeline orchestration.

This module is the only place that sequences the library's subsystems:
source resolution → profiling → feature construction → detection →
calibration → scoring → explanation → persistence → reporting.

The client (cli.py) calls run_detection() and score_forward() and inspects
their return values. It does not call any subsystem directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from sorethumb.config import Config, SourceConfig
from sorethumb.detectors import registry
from sorethumb.errors import SchemaError, SorethumbWarning, StoreError
from sorethumb.explain.blend import blend
from sorethumb.explain.project import aggregate_to_original, top_n_reasons
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.features.space import FeatureSpace
from sorethumb.io.fingerprint import content_fingerprint, schema_fingerprint
from sorethumb.io.nested import unnest_all
from sorethumb.io.readers import read_frame
from sorethumb.io.source import resolve_source
from sorethumb.profiling.plan import FeaturePlan, build_feature_plan
from sorethumb.report.html import GroupSection, RunMeta, render_report
from sorethumb.scoring.calibrate import Calibrator
from sorethumb.scoring.combine import ScoreEnsemble
from sorethumb.store.models import load_plan, save_model, save_plan, score_with_existing
from sorethumb.store.results import write_results
from sorethumb.store.workspace import Workspace, make_group_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass
class GroupSummary:
    """Per-group outcome from a run."""

    group_key: str
    group_label: str
    n_records: int
    n_anomalies: int
    anomaly_rate: float | None
    results_path: Path | None
    status: str  # "success" | "skipped" | "failed" | "too_few_records"
    error: str | None
    elapsed_seconds: float
    drifted: bool
    refit_reason: str | None
    warnings_issued: list[str]


@dataclass
class RunResult:
    """Return value from run_detection / score_forward."""

    run_id: str
    dataset_uri: str
    dataset_fp: str
    config_hash: str
    period_label: str | None
    workspace_path: Path
    groups: list[GroupSummary]
    report_path: Path | None
    started_at: str
    finished_at: str
    warnings_issued: list[str] = field(default_factory=list)

    # Convenience helpers

    @property
    def n_succeeded(self) -> int:
        return sum(1 for g in self.groups if g.status == "success")

    @property
    def n_skipped(self) -> int:
        return sum(1 for g in self.groups if g.status == "skipped")

    @property
    def n_failed(self) -> int:
        return sum(1 for g in self.groups if g.status == "failed")

    @property
    def n_anomalies(self) -> int:
        return sum(g.n_anomalies for g in self.groups)


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def load_dataset(config: SourceConfig, cache_dir: Path | None = None) -> pl.DataFrame:
    """Resolve a source, read it, and unnest structs.

    Parameters
    ----------
    config:
        Source configuration.
    cache_dir:
        Where to cache downloaded files. Defaults to the current directory's
        ``.sorethumb_cache/`` when not given.

    Returns
    -------
    Flat Polars DataFrame (no struct columns).

    """
    _cache = cache_dir or Path(".sorethumb_cache")
    _cache.mkdir(parents=True, exist_ok=True)
    local_path = resolve_source(config, _cache)
    lf = read_frame(local_path, config)
    df = lf.collect()
    if config.max_nesting_depth > 0:
        df = unnest_all(df, config.max_nesting_depth)
    return df


def list_detectors() -> list[str]:
    """Return sorted names of all registered detectors."""
    return sorted(registry.keys())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _period_window_expr(time_col: str, dtype: pl.DataType, period_from: str, period_to: str) -> pl.Expr:
    """Return a ``[period_from, period_to)`` filter for *time_col*.

    ``resolve_period`` / ``period_bounds`` hand back ISO **strings**; Polars will
    not compare a temporal column to a string, so the bounds are coerced to the
    column's own dtype (Date, Datetime naive/tz, or left as strings for a
    still-unparsed Utf8 column — lexical order is valid for ISO strings).
    """
    col = pl.col(time_col)
    if dtype in (pl.String, pl.Categorical):
        lo: object
        hi: object
        lo, hi = period_from, period_to
    elif dtype == pl.Date:
        lo, hi = date.fromisoformat(period_from), date.fromisoformat(period_to)
    else:  # Datetime — any time unit, with or without a time zone
        lo, hi = datetime.fromisoformat(period_from), datetime.fromisoformat(period_to)
        if getattr(dtype, "time_zone", None):
            # Period bounds are UTC by construction (see resolve_period).
            lo, hi = lo.replace(tzinfo=UTC), hi.replace(tzinfo=UTC)
    return (col >= lo) & (col < hi)


def _resolve_and_filter_period(
    df_raw: pl.DataFrame,
    config: Config,
    period_label_override: str | None,
) -> tuple[pl.DataFrame, str | None, tuple[str, str] | None]:
    """Resolve the period label and filter *df_raw* to that period's window.

    - No ``columns.time_column``: the override (or None) passes through, nothing
      is filtered, and the window is ``None`` (no history is recorded).
    - No override: resolve the current period from the wall clock.
    - Override given (e.g. a ``sorethumb backfill`` label): treat it as an
      existing label and derive its ``[period_from, period_to)`` window.

    Returns ``(filtered_df, period_label, period_window)`` where ``period_window``
    is ``(period_from, period_to)`` (or ``None`` when there is no time column).
    Whenever a time column exists the frame is filtered to that window, so a
    historical label processes only its own rows — not the whole dataset.
    """
    time_col = config.columns.time_column
    if not time_col:
        if period_label_override is not None:
            logger.warning(
                "period_label_override=%r ignored: no columns.time_column configured.",
                period_label_override,
            )
        return df_raw, period_label_override, None

    from sorethumb.history.periods import period_bounds, resolve_period  # noqa: PLC0415

    granularity = config.history.period_granularity
    if period_label_override is None:
        period_from, period_to, label = resolve_period(
            datetime.now(UTC), granularity, config.history.roll_non_business
        )
    else:
        label = period_label_override
        period_from, period_to = period_bounds(label, granularity)

    if time_col not in df_raw.columns:
        msg = f"columns.time_column={time_col!r} is not in the dataset."
        raise SchemaError(msg)

    filtered = df_raw.filter(_period_window_expr(time_col, df_raw[time_col].dtype, period_from, period_to))
    logger.info(
        "Period %s: window [%s, %s) selects %d of %d rows.",
        label,
        period_from,
        period_to,
        len(filtered),
        len(df_raw),
    )
    return filtered, label, (period_from, period_to)


def _record_period_history(
    ws: Workspace,
    dataset_fp: str,
    config_hash: str,
    run_id: str,
    period_label: str,
    period_window: tuple[str, str],
    group_results: list[GroupSummary],
) -> None:
    """Write the ``period`` row and one ``totals`` row per processed group.

    Built straight from the :class:`GroupSummary` objects: each one already
    carries ``n_records`` (the rows that entered the pipeline for this
    group × period — i.e. the population) and ``n_anomalies`` (the count).

    We deliberately do **not** feed the persisted results frame to
    ``compute_totals``: that frame holds only the flagged rows, with a
    ``flagged`` column and the group *digest* — not the boolean ``anomaly_flag``
    and raw grouping columns ``compute_totals`` needs, and it has no population.
    The summaries are the right source.

    ``skipped`` groups (ledger already had them) and ``failed`` groups are left
    out so their existing totals row is not overwritten / a gap stays a gap.
    ``too_few_records`` groups are recorded with a zero count so the period is
    not re-queued forever.
    """
    ws.store.upsert_period(dataset_fp, period_label, period_window[0], period_window[1])
    for g in group_results:
        if g.status not in ("success", "too_few_records"):
            continue
        ws.store.upsert_total(
            dataset_fp=dataset_fp,
            group_key=g.group_key,
            period_label=period_label,
            anomaly_count=g.n_anomalies,
            population=g.n_records,
            rate=g.anomaly_rate,
            run_id=run_id,
            config_hash=config_hash,
        )


def run_detection(
    config: Config,
    *,
    only_groups: list[str] | None = None,
    group_filter_regex: str | None = None,
    force: bool = False,
    no_report: bool = False,
    dry_run: bool = False,
    period_label_override: str | None = None,
) -> RunResult:
    """Run the full anomaly detection pipeline.

    Parameters
    ----------
    config:
        Fully resolved configuration.
    only_groups:
        When set, run only these group labels. Applied before group_filter_regex.
    group_filter_regex:
        Regex applied to group labels; only matching groups run.
        Unanchored search semantics — use ``^`` / ``$`` to anchor explicitly.
    force:
        Re-run groups that are already marked complete in the ledger.
    no_report:
        Skip HTML report rendering.
    dry_run:
        Resolve everything, print the work plan, and exit without writing anything.
    period_label_override:
        Force a specific period label instead of resolving from the reference date.

    Returns
    -------
    RunResult with per-group outcomes and the report path.

    """
    started_at = datetime.now(UTC).isoformat()
    ws_path = Path(config.run.workdir)

    if ws_path.exists() and (ws_path / "sorethumb.db").exists():
        ws = Workspace.open(ws_path)
    else:
        ws = Workspace.init(ws_path)

    issued_warnings: list[str] = []

    def _capture_warning(message: warnings.WarningMessage) -> None:
        issued_warnings.append(str(message.message))

    with ws:
        cache_dir = ws.root / "cache" / "datasets"

        # ── 1. Load dataset ──────────────────────────────────────────────
        local_path = resolve_source(config.source, cache_dir)
        lf = read_frame(local_path, config.source)
        df_raw = lf.collect()
        if config.source.max_nesting_depth > 0:
            df_raw = unnest_all(df_raw, config.source.max_nesting_depth)

        content_fp = content_fingerprint(local_path)
        schema_fp = schema_fingerprint(df_raw)
        dataset_fp = f"{content_fp[:32]}_{schema_fp[:16]}"

        ws.store.upsert_dataset(
            dataset_fp=dataset_fp,
            source_uri=config.source.uri,
            schema_fingerprint=schema_fp,
            content_fingerprint=content_fp,
            n_rows=len(df_raw),
            n_cols=len(df_raw.columns),
        )

        # ── 2. Period resolution + window filter ────────────────────────
        df_raw, period_label, period_window = _resolve_and_filter_period(
            df_raw, config, period_label_override
        )

        # ── 3. Register run ──────────────────────────────────────────────
        # run_id is derived deterministically so that a repeat call with identical
        # dataset + config + period finds the same ledger entries and can skip
        # already-complete groups (resume behaviour).
        run_id = _make_run_id(dataset_fp, config.config_hash(), period_label)
        config_json = config.model_dump_json()
        ws.store.insert_run(
            run_id=run_id,
            dataset_fp=dataset_fp,
            config_json=config_json,
            seed=config.run.seed,
        )

        if dry_run:
            logger.info(
                "DRY RUN — run_id=%s dataset_fp=%s period=%s rows=%d cols=%d",
                run_id,
                dataset_fp,
                period_label,
                len(df_raw),
                len(df_raw.columns),
            )
            return RunResult(
                run_id=run_id,
                dataset_uri=config.source.uri,
                dataset_fp=dataset_fp,
                config_hash=config.config_hash(),
                period_label=period_label,
                workspace_path=ws_path,
                groups=[],
                report_path=None,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
            )

        # ── 4. Build feature plan on full dataset ────────────────────────
        plan = build_feature_plan(df_raw, config)

        # Fit scaler / correlation / PCA on the full frame so parameters are
        # stable across groups — each group's matrix is a subset of this space.
        full_space = fit_features(df_raw, plan, config)

        # Persist the fitted plan so a later `sorethumb score --from-run` can
        # reload and apply it without re-fitting.
        save_plan(ws, run_id, plan.to_json())

        # ── 5. Discover groups ───────────────────────────────────────────
        group_by = config.columns.group_by
        if group_by:
            # Single pass: get distinct groups and their row counts together.
            groups_info = df_raw.group_by(group_by).agg(pl.len().alias("__n__")).to_dicts()
        else:
            groups_info = [{"__n__": len(df_raw)}]

        # Apply user filters (only_groups first, then regex)
        if only_groups:
            groups_info = [g for g in groups_info if _group_label(g, group_by) in only_groups]
        if group_filter_regex:
            pat = re.compile(group_filter_regex)
            groups_info = [g for g in groups_info if pat.search(_group_label(g, group_by))]

        # ── 6. Per-group pipeline ────────────────────────────────────────
        group_results: list[GroupSummary] = []

        for ginfo in groups_info:
            n_records: int = ginfo.pop("__n__", 0)
            group_values = {col: str(ginfo[col]) for col in group_by} if group_by else {}
            group_key = make_group_key(group_values)
            group_label = _group_label(ginfo, group_by) if group_by else "__all__"

            gsummary = _execute_group(
                ws,
                run_id,
                group_key,
                group_label,
                group_values,
                n_records,
                force=force,
                body=partial(
                    _run_group,
                    ws=ws,
                    run_id=run_id,
                    config=config,
                    plan=plan,
                    full_space=full_space,
                    df_raw=df_raw,
                    group_by=group_by,
                    group_values=group_values,
                    group_key=group_key,
                    group_label=group_label,
                    n_records=n_records,
                    period_label=period_label,
                    dataset_fp=dataset_fp,
                ),
            )
            issued_warnings.extend(gsummary.warnings_issued)
            group_results.append(gsummary)

        # ── 7. Mark run complete / failed ────────────────────────────────
        any_failed = any(g.status == "failed" for g in group_results)
        if any_failed:
            ws.store.mark_run_failed(run_id, "one or more groups failed")
        else:
            ws.store.mark_run_complete(run_id)

        # ── 7b. History ledger: period + per-group totals rows ──────────
        # This is what makes `sorethumb backfill` idempotent (a processed
        # period gets a totals row, so it is not re-queued) and gives
        # `sorethumb history` something to aggregate.
        if period_label is not None and period_window is not None:
            _record_period_history(
                ws, dataset_fp, config.config_hash(), run_id, period_label, period_window, group_results
            )

        # ── 8. Render report ─────────────────────────────────────────────
        report_path: Path | None = None
        if not no_report and group_results:
            report_path = _render_run_report(ws, run_id, config, plan, group_results)

        finished_at = datetime.now(UTC).isoformat()
        return RunResult(
            run_id=run_id,
            dataset_uri=config.source.uri,
            dataset_fp=dataset_fp,
            config_hash=config.config_hash(),
            period_label=period_label,
            workspace_path=ws_path,
            groups=group_results,
            report_path=report_path,
            started_at=started_at,
            finished_at=finished_at,
            warnings_issued=issued_warnings,
        )


# ---------------------------------------------------------------------------
# Score-forward pipeline
# ---------------------------------------------------------------------------


def _make_score_run_id(
    dataset_fp: str, config_hash: str, period_label: str | None, source_run_id: str
) -> str:
    """Deterministic id for a score-forward run.

    Distinct from a fitted run's id (``score_`` vs ``run_`` prefix) and keyed on
    the source run, so re-scoring the same new data against the same source is
    idempotent.
    """
    key = f"{dataset_fp}:{config_hash}:{period_label or '__no_period__'}:{source_run_id}"
    return "score_" + hashlib.sha256(key.encode()).hexdigest()[:32]


def score_forward(
    config: Config,
    source_run_id: str,
    *,
    strict: bool = False,
    force: bool = False,
    no_report: bool = False,
    period_label_override: str | None = None,
) -> RunResult:
    """Score new data with a previous run's persisted artefacts — no re-fitting.

    Loads *source_run_id*'s fitted FeaturePlan and, per group, its persisted
    per-detector models and calibrators. The plan is applied to the new data
    (``apply_feature_plan``); each detector is unpickled and only
    ``score_samples`` / ``natural_flag`` is called; the persisted calibrator's
    ``transform`` maps scores onto the source run's reference distribution so the
    numbers are comparable across runs. Schema drift and library-version drift
    are detected (``strict`` promotes both to errors). A new, distinct run is
    written with ``source_run_id`` recorded.

    Parameters
    ----------
    config:
        Configuration for the *new* data (``config.run.workdir`` must be the
        workspace that holds *source_run_id*).
    source_run_id:
        The fitted run to reuse.
    strict:
        Raise on schema / version drift instead of warning.
    force:
        Re-score groups already marked complete for this score-forward run.
    no_report:
        Skip HTML report rendering.
    period_label_override:
        Force a specific period label instead of resolving from the reference date.

    """
    started_at = datetime.now(UTC).isoformat()
    ws_path = Path(config.run.workdir)
    if not (ws_path.exists() and (ws_path / "sorethumb.db").exists()):
        msg = f"No workspace at {ws_path}; cannot score against run {source_run_id!r}."
        raise StoreError(msg)

    issued_warnings: list[str] = []

    with Workspace.open(ws_path) as ws:
        if ws.store.get_run(source_run_id) is None:
            msg = f"Source run {source_run_id!r} not found in workspace {ws_path}."
            raise StoreError(msg)

        # The exact fitted plan from the source run (frequency maps, scaler
        # params, PCA components, correlation-drop list). Raises if the source
        # run predates plan persistence.
        plan = load_plan(ws, source_run_id)

        # ── Load the new dataset ─────────────────────────────────────────
        cache_dir = ws.root / "cache" / "datasets"
        local_path = resolve_source(config.source, cache_dir)
        df_raw = read_frame(local_path, config.source).collect()
        if config.source.max_nesting_depth > 0:
            df_raw = unnest_all(df_raw, config.source.max_nesting_depth)

        content_fp = content_fingerprint(local_path)
        schema_fp = schema_fingerprint(df_raw)
        dataset_fp = f"{content_fp[:32]}_{schema_fp[:16]}"
        ws.store.upsert_dataset(
            dataset_fp=dataset_fp,
            source_uri=config.source.uri,
            schema_fingerprint=schema_fp,
            content_fingerprint=content_fp,
            n_rows=len(df_raw),
            n_cols=len(df_raw.columns),
        )

        # ── Period resolution + window filter (same rules as run_detection) ──
        # score-forward does not write history: its dataset_fp is the *new*
        # data's and its numbers come from a reused model, not a period compute.
        df_raw, period_label, _ = _resolve_and_filter_period(df_raw, config, period_label_override)

        # ── Register the score-forward run ──────────────────────────────
        new_run_id = _make_score_run_id(dataset_fp, config.config_hash(), period_label, source_run_id)
        ws.store.insert_run(
            run_id=new_run_id,
            dataset_fp=dataset_fp,
            config_json=config.model_dump_json(),
            seed=config.run.seed,
            source_run_id=source_run_id,
        )
        logger.info("score-forward run %s from source %s", new_run_id, source_run_id)

        # ── Discover groups (from the new data) ─────────────────────────
        group_by = config.columns.group_by
        if group_by:
            groups_info = df_raw.group_by(group_by).agg(pl.len().alias("__n__")).to_dicts()
        else:
            groups_info = [{"__n__": len(df_raw)}]

        group_results: list[GroupSummary] = []
        for ginfo in groups_info:
            n_records: int = ginfo.pop("__n__", 0)
            group_values = {col: str(ginfo[col]) for col in group_by} if group_by else {}
            group_key = make_group_key(group_values)
            group_label = _group_label(ginfo, group_by) if group_by else "__all__"

            gsummary = _execute_group(
                ws,
                new_run_id,
                group_key,
                group_label,
                group_values,
                n_records,
                force=force,
                body=partial(
                    _score_forward_group,
                    ws=ws,
                    source_run_id=source_run_id,
                    run_id=new_run_id,
                    config=config,
                    plan=plan,
                    df_raw=df_raw,
                    group_by=group_by,
                    group_values=group_values,
                    group_key=group_key,
                    group_label=group_label,
                    n_records=n_records,
                    period_label=period_label,
                    strict=strict,
                ),
            )
            issued_warnings.extend(gsummary.warnings_issued)
            group_results.append(gsummary)

        if any(g.status == "failed" for g in group_results):
            ws.store.mark_run_failed(new_run_id, "one or more groups failed")
        else:
            ws.store.mark_run_complete(new_run_id)

        report_path: Path | None = None
        if not no_report and group_results:
            report_path = _render_run_report(ws, new_run_id, config, plan, group_results)

        return RunResult(
            run_id=new_run_id,
            dataset_uri=config.source.uri,
            dataset_fp=dataset_fp,
            config_hash=config.config_hash(),
            period_label=period_label,
            workspace_path=ws_path,
            groups=group_results,
            report_path=report_path,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            warnings_issued=issued_warnings,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_id(dataset_fp: str, config_hash: str, period_label: str | None) -> str:
    """Stable run identifier derived from dataset content, config, and period.

    Same inputs always produce the same ID, so a repeat invocation can find
    prior ledger entries and skip already-complete groups (resume behaviour).
    """
    key = f"{dataset_fp}:{config_hash}:{period_label or '__no_period__'}"
    return "run_" + hashlib.sha256(key.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Per-group sub-pipeline
# ---------------------------------------------------------------------------


def _slice_group_frame(
    df_raw: pl.DataFrame,
    group_by: list[str],
    group_values: dict[str, str],
    plan: FeaturePlan,
) -> pl.DataFrame:
    """Filter *df_raw* to one group and time-sort it.

    Sorting on the plan's time column keeps raw-value lookups and the feature
    matrix in the same row order.
    """
    if group_by:
        expr = pl.lit(True)
        for col_name, col_val in group_values.items():
            expr = expr & (pl.col(col_name).cast(pl.Utf8) == col_val)
        df_group = df_raw.filter(expr)
    else:
        df_group = df_raw
    if plan.chosen_time_column and plan.chosen_time_column in df_group.columns:
        df_group = df_group.sort(plan.chosen_time_column)
    return df_group


def _group_summary(
    group_key: str,
    group_label: str,
    n_records: int,
    *,
    status: str,
    error: str | None = None,
    n_anomalies: int = 0,
    anomaly_rate: float | None = None,
    results_path: Path | None = None,
    drifted: bool = False,
    refit_reason: str | None = None,
) -> GroupSummary:
    return GroupSummary(
        group_key=group_key,
        group_label=group_label,
        n_records=n_records,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        results_path=results_path,
        status=status,
        error=error,
        elapsed_seconds=0.0,
        drifted=drifted,
        refit_reason=refit_reason,
        warnings_issued=[],
    )


def _execute_group(
    ws: Workspace,
    run_id: str,
    group_key: str,
    group_label: str,
    group_values: dict[str, str],
    n_records: int,
    *,
    force: bool,
    body: Callable[[], GroupSummary],
) -> GroupSummary:
    """Run one group's *body* inside the shared ledger/status bookkeeping.

    Used by both ``run_detection`` and ``score_forward``: skip if already
    complete, mark running, capture warnings + timing, mark complete or failed.
    ``body`` returns a GroupSummary; its ``elapsed_seconds`` and
    ``warnings_issued`` are filled in here.
    """
    gv_json = json.dumps(group_values)

    if not force and ws.store.group_status(run_id, group_key) == "complete":
        return _group_summary(group_key, group_label, n_records, status="skipped")

    ws.store.upsert_run_group(
        run_id=run_id,
        group_key=group_key,
        group_values_json=gv_json,
        group_label=group_label,
        status="running",
        record_count=n_records,
    )

    t0 = time.time()
    warns: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SorethumbWarning)
            gsummary = body()
            warns = [str(w.message) for w in caught]
        gsummary.elapsed_seconds = time.time() - t0
        gsummary.warnings_issued = warns
        ws.store.upsert_run_group(
            run_id=run_id,
            group_key=group_key,
            group_values_json=gv_json,
            group_label=group_label,
            status="complete",
            record_count=n_records,
            anomaly_count=gsummary.n_anomalies,
            rate=gsummary.anomaly_rate,
            timing_seconds=gsummary.elapsed_seconds,
        )
        return gsummary
    except Exception as exc:
        elapsed = time.time() - t0
        err_msg = f"{type(exc).__name__}: {exc!s}"
        logger.exception("Group %s failed: %s", group_key, err_msg)
        ws.store.upsert_run_group(
            run_id=run_id,
            group_key=group_key,
            group_values_json=gv_json,
            group_label=group_label,
            status="failed",
            error=err_msg[:500],
            timing_seconds=elapsed,
        )
        summary = _group_summary(group_key, group_label, n_records, status="failed", error=err_msg[:500])
        summary.elapsed_seconds = elapsed
        summary.warnings_issued = warns
        return summary


def _run_group(
    ws: Workspace,
    run_id: str,
    config: Config,
    plan: FeaturePlan,
    full_space: FeatureSpace,  # noqa: ARG001 — reserved for future row-selection optimisation
    df_raw: pl.DataFrame,
    group_by: list[str],
    group_values: dict[str, str],
    group_key: str,
    group_label: str,
    n_records: int,
    period_label: str | None,
    dataset_fp: str,  # noqa: ARG001 — passed for future ledger integration
) -> GroupSummary:
    """Fit + score one group.

    Fits each enabled detector, persists the models, then hands off to
    :func:`_finalize_group` for calibrate / combine / explain / write.
    """
    df_group = _slice_group_frame(df_raw, group_by, group_values, plan)

    if len(df_group) < config.scoring.min_records:
        logger.info(
            "Skipping group %s: %d records < min_records=%d.",
            group_key,
            len(df_group),
            config.scoring.min_records,
        )
        return _group_summary(group_key, group_label, n_records, status="too_few_records")

    group_space = apply_feature_plan(df_group, plan)
    X = group_space.matrix.astype(np.float64)
    n_rows = len(X)

    # ── Detectors ─────────────────────────────────────────────────────────
    enabled_names = [d.name for d in config.detectors if d.enabled]
    det_instances: dict[str, Any] = {}
    raw_scores_map: dict[str, np.ndarray] = {}
    natural_flags_map: dict[str, np.ndarray] = {}
    calibrators: dict[str, Calibrator] = {}

    for det_name in enabled_names:
        if det_name not in registry:
            logger.warning("Detector %r not in registry; skipping.", det_name)
            continue

        det_cfg = next((d for d in config.detectors if d.name == det_name), None)
        params = dict(det_cfg.params) if det_cfg else {}
        # Wire scoring contamination → OCSVM nu when not explicitly overridden.
        # nu is the training-time equivalent of contamination for OneClassSVM.
        if det_name == "one_class_svm" and "nu" not in params:
            if isinstance(config.scoring.contamination, float):
                params["nu"] = config.scoring.contamination
        det = registry[det_name](**params)
        train_X = X
        if det.default_train_row_cap and n_rows > det.default_train_row_cap:
            rng = np.random.default_rng(config.run.seed)
            idx = rng.choice(n_rows, det.default_train_row_cap, replace=False)
            train_X = X[idx]

        det.fit(train_X, seed=config.run.seed)
        raw = det.score_samples(X)  # higher = more normal
        flags = det.natural_flag(raw)

        cal = Calibrator(mode="self")
        cal.fit(raw)

        det_instances[det_name] = det
        raw_scores_map[det_name] = raw
        natural_flags_map[det_name] = flags
        calibrators[det_name] = cal

        save_model(
            workspace=ws,
            run_id=run_id,
            group_key=group_key,
            detector=det,
            calibrator=cal,
            plan_json=plan.to_json(),
            feature_schema_hash=group_space.feature_schema_hash,
            train_row_count=len(train_X),
            seed=config.run.seed,
        )

    if not raw_scores_map:
        return _group_summary(
            group_key, group_label, n_records, status="failed", error="No detectors produced scores."
        )

    calibrated_map = {d: calibrators[d].transform(raw_scores_map[d]) for d in raw_scores_map}
    return _finalize_group(
        ws=ws,
        run_id=run_id,
        config=config,
        plan=plan,
        df_group=df_group,
        group_space=group_space,
        group_key=group_key,
        group_label=group_label,
        n_records=n_records,
        period_label=period_label,
        X=X,
        raw_scores_map=raw_scores_map,
        calibrated_map=calibrated_map,
        natural_flags_map=natural_flags_map,
        det_instances=det_instances,
        drifted=False,
    )


def _score_forward_group(
    ws: Workspace,
    source_run_id: str,
    run_id: str,
    config: Config,
    plan: FeaturePlan,
    df_raw: pl.DataFrame,
    group_by: list[str],
    group_values: dict[str, str],
    group_key: str,
    group_label: str,
    n_records: int,
    period_label: str | None,
    *,
    strict: bool,
) -> GroupSummary:
    """Score one group against a previous run's persisted models — no fitting.

    Applies *plan* (the source run's fitted FeaturePlan) to the new data, then
    calls :func:`score_with_existing`, which unpickles each detector + calibrator
    and calls only ``score_samples`` / ``natural_flag`` / ``transform``. Schema
    and library-version drift are detected there.
    """
    df_group = _slice_group_frame(df_raw, group_by, group_values, plan)

    if len(df_group) < config.scoring.min_records:
        return _group_summary(group_key, group_label, n_records, status="too_few_records")

    group_space = apply_feature_plan(df_group, plan)
    X = group_space.matrix.astype(np.float64)

    enabled_names = [d.name for d in config.detectors if d.enabled]
    res = score_with_existing(
        ws,
        source_run_id,
        group_key,
        X,
        group_space.feature_schema_hash,
        enabled_names,
        strict=strict,
    )
    raw_scores_map: dict[str, np.ndarray] = res["scores"]
    if not raw_scores_map:
        return _group_summary(
            group_key,
            group_label,
            n_records,
            status="failed",
            error=f"No persisted models for group {group_key} in source run {source_run_id}.",
        )
    if res["missing"]:
        logger.warning(
            "Group %s: source run %s has no model for %s; scoring with the rest.",
            group_key,
            source_run_id,
            res["missing"],
        )

    return _finalize_group(
        ws=ws,
        run_id=run_id,
        config=config,
        plan=plan,
        df_group=df_group,
        group_space=group_space,
        group_key=group_key,
        group_label=group_label,
        n_records=n_records,
        period_label=period_label,
        X=X,
        raw_scores_map=raw_scores_map,
        calibrated_map=res["calibrated"],
        natural_flags_map=res["natural_flags"],
        det_instances=res["detectors"],
        drifted=bool(res["drifted"]),
    )


def _finalize_group(
    *,
    ws: Workspace,
    run_id: str,
    config: Config,
    plan: FeaturePlan,
    df_group: pl.DataFrame,
    group_space: FeatureSpace,
    group_key: str,
    group_label: str,
    n_records: int,
    period_label: str | None,
    X: np.ndarray,
    raw_scores_map: dict[str, np.ndarray],
    calibrated_map: dict[str, np.ndarray],
    natural_flags_map: dict[str, np.ndarray],
    det_instances: dict[str, Any],
    drifted: bool,
) -> GroupSummary:
    """Ensemble-combine, threshold, explain, and write one group's results.

    The shared tail of both paths (fit and score-forward), so they cannot
    diverge on how scores become a results frame.
    """
    n_rows = len(X)

    # ── Ensemble combination ───────────────────────────────────────────────
    ensemble = ScoreEnsemble(
        weighting=config.scoring.weighting,
        combination=config.scoring.combination,
        contamination=config.scoring.contamination,
        manual_weights=config.scoring.weights or None,
    )
    result_dict = ensemble.combine(calibrated_map, natural_flags_map)
    composite_score: np.ndarray = result_dict["combined_score"]
    anomaly_flag: np.ndarray = result_dict["anomaly_flag"]
    weights_used: dict[str, float] = result_dict["weights"]

    flagged_idx = np.where(anomaly_flag)[0]
    n_anomalies = int(anomaly_flag.sum())
    anomaly_rate = n_anomalies / n_rows if n_rows > 0 else None

    # ── Explanations ────────────────────────────────────────────────────────
    top_n = config.explain.top_n
    attribution_cols = _compute_attributions(
        config=config,
        det_instances=det_instances,
        weights_used=weights_used,
        X=X,
        flagged_idx=flagged_idx,
        plan=plan,
        group_space=group_space,
    )

    # ── Build result DataFrame ────────────────────────────────────────────
    # Use actual id_column values when configured (joinable back to source);
    # fall back to positional index from the FeatureSpace otherwise.
    id_col = config.columns.id_column
    row_ids = df_group[id_col].to_numpy() if id_col and id_col in df_group.columns else group_space.row_ids

    rank_arr = np.zeros(n_rows, dtype=int)
    if n_anomalies > 0:
        order = np.argsort(composite_score)[::-1]
        rank_arr[order[:n_anomalies]] = np.arange(1, n_anomalies + 1)

    records: dict[str, Any] = {
        "row_id": row_ids.tolist(),
        "group_key": [group_key] * n_rows,
        "group_label": [group_label] * n_rows,
        "period_label": [period_label] * n_rows,
        "composite_score": composite_score.tolist(),
        "rank": rank_arr.tolist(),
        "flagged": anomaly_flag.tolist(),
        "attribution_kind": ["none"] * n_rows,
    }

    # Per-detector scores
    for det_name, raw in raw_scores_map.items():
        records[f"score_raw_{det_name}"] = raw.tolist()
    for det_name, cal_arr in calibrated_map.items():
        records[f"score_cal_{det_name}"] = cal_arr.tolist()

    # Reason columns (null for non-flagged rows)
    for i in range(top_n):
        records[f"reason_{i + 1}"] = [None] * n_rows

    # Fill in reasons for flagged rows
    if attribution_cols is not None and n_anomalies > 0:
        orig_attributions, attr_kind = attribution_cols
        orig_df_rows = df_group.to_dicts()  # pre-encoding values

        for row_pos in flagged_idx:
            raw_row_data = orig_df_rows[int(row_pos)] if int(row_pos) < len(orig_df_rows) else {}
            reasons = top_n_reasons(
                row_idx=int(row_pos),
                original_attributions=orig_attributions,
                raw_row=raw_row_data,
                top_n=top_n,
            )
            records["attribution_kind"][int(row_pos)] = attr_kind
            for r_i, reason in enumerate(reasons):
                reason_col = cast("str | None", reason.get("column"))
                raw_val = reason.get("raw_value")
                label = f"{reason_col}={raw_val}" if reason_col else None
                records[f"reason_{r_i + 1}"][int(row_pos)] = label

    df_results = pl.DataFrame(records)
    df_anomalies = df_results.filter(pl.col("flagged"))

    results_path = write_results(ws, run_id, group_key, df_anomalies)

    return _group_summary(
        group_key,
        group_label,
        n_records,
        status="success",
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        results_path=results_path,
        drifted=drifted,
    )


def _compute_attributions(
    config: Config,
    det_instances: dict[str, Any],
    weights_used: dict[str, float],
    X: np.ndarray,
    flagged_idx: np.ndarray,
    plan: FeaturePlan,
    group_space: FeatureSpace,
) -> tuple[dict[str, np.ndarray], str] | None:
    """Compute blended attributions for flagged rows. Returns None on failure."""
    from sorethumb.detectors.isolation_forest import IsolationForestDetector  # noqa: PLC0415
    from sorethumb.detectors.kmeans_distance import KMeansDetector  # noqa: PLC0415
    from sorethumb.explain.centroid import centroid_attributions  # noqa: PLC0415
    from sorethumb.explain.gradient import gradient_attributions  # noqa: PLC0415
    from sorethumb.explain.shap_tree import tree_shap_attributions  # noqa: PLC0415

    if len(flagged_idx) == 0:
        return None

    n_rows = len(X)
    n_features = X.shape[1]
    X_flagged = X[flagged_idx]

    sources: list[tuple[np.ndarray, str]] = []
    source_weights: list[float] = []

    for det_name, det in det_instances.items():
        try:
            if isinstance(det, IsolationForestDetector):
                # Full matrix needed; SHAP attributes all rows
                full_attr, tag = tree_shap_attributions(det, X, group_name=det_name)
                attr = full_attr[flagged_idx]
            elif isinstance(det, KMeansDetector):
                full_attr, tag = centroid_attributions(det, X)
                attr = full_attr[flagged_idx]
            else:
                # Gradient: operate only on flagged rows for cost control
                max_rows = config.explain.max_rows if hasattr(config.explain, "max_rows") else 5000
                attr, tag = gradient_attributions(det, X_flagged, max_rows=max_rows)
        except Exception:
            logger.debug("Attribution skipped for detector %r.", det_name, exc_info=True)
            continue

        # Pad to full n_rows shape centred on flagged positions
        # (blend expects same shape; flagged_idx positions us correctly)
        full = np.zeros((len(flagged_idx), n_features), dtype=np.float64)
        full[: len(attr)] = attr
        sources.append((full, tag))
        source_weights.append(weights_used.get(det_name, 1.0))

    if not sources:
        return None

    blended, blend_tag = blend(sources, source_weights)

    # PCA back-projection if applicable
    feature_attributions = blended
    feature_names = group_space.feature_names
    if plan.pca_components is not None:
        from sorethumb.explain.project import back_project_pca  # noqa: PLC0415

        try:
            feature_attributions = back_project_pca(
                blended,
                np.array(plan.pca_components),
                n_features=len(plan.output_features),
            )
            feature_names = plan.output_features
        except Exception:
            logger.debug("PCA back-projection failed; using feature-space attributions.", exc_info=True)

    orig_attributions = aggregate_to_original(feature_attributions, feature_names, plan.derived_to_original)

    # Expand to full-frame indexing so top_n_reasons can look up row_pos directly
    n_flagged = len(flagged_idx)
    expanded: dict[str, np.ndarray] = {}
    for orig_col, arr in orig_attributions.items():
        full_arr = np.zeros(n_rows, dtype=np.float64)
        full_arr[flagged_idx[:n_flagged]] = arr[:n_flagged]
        expanded[orig_col] = full_arr

    return expanded, blend_tag


# ---------------------------------------------------------------------------
# Report rendering helper
# ---------------------------------------------------------------------------


def _render_run_report(
    ws: Workspace,
    run_id: str,
    config: Config,
    plan: FeaturePlan,
    group_results: list[GroupSummary],
) -> Path | None:
    try:
        import sorethumb as _st  # noqa: PLC0415

        meta = RunMeta(
            run_id=run_id,
            dataset_uri=config.source.uri,
            dataset_fp="",
            config_hash=config.config_hash(),
            seed=config.run.seed,
            library_version=_st.__version__,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            config_json=config.model_dump_json(),
            started_at=datetime.now(UTC).isoformat(),
        )

        group_sections: list[GroupSection] = []
        for gsummary in group_results:
            if gsummary.results_path and gsummary.results_path.exists():
                from sorethumb.store.results import read_results  # noqa: PLC0415

                _rdf = read_results(ws, run_id, gsummary.group_key)
                records_df = _rdf if _rdf is not None else pl.DataFrame()
            else:
                records_df = pl.DataFrame()

            dropped = [
                {"column": d.column, "reason": d.reason, "class": d.col_class.value}
                for d in (plan.decisions or [])
                if d.treatment.value == "drop"
            ]

            group_sections.append(
                GroupSection(
                    group_key=gsummary.group_key,
                    group_label=gsummary.group_label,
                    records=records_df,
                    plan_dropped=dropped,
                )
            )

        report_dir = ws.root / "reports" / run_id
        return render_report(meta, group_sections, report_dir)

    except Exception:
        logger.warning("Report rendering failed; skipping.", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _group_label(ginfo: dict[str, Any], group_by: list[str]) -> str:
    if not group_by:
        return "__all__"
    return "_".join(str(ginfo.get(col, "")) for col in group_by)
