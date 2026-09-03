"""End-to-end pipeline orchestration.

This module is the only place that sequences the library's subsystems:
source resolution → profiling → feature construction → detection →
calibration → scoring → explanation → persistence → reporting.

The client (cli.py) calls run_detection() and score_with_existing() and
inspects their return values. It does not call any subsystem directly.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sorethumb.config import Config, SourceConfig
from sorethumb.detectors import registry
from sorethumb.errors import SorethumbWarning
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
from sorethumb.store.models import save_model
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
    """Return value from run_detection / score_with_existing."""

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
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
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
        dataset_fp = f"{content_fp[:16]}_{schema_fp[:8]}"

        ws.store.upsert_dataset(
            dataset_fp=dataset_fp,
            source_uri=config.source.uri,
            schema_fingerprint=schema_fp,
            content_fingerprint=content_fp,
            n_rows=len(df_raw),
            n_cols=len(df_raw.columns),
        )

        # ── 2. Period resolution ─────────────────────────────────────────
        period_label: str | None = period_label_override
        if period_label is None and config.columns.time_column:
            from sorethumb.history.periods import resolve_period  # noqa: PLC0415

            _ref = datetime.now(UTC)
            _pf, _pt, period_label = resolve_period(
                _ref,
                config.history.period_granularity,
                config.history.roll_non_business,
            )
            df_raw = df_raw.filter(
                (pl.col(config.columns.time_column) >= _pf) & (pl.col(config.columns.time_column) < _pt)
            )

        # ── 3. Register run ──────────────────────────────────────────────
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

            # Skip ledger-complete groups unless forced
            if not force:
                status = ws.store.group_status(run_id, group_key)
                if status == "complete":
                    group_results.append(
                        GroupSummary(
                            group_key=group_key,
                            group_label=group_label,
                            n_records=n_records,
                            n_anomalies=0,
                            anomaly_rate=None,
                            results_path=None,
                            status="skipped",
                            error=None,
                            elapsed_seconds=0.0,
                            drifted=False,
                            refit_reason=None,
                            warnings_issued=[],
                        )
                    )
                    continue

            # Record as running
            ws.store.upsert_run_group(
                run_id=run_id,
                group_key=group_key,
                group_values_json=json.dumps(group_values),
                group_label=group_label,
                status="running",
                record_count=n_records,
            )

            t0 = time.time()
            group_warns: list[str] = []

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", SorethumbWarning)
                    gsummary = _run_group(
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
                    )
                    group_warns = [str(w.message) for w in caught]

                gsummary.elapsed_seconds = time.time() - t0
                gsummary.warnings_issued = group_warns
                issued_warnings.extend(group_warns)

                ws.store.upsert_run_group(
                    run_id=run_id,
                    group_key=group_key,
                    group_values_json=json.dumps(group_values),
                    group_label=group_label,
                    status="complete",
                    record_count=n_records,
                    anomaly_count=gsummary.n_anomalies,
                    rate=gsummary.anomaly_rate,
                    timing_seconds=gsummary.elapsed_seconds,
                )
                group_results.append(gsummary)

            except Exception as exc:
                elapsed = time.time() - t0
                err_msg = f"{type(exc).__name__}: {exc!s}"
                logger.exception("Group %s failed: %s", group_key, err_msg)
                ws.store.upsert_run_group(
                    run_id=run_id,
                    group_key=group_key,
                    group_values_json=json.dumps(group_values),
                    group_label=group_label,
                    status="failed",
                    error=err_msg[:500],
                    timing_seconds=elapsed,
                )
                group_results.append(
                    GroupSummary(
                        group_key=group_key,
                        group_label=group_label,
                        n_records=n_records,
                        n_anomalies=0,
                        anomaly_rate=None,
                        results_path=None,
                        status="failed",
                        error=err_msg[:500],
                        elapsed_seconds=elapsed,
                        drifted=False,
                        refit_reason=None,
                        warnings_issued=group_warns,
                    )
                )

        # ── 7. Mark run complete / failed ────────────────────────────────
        any_failed = any(g.status == "failed" for g in group_results)
        if any_failed:
            ws.store.mark_run_failed(run_id, "one or more groups failed")
        else:
            ws.store.mark_run_complete(run_id)

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
# Per-group sub-pipeline
# ---------------------------------------------------------------------------


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
    # Filter raw df to this group
    if group_by:
        filter_expr = pl.lit(True)
        for col_name, col_val in group_values.items():
            filter_expr = filter_expr & (pl.col(col_name).cast(pl.Utf8) == col_val)
        df_group = df_raw.filter(filter_expr)
    else:
        df_group = df_raw

    # Skip group below minimum records
    if len(df_group) < config.scoring.min_records:
        logger.info(
            "Skipping group %s: %d records < min_records=%d.",
            group_key,
            len(df_group),
            config.scoring.min_records,
        )
        return GroupSummary(
            group_key=group_key,
            group_label=group_label,
            n_records=n_records,
            n_anomalies=0,
            anomaly_rate=None,
            results_path=None,
            status="too_few_records",
            error=None,
            elapsed_seconds=0.0,
            drifted=False,
            refit_reason=None,
            warnings_issued=[],
        )

    # Apply the pre-fitted plan to the group's rows
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
        params = det_cfg.params if det_cfg else {}
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
        return GroupSummary(
            group_key=group_key,
            group_label=group_label,
            n_records=n_records,
            n_anomalies=0,
            anomaly_rate=None,
            results_path=None,
            status="failed",
            error="No detectors produced scores.",
            elapsed_seconds=0.0,
            drifted=False,
            refit_reason=None,
            warnings_issued=[],
        )

    # Calibrated scores: higher = more anomalous
    calibrated_map = {d: calibrators[d].transform(raw_scores_map[d]) for d in raw_scores_map}

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
    # Row IDs come from the FeatureSpace
    row_ids = group_space.row_ids

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
    for det_name, cal in calibrated_map.items():
        records[f"score_cal_{det_name}"] = cal.tolist()

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
                col_name = reason.get("column")
                raw_val = reason.get("raw_value")
                label = f"{col_name}={raw_val}" if col_name else None
                records[f"reason_{r_i + 1}"][int(row_pos)] = label

    df_results = pl.DataFrame(records)
    df_anomalies = df_results.filter(pl.col("flagged"))

    results_path = write_results(ws, run_id, group_key, df_anomalies)

    return GroupSummary(
        group_key=group_key,
        group_label=group_label,
        n_records=n_records,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        results_path=results_path,
        status="success",
        error=None,
        elapsed_seconds=0.0,
        drifted=False,
        refit_reason=None,
        warnings_issued=[],
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
                # last_contributions set after score_samples; index flagged rows
                full_attr, tag = centroid_attributions(det)
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

                records_df = read_results(ws, run_id, gsummary.group_key) or pl.DataFrame()
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
