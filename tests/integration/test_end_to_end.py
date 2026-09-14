"""End-to-end regression harness for the sorethumb pipeline.

Each test targets one specific correctness property. Tests marked xfail document
known bugs; the xfail mark is removed when the corresponding fix lands:

  Phase 1 → test_kmeans_auc_above_half
  Phase 2 → test_explanation_references_perturbed_column
  Phase 3 → test_resume_skips_completed_group, test_run_id_is_deterministic
  Phase 4 → test_fit_apply_schema_is_stable

No network access; all data is generated in-process.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb import Config
from sorethumb._pipeline import run_detection
from sorethumb.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    FeaturesConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.profiling.plan import build_feature_plan
from tests.factories.configs import make_config
from tests.factories.detectors import detector_auc
from tests.factories.frames import write_planted_csv as _make_planted_csv
from tests.factories.frames import write_time_sorted_parquet as _make_time_sorted_parquet
from tests.factories.frames import write_two_period_parquet as _two_period_parquet
from tests.factories.runs import run_planted_detection

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(
    csv_path: Path,
    workdir: Path,
    *,
    contamination: str | float = "auto",
    combination: str = "composite",
    detectors: list[DetectorConfig] | None = None,
) -> Config:
    """Build a minimal Config suitable for fast integration tests."""
    return make_config(
        csv_path,
        workdir,
        contamination=contamination,
        combination=combination,
        detectors=detectors,
    )


# ---------------------------------------------------------------------------
# Phase 0 baseline: basic detection works
# ---------------------------------------------------------------------------


def test_planted_anomalies_are_detected(tmp_path: Path) -> None:
    """Planted num_a=999 rows must appear in the flagged output.

    This is the fundamental correctness baseline. It uses only
    isolation_forest (reliable) and explicit contamination so the
    threshold is not data-driven.
    """
    planted = run_planted_detection(
        tmp_path,
        n_normal=280,
        n_anomaly=5,
        contamination=0.03,  # ~5% leaves room; 5/285 ≈ 1.75%
    )
    result = planted.result

    assert result.n_succeeded == 1, f"Expected 1 successful group; got {result.groups}"
    assert result.n_anomalies > 0, "No anomalies found at all — pipeline may be broken"

    # Load the anomalies parquet and check at least one planted row appears
    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)
    flagged_ids = set(df_anomalies["row_id"].to_list())

    # row_id is the actual id column value (id = positional index in this CSV,
    # so 0-based, matching planted.anomaly_indices).
    planted_positions = set(planted.anomaly_indices)
    overlap = flagged_ids & planted_positions
    assert len(overlap) > 0, (
        f"None of the planted anomaly rows ({planted_positions}) appeared in "
        f"flagged set ({sorted(flagged_ids)[:20]})"
    )


# ---------------------------------------------------------------------------
# P0-3: intersection is a genuine three-way vote; rank tracks anomaly_flag
# ---------------------------------------------------------------------------


def test_intersection_rank_is_dense_and_positive_iff_flagged(tmp_path: Path) -> None:
    """For combination="intersection", anomaly_flag is a per-detector vote
    (AND of three independent thresholds), not the same statistic as
    composite_score (min across detectors). `rank` must be positive exactly
    for flagged rows, dense 1..n_flagged, ordered by composite_score
    descending -- never assigned to a row the vote did not actually flag.
    """
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=5, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(
        csv,
        workdir,
        contamination=0.05,
        combination="intersection",
        detectors=[
            DetectorConfig(name="isolation_forest"),
            DetectorConfig(name="kmeans_distance"),
            DetectorConfig(name="one_class_svm"),
        ],
    )

    result = run_detection(cfg, no_report=True)
    assert result.n_succeeded == 1
    group = result.groups[0]
    assert group.n_anomalies > 0, "test needs at least one flagged row to be meaningful"
    assert group.results_path is not None

    df = pl.read_parquet(group.results_path)
    # write_results persists only flagged rows, so every row here has
    # flagged=True by construction -- the meaningful check is that every one
    # of them also has a positive, dense rank (the invariant the fix
    # restores; the pre-fix code could instead assign these ranks to
    # whichever rows had the globally highest composite_score, flagged or not).
    assert (df["flagged"]).all()
    assert (df["rank"] > 0).all()
    assert sorted(df["rank"].to_list()) == list(range(1, group.n_anomalies + 1))

    # Dense ranks are ordered by composite_score descending.
    ordered = df.sort("rank")
    scores = ordered["composite_score"].to_list()
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Phase 1 → kmeans AUC guard
# ---------------------------------------------------------------------------


def test_kmeans_auc_above_half() -> None:
    """Every default detector must achieve AUC > 0.5 on a simple labelled set.

    A planted tight anomaly cluster is placed far from the inlier cloud.
    Any detector that cannot score those rows higher than inliers has
    directional failure (random baseline = 0.5).
    """
    rng = np.random.default_rng(42)
    n_inliers = 270
    n_anomalies = 30

    # 5-dimensional: inliers near origin, anomalies tightly clustered far away
    X_in = rng.normal(0.0, 1.0, (n_inliers, 5))
    X_an = rng.normal(10.0, 0.2, (n_anomalies, 5))  # tight cluster far from inliers
    X = np.vstack([X_in, X_an]).astype(np.float32)
    y_true = np.array([0] * n_inliers + [1] * n_anomalies)

    detectors: list[tuple[str, object]] = [
        ("isolation_forest", IsolationForestDetector()),
        ("kmeans_distance", KMeansDetector()),
        ("one_class_svm", OneClassSVMDetector()),
    ]

    for name, det in detectors:
        auc = detector_auc(det, X, y_true, seed=42)  # type: ignore[arg-type]
        assert auc > 0.5, (
            f"Detector '{name}' AUC={auc:.4f} on planted anomaly cluster — "
            f"should exceed 0.5 (random baseline). "
            f"kmeans collapses because anomalies get their own centroid."
        )


# ---------------------------------------------------------------------------
# Phase 2 → explanation references correct column
# ---------------------------------------------------------------------------


def test_explanation_references_perturbed_column(tmp_path: Path) -> None:
    """When num_a is perturbed in a time-sorted frame, reason_1 must say 'num_a=…'.

    The bug: build.py sorts by ts, but _pipeline.py reads raw values from
    the unsorted df_group. If the anomaly moves when sorted, the reason
    lookup hits the wrong original row and reason_1 is about a different column.
    """
    parquet = tmp_path / "data.parquet"
    n = 30
    anomaly_orig_idx = 0  # first row in file = latest timestamp = last after time-sort
    _make_time_sorted_parquet(parquet, n=n, anomaly_orig_idx=anomaly_orig_idx, seed=7)

    cfg = Config(
        source=SourceConfig(uri=str(parquet), format="parquet"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        scoring=ScoringConfig(
            combination="composite",
            contamination=0.07,  # ~2 rows flagged from 30
            weighting="equal",
            min_records=5,
        ),
    )
    result = run_detection(cfg, no_report=True)

    assert result.n_anomalies > 0, "No anomalies detected; check contamination threshold"

    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)

    # At least one flagged row should have reason_1 referencing num_a
    if "reason_1" not in df_anomalies.columns:
        pytest.skip("explain not enabled or reason columns absent")

    reason_values = df_anomalies["reason_1"].drop_nulls().to_list()
    # The reason must not only mention num_a but show the planted value (999).
    # If there is a sort/lookup mismatch the raw value will be from the wrong row
    # (a normal row, num_a ≈ N(0,1)) so the reason will say e.g. "num_a=0.31"
    # instead of "num_a=999.0".
    assert any("num_a=999" in str(r) for r in reason_values), (
        f"reason_1 should contain 'num_a=999' (the perturbed column=value) "
        f"but got: {reason_values}. "
        f"This indicates a sort/lookup mismatch: the raw value is read from the "
        f"wrong original frame position when a time column is present."
    )


def test_rows_beyond_max_rows_cap_are_marked_unavailable_not_fabricated(tmp_path: Path) -> None:
    """explain.max_rows caps per-row attribution methods (gradient/KernelSHAP).

    Rows beyond the cap must come back with attribution_kind="unavailable" and
    an explicit reason_1 -- not an all-zero vector that top_n_reasons turns into
    fabricated "column=value" reasons picked from arbitrary leading columns. The
    rows that DO get explained must be the most anomalous of the flagged set.
    """
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=180, n_anomaly=10, seed=3)

    cfg = Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        # one_class_svm has no native attributor: every source goes through the
        # capped gradient path, so exceeding max_rows leaves rows with zero
        # covering sources -- the scenario the guard exists for.
        detectors=[DetectorConfig(name="one_class_svm")],
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
        explain=ExplainConfig(max_rows=3, top_n=2),
    )
    result = run_detection(cfg, no_report=True)
    assert result.n_anomalies > 3, "fixture must flag more rows than explain.max_rows to exercise the cap"

    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df = pl.read_parquet(parquet_path).sort("composite_score", descending=True)

    kinds = df["attribution_kind"].to_list()
    assert all(k != "unavailable" for k in kinds[:3]), (
        f"the most anomalous rows must be explained first; got kinds={kinds}"
    )
    assert all(k == "unavailable" for k in kinds[3:]), (
        f"rows beyond explain.max_rows must be marked unavailable; got kinds={kinds}"
    )

    unavailable_reasons = df.filter(pl.col("attribution_kind") == "unavailable")["reason_1"].to_list()
    assert unavailable_reasons
    assert all(r is not None and "unavailable" in r for r in unavailable_reasons)

    explained_reasons = df.filter(pl.col("attribution_kind") != "unavailable")["reason_1"].to_list()
    assert explained_reasons
    assert all(r is not None and "unavailable" not in r and "=" in r for r in explained_reasons)


def test_pca_back_projection_uses_pre_pca_feature_names(tmp_path: Path) -> None:
    """With PCA on, reasons must name real original columns (num_a, num_b, cat)

    -- never a PCA component (pc_0, pc_1, ...). back_project_pca needs the
    exact column list the matrix had right before PCA ran (post width-control
    demotion, post correlation-drop); plan.output_features is the pre-demotion
    list and can silently mismatch it.
    """
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=180, n_anomaly=10, seed=11)

    cfg = Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name="isolation_forest")],
        features=FeaturesConfig(pca=True, pca_max_components=2, pca_min_explained_variance=0.5),
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
    )
    result = run_detection(cfg, no_report=True)
    assert result.n_anomalies > 0

    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)

    reasons = df_anomalies["reason_1"].drop_nulls().to_list()
    assert reasons, "expected at least one non-null reason"
    assert not any(r.split("=", 1)[0].startswith("pc_") for r in reasons), (
        f"reasons must reference original columns, not PCA components: {reasons}"
    )
    assert any("num_a" in r for r in reasons), f"expected the planted column to surface: {reasons}"


def test_pca_back_projection_failure_marks_all_flagged_rows_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A back-projection failure is not per-row: if the PCA loadings can't be

    applied to the blended attribution, no flagged row in the group has a
    trustworthy original-column attribution. Every flagged row must come back
    attribution_kind="unavailable" with an explicit reason -- never silently
    attributed in raw PCA-component space (pc_3="high" is not a real reason).
    """
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=180, n_anomaly=10, seed=11)

    cfg = Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name="isolation_forest")],
        features=FeaturesConfig(pca=True, pca_max_components=2, pca_min_explained_variance=0.5),
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced back-projection failure")

    monkeypatch.setattr("sorethumb.explain.project.back_project_pca", _raise)

    result = run_detection(cfg, no_report=True)
    assert result.n_anomalies > 0

    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)

    kinds = df_anomalies["attribution_kind"].to_list()
    reasons = df_anomalies["reason_1"].to_list()
    assert all(k == "unavailable" for k in kinds), f"expected every flagged row unavailable; got {kinds}"
    assert all(r == "unavailable (PCA back-projection failed)" for r in reasons), reasons


def test_ecod_hbos_get_exact_native_attributions(tmp_path: Path) -> None:
    """ECOD and HBOS decompose their score into per-feature terms exactly --

    no finite-difference gradient involved. Both flagged their attributions
    must come back attribution_kind="exact" with a real column=value reason,
    never "heuristic" (which the old gradient-fallback path would have given,
    routinely as an all-zero vector for a far-tail row landing in the same
    histogram bin / empirical-CDF rank as its unperturbed position).
    """
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=180, n_anomaly=10, seed=13)

    cfg = Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name="ecod"), DetectorConfig(name="hbos")],
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
    )
    result = run_detection(cfg, no_report=True)
    assert result.n_anomalies > 0

    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)

    kinds = df_anomalies["attribution_kind"].to_list()
    reasons = df_anomalies["reason_1"].to_list()
    assert all(k == "exact" for k in kinds), f"expected every flagged row exact; got {kinds}"
    assert all(r is not None and "=" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Phase 3 → resume / deterministic run_id
# ---------------------------------------------------------------------------


def test_run_id_is_deterministic(tmp_path: Path) -> None:
    """Two calls with identical config and data must return the same run_id."""
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"

    cfg = _minimal_config(csv, workdir, contamination=0.02)

    r1 = run_detection(cfg, no_report=True)
    r2 = run_detection(cfg, no_report=True)

    assert r1.run_id == r2.run_id, (
        f"Expected identical run_ids for identical inputs; got {r1.run_id!r} vs {r2.run_id!r}"
    )


def test_resume_skips_completed_group(tmp_path: Path) -> None:
    """A second run with identical inputs must skip already-complete groups."""
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"

    cfg = _minimal_config(csv, workdir, contamination=0.02)

    r1 = run_detection(cfg, no_report=True)
    assert r1.n_succeeded >= 1

    r2 = run_detection(cfg, no_report=True)
    assert r2.n_skipped >= 1, (
        f"Second identical run should skip completed groups; "
        f"got n_skipped={r2.n_skipped}, n_succeeded={r2.n_succeeded}"
    )


# ---------------------------------------------------------------------------
# P0-1: a failed group must be persisted as failed, not complete -- a resumed
# run must retry it, and must never report the run as complete because a
# failed group was later (wrongly) treated as already done.
# ---------------------------------------------------------------------------


def test_group_with_no_scoring_detectors_is_failed_not_complete(tmp_path: Path) -> None:
    """A group where no configured detector produces scores must be recorded
    as failed in both the RunResult and the ledger, and must be retried (not
    silently skipped as already complete) on the next call with the same
    inputs.
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02, detectors=[DetectorConfig(name="does_not_exist")])

    r1 = run_detection(cfg, no_report=True)
    assert r1.n_failed == 1
    assert r1.n_succeeded == 0
    g1 = r1.groups[0]
    assert g1.status == "failed"
    assert g1.error is not None
    assert "No detectors produced scores" in g1.error

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r1.run_id) == "failed"
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    # Same inputs -> same deterministic run_id. The group must be re-executed,
    # not treated as already complete, and the run must still report failure.
    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_failed == 1
    assert r2.n_succeeded == 0
    assert r2.groups[0].status == "failed", "a failed group must be retried, not skipped"

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r2.run_id) == "failed"


def test_group_ledger_status_survives_detector_fit_failure_and_retry(tmp_path: Path) -> None:
    """A detector.fit() exception must fail the group and the run; once the
    underlying problem is gone, retrying the same run_id must actually
    re-execute the group (not report false success from a stale 'complete'
    ledger row, and not report false failure once it truly works).
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02)

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("synthetic fit failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "fit", _boom)
        r1 = run_detection(cfg, no_report=True)

    assert r1.n_failed == 1
    assert r1.n_succeeded == 0
    g1 = r1.groups[0]
    assert g1.status == "failed"
    assert "synthetic fit failure" in (g1.error or "")

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r1.run_id) == "failed"
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    # fit() is no longer patched -- the retry must actually re-run the group
    # (the ledger must not have recorded it as already complete) and succeed.
    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_failed == 0
    assert r2.n_succeeded == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.run_status(r2.run_id) == "complete"
        assert ws.store.group_status(r2.run_id, g1.group_key) == "complete"


def test_group_ledger_status_survives_score_samples_failure_and_retry(tmp_path: Path) -> None:
    """Same as the fit-failure case, but the exception comes from
    score_samples (after a successful fit) instead of fit itself.
    """
    from sorethumb import Workspace

    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=3, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.02)

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("synthetic scoring failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "score_samples", _boom)
        r1 = run_detection(cfg, no_report=True)

    assert r1.n_failed == 1
    g1 = r1.groups[0]
    assert g1.status == "failed"

    with Workspace.open(workdir) as ws:
        assert ws.store.group_status(r1.run_id, g1.group_key) == "failed"

    r2 = run_detection(cfg, no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_succeeded == 1
    assert r2.n_failed == 0


def test_repeat_run_does_not_blank_the_report(tmp_path: Path) -> None:
    """A resumed run (every group skipped) must re-render the same report, not an empty one."""
    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=5, seed=0)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.03)

    r1 = run_detection(cfg, no_report=False)
    assert r1.n_anomalies > 0
    assert r1.report_path is not None
    assert r1.report_path.exists()
    html1 = r1.report_path.read_text(encoding="utf-8")
    csvs1 = sorted(r1.report_path.parent.glob("*.csv"))
    assert any(p.stat().st_size > 0 for p in csvs1), "first run wrote a non-empty group CSV"
    assert "No anomalies flagged." not in html1, "first report has a real records table"

    r2 = run_detection(cfg, no_report=False)
    assert r2.run_id == r1.run_id
    assert r2.n_succeeded == 0
    assert r2.n_skipped >= 1
    # Skipped-group summaries carry the persisted counts + results path forward.
    assert r2.n_anomalies == r1.n_anomalies
    assert r2.groups[0].results_path is not None
    assert r2.groups[0].results_path.exists()

    assert r2.report_path is not None
    html2 = r2.report_path.read_text(encoding="utf-8")
    assert "No anomalies flagged." not in html2, "re-rendered report still has the records table"
    assert html2 == html1, "resumed run reproduced the report byte-for-byte instead of blanking it"


def test_render_report_for_run_rebuilds_a_deleted_report(tmp_path: Path) -> None:
    """render_report_for_run recreates index.html + group CSVs purely from persisted state."""
    from sorethumb import Workspace, render_report_for_run

    csv = tmp_path / "data.csv"
    _make_planted_csv(csv, n_normal=200, n_anomaly=5, seed=1)
    workdir = tmp_path / "ws"
    cfg = _minimal_config(csv, workdir, contamination=0.03)

    r1 = run_detection(cfg, no_report=False)
    assert r1.report_path is not None
    report_dir = r1.report_path.parent
    original = r1.report_path.read_text(encoding="utf-8")

    # Nuke the rendered artefacts; the DB + results Parquet are untouched.
    for p in report_dir.iterdir():
        p.unlink()

    with Workspace.open(workdir) as ws:
        out = render_report_for_run(ws, r1.run_id)
        assert render_report_for_run(ws, "run_does_not_exist") is None

    assert out is not None
    assert out.exists()
    assert out.read_text(encoding="utf-8") == original
    assert list(report_dir.glob("*.csv"))  # group CSV siblings rebuilt too


# ---------------------------------------------------------------------------
# Phase 4 → fit/apply schema stability
# ---------------------------------------------------------------------------


def test_fit_apply_schema_is_stable(tmp_path: Path) -> None:
    """apply_feature_plan must produce the same feature_schema_hash as build_features.

    Uses a column with enough unique values to trigger demotion
    (n_unique > max_feature_width, default 50).
    """
    rng = np.random.default_rng(0)
    n = 200
    # 80 unique categories → above default max_feature_width=50 → triggers demotion
    cats = [f"cat_{i % 80}" for i in range(n)]
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "high_card": cats,
        }
    )

    cfg = Config(
        source=SourceConfig(uri="dummy"),
        run=RunConfig(workdir=str(tmp_path), seed=0),
    )
    plan = build_feature_plan(df, cfg)
    fit_space = fit_features(df, plan, cfg)
    apply_space = apply_feature_plan(df, plan)

    assert fit_space.feature_schema_hash == apply_space.feature_schema_hash, (
        f"Fit hash {fit_space.feature_schema_hash!r} != "
        f"apply hash {apply_space.feature_schema_hash!r}. "
        f"Fit features: {fit_space.feature_names[:10]}… "
        f"Apply features: {apply_space.feature_names[:10]}…"
    )


# ---------------------------------------------------------------------------
# Historical period selection: an override must filter to that period's window
# ---------------------------------------------------------------------------


def _period_config(parquet: Path, workdir: Path, *, detectors: list[DetectorConfig] | None = None) -> Config:
    return make_config(
        parquet,
        workdir,
        source_format="parquet",
        combination="composite",
        detectors=detectors,
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )


def _one_day_two_group_parquet(path: Path, *, per_group: int = 60, seed: int = 0) -> None:
    """One calendar day, two groups ("cat" A/B), each large enough to fit a detector."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = per_group * 2
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    pl.DataFrame(
        {
            "id": list(range(n)),
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "cat": ["A"] * per_group + ["B"] * per_group,
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(path))


def _period_config_with_groups(parquet: Path, workdir: Path) -> Config:
    return make_config(
        parquet,
        workdir,
        source_format="parquet",
        combination="composite",
        time_column="ts",
        group_by=["cat"],
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )


@pytest.mark.parametrize("label", ["2024-01-15", "2024-01-16"])
def test_period_override_filters_to_that_window(tmp_path: Path, label: str) -> None:
    parquet = tmp_path / "two_periods.parquet"
    per_day = 100
    planted = _two_period_parquet(parquet, per_day=per_day)
    cfg = _period_config(parquet, tmp_path / "ws")

    result = run_detection(cfg, period_label_override=label, no_report=True)

    assert result.period_label == label
    assert result.n_succeeded == 1
    group = result.groups[0]
    # Only that day's rows entered the pipeline — not the whole 200-row dataset.
    assert group.n_records == per_day

    flagged = set(pl.read_parquet(group.results_path)["row_id"].to_list())
    this_day = set(planted[label])
    other_day = set(planted["2024-01-16" if label == "2024-01-15" else "2024-01-15"])

    # This period's planted anomalies are caught; the other period's ids are
    # absent entirely (they were never in the filtered frame).
    assert this_day <= flagged, f"missed planted anomalies {this_day - flagged}"
    assert not (flagged & other_day)
    assert all(rid < per_day for rid in flagged) == (label == "2024-01-15")


def test_period_overrides_produce_distinct_runs(tmp_path: Path) -> None:
    parquet = tmp_path / "two_periods.parquet"
    _two_period_parquet(parquet)
    cfg = _period_config(parquet, tmp_path / "ws")

    r15 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    r16 = run_detection(cfg, period_label_override="2024-01-16", no_report=True)

    assert r15.run_id != r16.run_id
    assert r15.n_anomalies > 0
    assert r16.n_anomalies > 0


def test_period_run_records_history_ledger(tmp_path: Path) -> None:
    """run_detection writes the period row + per-group totals so backfill is idempotent."""
    from sorethumb.history.ledger import iter_pending_periods
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "two_periods.parquet"
    per_day = 100
    _two_period_parquet(parquet, per_day=per_day)
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    result = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    dataset_fp = result.dataset_fp

    with Workspace.open(workdir) as ws:
        # period row carries the [from, to) window for the label
        prow = ws.store._conn.execute(
            "SELECT period_from, period_to FROM period WHERE dataset_fp=? AND period_label=?",
            (dataset_fp, "2024-01-15"),
        ).fetchone()
        assert prow is not None
        assert (prow["period_from"], prow["period_to"]) == ("2024-01-15", "2024-01-16")

        # one totals row per processed group, population == rows that entered the pipeline
        totals = ws.store.totals_for_periods(dataset_fp, ["2024-01-15"], result.config_hash)
        assert len(totals) == result.n_succeeded == 1
        row = totals[0]
        assert row["population"] == per_day
        assert row["anomaly_count"] == result.groups[0].n_anomalies
        assert row["run_id"] == result.run_id

        # the label is now complete: backfill would not re-queue it
        assert iter_pending_periods(
            ws.store, dataset_fp, result.config_hash, ["2024-01-15", "2024-01-16"]
        ) == ["2024-01-16"]


# ---------------------------------------------------------------------------
# P0-2: history completion is atomic and scoped to (dataset_fp, period_label,
# config_hash) -- a partial or crashed attempt must be retried, two
# configurations sharing a period must never blend or block each other, and a
# clean repeat must be idempotent.
# ---------------------------------------------------------------------------


def test_repeat_period_run_is_idempotent_in_history_ledger(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "two_periods.parquet"
    _two_period_parquet(parquet, per_day=100)
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r1.run_id == r2.run_id

    with Workspace.open(workdir) as ws:
        totals = ws.store.totals_for_periods(r1.dataset_fp, ["2024-01-15"], r1.config_hash)
        assert len(totals) == 1, "a repeat run must not duplicate the totals row"
        assert ws.store.period_is_complete(r1.dataset_fp, "2024-01-15", r1.config_hash) is True


def test_two_configs_on_one_period_do_not_double_count_or_block_each_other(tmp_path: Path) -> None:
    """Two configurations processing the same period_label must each get
    their own totals row (scoped by config_hash), and running one must not
    make the ledger think the other is already done.
    """
    from sorethumb.history.ledger import iter_pending_periods
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "two_periods.parquet"
    _two_period_parquet(parquet, per_day=100)
    workdir = tmp_path / "ws"
    cfg_a = _period_config(parquet, workdir, detectors=[DetectorConfig(name="isolation_forest")])
    cfg_b = _period_config(parquet, workdir, detectors=[DetectorConfig(name="kmeans_distance")])
    assert cfg_a.config_hash() != cfg_b.config_hash()

    ra = run_detection(cfg_a, period_label_override="2024-01-15", no_report=True)
    assert ra.n_succeeded == 1

    with Workspace.open(workdir) as ws:
        # Config B has never run this period -- must still be pending under
        # its own hash, regardless of config A having just completed it.
        pending_b = iter_pending_periods(ws.store, ra.dataset_fp, cfg_b.config_hash(), ["2024-01-15"])
        assert pending_b == ["2024-01-15"]

    rb = run_detection(cfg_b, period_label_override="2024-01-15", no_report=True)
    assert rb.n_succeeded == 1
    assert ra.run_id != rb.run_id

    with Workspace.open(workdir) as ws:
        totals_a = ws.store.totals_for_periods(ra.dataset_fp, ["2024-01-15"], ra.config_hash)
        totals_b = ws.store.totals_for_periods(rb.dataset_fp, ["2024-01-15"], rb.config_hash)
        assert len(totals_a) == 1
        assert len(totals_b) == 1
        # Each config's own read never includes the other's row.
        assert {t["config_hash"] for t in totals_a} == {ra.config_hash}
        assert {t["config_hash"] for t in totals_b} == {rb.config_hash}

        assert ws.store.period_is_complete(ra.dataset_fp, "2024-01-15", ra.config_hash) is True
        assert ws.store.period_is_complete(rb.dataset_fp, "2024-01-15", rb.config_hash) is True

        # Rolling-window aggregation for A must not pick up B's contribution.
        other = ws.store.other_config_hashes_for_periods(ra.dataset_fp, ["2024-01-15"], ra.config_hash)
        assert other == {rb.config_hash}


def test_period_with_one_failed_group_is_not_marked_complete(tmp_path: Path) -> None:
    """A period where one of several groups fails must not be recorded
    complete -- only the succeeding group's totals are written -- and a retry
    with the same inputs must re-execute the failed group and only then
    complete the period.
    """
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "grouped.parquet"
    _one_day_two_group_parquet(parquet)
    workdir = tmp_path / "ws"
    cfg = _period_config_with_groups(parquet, workdir)

    calls = {"n": 0}
    orig_fit = IsolationForestDetector.fit

    def _flaky_fit(self: object, *a: object, **k: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("synthetic failure for the second group")
        return orig_fit(self, *a, **k)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "fit", _flaky_fit)
        r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    assert r1.n_succeeded == 1
    assert r1.n_failed == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r1.dataset_fp, "2024-01-15", r1.config_hash) is False
        totals = ws.store.totals_for_periods(r1.dataset_fp, ["2024-01-15"], r1.config_hash)
        assert len(totals) == 1, "only the succeeding group's totals are recorded"

    # Retry with the same inputs: the failed group must be re-executed (fit is
    # no longer patched), and only then does the period become complete.
    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_succeeded == 1
    assert r2.n_skipped == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r2.dataset_fp, "2024-01-15", r2.config_hash) is True
        totals = ws.store.totals_for_periods(r2.dataset_fp, ["2024-01-15"], r2.config_hash)
        assert len(totals) == 2


def test_interruption_between_groups_recovers_all_totals_on_retry(tmp_path: Path) -> None:
    """A crash after one group is marked complete in the run ledger but
    before the next group even starts -- so _record_period_history never
    runs at all -- must not permanently lose the finished group's totals: a
    retry must record every group, including the one that was only resumed
    (skipped) this time, not just the one it (re-)executes.
    """
    import sorethumb._pipeline as pipe
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "grouped.parquet"
    _one_day_two_group_parquet(parquet)
    workdir = tmp_path / "ws"
    cfg = _period_config_with_groups(parquet, workdir)

    real_execute_group = pipe._execute_group
    calls = {"n": 0}

    def _crash_on_second_group(*a: object, **k: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash between groups")
        return real_execute_group(*a, **k)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipe, "_execute_group", _crash_on_second_group)
        with pytest.raises(RuntimeError, match="simulated crash between groups"):
            run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    with Workspace.open(workdir) as ws:
        run_row = ws.store.list_runs(limit=1)[0]
        run_id = str(run_row["run_id"])
        dataset_fp = str(run_row["dataset_fp"])
        config_hash = cfg.config_hash()

        # The first group committed to the run ledger before the crash; the
        # second group was never attempted. Neither has a totals row yet, and
        # the period is not complete -- _record_period_history never ran.
        run_groups = ws.store.all_run_groups(run_id)
        assert len(run_groups) == 1
        assert run_groups[0]["status"] == "complete"
        assert ws.store.totals_for_periods(dataset_fp, ["2024-01-15"], config_hash) == []
        assert ws.store.period_is_complete(dataset_fp, "2024-01-15", config_hash) is False

    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r2.run_id == run_id
    assert r2.n_skipped == 1, "the first group must be resumed, not re-executed"
    assert r2.n_succeeded == 1, "the second group runs for the first time"

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r2.dataset_fp, "2024-01-15", r2.config_hash) is True
        totals = ws.store.totals_for_periods(r2.dataset_fp, ["2024-01-15"], r2.config_hash)
        assert len(totals) == 2, "the resumed (skipped) group's totals must be recorded too"


def _write_days_parquet(path: Path, *, days: list[str], per_day: int = 80, seed: int = 0) -> None:
    """Write ``per_day`` rows for each ISO day in *days*; row 3 of each day is a +999 anomaly."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    ts: list[datetime] = []
    num_a: list[float] = []
    for d, day in enumerate(days):
        base = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(hours=8)
        col = rng.normal(0.0, 1.0, per_day).tolist()
        col[3] = 999.0
        for r in range(per_day):
            ids.append(d * per_day + r)
            ts.append(base + timedelta(minutes=r))
            num_a.append(col[r])
    pl.DataFrame(
        {
            "id": ids,
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, len(ids)).tolist(),
        }
    ).write_parquet(str(path))


def test_appending_a_snapshot_keeps_dataset_identity_and_history(tmp_path: Path) -> None:
    """Appending a day of rows must not orphan the previous period's history."""
    from sorethumb.history.ledger import iter_pending_periods
    from sorethumb.store.workspace import Workspace

    parquet = tmp_path / "growing.parquet"
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    # Snapshot 1: only 2024-01-15 exists. Process that period.
    _write_days_parquet(parquet, days=["2024-01-15"])
    r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    # Snapshot 2: 2024-01-16 appended. Process the new period.
    _write_days_parquet(parquet, days=["2024-01-15", "2024-01-16"])
    r2 = run_detection(cfg, period_label_override="2024-01-16", no_report=True)

    # Logical identity is unchanged; the snapshot fingerprint moved.
    assert r1.dataset_fp == r2.dataset_fp
    assert r1.snapshot_fp
    assert r2.snapshot_fp
    assert r1.snapshot_fp != r2.snapshot_fp
    assert r1.run_id != r2.run_id  # snapshot_fp is part of the run id

    with Workspace.open(workdir) as ws:
        dfp = r2.dataset_fp
        # Both periods' totals live under the one logical dataset — nothing orphaned.
        labels = {
            r["period_label"]
            for r in ws.store.totals_for_periods(dfp, ["2024-01-15", "2024-01-16"], r2.config_hash)
        }
        assert labels == {"2024-01-15", "2024-01-16"}
        assert iter_pending_periods(ws.store, dfp, r2.config_hash, ["2024-01-15", "2024-01-16"]) == []

        # Both snapshots are recorded against the same dataset_fp.
        snaps = {s["snapshot_fp"] for s in ws.store.dataset_snapshots(dfp)}
        assert snaps == {r1.snapshot_fp, r2.snapshot_fp}
