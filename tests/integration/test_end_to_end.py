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
    HistoryConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.profiling.plan import build_feature_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_planted_csv(path: Path, *, n_normal: int = 280, n_anomaly: int = 5, seed: int = 0) -> list[int]:
    """Write a CSV with planted anomalies and return their 0-based row indices.

    Anomalies have num_a=999 (far outside the normal range of ~N(0,1)).
    The dataset includes:
      - ``id``: integer row identifier (joinable back to source)
      - ``num_a``, ``num_b``: numeric features
      - ``cat``: low-cardinality string
    """
    rng = np.random.default_rng(seed)
    n_total = n_normal + n_anomaly
    path.parent.mkdir(parents=True, exist_ok=True)

    num_a = rng.normal(0.0, 1.0, n_total).tolist()
    anomaly_indices = list(range(n_normal, n_total))  # last rows are anomalies
    for i in anomaly_indices:
        num_a[i] = 999.0

    df = pl.DataFrame(
        {
            "id": list(range(n_total)),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n_total).tolist(),
            "cat": ["A" if i % 3 != 0 else "B" for i in range(n_total)],
        }
    )
    df.write_csv(str(path))
    return anomaly_indices


def _make_time_sorted_parquet(path: Path, *, n: int = 20, anomaly_orig_idx: int = 0, seed: int = 0) -> None:
    """Write a Parquet file where sorting by ``ts`` reorders rows.

    Row at ``anomaly_orig_idx`` has num_a=999. The timestamp is DESCENDING in
    file order, so ascending time-sort moves row 0 (ts=latest) to the end.
    The sort mismatch bug surfaces when the pipeline looks up the raw value at
    the wrong original-frame position.

    Written as Parquet (not CSV) so that the ``ts`` column is preserved as a
    proper Datetime type — Polars CSV inference may not parse date strings,
    which would prevent the time-sort from firing.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    # Timestamps in descending order so ascending sort moves row 0 to the end
    timestamps = [base + timedelta(days=n - 1 - i) for i in range(n)]
    num_a = rng.normal(0.0, 1.0, n).tolist()
    num_a[anomaly_orig_idx] = 999.0

    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "ts": pl.Series(timestamps).dt.cast_time_unit("us"),  # proper Datetime dtype
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    )
    df.write_parquet(str(path))


def _minimal_config(
    csv_path: Path,
    workdir: Path,
    *,
    contamination: str | float = "auto",
    combination: str = "composite",
    detectors: list[DetectorConfig] | None = None,
) -> Config:
    """Build a minimal Config suitable for fast integration tests."""
    if detectors is None:
        # Default: isolation_forest only, avoiding kmeans contamination of composite score
        detectors = [DetectorConfig(name="isolation_forest")]
    return Config(
        source=SourceConfig(uri=str(csv_path), format="csv"),
        run=RunConfig(workdir=str(workdir), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=detectors,
        scoring=ScoringConfig(
            combination=combination,
            contamination=contamination,
            weighting="equal",
            min_records=5,
        ),
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
    csv = tmp_path / "data.csv"
    anomaly_idx = _make_planted_csv(csv, n_normal=280, n_anomaly=5)

    cfg = _minimal_config(
        csv,
        tmp_path / "ws",
        contamination=0.03,  # ~5% leaves room; 5/285 ≈ 1.75%
    )
    result = run_detection(cfg, no_report=True)

    assert result.n_succeeded == 1, f"Expected 1 successful group; got {result.groups}"
    assert result.n_anomalies > 0, "No anomalies found at all — pipeline may be broken"

    # Load the anomalies parquet and check at least one planted row appears
    parquet_path = result.groups[0].results_path
    assert parquet_path is not None
    df_anomalies = pl.read_parquet(parquet_path)
    flagged_ids = set(df_anomalies["row_id"].to_list())

    # row_id is the actual id column value (id = positional index in this CSV,
    # so 0-based). Planted anomalies are at the last n_anomaly rows (ids 280-284).
    n_total = 285
    n_anomaly = 5
    planted_positions = set(range(n_total - n_anomaly, n_total))
    overlap = flagged_ids & planted_positions
    assert len(overlap) > 0, (
        f"None of the planted anomaly rows ({planted_positions}) appeared in "
        f"flagged set ({sorted(flagged_ids)[:20]})"
    )


# ---------------------------------------------------------------------------
# Phase 1 → kmeans AUC guard
# ---------------------------------------------------------------------------


def test_kmeans_auc_above_half() -> None:
    """Every default detector must achieve AUC > 0.5 on a simple labelled set.

    A planted tight anomaly cluster is placed far from the inlier cloud.
    Any detector that cannot score those rows higher than inliers has
    directional failure (random baseline = 0.5).
    """
    from sklearn.metrics import roc_auc_score

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
        det.fit(X, seed=42)  # type: ignore[union-attr]
        scores = det.score_samples(X)  # type: ignore[union-attr]
        # Calibrated: higher = more anomalous. Negate because protocol says higher = more NORMAL.
        # After calibration the direction flips; here we work with raw scores directly.
        # The protocol says score_samples returns higher=more_normal, so anomalies = lower scores.
        # AUC is computed with anomaly=1 meaning we need lower scores → invert for roc_auc_score.
        auc = float(roc_auc_score(y_true, -scores))
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


def _two_period_parquet(path: Path, *, per_day: int = 100, seed: int = 0) -> dict[str, list[int]]:
    """Two calendar days of data with distinct planted anomalies per day.

    Day 2024-01-15: ids [n_a0..] have num_a = +999.
    Day 2024-01-16: a *different* set of ids have num_a = -999.
    Returns {"2024-01-15": [ids...], "2024-01-16": [ids...]}.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = per_day * 2
    ids = list(range(n))
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(per_day)]
    ts += [datetime(2024, 1, 16, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(per_day)]
    num_a = rng.normal(0.0, 1.0, n).tolist()

    planted = {
        "2024-01-15": [3, 17, 42],
        "2024-01-16": [per_day + 5, per_day + 8, per_day + 60, per_day + 91],
    }
    for i in planted["2024-01-15"]:
        num_a[i] = 999.0
    for i in planted["2024-01-16"]:
        num_a[i] = -999.0

    pl.DataFrame(
        {
            "id": ids,
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(path))
    return planted


def _period_config(parquet: Path, workdir: Path) -> Config:
    return Config(
        source=SourceConfig(uri=str(parquet), format="parquet"),
        run=RunConfig(workdir=str(workdir), seed=42),
        columns=ColumnsConfig(id_column="id", time_column="ts"),
        history=HistoryConfig(period_granularity="day", roll_non_business=False),
        detectors=[DetectorConfig(name="isolation_forest")],
        scoring=ScoringConfig(
            combination="composite", contamination="auto", weighting="equal", min_records=5
        ),
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
        totals = ws.store.totals_for_periods(dataset_fp, ["2024-01-15"])
        assert len(totals) == result.n_succeeded == 1
        row = totals[0]
        assert row["population"] == per_day
        assert row["anomaly_count"] == result.groups[0].n_anomalies
        assert row["run_id"] == result.run_id

        # the label is now complete: backfill would not re-queue it
        assert iter_pending_periods(ws.store, dataset_fp, ["2024-01-15", "2024-01-16"]) == ["2024-01-16"]


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
        labels = {r["period_label"] for r in ws.store.totals_for_periods(dfp, ["2024-01-15", "2024-01-16"])}
        assert labels == {"2024-01-15", "2024-01-16"}
        assert iter_pending_periods(ws.store, dfp, ["2024-01-15", "2024-01-16"]) == []

        # Both snapshots are recorded against the same dataset_fp.
        snaps = {s["snapshot_fp"] for s in ws.store.dataset_snapshots(dfp)}
        assert snaps == {r1.snapshot_fp, r2.snapshot_fp}
