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
# Phase 4 → fit/apply schema stability
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Phase 4: width demotion is computed at fit time but never persisted to "
        "FeaturePlan. apply_feature_plan passes an empty demoted set, so a "
        "high-cardinality column is one-hot encoded on apply where it was "
        "frequency-encoded on fit, producing a different feature_schema_hash."
    ),
)
def test_fit_apply_schema_is_stable(tmp_path: Path) -> None:
    """apply_feature_plan must produce the same feature_schema_hash as build_features.

    Uses a column with enough unique values to trigger demotion
    (n_unique > max_feature_width, default 50).
    """
    from sorethumb.config import FeaturesConfig, ProfilingConfig

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

    profiling_cfg = ProfilingConfig()
    features_cfg = FeaturesConfig()  # max_feature_width=50 by default

    plan = build_feature_plan(df, profiling_cfg, features_cfg, ColumnsConfig())

    # Use a dummy config to pass features config
    dummy_cfg = Config(
        source=SourceConfig(uri="dummy"),
        run=RunConfig(workdir=str(tmp_path), seed=0),
    )

    fit_space = fit_features(df, plan, dummy_cfg)
    apply_space = apply_feature_plan(df, plan)

    assert fit_space.feature_schema_hash == apply_space.feature_schema_hash, (
        f"Fit hash {fit_space.feature_schema_hash!r} != "
        f"apply hash {apply_space.feature_schema_hash!r}. "
        f"Fit features: {fit_space.feature_names[:10]}… "
        f"Apply features: {apply_space.feature_names[:10]}…"
    )
