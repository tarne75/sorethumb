"""Core pipeline correctness: planted anomalies are actually detected, the
default three-way intersection produces dense positive ranks (P0-3), every
default detector clears random baseline, and fit/apply produce the same
feature schema hash.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb import Config
from sorethumb._pipeline import run_detection
from sorethumb.config import DetectorConfig, FeaturesConfig, RunConfig, SourceConfig
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector
from sorethumb.errors import FeatureWidthWarning
from sorethumb.features.build import apply_feature_plan, fit_features
from sorethumb.profiling.plan import build_feature_plan
from tests.factories.configs import make_config
from tests.factories.detectors import detector_auc
from tests.factories.frames import write_planted_csv
from tests.factories.runs import run_planted_detection

pytestmark = pytest.mark.integration


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


def test_intersection_rank_is_dense_and_positive_iff_flagged(tmp_path: Path) -> None:
    """For combination="intersection", anomaly_flag is a per-detector vote
    (AND of three independent thresholds), not the same statistic as
    composite_score (min across detectors). `rank` must be positive exactly
    for flagged rows, dense 1..n_flagged, ordered by composite_score
    descending -- never assigned to a row the vote did not actually flag.
    """
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=200, n_anomaly=5, seed=0)
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


def test_fit_apply_schema_is_stable(tmp_path: Path) -> None:
    """apply_feature_plan must produce the same feature_schema_hash as fit_features
    when width-control demotion actually fires -- not just when the fixture happens
    to already be under budget and demotion is a no-op on both sides.

    Three low-cardinality categorical columns (each one-hot on its own) push the
    matrix well past a deliberately restrictive max_feature_width, so at least one
    must be demoted to frequency encoding for both fit and apply to agree on.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "cat_a": [f"a{i % 8}" for i in range(n)],
            "cat_b": [f"b{i % 8}" for i in range(n)],
            "cat_c": [f"c{i % 8}" for i in range(n)],
        }
    )

    cfg = Config(
        source=SourceConfig(uri="dummy"),
        run=RunConfig(workdir=str(tmp_path), seed=0),
        features=FeaturesConfig(max_feature_width=10),  # 3 one-hot columns need ~27
    )
    plan = build_feature_plan(df, cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_space = fit_features(df, plan, cfg)

    assert plan.demoted_columns, "fixture must actually trigger demotion to exercise this path"
    assert any(issubclass(w.category, FeatureWidthWarning) for w in caught)

    apply_space = apply_feature_plan(df, plan)

    assert fit_space.feature_schema_hash == apply_space.feature_schema_hash, (
        f"Fit hash {fit_space.feature_schema_hash!r} != "
        f"apply hash {apply_space.feature_schema_hash!r}. "
        f"Fit features: {fit_space.feature_names[:10]}… "
        f"Apply features: {apply_space.feature_names[:10]}…"
    )
