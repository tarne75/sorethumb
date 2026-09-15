"""Attribution correctness through the full pipeline: reasons reference the
actual perturbed/original column (never a raw PCA component or a fabricated
zero-vector guess), rows beyond explain.max_rows are marked unavailable
rather than fabricated, and ECOD/HBOS's native decomposition is exact.
"""

from __future__ import annotations

from pathlib import Path

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
from tests.factories.frames import write_planted_csv, write_time_sorted_parquet

pytestmark = pytest.mark.integration


def test_explanation_references_perturbed_column(tmp_path: Path) -> None:
    """When num_a is perturbed in a time-sorted frame, reason_1 must say 'num_a=…'.

    The bug: build.py sorts by ts, but _pipeline.py reads raw values from
    the unsorted df_group. If the anomaly moves when sorted, the reason
    lookup hits the wrong original row and reason_1 is about a different column.
    """
    parquet = tmp_path / "data.parquet"
    n = 30
    anomaly_orig_idx = 0  # first row in file = latest timestamp = last after time-sort
    write_time_sorted_parquet(parquet, n=n, anomaly_orig_idx=anomaly_orig_idx, seed=7)

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

    # explain is enabled by default, so reason columns must be present -- a
    # missing reason_1 here is itself a bug, not a reason to skip the test.
    assert "reason_1" in df_anomalies.columns, (
        f"explain is enabled by default; reason_1 must be present. Got columns: {df_anomalies.columns}"
    )

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
    write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=3)

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
    write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=11)

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
    write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=11)

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
    write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=13)

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
