"""Integration test for scripts.validation.runner: real fit + score_forward + evaluate.

Unlike the rest of tests/scripts/validation/ (pure, fast unit tests), this
exercises the actual pipeline end to end -- a real workspace, SQLite store,
and model (de)serialisation -- so it is marked `integration`, not `unit`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scripts.validation.schema import CaseKey, ComboSpec, DatasetSpec

pytestmark = pytest.mark.integration


def _is_anomaly(label: pl.Series) -> pl.Series:
    return label == 1


def _make_labelled_dataset(tmp_path):
    # 400 rows total: a 70/30 split gives train=280, holdout=120, both above
    # the pipeline's default scoring.min_records=100 -- see
    # test_run_case_too_few_records_status_when_split_is_below_min_records
    # for what happens when a split falls below that floor.
    rng = np.random.default_rng(0)
    n_normal, n_anom = 360, 40
    normal = rng.normal(size=(n_normal, 4))
    anom = rng.uniform(-8, -6, size=(n_anom, 4))
    X = np.vstack([normal, anom])
    y = np.array([0] * n_normal + [1] * n_anom)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    df = pl.DataFrame({f"f{i}": X[:, i] for i in range(X.shape[1])}).with_columns(pl.Series("label", y))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.write_parquet(data_dir / "tiny_labelled.parquet")
    return data_dir


def test_run_case_end_to_end_with_labels(tmp_path):
    from scripts.validation.runner import run_case

    data_dir = _make_labelled_dataset(tmp_path)
    dataset = DatasetSpec(
        name="tiny_labelled",
        file="tiny_labelled.parquet",
        ignore=["label"],
        source="synthetic",
        description="synthetic labelled dataset for runner integration test",
        rows=100,
        cols=5,
        label_column="label",
        label_is_anomaly=_is_anomaly,
    )
    combo = ComboSpec(name="baseline", detectors=("isolation_forest", "kmeans_distance"))
    case = CaseKey(dataset="tiny_labelled", pca=False, combo="baseline", seed=0)
    workdir_base = tmp_path / "runs"

    result = run_case(case, dataset, combo, workdir_base, data_dir, code_rev="testrev", deps={"numpy": "0"})

    assert result.status == "success", result.error
    assert result.n_train + result.n_holdout == 400
    assert result.n_holdout > 0
    assert result.run_id is not None
    assert result.score_run_id is not None
    # A genuine label column was supplied -- accuracy metrics must be computed,
    # not left None as they are for an unlabelled dataset.
    assert result.n_holdout_anomalies_true is not None
    assert result.roc_auc is not None
    assert result.average_precision is not None
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.average_precision <= 1.0


def test_run_case_without_label_column_reports_no_accuracy_metrics(tmp_path):
    from scripts.validation.runner import run_case

    data_dir = _make_labelled_dataset(tmp_path)
    dataset = DatasetSpec(
        name="tiny_labelled",
        file="tiny_labelled.parquet",
        ignore=["label"],
        source="synthetic",
        description="same data, but registered with no label_column",
        rows=100,
        cols=5,
    )
    combo = ComboSpec(name="baseline", detectors=("isolation_forest",))
    case = CaseKey(dataset="tiny_labelled", pca=False, combo="baseline", seed=1)
    workdir_base = tmp_path / "runs"

    result = run_case(case, dataset, combo, workdir_base, data_dir, code_rev="testrev", deps={"numpy": "0"})

    assert result.status == "success", result.error
    assert result.roc_auc is None
    assert result.average_precision is None
    assert result.n_holdout_anomalies_true is None
    # Operational metrics must still be populated.
    assert result.n_holdout > 0


def test_run_case_reports_error_status_on_bad_source_file(tmp_path):
    from scripts.validation.runner import run_case

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = DatasetSpec(
        name="missing",
        file="does_not_exist.parquet",
        ignore=[],
        source="synthetic",
        description="deliberately missing file",
        rows=0,
        cols=0,
    )
    combo = ComboSpec(name="baseline", detectors=("isolation_forest",))
    case = CaseKey(dataset="missing", pca=False, combo="baseline", seed=0)

    result = run_case(case, dataset, combo, tmp_path / "runs", data_dir, code_rev="testrev", deps={})

    assert result.status == "error"
    assert result.error is not None


def test_run_case_too_few_records_status_when_split_is_below_min_records(tmp_path):
    """A split below scoring.min_records=100 must be reported distinctly, not as a
    misleading "0 anomalies found" success (see runner.run_case's status field).
    """
    from scripts.validation.runner import run_case

    rng = np.random.default_rng(0)
    df = pl.DataFrame({f"f{i}": rng.normal(size=50) for i in range(4)})  # 50 rows: both splits < 100
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.write_parquet(data_dir / "tiny.parquet")

    dataset = DatasetSpec(
        name="tiny", file="tiny.parquet", ignore=[], source="synthetic", description="d", rows=50, cols=4
    )
    combo = ComboSpec(name="baseline", detectors=("isolation_forest",))
    case = CaseKey(dataset="tiny", pca=False, combo="baseline", seed=0)

    result = run_case(case, dataset, combo, tmp_path / "runs", data_dir, code_rev="testrev", deps={})

    assert result.status == "too_few_records"
    assert result.error is None
    assert result.n_train + result.n_holdout == 50
    assert result.n_holdout_flagged == 0
    assert result.roc_auc is None
