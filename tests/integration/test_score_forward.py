"""`sorethumb score --from-run` / ``score_forward`` — reuse a prior run's fitted
FeaturePlan, per-detector models and calibrators, and re-fit *nothing*.

No network; all data generated in-process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb import Config, score_forward
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
from sorethumb.errors import StoreError
from sorethumb.store.workspace import Workspace


class _FitAttempted(BaseException):
    """Raised if any fitting path runs. A BaseException so the per-group
    ``except Exception`` handler cannot swallow it — it must reach the test."""


def _planted_csv(path: Path, *, n_normal: int = 200, n_anomaly: int = 6, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    n = n_normal + n_anomaly
    num_a = rng.normal(0.0, 1.0, n).tolist()
    for i in range(n_normal, n):
        num_a[i] = 999.0  # far outside ~N(0, 1)
    pl.DataFrame(
        {
            "id": list(range(n)),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
            "cat": ["A" if i % 3 else "B" for i in range(n)],
        }
    ).write_csv(str(path))
    return list(range(n_normal, n))


def _cfg(csv: Path, ws: Path) -> Config:
    return Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(ws), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[
            DetectorConfig(name="isolation_forest"),
            DetectorConfig(name="kmeans_distance"),
        ],
        scoring=ScoringConfig(
            combination="composite", contamination="auto", weighting="equal", min_records=5
        ),
    )


def _ban_all_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    import sorethumb._pipeline as pipe

    def _boom(*_a: object, **_k: object) -> None:
        raise _FitAttempted

    # Pipeline-level fitting entry points.
    monkeypatch.setattr(pipe, "fit_features", _boom)
    monkeypatch.setattr(pipe, "build_feature_plan", _boom)
    monkeypatch.setattr(pipe, "save_model", _boom)
    # Every detector wrapper's fit …
    for cls in (IsolationForestDetector, KMeansDetector, OneClassSVMDetector):
        monkeypatch.setattr(cls, "fit", _boom)
    # … and the underlying sklearn estimators, in case a wrapper is bypassed.
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    for cls in (IsolationForest, KMeans, OneClassSVM):
        monkeypatch.setattr(cls, "fit", _boom)


def test_score_forward_does_not_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    src_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    _planted_csv(src_csv, seed=0)
    _planted_csv(new_csv, seed=1)  # same schema, fresh draw, same 6 planted anomalies

    src = run_detection(_cfg(src_csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    _ban_all_fitting(monkeypatch)  # any fit from here raises _FitAttempted

    fwd = score_forward(_cfg(new_csv, ws), src.run_id, no_report=True)

    # It ran, wrote a distinct run, and touched no fitting path.
    assert fwd.run_id != src.run_id
    assert fwd.run_id.startswith("score_")
    assert fwd.n_failed == 0, [g.error for g in fwd.groups if g.error]
    assert fwd.n_succeeded >= 1
    good = next(g for g in fwd.groups if g.status == "success")
    assert good.results_path is not None
    assert good.results_path.exists()

    # Provenance is recorded on the run row.
    with Workspace.open(ws) as w:
        assert w.store.get_run(fwd.run_id)["source_run_id"] == src.run_id
        # No new model dirs were created for the score-forward run.
        assert not (ws / "models" / fwd.run_id).exists() or not any(
            (ws / "models" / fwd.run_id).glob("*/*.joblib")
        )


def test_score_forward_reuses_reference_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Score-forward on the *same* data as the source run must reproduce the
    source run's calibrated scores (same models, same calibrator)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)

    src = run_detection(_cfg(csv, ws), no_report=True)
    src_group = next(g for g in src.groups if g.status == "success")
    src_scores = pl.read_parquet(src_group.results_path).sort("row_id")

    _ban_all_fitting(monkeypatch)
    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    fwd_group = next(g for g in fwd.groups if g.status == "success")
    fwd_scores = pl.read_parquet(fwd_group.results_path).sort("row_id")

    assert src_scores["row_id"].to_list() == fwd_scores["row_id"].to_list()
    np.testing.assert_allclose(
        src_scores["composite_score"].to_numpy(),
        fwd_scores["composite_score"].to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_score_forward_missing_source_run_raises(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    run_detection(_cfg(csv, ws), no_report=True)

    with pytest.raises(StoreError, match="not found"):
        score_forward(_cfg(csv, ws), "run_does_not_exist", no_report=True)


def test_score_forward_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)

    _ban_all_fitting(monkeypatch)
    a = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    b = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert a.run_id == b.run_id  # deterministic id
    # second call skips the already-complete group
    assert any(g.status == "skipped" for g in b.groups)
