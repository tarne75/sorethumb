"""The explanation reference end to end: saved by a fitting run, reused by score-forward, pinned."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sorethumb_ml import Config, score_forward
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import DetectorConfig
from sorethumb_ml.explain.record import EXPLAINER_VERSION
from sorethumb_ml.explain.reference import REFERENCE_FILENAME, ReferenceSummary
from sorethumb_ml.store.models import load_reference_summary
from sorethumb_ml.store.results import read_attributions
from sorethumb_ml.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv
from tests.factories.results import joined_results

pytestmark = pytest.mark.integration


def _cfg(csv: Path, ws: Path, *, explain: bool = True, detectors: list[str] | None = None) -> Config:
    return make_config(
        csv,
        ws,
        combination="composite",
        contamination=0.05,
        detectors=[DetectorConfig(name=n) for n in (detectors or ["isolation_forest", "ecod"])],
        explain_kwargs={"enabled": explain, "top_n": 2},
    )


def _records(ws: Path, run_id: str, group_key: str) -> pl.DataFrame:
    with Workspace.open(ws) as w:
        records = read_attributions(w, run_id, group_key)
    assert records is not None
    return records


def test_a_fitting_run_saves_its_reference_summary(tmp_path: Path) -> None:
    csv, ws = tmp_path / "train.csv", tmp_path / "ws"
    write_planted_csv(csv, n_normal=200, n_anomaly=6, seed=0)
    result = run_detection(_cfg(csv, ws), no_report=True)
    group = result.groups[0]
    path = ws / "models" / result.run_id / group.group_key / REFERENCE_FILENAME
    assert path.is_file()
    with Workspace.open(ws) as w:
        summary, problem = load_reference_summary(w, result.run_id, group.group_key)
    assert problem is None
    assert summary is not None
    assert summary.reference_id == "consensus_normal"
    assert summary.n_rows <= 200 - group.n_anomalies
    assert summary.label == f"consensus_normal:{result.run_id}"
    # Raw typical values are kept per original field, on the original scale.
    value, kind = summary.raw_values["num_a"]
    assert kind == "ordered"
    assert abs(float(value)) < 1.0  # the planted 999s are not part of "normal"
    assert summary.raw_values["cat"][1] == "categorical"


def test_score_forward_explains_against_the_fitting_runs_reference(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    train, new = tmp_path / "train.csv", tmp_path / "new.csv"
    write_planted_csv(train, n_normal=200, n_anomaly=6, seed=0)
    write_planted_csv(new, n_normal=60, n_anomaly=6, seed=1)
    src = run_detection(_cfg(train, ws), no_report=True)
    fwd = score_forward(_cfg(new, ws), src.run_id, no_report=True)
    group = fwd.groups[0]
    assert group.n_anomalies > 0
    assert "explain.reference_fallback" not in {r["code"] for r in group.warning_records}

    records = _records(ws, fwd.run_id, group.group_key)
    assert set(records["reference_id"].to_list()) == {f"consensus_normal:{src.run_id}"}
    with Workspace.open(ws) as w:
        summary, _ = load_reference_summary(w, src.run_id, group.group_key)
    assert summary is not None
    num_a = records.filter(pl.col("original_field") == "num_a")
    assert num_a.height > 0
    assert set(num_a["reference_value"].to_list()) == {str(summary.raw_values["num_a"][0])}
    # A score-forward run does not write a summary of its own.
    assert not (ws / "models" / fwd.run_id / group.group_key / REFERENCE_FILENAME).exists()


def test_score_forward_without_a_saved_reference_falls_back_with_a_warning(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    train, new = tmp_path / "train.csv", tmp_path / "new.csv"
    write_planted_csv(train, n_normal=200, n_anomaly=6, seed=0)
    write_planted_csv(new, n_normal=200, n_anomaly=6, seed=1)
    src = run_detection(_cfg(train, ws, explain=False), no_report=True)
    assert not (ws / "models" / src.run_id / src.groups[0].group_key / REFERENCE_FILENAME).exists()

    fwd = score_forward(_cfg(new, ws), src.run_id, no_report=True)
    group = fwd.groups[0]
    (rec,) = [r for r in group.warning_records if r["code"] == "explain.reference_fallback"]
    assert "none was saved" in str(rec["message"])
    records = _records(ws, fwd.run_id, group.group_key)
    assert set(records["reference_id"].to_list()) == {"consensus_normal"}


def test_an_unreadable_reference_summary_falls_back_too(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    train = tmp_path / "train.csv"
    write_planted_csv(train, n_normal=200, n_anomaly=6, seed=0)
    src = run_detection(_cfg(train, ws), no_report=True)
    path = ws / "models" / src.run_id / src.groups[0].group_key / REFERENCE_FILENAME
    path.write_text("{not json", encoding="utf-8")
    fwd = score_forward(_cfg(train, ws), src.run_id, no_report=True)
    messages = [
        str(r["message"]) for r in fwd.groups[0].warning_records if r["code"] == "explain.reference_fallback"
    ]
    assert messages
    assert "unreadable" in messages[0]


def test_a_group_with_nothing_flagged_still_saves_its_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    import sorethumb_ml._pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "_flagged_idx_by_score_desc", lambda *_a, **_k: np.array([], dtype=np.int64)
    )
    csv, ws = tmp_path / "train.csv", tmp_path / "ws"
    write_planted_csv(csv, n_normal=200, n_anomaly=0, seed=0)
    result = run_detection(_cfg(csv, ws), no_report=True)
    group = result.groups[0]
    assert group.n_anomalies == 0
    assert group.explain_status == "no_flagged_rows"
    assert (ws / "models" / result.run_id / group.group_key / REFERENCE_FILENAME).is_file()


def test_golden_explanations_for_the_planted_anomalies(tmp_path: Path) -> None:
    """Pins what the planted anomalies are explained by, against which reference, with which logic.

    Labels and structure only: numeric attributions differ in the last digits across platforms.
    """
    csv, ws = tmp_path / "train.csv", tmp_path / "ws"
    planted = write_planted_csv(csv, n_normal=200, n_anomaly=6, seed=0)
    result = run_detection(_cfg(csv, ws), no_report=True)
    group = result.groups[0]
    assert group.results_path is not None
    rows = joined_results(group.results_path).filter(pl.col("row_id").is_in(planted) & pl.col("flagged"))
    assert rows.height > 0
    assert set(rows["explain_status"].to_list()) == {"ok"}
    assert set(rows["explainer_version"].to_list()) == {EXPLAINER_VERSION} == {"explain@2"}
    assert set(rows["reason_1"].to_list()) == {"num_a=999.0"}

    records = _records(ws, result.run_id, group.group_key).filter(
        pl.col("row_id").is_in(rows["row_id"].to_list()) & (pl.col("reason_rank") == 1)
    )
    assert records.height == rows.height
    assert set(records["original_field"].to_list()) == {"num_a"}
    assert set(records["direction"].to_list()) == {"above"}
    assert set(records["reference_id"].to_list()) == {"consensus_normal"}
    assert (records["signed_contribution"] > 0).all()
    methods = {(m["detector"], m["method"]) for ms in records["methods"].to_list() for m in ms}
    assert ("ecod", "ecod_decomposition") in methods
    assert {d for d, _ in methods} == {"isolation_forest", "ecod"}


def test_the_summary_file_is_plain_json(tmp_path: Path) -> None:
    csv, ws = tmp_path / "train.csv", tmp_path / "ws"
    write_planted_csv(csv, n_normal=120, n_anomaly=4, seed=2)
    result = run_detection(_cfg(csv, ws, detectors=["isolation_forest"]), no_report=True)
    path = ws / "models" / result.run_id / result.groups[0].group_key / REFERENCE_FILENAME
    summary = ReferenceSummary.from_json(path.read_text(encoding="utf-8"))
    assert summary.n_features == len(summary.feature_names)
    assert 0 < len(summary.background) <= 50
