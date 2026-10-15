"""Detection output is final before explanation; explanation failures never change it.

The detection stage writes ``anomalies.parquet`` (and records its sha256)
before any explanation is computed; the explanation stage writes only
``explanations.parquet`` and its own status, inside its own error boundary.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import sorethumb_ml._pipeline as pipeline_mod
from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import ExitCode, _exit_code_for
from sorethumb_ml.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb_ml.store.results import file_digest, read_results
from sorethumb_ml.store.workspace import Workspace
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration


def _cfg(tmp_path: Path, name: str, *, explain: bool = True, strict: bool = False) -> Config:
    csv = tmp_path / "data.csv"
    if not csv.exists():
        write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=7)
    return Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / name), seed=42, strict=strict),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name="isolation_forest"), DetectorConfig(name="kmeans_distance")],
        scoring=ScoringConfig(contamination=0.05, min_records=5),
        explain=ExplainConfig(enabled=explain, top_n=2),
    )


def _break_explanation_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs: object) -> object:
        raise RuntimeError("synthetic explanation-stage failure")

    monkeypatch.setattr(pipeline_mod, "_explanation_frame", _boom)


def test_detection_output_is_byte_identical_whatever_the_explanation_stage_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explained = run_detection(_cfg(tmp_path, "a"), no_report=True)
    disabled = run_detection(_cfg(tmp_path, "b", explain=False), no_report=True)
    _break_explanation_stage(monkeypatch)
    broken = run_detection(_cfg(tmp_path, "c"), no_report=True)

    digests = []
    for result in (explained, disabled, broken):
        (group,) = result.groups
        assert group.status == "success"
        assert group.results_path is not None
        assert group.detection_digest == file_digest(group.results_path)  # recorded as written
        digests.append(group.detection_digest)
    assert len(set(digests)) == 1
    assert explained.groups[0].n_anomalies == disabled.groups[0].n_anomalies == broken.groups[0].n_anomalies
    assert [g.explain_status for g in (explained, disabled, broken) for g in g.groups] == [
        "ok",
        "not_requested",
        "failed",
    ]


def test_an_explanation_stage_failure_is_a_warning_and_a_partial_exit_not_a_group_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_explanation_stage(monkeypatch)
    result = run_detection(_cfg(tmp_path, "ws"), no_report=True)
    (group,) = result.groups
    assert result.n_failed == 0
    assert group.n_anomalies > 0
    assert group.explain_status == "failed"
    assert any("Explanation stage failed" in w and "synthetic" in w for w in group.warnings_issued)
    (rec,) = [r for r in group.warning_records if r["code"] == "explain.stage_failed"]
    assert rec["rows_affected"] == group.n_anomalies
    assert _exit_code_for(result) is ExitCode.PARTIAL
    with Workspace.open(tmp_path / "ws") as ws:
        row = ws.store.get_run_group(result.run_id, group.group_key)
        frame = read_results(ws, result.run_id, group.group_key)
    assert row is not None
    assert row["status"] == "complete"
    assert row["explain_status"] == "failed"
    assert row["anomaly_count"] == group.n_anomalies
    assert frame is not None
    assert frame.height == group.n_anomalies


def test_run_strict_never_fails_a_group_over_its_explanations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_explanation_stage(monkeypatch)
    result = run_detection(_cfg(tmp_path, "ws", strict=True), no_report=True)
    assert result.n_failed == 0
    assert result.groups[0].n_anomalies > 0
    assert _exit_code_for(result) is ExitCode.PARTIAL


def test_explanations_disabled_writes_no_explanation_file_and_exits_ok(tmp_path: Path) -> None:
    result = run_detection(_cfg(tmp_path, "ws", explain=False), no_report=True)
    (group,) = result.groups
    assert group.explain_status == "not_requested"
    assert group.results_path is not None
    assert not group.results_path.with_name("explanations.parquet").exists()
    assert _exit_code_for(result) is ExitCode.OK


def test_explained_results_join_on_and_a_resumed_run_keeps_the_explain_outcome(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "ws")
    first = run_detection(cfg, no_report=True)
    (group,) = first.groups
    assert group.explain_status == "ok"
    assert group.n_explained == group.n_anomalies
    with Workspace.open(tmp_path / "ws") as ws:
        joined = read_results(ws, first.run_id, group.group_key)
        bare = read_results(ws, first.run_id, group.group_key, with_explanations=False)
    assert joined is not None
    assert bare is not None
    assert {"attribution_kind", "reason_1", "reason_2"} <= set(joined.columns)
    assert "reason_1" not in bare.columns  # the detection file carries no explanation columns
    assert joined["reason_1"].is_not_null().all()

    again = run_detection(cfg, no_report=True)  # same deterministic run_id: the group is skipped
    (resumed,) = again.groups
    assert resumed.status == "skipped"
    assert (resumed.explain_status, resumed.n_explained, resumed.detection_digest) == (
        "ok",
        group.n_explained,
        group.detection_digest,
    )


def test_results_written_before_the_split_read_back_unchanged(tmp_path: Path) -> None:
    """Older runs stored reasons inside anomalies.parquet and have no explanations file."""
    from sorethumb_ml.store.results import write_results

    legacy = pl.DataFrame({"row_id": [1, 2], "flagged": [True, True], "reason_1": ["a=1", "b=2"]})
    with Workspace.init(tmp_path / "ws") as ws:
        write_results(ws, "run_x", "group_y", legacy)
        frame = read_results(ws, "run_x", "group_y")
    assert frame is not None
    assert frame.equals(legacy)


# ---------------------------------------------------------------------------
# Structured explanation warnings and per-row status (E2)
# ---------------------------------------------------------------------------


def test_a_failing_explainer_is_a_structured_warning_and_its_rows_are_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import sorethumb_ml.explain.centroid as centroid_mod

    def _boom(*_a: object, **_k: object) -> object:
        raise ValueError("centroid failure")

    monkeypatch.setattr(centroid_mod, "centroid_attributions", _boom)
    result = run_detection(_cfg(tmp_path, "ws"), no_report=True)
    (group,) = result.groups
    (rec,) = [r for r in group.warning_records if r["code"] == "explain.backend_failed"]
    assert rec["detector"] == "kmeans_distance"
    assert rec["stage"] == "explain"
    assert rec["rows_affected"] == group.n_anomalies
    assert rec in result.warning_records
    assert group.explain_status == "partial"
    with Workspace.open(tmp_path / "ws") as ws:
        frame = read_results(ws, result.run_id, group.group_key)
        stored = json.loads(str(ws.store.get_run_group(result.run_id, group.group_key)["warnings_json"]))
    assert frame is not None
    assert (frame["explain_status"] == "partial").all()
    assert (frame["explain_issue"] == "detectors_unexplained:kmeans_distance").all()
    assert frame["reason_1"].is_not_null().all()  # isolation_forest still explains every row
    assert any(r["code"] == "explain.backend_failed" for r in stored)  # persisted as records


def test_a_detector_with_no_vetted_explainer_is_reported_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sorethumb_ml.detectors import registry
    from tests.factories.detectors import UnexplainablePluginDetector

    monkeypatch.setitem(registry, "plugin_detector", UnexplainablePluginDetector)
    cfg = _cfg(tmp_path, "ws").model_copy(
        update={"detectors": [DetectorConfig(name="kmeans_distance"), DetectorConfig(name="plugin_detector")]}
    )
    result = run_detection(cfg, no_report=True)
    (group,) = result.groups
    (rec,) = [r for r in group.warning_records if r["code"] == "explain.detector_unsupported"]
    assert rec["detector"] == "plugin_detector"
    assert group.explain_status == "partial"


def test_a_group_with_too_few_normal_rows_says_its_reference_includes_flagged_rows(tmp_path: Path) -> None:
    csv = tmp_path / "small.csv"
    write_planted_csv(csv, n_normal=10, n_anomaly=4, seed=3)
    cfg = Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name="isolation_forest"), DetectorConfig(name="kmeans_distance")],
        scoring=ScoringConfig(combination="composite", contamination=0.4, min_records=5),
        explain=ExplainConfig(top_n=2),
    )
    result = run_detection(cfg, no_report=True)
    codes = [r["code"] for r in result.groups[0].warning_records]
    assert "explain.reference_fallback" in codes


def test_the_run_summary_and_json_show_warning_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from typer.testing import CliRunner

    import sorethumb_ml.explain.centroid as centroid_mod
    from sorethumb_ml.cli import app

    def _boom(*_a: object, **_k: object) -> object:
        raise ValueError("centroid failure")

    monkeypatch.setattr(centroid_mod, "centroid_attributions", _boom)
    cfg = _cfg(tmp_path, "ws")
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f"""[source]
uri = {json.dumps(cfg.source.uri)}
format = "csv"
[run]
workdir = {json.dumps(cfg.run.workdir)}
seed = 42
[columns]
id_column = "id"
[scoring]
contamination = 0.05
min_records = 5
[explain]
top_n = 2
[[detectors]]
name = "isolation_forest"
[[detectors]]
name = "kmeans_distance"
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    human = runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert "[explain.backend_failed]" in " ".join(human.stdout.split())
    doc = json.loads(
        runner.invoke(app, ["run", "--config", str(toml), "--no-report", "--json", "--force"]).stdout
    )
    assert any(w["code"] == "explain.backend_failed" for w in doc["warnings"])
    assert any(w["code"] == "explain.backend_failed" for w in doc["groups"][0]["warnings"])
    shown = runner.invoke(app, ["show", doc["run_id"], "--config", str(toml)])
    assert "explain.backend_failed" in " ".join(shown.stdout.split())
