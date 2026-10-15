"""An attribution backend that raises is reported, not silently dropped."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import polars as pl
import pytest

import sorethumb_ml.explain.gradient as gradient_mod
from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import (
    ColumnsConfig,
    DetectorConfig,
    ExplainConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb_ml.errors import AttributionBackendWarning, SorethumbWarning
from tests.factories.frames import write_planted_csv
from tests.factories.policy_evidence import record_evidence

pytestmark = pytest.mark.integration

_BOOM = "synthetic gradient backend failure"


def _cfg(
    tmp_path: Path, *, strict: bool = False, detectors: tuple[str, ...] = ("kmeans_distance", "one_class_svm")
) -> Config:
    csv = tmp_path / "data.csv"
    write_planted_csv(csv, n_normal=180, n_anomaly=10, seed=3)
    return Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(tmp_path / "ws"), seed=42, strict=strict),
        columns=ColumnsConfig(id_column="id"),
        detectors=[DetectorConfig(name=n) for n in detectors],
        scoring=ScoringConfig(combination="composite", contamination=0.05, weighting="equal", min_records=5),
        explain=ExplainConfig(top_n=2),
    )


def _break_gradient(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def _boom(*_a: object, **_k: object) -> object:
        calls.append(1)
        raise RuntimeError(_BOOM)

    monkeypatch.setattr(gradient_mod, "gradient_attributions", _boom)
    return calls


def test_failing_backend_emits_one_warning_with_cause_and_keeps_other_explanations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _break_gradient(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="sorethumb_ml"):
        result = run_detection(_cfg(tmp_path), no_report=True)

    assert calls, "the broken backend was never reached"
    assert result.n_succeeded == 1
    group = result.groups[0]
    hits = [w for w in group.warnings_issued if "Attribution backend failed" in w]
    assert len(hits) == 1, group.warnings_issued  # one per detector and group
    assert "one_class_svm" in hits[0]
    assert _BOOM in hits[0]
    assert "RuntimeError" in hits[0]
    assert "kmeans_distance" not in hits[0]

    records = [r for r in caplog.records if "Attribution backend failed" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is not None  # traceback preserved for the log
    assert _BOOM in records[0].getMessage()

    # The surviving detector (kmeans centroid) still explains every flagged row.
    assert group.explain_status == "partial"  # one explainer failed
    flagged = _results(tmp_path, result)
    assert flagged.height > 0
    assert (flagged["attribution_kind"] != "unavailable").all()
    assert flagged["reason_1"].is_not_null().all()


def test_each_failing_detector_gets_its_own_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sorethumb_ml.explain.centroid as centroid_mod

    _break_gradient(monkeypatch)

    def _boom(*_a: object, **_k: object) -> object:
        raise ValueError("centroid failure")

    monkeypatch.setattr(centroid_mod, "centroid_attributions", _boom)
    result = run_detection(_cfg(tmp_path), no_report=True)

    hits = [w for w in result.groups[0].warnings_issued if "Attribution backend failed" in w]
    assert len(hits) == 2
    assert {("one_class_svm" in h, "kmeans_distance" in h) for h in hits} == {(True, False), (False, True)}
    # Nothing explains the flagged rows now. The group says so -- explain_status
    # "failed" and a warning -- and its detection output stands untouched.
    group = result.groups[0]
    assert group.status == "success"
    assert group.n_anomalies > 0
    assert group.explain_status == "failed"
    assert group.n_explained == 0
    assert any("No attribution source explained" in w for w in group.warnings_issued)
    assert any(r["code"] == "explain.no_source" for r in group.warning_records)
    flagged = _results(tmp_path, result)
    assert flagged.height == group.n_anomalies
    assert "attribution_kind" not in flagged.columns  # no explanation file was written


def test_strict_mode_keeps_detection_output_and_exits_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explanation warning never fails a group, strict or not: the detection
    output is final before explanation starts. Under run.strict the run exits
    partial instead."""
    from sorethumb_ml.cli import ExitCode, _exit_code_for

    _break_gradient(monkeypatch)
    lenient = run_detection(_cfg(tmp_path / "a"), no_report=True)
    assert lenient.n_failed == 0
    assert _exit_code_for(lenient) is ExitCode.OK

    strict_cfg = _cfg(tmp_path / "b", strict=True)
    record_evidence(strict_cfg)  # run.strict otherwise refuses the composite policy before any group runs
    strict = run_detection(strict_cfg, no_report=True)
    assert strict.n_failed == 0
    assert strict.n_succeeded == 1
    group = strict.groups[0]
    assert group.n_anomalies == lenient.groups[0].n_anomalies
    assert any(_BOOM in w for w in group.explain_warnings)
    assert _exit_code_for(strict) is ExitCode.PARTIAL


def _results(tmp_path: Path, result: object) -> pl.DataFrame:
    """The group's flagged rows with explanations joined on (store.results.read_results)."""
    from sorethumb_ml.store.results import read_results
    from sorethumb_ml.store.workspace import Workspace

    group = result.groups[0]  # type: ignore[attr-defined]
    with Workspace.open(tmp_path / "ws") as ws:
        df = read_results(ws, result.run_id, group.group_key)  # type: ignore[attr-defined]
    assert df is not None
    return df.filter(pl.col("flagged"))


def test_healthy_backends_emit_no_attribution_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_detection(_cfg(tmp_path), no_report=True)
    assert not [w for w in result.groups[0].warnings_issued if "Attribution backend failed" in w]


def test_warning_is_a_project_warning() -> None:
    assert issubclass(AttributionBackendWarning, SorethumbWarning)
