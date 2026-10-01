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
    assert group.results_path is not None
    flagged = pl.read_parquet(group.results_path).filter(pl.col("flagged"))
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
    # Nothing explains the flagged rows now: the group says so ("none") instead of inventing a source.
    assert result.groups[0].results_path is not None
    flagged = pl.read_parquet(result.groups[0].results_path).filter(pl.col("flagged"))
    assert (flagged["attribution_kind"] == "none").all()


def test_strict_mode_promotes_the_warning_to_a_group_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _break_gradient(monkeypatch)
    lenient = run_detection(_cfg(tmp_path / "a"), no_report=True)
    assert lenient.n_failed == 0

    strict = run_detection(_cfg(tmp_path / "b", strict=True), no_report=True)
    assert strict.n_failed == 1
    assert strict.n_succeeded == 0
    assert "AttributionBackendWarning" in (strict.groups[0].error or "")
    assert _BOOM in (strict.groups[0].error or "")


def test_healthy_backends_emit_no_attribution_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_detection(_cfg(tmp_path), no_report=True)
    assert not [w for w in result.groups[0].warnings_issued if "Attribution backend failed" in w]


def test_warning_is_a_project_warning() -> None:
    assert issubclass(AttributionBackendWarning, SorethumbWarning)
