"""The run-time evidence gate for a non-default ensemble policy, and the backtest that feeds it.

A non-default policy without an ``outperforms`` record for this dataset and
configuration scope warns (and is still run); ``run.strict`` refuses it before
anything is registered; with such a record the run names the evidence it relied on.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import Config
from sorethumb_ml.errors import (
    ConfigError,
    SchemaError,
    UnevidencedPolicyError,
    UnevidencedPolicyWarning,
)
from sorethumb_ml.evaluate.user_backtest import backtest_dataset
from sorethumb_ml.store.workspace import Workspace
from tests.factories.policy_evidence import record_evidence

pytestmark = pytest.mark.integration


def _write_labelled_csv(path: Path, *, n_normal: int = 380, n_anomaly: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_normal + n_anomaly, 3))
    x[n_normal:] += 6.0
    pl.DataFrame(
        {
            "a": x[:, 0],
            "b": x[:, 1],
            "c": x[:, 2],
            "label": [0] * n_normal + [1] * n_anomaly,
        }
    ).write_csv(path)


def _cfg(
    tmp_path: Path, *, combination: str = "composite", weighting: str = "agreement", **over: object
) -> Config:
    csv = tmp_path / "data.csv"
    if not csv.exists():
        _write_labelled_csv(csv)
    raw: dict[str, object] = {
        "source": {"uri": str(csv), "format": "csv"},
        "run": {"workdir": str(tmp_path / "ws"), "seed": 0, **over.pop("run", {})},  # type: ignore[dict-item]
        "columns": {"reference_column": "label"},
        "detectors": [{"name": "isolation_forest"}, {"name": "kmeans_distance"}],
        "scoring": {"contamination": 0.05, "combination": combination, "weighting": weighting},
        "explain": {"enabled": False},
    }
    raw.update(over)
    return Config.model_validate(raw)


def _run_row(cfg: Config, run_id: str) -> dict[str, object]:
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        row = ws.store.get_run(run_id)
    assert row is not None
    return row


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_default_policy_needs_no_evidence(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, combination="intersection", weighting="equal")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnevidencedPolicyWarning)
        result = run_detection(cfg, no_report=True)
    assert result.policy_evidence_id is None
    assert not [w for w in result.warnings_issued if "backtest" in w]


def test_a_non_default_policy_without_evidence_warns_and_still_runs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.warns(UnevidencedPolicyWarning, match="no backtest has been run"):
        result = run_detection(cfg, no_report=True)
    assert result.n_succeeded == 1
    assert result.policy_evidence_id is None
    assert any("composite-agreement@1" in w for w in result.warnings_issued)
    assert _run_row(cfg, result.run_id)["policy_evidence_id"] is None


def test_run_strict_refuses_a_policy_without_evidence_before_registering_the_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run={"strict": True})
    with pytest.raises(UnevidencedPolicyError, match=r"run\.strict refuses"):
        run_detection(cfg, no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        assert ws.store.list_runs() == []


def test_outperforming_evidence_in_scope_is_recorded_on_the_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run={"strict": True})
    evidence_id = record_evidence(cfg)
    result = run_detection(cfg, no_report=True)
    assert result.policy_evidence_id == evidence_id
    assert _run_row(cfg, result.run_id)["policy_evidence_id"] == evidence_id


@pytest.mark.parametrize(
    ("verdict", "message"),
    [
        ("does_not_outperform", "does not outperform"),
        ("not_assessable", "had no labels"),
    ],
)
def test_evidence_against_the_policy_does_not_satisfy_the_gate(
    tmp_path: Path, verdict: str, message: str
) -> None:
    cfg = _cfg(tmp_path)
    record_evidence(cfg, verdict)
    with pytest.warns(UnevidencedPolicyWarning, match=message):
        result = run_detection(cfg, no_report=True)
    assert result.policy_evidence_id is None


def test_evidence_from_another_configuration_does_not_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    record_evidence(cfg, scope_hash="another-configuration")
    with pytest.warns(UnevidencedPolicyWarning, match="different configuration"):
        run_detection(cfg, no_report=True)


# ---------------------------------------------------------------------------
# The backtest that produces the evidence
# ---------------------------------------------------------------------------


def test_a_labelled_backtest_records_evidence_a_run_then_relies_on(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    result = backtest_dataset(cfg, policies=["composite-agreement@1"], replicates=6, n_resamples=2000)
    assert result.labelled
    (verdict,) = result.verdicts
    assert verdict.policy_id == "composite-agreement@1"
    assert verdict.evidence_id is not None
    assert {s.policy_id for s in result.summaries} == {"intersection@1", "composite-agreement@1"}
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        (row,) = ws.store.list_policy_evidence(result.dataset_fp)
    assert row["evidence_id"] == verdict.evidence_id
    assert row["verdict"] == verdict.verdict
    details = json.loads(str(row["details_json"]))
    assert len(details["per_replicate"]["composite-agreement@1"]) == 6

    if verdict.verdict == "outperforms":
        run = run_detection(cfg, no_report=True)
        assert run.policy_evidence_id == verdict.evidence_id
    else:  # the data decides; either way the gate must agree with the record
        with pytest.warns(UnevidencedPolicyWarning, match="does not outperform"):
            run_detection(cfg, no_report=True)


def test_an_unlabelled_backtest_reports_volume_and_stability_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, columns={})
    result = backtest_dataset(cfg, policies=["union@1"], replicates=5, record=False)
    assert not result.labelled
    assert result.verdicts[0].verdict == "not_assessable"
    assert result.verdicts[0].evidence_id is None
    assert result.notes
    union = next(s for s in result.summaries if s.policy_id == "union@1")
    assert union.over_flag_ratio > 0
    assert np.isnan(union.flag_f1)


def test_a_backtest_needs_a_numeric_review_budget(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, scoring={"contamination": "auto"})
    with pytest.raises(ConfigError, match="fixed review budget"):
        backtest_dataset(cfg, replicates=5)


def test_a_backtest_rejects_labels_that_are_not_binary(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_labelled_csv(csv)
    df = pl.read_csv(csv).with_columns(pl.col("label") * 2)
    df.write_csv(csv)
    with pytest.raises(SchemaError, match="only 0/1"):
        backtest_dataset(_cfg(tmp_path), replicates=5)


# ---------------------------------------------------------------------------
# sorethumb backtest
# ---------------------------------------------------------------------------


def _toml(tmp_path: Path, contamination: str = "0.05") -> Path:
    _write_labelled_csv(tmp_path / "data.csv")
    path = tmp_path / "sorethumb.toml"
    path.write_text(
        f"""[source]
uri = {json.dumps(str(tmp_path / "data.csv"))}
[run]
workdir = {json.dumps(str(tmp_path / "ws"))}
[columns]
reference_column = "label"
[scoring]
contamination = {contamination}
[[detectors]]
name = "isolation_forest"
[[detectors]]
name = "kmeans_distance"
""",
        encoding="utf-8",
    )
    return path


def test_backtest_cli_json_reports_and_records_each_candidate(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    toml = _toml(tmp_path)
    result = CliRunner().invoke(
        app, ["backtest", "--config", str(toml), "--replicates", "5", "--policy", "union@1", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    doc = json.loads(result.stdout)
    assert doc["labelled"] is True
    assert [v["policy_id"] for v in doc["verdicts"]] == ["union@1"]
    assert doc["verdicts"][0]["evidence_id"].startswith("ev_")


def test_backtest_cli_human_output_names_the_verdict(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    toml = _toml(tmp_path)
    result = CliRunner().invoke(
        app, ["backtest", "--config", str(toml), "--replicates", "5", "--policy", "union@1", "--no-record"]
    )
    assert result.exit_code == 0, result.stdout
    assert "Against intersection@1" in result.stdout
    assert "--no-record" in result.stdout
    with Workspace.open(tmp_path / "ws") as ws:
        assert ws.store.list_policy_evidence() == []


def test_backtest_cli_refuses_auto_contamination_as_preflight(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    toml = _toml(tmp_path, contamination='"auto"')
    result = CliRunner().invoke(app, ["backtest", "--config", str(toml), "--json"])
    assert result.exit_code == 2
    assert "fixed review budget" in json.loads(result.stdout)["error"]
