"""Every run records the versioned ensemble policy that produced its flags."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection, score_forward
from sorethumb_ml.cli import app
from sorethumb_ml.config import RunConfig, ScoringConfig, SourceConfig
from sorethumb_ml.store.workspace import Workspace

pytestmark = pytest.mark.integration


def _write(csv: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    a = rng.normal(50, 10, n)
    a[:4] = 900.0
    pl.DataFrame({"amount": a, "other": rng.normal(0, 1, n)}).write_csv(csv)


def _cfg(tmp_path: Path, **scoring_kw: object) -> Config:
    return Config(
        source=SourceConfig(uri=str(tmp_path / "d.csv")),
        run=RunConfig(workdir=str(tmp_path / "ws")),
        scoring=ScoringConfig(**scoring_kw),  # type: ignore[arg-type]
    )


def _stored_policy(tmp_path: Path, run_id: str) -> str | None:
    with Workspace.open(tmp_path / "ws") as ws:
        row = ws.store.get_run(run_id)
    assert row is not None
    return row["ensemble_policy"]


def test_a_default_run_records_the_default_policy(tmp_path: Path) -> None:
    _write(tmp_path / "d.csv")
    result = run_detection(_cfg(tmp_path), no_report=True)
    assert result.ensemble_policy == "intersection@1"
    assert _stored_policy(tmp_path, result.run_id) == "intersection@1"


@pytest.mark.parametrize(
    ("scoring_kw", "expected"),
    [
        ({"combination": "union"}, "union@1"),
        ({"combination": "composite", "weighting": "agreement"}, "composite-agreement@1"),
    ],
)
def test_an_overridden_policy_is_recorded_as_the_one_actually_used(
    tmp_path: Path, scoring_kw: dict[str, object], expected: str
) -> None:
    _write(tmp_path / "d.csv")
    result = run_detection(_cfg(tmp_path, **scoring_kw), no_report=True)
    assert result.ensemble_policy == expected
    assert _stored_policy(tmp_path, result.run_id) == expected


def test_dry_run_and_score_forward_record_their_policy(tmp_path: Path) -> None:
    _write(tmp_path / "d.csv")
    cfg = _cfg(tmp_path)
    dry = run_detection(cfg, dry_run=True)
    assert dry.ensemble_policy == "intersection@1"

    fitted = run_detection(cfg, no_report=True, force=True)
    _write(tmp_path / "d.csv", seed=1)
    forward = score_forward(cfg, source_run_id=fitted.run_id, no_report=True)
    assert forward.ensemble_policy == "intersection@1"
    assert _stored_policy(tmp_path, forward.run_id) == "intersection@1"


def test_run_json_and_show_report_the_policy(tmp_path: Path) -> None:
    _write(tmp_path / "d.csv")
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f'[source]\nuri = "{(tmp_path / "d.csv").as_posix()}"\n\n'
        f'[run]\nworkdir = "{(tmp_path / "ws").as_posix()}"\n\n'
        '[scoring]\ncombination = "composite"\n'
    )
    runner = CliRunner()
    ran = runner.invoke(app, ["run", "--config", str(toml), "--json", "--no-report"])
    assert ran.exit_code == 0, ran.stdout
    payload = json.loads(ran.stdout)
    assert payload["ensemble_policy"] == "composite-equal@1"

    shown = runner.invoke(app, ["show", payload["run_id"], "--config", str(toml)])
    assert shown.exit_code == 0, shown.stdout
    assert "composite-equal@1" in shown.stdout


def test_show_says_when_a_run_predates_policy_recording(tmp_path: Path) -> None:
    """A run row written before migration 010 has no policy; show says so, not 'None'."""
    ws_path = tmp_path / "ws"
    with Workspace.init(ws_path) as ws:
        ws.store.upsert_dataset("d", "x.csv", "schema", "content", 1, 1)
        ws.store.insert_run(run_id="run_legacy", dataset_fp="d", config_json="{}", seed=0)
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(f'[source]\nuri = "x.csv"\n\n[run]\nworkdir = "{ws_path.as_posix()}"\n')
    shown = CliRunner().invoke(app, ["show", "run_legacy", "--config", str(toml)])
    assert shown.exit_code == 0, shown.stdout
    assert "not recorded" in shown.stdout
