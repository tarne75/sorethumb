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


def test_a_repeat_run_records_evidence_recorded_since_its_first_attempt(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.warns(UnevidencedPolicyWarning):
        first = run_detection(cfg, no_report=True)
    assert _run_row(cfg, first.run_id)["policy_evidence_id"] is None
    evidence_id = record_evidence(cfg)
    again = run_detection(cfg, no_report=True)  # same deterministic run_id; groups already complete
    assert again.run_id == first.run_id
    assert again.policy_evidence_id == evidence_id
    assert _run_row(cfg, first.run_id)["policy_evidence_id"] == evidence_id


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"replicates": 3}, "at least 5"), ({"subsample": 1.0}, "subsample"), ({"subsample": 0.1}, "subsample")],
)
def test_a_backtest_rejects_unusable_settings_before_loading_anything(
    tmp_path: Path, kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        backtest_dataset(_cfg(tmp_path), **kwargs)  # type: ignore[arg-type]


def test_a_backtest_needs_every_row_labelled_in_an_existing_column(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="not in the dataset"):
        backtest_dataset(_cfg(tmp_path, columns={"reference_column": "nope"}), replicates=5)
    csv = tmp_path / "data.csv"
    pl.read_csv(csv).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("label")).alias("label")
    ).write_csv(csv)
    with pytest.raises(SchemaError, match="missing value"):
        backtest_dataset(_cfg(tmp_path), replicates=5)


def test_a_grouped_backtest_fits_each_group_and_skips_small_ones_as_a_run_does(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_labelled_csv(csv)
    df = pl.read_csv(csv)
    shop = ["big"] * (len(df) - 6) + ["tiny"] * 6
    df.with_columns(pl.Series("shop", shop)).write_csv(csv)
    cfg = _cfg(
        tmp_path,
        columns={"reference_column": "label", "group_by": ["shop"]},
        scoring={"contamination": 0.05, "min_records": 20},
    )
    result = backtest_dataset(cfg, policies=["union@1"], replicates=5, record=False)
    as_json = json.loads(json.dumps(result.as_dict()))
    assert as_json["labelled"] is True
    union = next(s for s in result.summaries if s.policy_id == "union@1")
    assert union.flag_count > 0  # only the big group is fitted and flagged


# ---------------------------------------------------------------------------
# Seeing the evidence: sorethumb evidence, show, the run summary
# ---------------------------------------------------------------------------


def _toml_for(cfg: Config, tmp_path: Path) -> Path:
    path = tmp_path / "sorethumb.toml"
    path.write_text(
        f"""[source]
uri = {json.dumps(cfg.source.uri)}
format = "csv"
[run]
workdir = {json.dumps(cfg.run.workdir)}
seed = 0
[columns]
reference_column = "label"
[scoring]
contamination = 0.05
combination = "composite"
weighting = "agreement"
[explain]
enabled = false
[[detectors]]
name = "isolation_forest"
[[detectors]]
name = "kmeans_distance"
""",
        encoding="utf-8",
    )
    return path


def test_evidence_cli_lists_records_and_marks_the_current_scope(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    cfg = _cfg(tmp_path)
    toml = _toml_for(cfg, tmp_path)
    in_scope = record_evidence(cfg)
    record_evidence(cfg, "does_not_outperform", scope_hash="another-configuration")

    result = CliRunner().invoke(app, ["evidence", "--config", str(toml), "--json"])
    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    applies = {r["evidence_id"]: r["applies_to_current_config"] for r in rows}
    assert applies[in_scope] is True
    assert sorted(applies.values()) == [False, True]
    assert all("details_json" not in r for r in rows)

    human = CliRunner().invoke(app, ["evidence", "--config", str(toml), "--all"])
    assert human.exit_code == 0, human.stdout
    assert "Backtest evidence (2)" in human.stdout


def test_evidence_cli_says_when_nothing_is_recorded(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    cfg = _cfg(tmp_path)
    toml = _toml_for(cfg, tmp_path)
    Workspace.init(Path(cfg.run.workdir)).close()
    result = CliRunner().invoke(app, ["evidence", "--config", str(toml)])
    assert result.exit_code == 0, result.stdout
    assert "No backtest evidence" in result.stdout


def test_run_summary_and_show_name_the_evidence_or_its_absence(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    cfg = _cfg(tmp_path)
    toml = _toml_for(cfg, tmp_path)
    runner = CliRunner()

    bare = runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert bare.exit_code == 0, bare.stdout
    assert "No backtest evidence" in bare.stdout
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        run_id = str(ws.store.list_runs(limit=1)[0]["run_id"])
    shown = runner.invoke(app, ["show", run_id, "--config", str(toml)])
    assert "none recorded" in shown.stdout

    evidence_id = record_evidence(cfg)
    rerun = runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert evidence_id in rerun.stdout
    shown = runner.invoke(app, ["show", run_id, "--config", str(toml)])
    assert f"Evidence: {evidence_id}" in shown.stdout
    doc = json.loads(runner.invoke(app, ["show", run_id, "--config", str(toml), "--json"]).stdout)
    assert doc["policy_evidence"]["evidence_id"] == evidence_id


# ---------------------------------------------------------------------------
# Forward (out-of-time) backtests
# ---------------------------------------------------------------------------


def _write_timed_csv(path: Path, *, days: int = 7, per_day: int = 200, seed: int = 0) -> None:
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)
    n = days * per_day
    x = rng.normal(size=(n, 3))
    y = np.zeros(n, dtype=int)
    hit = rng.choice(n, n // 25, replace=False)
    x[hit] += 5.0
    y[hit] = 1
    minutes = np.sort(rng.integers(0, days * 24 * 60, n))
    ts = [datetime(2026, 1, 5) + timedelta(minutes=int(m)) for m in minutes]
    pl.DataFrame({"ts": ts, "a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "label": y}).write_csv(path)


def _timed_cfg(tmp_path: Path, **over: object) -> Config:
    csv = tmp_path / "timed.csv"
    if not csv.exists():
        _write_timed_csv(csv)
    raw: dict[str, object] = {
        "source": {"uri": str(csv), "format": "csv"},
        "run": {"workdir": str(tmp_path / "ws"), "seed": 0},
        "columns": {"reference_column": "label", "time_column": "ts"},
        "detectors": [{"name": "isolation_forest"}, {"name": "kmeans_distance"}],
        "scoring": {"contamination": 0.05, "combination": "composite", "weighting": "agreement"},
        "history": {"period_granularity": "day", "roll_non_business": False},
        "explain": {"enabled": False},
    }
    raw.update(over)
    return Config.model_validate(raw)


def test_period_frames_match_what_a_run_would_process_per_period(tmp_path: Path) -> None:
    from sorethumb_ml.evaluate.user_backtest import period_frames
    from sorethumb_ml.profiling.temporal import parse_temporal_columns

    cfg = _timed_cfg(tmp_path)
    df, _ = parse_temporal_columns(pl.read_csv(tmp_path / "timed.csv"), cfg.columns)
    frames = period_frames(df, cfg)
    assert [label for label, _ in frames] == [f"2026-01-{d:02d}" for d in range(5, 12)]
    assert sum(len(f) for _, f in frames) == len(df)


def test_a_forward_backtest_fits_each_period_and_scores_the_next(tmp_path: Path) -> None:
    cfg = _timed_cfg(tmp_path)
    result = backtest_dataset(
        cfg, policies=["composite-agreement@1"], mode="forward", record=True, n_resamples=2000
    )
    assert result.mode == "forward"
    assert result.periods == tuple(f"2026-01-{d:02d}" for d in range(6, 12))  # day 1 is only ever fitted
    assert result.replicates == 6
    assert len(result.per_replicate["composite-agreement@1"]) == 6
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        (row,) = ws.store.list_policy_evidence(result.dataset_fp)
    details = json.loads(str(row["details_json"]))
    assert details["mode"] == "forward"
    assert details["periods"] == list(result.periods)

    last5 = backtest_dataset(
        cfg, policies=["union@1"], mode="forward", folds=5, record=False, n_resamples=2000
    )
    assert last5.periods == result.periods[-5:]


@pytest.mark.parametrize(
    ("over", "kwargs", "error", "message"),
    [
        ({"columns": {"reference_column": "label"}}, {}, ConfigError, "time_column"),
        ({"history": {"period_granularity": "week"}}, {}, ConfigError, "at least 5 scorable"),
        ({}, {"folds": 3}, ConfigError, "folds must be at least 5"),
        ({}, {"mode": "sideways"}, ConfigError, "mode must be"),
    ],
)
def test_a_forward_backtest_refuses_what_it_cannot_measure(
    tmp_path: Path, over: dict[str, object], kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    cfg = _timed_cfg(tmp_path, **over)
    with pytest.raises(error, match=message):
        backtest_dataset(cfg, **{"mode": "forward", "record": False, **kwargs})  # type: ignore[arg-type]


def test_backtest_cli_forward_reports_the_scored_periods(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    cfg = _timed_cfg(tmp_path)
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f"""[source]
uri = {json.dumps(cfg.source.uri)}
format = "csv"
[run]
workdir = {json.dumps(cfg.run.workdir)}
[columns]
reference_column = "label"
time_column = "ts"
[scoring]
contamination = 0.05
[history]
period_granularity = "day"
roll_non_business = false
[[detectors]]
name = "isolation_forest"
[[detectors]]
name = "kmeans_distance"
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app, ["backtest", "--config", str(toml), "--forward", "--policy", "union@1", "--no-record", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    doc = json.loads(result.stdout)
    assert doc["mode"] == "forward"
    assert len(doc["periods"]) == 6
    assert "per_replicate" not in doc
    human = CliRunner().invoke(
        app, ["backtest", "--config", str(toml), "--forward", "--policy", "union@1", "--no-record"]
    )
    assert "periods scored forward" in human.stdout
