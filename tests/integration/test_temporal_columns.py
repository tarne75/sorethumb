"""End-to-end: date/time stored as text, and Duration, go through the real pipeline.

Covers what profiling/temporal.py unit tests cannot: that the parsed dtype
actually reaches period-window filtering, that every temporal column (not just
one "chosen" one) gets derive_time features, that score-forward replays the
same parse on a fresh raw (still-text) load, and that a period window
selecting zero rows is diagnosed clearly rather than failing deep in feature
building.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection, score_forward
from sorethumb_ml.config import ColumnsConfig, RunConfig, SourceConfig
from sorethumb_ml.errors import AmbiguousDateFormatWarning, SchemaError
from sorethumb_ml.store.models import load_plan
from sorethumb_ml.store.workspace import Workspace

pytestmark = pytest.mark.integration


def _cfg(tmp_path: Path, path: Path, **columns_kw: object) -> Config:
    return Config(
        source=SourceConfig(uri=str(path)),
        run=RunConfig(workdir=str(tmp_path / "ws")),
        columns=ColumnsConfig(**columns_kw),  # type: ignore[arg-type]
    )


def _write_mixed_temporal_csv(path: Path, n: int = 300, seed: int = 0) -> list[datetime]:
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ts = [base + timedelta(hours=int(h)) for h in rng.integers(0, 24 * 90, n)]
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    pl.DataFrame(
        {
            "amount": amount,
            "iso_date": [t.strftime("%Y-%m-%d") for t in ts],
            "iso_datetime": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in ts],
            "day_month_year": [t.strftime("%d/%m/%Y") for t in ts],
        }
    ).write_csv(path)
    return ts


def test_every_temporal_column_gets_derive_time_features_not_only_the_chosen_one(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "data.csv"
    _write_mixed_temporal_csv(csv)
    cfg = _cfg(tmp_path, csv)

    result = run_detection(cfg, no_report=True)
    assert result.groups[0].status == "success"

    with Workspace.open(tmp_path / "ws") as ws:
        plan = load_plan(ws, result.run_id)

    assert set(plan.parsed_temporal) == {"iso_date", "iso_datetime", "day_month_year"}
    for col in ("iso_date", "iso_datetime", "day_month_year"):
        assert any(f.startswith(f"{col}__") for f in plan.output_features), (
            f"{col} produced no derive_time feature: {plan.output_features}"
        )
    # Only one column drives sort/period alignment, but every temporal column is still derived.
    assert plan.chosen_time_column in plan.parsed_temporal


def test_configured_time_column_as_text_is_parsed_before_period_filtering(tmp_path: Path) -> None:
    """Regression: a day/month/year time_column compared *lexically* against an ISO
    period window would never match real data -- the window must compare real dates."""
    csv = tmp_path / "data.csv"
    rng = np.random.default_rng(0)
    n = 200
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    # Every row falls on 2026-01-15: the period window must select all of them, which it
    # can only do by comparing real dates -- compared lexically, "15/01/2026" never
    # matches the ISO window ["2026-01-15", "2026-01-16") at all.
    pl.DataFrame({"amount": amount, "day_month_year": ["15/01/2026"] * n}).write_csv(csv)
    cfg = _cfg(tmp_path, csv, time_column="day_month_year")

    result = run_detection(cfg, no_report=True, period_label_override="2026-01-15")
    assert result.groups[0].status == "success"
    assert result.period_label == "2026-01-15"


def test_period_window_with_zero_rows_names_the_time_column_and_available_range(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "data.csv"
    _write_mixed_temporal_csv(csv)  # data spans roughly 2026-01-01 .. 2026-03-31
    cfg = _cfg(tmp_path, csv, time_column="iso_date")

    with pytest.raises(SchemaError, match="selected 0 of") as exc_info:
        run_detection(cfg, no_report=True, period_label_override="2030-01-01")
    assert "iso_date" in str(exc_info.value)
    assert "2026" in str(exc_info.value)  # the available range is named


def test_duration_column_is_converted_to_a_numeric_feature(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    df = pl.DataFrame(
        {
            "amount": amount,
            "wait": [timedelta(minutes=int(m)) for m in rng.integers(1, 500, n)],
        }
    )
    parquet = tmp_path / "data.parquet"
    df.write_parquet(parquet)
    cfg = _cfg(tmp_path, parquet)

    result = run_detection(cfg, no_report=True)
    assert result.groups[0].status == "success"

    with Workspace.open(tmp_path / "ws") as ws:
        plan = load_plan(ws, result.run_id)
    assert "wait" in plan.output_features  # a single numeric feature, not dropped
    decision = next(d for d in plan.decisions if d.column == "wait")
    assert decision.col_class.value == "numeric"


def test_score_forward_reparses_a_fresh_raw_text_load(tmp_path: Path) -> None:
    """score-forward must replay the training run's own parse on new, still-raw-text data."""
    csv = tmp_path / "data.csv"
    _write_mixed_temporal_csv(csv, seed=1)
    cfg = _cfg(tmp_path, csv)
    fit_result = run_detection(cfg, no_report=True)
    assert fit_result.groups[0].status == "success"

    # Fresh data, still raw text on disk, same schema.
    _write_mixed_temporal_csv(csv, seed=2)
    forward_result = score_forward(cfg, source_run_id=fit_result.run_id, no_report=True)
    assert forward_result.groups[0].status == "success"
    assert forward_result.groups[0].n_anomalies >= 0  # ran to completion, didn't crash


def test_ambiguous_date_order_is_left_as_categorical_text_end_to_end(tmp_path: Path) -> None:
    """A column too ambiguous to parse is simply frequency/one-hot encoded, not refused."""
    rng = np.random.default_rng(0)
    n = 300
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    ambiguous = [f"{rng.integers(1, 13):02d}/{rng.integers(1, 13):02d}/2026" for _ in range(n)]
    df = pl.DataFrame({"amount": amount, "when": ambiguous})
    csv = tmp_path / "data.csv"
    df.write_csv(csv)
    cfg = _cfg(tmp_path, csv)

    with pytest.warns(AmbiguousDateFormatWarning, match="when"):
        result = run_detection(cfg, no_report=True)
    assert result.groups[0].status == "success"

    with Workspace.open(tmp_path / "ws") as ws:
        plan = load_plan(ws, result.run_id)
    assert "when" not in plan.parsed_temporal
    decision = next(d for d in plan.decisions if d.column == "when")
    assert decision.col_class.value == "categorical"


def test_grouping_by_a_text_date_column_keeps_it_as_text(tmp_path: Path) -> None:
    """Regression: a text-date group_by column was recast to Date, changing every group key
    and orphaning earlier runs' history and models for those groups."""
    rng = np.random.default_rng(0)
    n = 400
    amount = rng.normal(50, 10, n)
    amount[:4] = 900.0
    csv = tmp_path / "data.csv"
    pl.DataFrame({"amount": amount, "day": rng.choice(["2026-01-01", "2026-01-02"], n)}).write_csv(csv)
    cfg = _cfg(tmp_path, csv, group_by=["day"])
    cfg = cfg.model_copy(update={"scoring": cfg.scoring.model_copy(update={"min_records": 50})})

    result = run_detection(cfg, no_report=True)
    assert sorted(g.group_label for g in result.groups) == ["2026-01-01", "2026-01-02"]
    with Workspace.open(tmp_path / "ws") as ws:
        plan = load_plan(ws, result.run_id)
    assert "day" not in plan.parsed_temporal


def test_strict_run_with_an_ambiguous_text_date_still_succeeds(tmp_path: Path) -> None:
    """Regression: run.strict promoted the ambiguity warning and failed a run that, before
    temporal parsing existed, succeeded with the column as text -- as it still is."""
    rng = np.random.default_rng(0)
    n = 300
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    ambiguous = [f"{rng.integers(1, 13):02d}/{rng.integers(1, 13):02d}/2026" for _ in range(n)]
    csv = tmp_path / "data.csv"
    pl.DataFrame({"amount": amount, "when": ambiguous}).write_csv(csv)
    cfg = _cfg(tmp_path, csv)
    cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"strict": True})})

    with pytest.warns(AmbiguousDateFormatWarning):
        result = run_detection(cfg, no_report=True)
    assert result.groups[0].status == "success"


def test_explain_plan_parses_text_dates_like_run_does(tmp_path: Path) -> None:
    """Regression: inspect / explain-plan built their plan from the unparsed frame, so a text
    date showed as categorical there while `run` derived temporal features from it."""
    import json

    from typer.testing import CliRunner

    from sorethumb_ml.cli import app

    csv = tmp_path / "data.csv"
    _write_mixed_temporal_csv(csv)
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f'[source]\nuri = "{csv.as_posix()}"\n\n[run]\nworkdir = "{(tmp_path / "ws").as_posix()}"\n'
    )
    result = CliRunner().invoke(app, ["explain-plan", "--config", str(toml), "--json"])
    assert result.exit_code == 0, result.stdout
    features = json.loads(result.stdout)["output_features"]
    assert any(f.startswith("iso_date__") for f in features), features

    shown = CliRunner().invoke(app, ["inspect", "--config", str(toml)])
    assert shown.exit_code == 0, shown.stdout
    assert "temporal" in shown.stdout
