"""`show --group` and `history --group` accept a group label (as `run --only-group` does) or a key."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import _resolve_group, app
from tests.factories.configs import make_config

pytestmark = pytest.mark.integration

runner = CliRunner()


def _grouped(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 240
    start = datetime(2024, 1, 15, 8, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "store": ["north" if i % 2 else "south" for i in range(n)],
            "ts": [start + timedelta(minutes=i) for i in range(n)],
            "amount": rng.normal(50, 10, n).tolist(),
            "qty": rng.normal(5, 1, n).tolist(),
        }
    )
    path = tmp_path / "grouped.parquet"
    df.write_parquet(path)
    cfg = make_config(
        path,
        tmp_path / "ws",
        source_format="parquet",
        combination="composite",
        contamination=0.05,
        group_by=["store"],
        time_column="ts",
        min_records=10,
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    result = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    return result, str(tmp_path / "ws")


def test_show_and_history_accept_a_label_or_a_key(tmp_path: Path) -> None:
    result, ws = _grouped(tmp_path)
    by_label = {g.group_label: g.group_key for g in result.groups}
    label = next(lbl for lbl in by_label if "north" in lbl)
    key = by_label[label]

    for selector in (label, key):
        res = runner.invoke(app, ["show", result.run_id, "--workdir", ws, "--group", selector, "--json"])
        assert res.exit_code == 0, res.output
        groups = json.loads(res.output)["groups"]
        assert [g["group_key"] for g in groups] == [key]

    hist = runner.invoke(
        app, ["history", "--workdir", ws, "--group", label, "--period", "2024-01-15", "--json"]
    )
    assert hist.exit_code == 0, hist.output
    counts = json.loads(hist.output)["windows"][0]["counts"]
    assert counts["current_population"] == 120  # one store's rows, not both


def test_an_unknown_group_is_not_found_and_lists_the_labels(tmp_path: Path) -> None:
    result, ws = _grouped(tmp_path)
    res = runner.invoke(app, ["show", result.run_id, "--workdir", ws, "--group", "west", "--json"])
    assert res.exit_code == 3
    error = json.loads(res.output)["error"]
    assert "west" in error
    assert all(g.group_label in error for g in result.groups)
    hist = runner.invoke(app, ["history", "--workdir", ws, "--group", "west", "--period", "2024-01-15"])
    assert hist.exit_code == 3


def test_resolve_group_cases() -> None:
    import typer

    known = {"k1": "store=a", "k2": "store=b", "k3": "store=b"}
    assert _resolve_group("k1", known, json_output=True) == "k1"
    assert _resolve_group("store=a", known, json_output=True) == "k1"
    with pytest.raises(typer.Exit) as ambiguous:
        _resolve_group("store=b", known, json_output=True)
    assert ambiguous.value.exit_code == 2
    with pytest.raises(typer.Exit) as none_recorded:
        _resolve_group("x", {}, json_output=True)
    assert none_recorded.value.exit_code == 3
