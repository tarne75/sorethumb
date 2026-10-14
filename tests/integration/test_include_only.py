"""columns.include_only end to end: only the listed columns (plus protected ones) are scored."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import ColumnsConfig, RunConfig, SourceConfig
from sorethumb_ml.store.models import load_plan
from sorethumb_ml.store.workspace import Workspace

pytestmark = pytest.mark.integration


def test_run_uses_only_the_included_columns(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    amount = rng.normal(50, 10, n)
    amount[:3] = 900.0
    noise = rng.normal(0, 1, n)
    csv = tmp_path / "data.csv"
    pl.DataFrame({"amount": amount, "noise": noise, "region": rng.choice(["us", "eu"], n)}).write_csv(csv)
    cfg = Config(
        source=SourceConfig(uri=str(csv)),
        run=RunConfig(workdir=str(tmp_path / "ws")),
        columns=ColumnsConfig(include_only=["amount"]),
    )

    result = run_detection(cfg, no_report=True)
    assert result.groups[0].status == "success"
    plan = load_plan(Workspace.open(tmp_path / "ws"), result.run_id)
    assert plan.output_features == ["amount"]
