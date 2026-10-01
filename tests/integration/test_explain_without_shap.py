"""Default-ensemble explanations name the right column on a core install (no shap).

Without the ``explain`` extra, IsolationForest falls back from TreeSHAP to a
finite-difference gradient, and One-Class SVM always uses one. Both scores go
flat for rows far outside the data, so the gradient there is zero in exactly
the column that made the row anomalous; before the saturating fallback (see
``explain/gradient.py``) and the negligible-row rule in ``explain/blend.py``,
about half of the rows planted with an extreme ``amount`` were explained as
``lat`` or ``region``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.config import ColumnsConfig, RunConfig, SourceConfig

pytestmark = pytest.mark.integration

_N_ROWS = 5_000
_N_PLANTED = 20


@pytest.fixture
def no_shap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import shap`` raise ImportError, as on ``pip install sorethumb-ml``."""
    monkeypatch.setitem(sys.modules, "shap", None)


def _write_planted(path: Path, planted: dict[str, tuple[range, float]]) -> None:
    rng = np.random.default_rng(0)
    columns = {
        "amount": rng.normal(50.0, 10.0, _N_ROWS),
        "lat": rng.normal(200.0, 30.0, _N_ROWS),
    }
    for column, (rows, value) in planted.items():
        columns[column][list(rows)] = value
    pl.DataFrame(
        {"rid": np.arange(_N_ROWS), **columns, "region": rng.choice(["us", "eu"], _N_ROWS)}
    ).write_csv(path)


def _reason_1_by_row(tmp_path: Path, planted: dict[str, tuple[range, float]]) -> dict[int, str]:
    """Run the default config on the planted data; map each flagged row to its reason_1."""
    csv = tmp_path / "planted.csv"
    _write_planted(csv, planted)
    config = Config(
        source=SourceConfig(uri=str(csv)),
        run=RunConfig(workdir=str(tmp_path / "ws")),
        columns=ColumnsConfig(id_column="rid"),
    )
    result = run_detection(config, no_report=True)
    results_path = result.groups[0].results_path
    assert results_path is not None
    frame = pl.read_parquet(results_path).filter(pl.col("flagged"))
    # Without shap no source is better than a heuristic, and the label must say so.
    assert set(frame["attribution_kind"].unique()) == {"heuristic"}
    return {
        int(row_id): str(reason) for row_id, reason in zip(frame["row_id"], frame["reason_1"], strict=True)
    }


def _assert_planted_column_first(reasons: dict[int, str], planted: dict[str, tuple[range, float]]) -> None:
    wrong = {}
    for column, (rows, _value) in planted.items():
        for row in rows:
            assert row in reasons, f"planted row {row} was not flagged"
            if not reasons[row].startswith(f"{column}="):
                wrong[row] = reasons[row]
    assert not wrong, f"reason_1 names the wrong column for planted rows: {wrong}"


@pytest.mark.usefixtures("no_shap")
def test_default_ensemble_names_the_planted_column_without_shap(tmp_path: Path) -> None:
    planted = {"amount": (range(_N_PLANTED), 900.0)}
    _assert_planted_column_first(_reason_1_by_row(tmp_path, planted), planted)


@pytest.mark.usefixtures("no_shap")
def test_default_ensemble_names_each_planted_column_without_shap(tmp_path: Path) -> None:
    planted = {
        "amount": (range(_N_PLANTED), 900.0),
        "lat": (range(_N_PLANTED, 2 * _N_PLANTED), 2_000.0),
    }
    _assert_planted_column_first(_reason_1_by_row(tmp_path, planted), planted)
