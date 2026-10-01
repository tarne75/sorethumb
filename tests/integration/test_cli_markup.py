"""Data-derived text is printed literally, never parsed as Rich markup.

Column names, group labels, category values, reasons, paths and warning text
all come from the user's data or environment. A column named "[/x]" used to
crash ``inspect`` and ``anomalies`` with MarkupError, "amt [usd]" lost its
"[usd]", and the run summary's shap hint printed ``pip install 'sorethumb-ml'``
with the ``[explain]`` swallowed as a tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import app

pytestmark = pytest.mark.integration

# Wide enough that Rich never folds a cell mid-string.
runner = CliRunner(env={"COLUMNS": "400", "NO_COLOR": "1"})

_ODD_COLUMNS = ("[/x]", "amt [usd]", "[bold]")
_ODD_GROUPS = ("[g1]", "[/g2]")
_ODD_CATEGORY = "[red]a"


def _write_odd_project(root: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 240
    frame = pl.DataFrame(
        {
            "id": np.arange(n),
            "grp": np.where(np.arange(n) % 2 == 0, *_ODD_GROUPS),
            _ODD_COLUMNS[0]: rng.normal(0.0, 1.0, n),
            _ODD_COLUMNS[1]: rng.normal(50.0, 5.0, n),
            # Cycles through four values (three would be dropped as near-constant),
            # so the planted rows (0-3, see below) include the odd one.
            _ODD_COLUMNS[2]: np.array([_ODD_CATEGORY, "[b]", "[/i]", "plain"])[np.arange(n) % 4],
        }
    )
    frame = frame.with_columns(
        pl.when(pl.col("id") < 4)
        .then(pl.lit(900.0))
        .otherwise(pl.col(_ODD_COLUMNS[1]))
        .alias(_ODD_COLUMNS[1])
    )
    csv = root / "odd.csv"
    frame.write_csv(csv)
    toml = root / "sorethumb.toml"
    toml.write_text(
        f"""\
[source]
uri = {json.dumps(str(csv))}

[run]
workdir = {json.dumps(str(root / "ws"))}

[columns]
id_column = "id"
group_by = ["grp"]

[scoring]
contamination = 0.05

[explain]
top_n = 4
""",
        encoding="utf-8",
    )
    return toml


def _invoke(*args: str) -> str:
    result = runner.invoke(app, list(args))
    output = result.stdout + (result.stderr or "")
    assert result.exception is None or isinstance(result.exception, SystemExit), output
    assert result.exit_code == 0, output
    return output


def test_bracketed_names_and_values_print_literally(tmp_path: Path) -> None:
    toml = str(_write_odd_project(tmp_path))

    inspected = _invoke("inspect", "--config", toml)
    for column in _ODD_COLUMNS:
        assert column in inspected

    ran = _invoke("run", "--config", toml, "--no-report")
    for label in _ODD_GROUPS:
        assert label in ran

    listed = _invoke("anomalies", "--config", toml)
    assert "amt [usd]=900" in listed
    for label in _ODD_GROUPS:
        assert label in listed

    runs = runner.invoke(app, ["runs", "--config", toml, "--json"])
    run_id = json.loads(runs.stdout)[0]["run_id"]
    shown = _invoke("show", run_id, "--config", toml)
    for label in _ODD_GROUPS:
        assert label in shown
    assert run_id in _invoke("runs", "--config", toml)


def test_category_value_with_markup_prints_literally(tmp_path: Path) -> None:
    toml = str(_write_odd_project(tmp_path))
    _invoke("run", "--config", toml, "--no-report")
    # explain.top_n = 4 covers every column, so each flagged row lists its [bold] value.
    listed = _invoke("anomalies", "--config", toml, "--reasons", "4")
    assert f"[bold]={_ODD_CATEGORY}" in listed


def test_run_summary_shap_hint_keeps_the_extra_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "shap", None)
    toml = str(_write_odd_project(tmp_path))
    ran = _invoke("run", "--config", toml, "--no-report")
    assert "sorethumb-ml[explain]" in ran
