"""Which third-party packages sorethumb actually needs, shown by running without them.

pyarrow and pydantic-settings used to be core dependencies. Nothing in the core
imports either, and polars' native readers, writers and numpy conversion do not use
pyarrow, so neither is a core dependency now. pyarrow is still needed by one thing:
polars' ``DataFrame.to_pandas()``, which the benchmark harness's sklearn-baseline
rows convert through, so it is declared in the ``benchmark`` extra.

Every CI lane installs ``--all-extras`` (which brings pyarrow back through the
``benchmark`` extra), so an installed-or-not check would prove nothing there. Each
test runs a fresh interpreter in which the packages are made unimportable before
anything else is imported. polars decides whether pyarrow is available from
``importlib.util.find_spec`` when it is first imported, and treats a
``ModuleNotFoundError`` there as "not installed", so this is what polars sees in an
environment without pyarrow.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]

_BLOCK = textwrap.dedent(
    """
    import importlib.abc
    import sys


    class _Absent(importlib.abc.MetaPathFinder):
        def __init__(self, names):
            self.names = names

        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in self.names:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None


    sys.meta_path.insert(0, _Absent({blocked!r}))
    """
)


def _run_without(blocked: set[str], body: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = _BLOCK.format(blocked=blocked) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_the_core_workflow_needs_neither_pyarrow_nor_pydantic_settings(tmp_path: Path) -> None:
    # Fit on CSV, score forward on Parquet, with explanations and every report
    # format: every reader, writer and conversion the core uses.
    csv = tmp_path / "train.csv"
    write_planted_csv(csv, seed=0)
    new_csv = tmp_path / "new.csv"
    write_planted_csv(new_csv, seed=1)

    proc = _run_without(
        {"pyarrow", "pydantic_settings"},
        """
        from pathlib import Path

        import polars as pl

        from sorethumb_ml import score_forward
        from sorethumb_ml._pipeline import run_detection
        from tests.factories.configs import make_config

        csv, new_csv, ws = (Path(a) for a in sys.argv[1:4])
        new_parquet = new_csv.with_suffix(".parquet")
        pl.read_csv(new_csv).write_parquet(new_parquet)
        reports = {"report_kwargs": {"formats": ["html", "csv", "json"]}}

        fitted = run_detection(make_config(csv, ws, **reports))
        assert fitted.n_succeeded == 1 and fitted.n_anomalies >= 1, fitted
        scored = score_forward(make_config(new_parquet, ws, source_format="parquet", **reports), fitted.run_id)
        assert scored.n_succeeded == 1 and scored.n_anomalies >= 1, scored
        for run_id in (fitted.run_id, scored.run_id):
            assert (ws / "reports" / run_id / "index.html").is_file(), run_id
        assert {"pyarrow", "pydantic_settings"}.isdisjoint(sys.modules)
        print("core ok")
        """,
        str(csv),
        str(new_csv),
        str(tmp_path / "ws"),
    )

    assert proc.returncode == 0, proc.stderr
    assert "core ok" in proc.stdout


def test_the_benchmark_baselines_need_pyarrow() -> None:
    # The sklearn-baseline rows one-hot encode through pandas, and polars'
    # to_pandas() cannot convert without pyarrow: hence pyarrow in the benchmark extra.
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    body = """
        import polars as pl

        from sorethumb_ml.evaluate.pipeline_benchmark import _encode_bare

        frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "colour": ["red", "blue", "red"]})
        try:
            train, holdout = _encode_bare(frame, frame)
        except ModuleNotFoundError as exc:
            print("missing:", exc.name)
        else:
            print("encoded:", train.shape)
        """

    with_pyarrow = _run_without(set(), body)
    without_pyarrow = _run_without({"pyarrow"}, body)

    assert with_pyarrow.returncode == 0, with_pyarrow.stderr
    assert "encoded: (3, 3)" in with_pyarrow.stdout
    assert without_pyarrow.returncode == 0, without_pyarrow.stderr
    assert "missing: pyarrow" in without_pyarrow.stdout
