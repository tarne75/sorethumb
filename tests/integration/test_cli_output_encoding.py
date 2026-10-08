"""Redirected CLI output never crashes on characters outside the stream's encoding.

Real subprocesses with PYTHONIOENCODING forcing a narrow encoding on stdout and
stderr, the way a Windows pipe or file gets cp1252. A column named "温度" and a
category value "naïve→x" flow into inspect tables, reasons and summaries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.integration

_COMMANDS = (("inspect",), ("run", "--no-report"), ("anomalies", "--top", "5"), ("runs",), ("explain-plan",))


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("encoding")
    rng = np.random.default_rng(0)
    n = 400
    amount = rng.normal(50.0, 10.0, n)
    amount[:8] = 900.0
    pl.DataFrame(
        {
            "温度": rng.normal(20.0, 3.0, n),
            "amount": amount,
            "cat": np.array(["naïve→x", "b", "c", "d"])[np.arange(n) % 4],
        }
    ).write_csv(root / "data.csv")
    (root / "sorethumb.toml").write_text(
        f"[source]\nuri = {json.dumps(str(root / 'data.csv'))}\n\n[run]\nworkdir = {json.dumps(str(root / 'ws'))}\n",
        encoding="utf-8",
    )
    return root


def _cli(cwd: Path, encoding: str, *args: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONIOENCODING": encoding, "COLUMNS": "200"}
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, "-c", "from sorethumb_ml.cli import app; app()", *args],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
def test_commands_survive_a_narrow_output_encoding(project: Path, encoding: str) -> None:
    for args in _COMMANDS:
        proc = _cli(project, encoding, *args)
        stderr = proc.stderr.decode("utf-8", errors="replace")
        assert proc.returncode == 0, (args, stderr)
        assert "Traceback" not in stderr, (args, stderr)
        assert "UnicodeEncodeError" not in stderr, (args, stderr)


@pytest.mark.parametrize("args", [("anomalies", "--json"), ("explain-plan", "--json"), ("runs", "--json")])
def test_json_output_is_ascii_and_lossless_under_any_encoding(project: Path, args: tuple[str, ...]) -> None:
    _cli(project, "utf-8", "run", "--no-report")
    proc = _cli(project, "ascii", *args)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    text = proc.stdout.decode("ascii")  # raises if any non-ASCII byte slipped into --json output
    assert "?" not in text.replace("\\u", "")
    if args[0] != "runs":
        assert "温度" in json.dumps(json.loads(text), ensure_ascii=False)


def test_report_csv_bom_reaches_the_written_report(project: Path, tmp_path: Path) -> None:
    toml = project / "sorethumb.toml"
    bom_toml = tmp_path / "sorethumb.toml"
    bom_toml.write_text(
        toml.read_text(encoding="utf-8") + "\n[report]\ncsv_bom = true\n",
        encoding="utf-8",
    )
    bom_toml.write_text(
        bom_toml.read_text(encoding="utf-8").replace(
            json.dumps(str(project / "ws")), json.dumps(str(tmp_path / "ws"))
        ),
        encoding="utf-8",
    )
    proc = _cli(tmp_path, "utf-8", "run", "--config", str(bom_toml))
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    csvs = list((tmp_path / "ws" / "reports").rglob("*.csv"))
    assert csvs
    assert all(c.read_bytes().startswith(b"\xef\xbb\xbf") for c in csvs)
