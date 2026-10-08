"""``~`` in a CLI path argument is expanded by sorethumb, not only by the shell.

POSIX shells expand an unquoted ``~``; cmd.exe never does, and no shell does
inside quotes. ``sorethumb init ~/analysis`` in cmd.exe used to create a
directory literally named "~" in the current directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import app
from tests.factories.frames import write_grouped_csv

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # what expanduser reads on Windows
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    return home


def test_init_expands_a_tilde(home: Path) -> None:
    result = runner.invoke(app, ["init", "~/analysis"])
    assert result.exit_code == 0, result.output
    assert (home / "analysis" / "sorethumb.toml").is_file()
    assert not (Path.cwd() / "~").exists()


def test_workdir_and_config_expand_a_tilde(home: Path) -> None:
    write_grouped_csv(home / "data.csv", n_rows=200)
    (home / "sorethumb.toml").write_text(
        f"[source]\nuri = {json.dumps(str(home / 'data.csv'))}\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["run", "--config", "~/sorethumb.toml", "--workdir", "~/ws", "--no-report"])
    assert result.exit_code == 0, result.output
    assert (home / "ws" / "sorethumb.db").is_file()
    assert not (Path.cwd() / "~").exists()
    listed = runner.invoke(app, ["runs", "--workdir", "~/ws", "--json"])
    assert listed.exit_code == 0, listed.output
    assert len(json.loads(listed.stdout)) == 1
