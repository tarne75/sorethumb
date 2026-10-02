"""The zero-config CLI flow works end to end with no sorethumb.toml.

``sorethumb run data.csv`` writes ./sorethumb-workspace/, and every follow-up
command it suggests (anomalies, runs, show, report, history, explain-plan
RUN_ID, workspace ls) must find that workspace without a config file, with or
without ``-w``. ``run`` must also finish under a closed stdin (cron, CI, a
pipe) instead of aborting at the "save settings?" prompt.

Commands run as real subprocesses with ``stdin=DEVNULL`` (the CLI's ``app()``
stands in for the installed console script), from a directory that holds no
config, so a non-TTY stdin is exactly what production sees.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.factories.frames import write_grouped_csv

pytestmark = pytest.mark.integration


def _cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "from sorethumb_ml.cli import app; app()", *args],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


@pytest.fixture(scope="module")
def zero_config_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One zero-config run, shared by the read-only follow-up tests below."""
    project = tmp_path_factory.mktemp("zero_config")
    write_grouped_csv(project / "data.csv", n_rows=300)
    proc = _cli(project, "run", "data.csv", "--no-report")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Aborted" not in proc.stdout + proc.stderr
    assert "sorethumb init" in proc.stderr  # says how to save a config instead of prompting
    assert not (project / "sorethumb.toml").exists()
    runs = _cli(project, "runs", "--json")
    assert runs.returncode == 0, runs.stderr
    return project, str(json.loads(runs.stdout)[0]["run_id"])


@pytest.mark.parametrize("workdir_args", [(), ("-w", "sorethumb-workspace")], ids=["default", "explicit-w"])
@pytest.mark.parametrize(
    "command",
    [
        ("anomalies", "--top", "3"),
        ("runs",),
        ("show", "{run_id}"),
        ("report",),
        ("history",),
        ("explain-plan", "{run_id}"),
        ("workspace", "ls"),
    ],
    ids=lambda c: c[0] if len(c) == 1 or c[0] != "workspace" else " ".join(c),
)
def test_read_only_commands_work_without_a_config(
    zero_config_project: tuple[Path, str], command: tuple[str, ...], workdir_args: tuple[str, ...]
) -> None:
    project, run_id = zero_config_project
    args = [a.format(run_id=run_id) for a in command]
    proc = _cli(project, *args, *workdir_args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Config file not found" not in proc.stdout + proc.stderr


def test_explicit_missing_config_is_still_exit_2(zero_config_project: tuple[Path, str]) -> None:
    project, _ = zero_config_project
    proc = _cli(project, "runs", "--config", "missing.toml")
    assert proc.returncode == 2
    assert "Config file not found" in proc.stdout + proc.stderr


def test_planning_current_data_still_needs_a_config(zero_config_project: tuple[Path, str]) -> None:
    project, _ = zero_config_project
    proc = _cli(project, "explain-plan")
    assert proc.returncode == 2
    assert "Config file not found" in proc.stdout + proc.stderr


def test_read_only_command_does_not_create_a_workspace(tmp_path: Path) -> None:
    proc = _cli(tmp_path, "runs")
    assert proc.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_read_only_command_keeps_the_legacy_dot_workspace_guard(tmp_path: Path) -> None:
    (tmp_path / "sorethumb.db").touch()
    proc = _cli(tmp_path, "runs")
    assert proc.returncode == 2
    assert "sorethumb-workspace" in proc.stdout + proc.stderr


def test_save_config_flag_saves_without_prompting(tmp_path: Path) -> None:
    write_grouped_csv(tmp_path / "data.csv", n_rows=300)
    proc = _cli(tmp_path, "run", "data.csv", "--no-report", "--save-config")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "sorethumb.toml").is_file()
    # The saved config is then picked up by a follow-up command.
    assert _cli(tmp_path, "runs").returncode == 0


def test_no_save_config_flag_skips_silently(tmp_path: Path) -> None:
    write_grouped_csv(tmp_path / "data.csv", n_rows=300)
    proc = _cli(tmp_path, "run", "data.csv", "--no-report", "--no-save-config")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (tmp_path / "sorethumb.toml").exists()
    assert "Not saving settings" not in proc.stderr
