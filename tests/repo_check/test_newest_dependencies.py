"""The newest release of every dependency is tested, not only the locked versions.

``uv.lock`` moves only when someone re-locks, while a fresh ``pip install sorethumb-ml``
resolves the newest releases: polars 2.0.0 shipped and was what new installs got while
CI kept testing the locked 1.44.1. These tests read ``newest-deps.yml`` (they do not run
CI) and pin that the workflow re-resolves at the newest versions, runs on a schedule,
installs every extra, covers the oldest and newest supported Python, and runs the test
lanes and accuracy floors -- so it cannot quietly turn into a no-op or a locked install.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "newest-deps.yml"


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _job() -> dict:
    jobs = _workflow()["jobs"]
    assert list(jobs) == ["newest"], f"expected one 'newest' job, got {list(jobs)}"
    return jobs["newest"]


def _runs(job: dict) -> str:
    return "\n".join(step["run"] for step in job["steps"] if "run" in step)


def test_it_runs_on_a_schedule() -> None:
    # PyYAML reads the bare key `on` as True.
    triggers = _workflow()[True]
    assert triggers["schedule"], "the workflow must run on a schedule"
    assert "workflow_dispatch" in triggers


def test_it_resolves_every_dependency_at_its_newest_release_and_installs_every_extra() -> None:
    runs = _runs(_job())
    assert "uv lock --upgrade\n" in runs + "\n"
    assert "--upgrade-package" not in runs
    # The committed lock is exactly what this workflow exists not to test.
    assert "--frozen" not in runs
    assert "uv sync --all-extras --locked" in runs


def test_it_covers_the_oldest_and_newest_supported_python() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    match = re.fullmatch(r">=\s*(\d+\.\d+)\s*(?:,\s*<\s*\d+\.\d+)?", project["requires-python"].strip())
    assert match, "requires-python must be '>=X.Y' or '>=X.Y,<X.Z' for this check"
    classifiers = [
        c for c in project["classifiers"] if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    newest = max((c.rsplit(" ", 1)[1] for c in classifiers), key=lambda v: tuple(map(int, v.split("."))))
    pythons = {str(leg["python"]) for leg in _job()["strategy"]["matrix"]["include"]}
    assert {match.group(1), newest} <= pythons


def test_it_runs_the_test_lanes_and_the_accuracy_floors_and_cannot_fail_open() -> None:
    job = _job()
    runs = _runs(job)
    assert '-m "unit or contract or integration or property"' in runs
    assert "tests/benchmark/test_accuracy_floors.py" in runs
    assert "tests/benchmark/test_pipeline_accuracy_floors.py" in runs
    assert job.get("continue-on-error") is not True
    assert all(step.get("continue-on-error") is not True for step in job["steps"])
