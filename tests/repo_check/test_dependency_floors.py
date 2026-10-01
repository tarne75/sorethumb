"""The declared dependency floors are actually exercised by the release gate.

Every other release-validation job installs from ``uv.lock`` (the newest resolvable
versions), so a lower bound in ``pyproject.toml`` can be wrong without anything
failing. These tests read the workflow and ``pyproject.toml`` (they do not run CI)
and pin that the ``lowest-direct`` job exists, is part of the gate, resolves every
direct runtime dependency at its floor on the minimum supported Python, and runs the
unit and contract lanes -- so the job cannot quietly turn into a no-op.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_JOB = "lowest-direct"


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _job() -> dict:
    workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "release-validation.yml").read_text(encoding="utf-8")
    )
    assert _JOB in workflow["jobs"], f"release-validation.yml has no {_JOB!r} job"
    return workflow["jobs"][_JOB]


def _runs(job: dict) -> str:
    return "\n".join(step["run"] for step in job["steps"] if "run" in step)


def _min_python() -> str:
    match = re.fullmatch(r">=\s*(\d+\.\d+)", _pyproject()["project"]["requires-python"].strip())
    assert match, "requires-python must be a simple '>=X.Y' bound for this check"
    return match.group(1)


def test_lowest_direct_job_resolves_at_the_declared_floors_on_the_minimum_python() -> None:
    job = _job()
    runs = _runs(job)
    assert "--resolution lowest-direct" in runs
    minimum = _min_python()
    assert f"--python-version {minimum}" in runs
    setup_uv = [s for s in job["steps"] if str(s.get("uses", "")).startswith("astral-sh/setup-uv")]
    assert setup_uv, "the job must set up uv"
    assert str(setup_uv[0].get("with", {}).get("python-version")) == minimum
    # Never the frozen lock: that is exactly the newest-versions install this job exists to avoid.
    assert "--frozen" not in runs
    assert "uv sync" not in runs


def test_lowest_direct_job_covers_every_runtime_extra() -> None:
    extras = set(_pyproject()["project"]["optional-dependencies"]) - {"dev"}
    assert extras, "expected runtime extras"
    runs = _runs(_job())
    for extra in sorted(extras):
        assert f"--extra {extra}" in runs, f"extra {extra!r} is not resolved at its floor"


def test_lowest_direct_job_runs_the_unit_and_contract_lanes_and_cannot_fail_open() -> None:
    job = _job()
    assert '-m "unit or contract"' in _runs(job)
    assert job.get("continue-on-error") is not True
    assert all(step.get("continue-on-error") is not True for step in job["steps"])


def test_test_tooling_is_installed_under_the_runtime_pins() -> None:
    # Dev tools must be constrained by the lowest-direct pins, not allowed to re-resolve them upward.
    assert re.search(r"-c\s+lowest-direct\.txt\b.*\.\[dev\]", _runs(_job()))


def test_every_direct_runtime_dependency_declares_a_floor() -> None:
    """``lowest-direct`` resolves to the *oldest release ever published* for an unbounded requirement."""
    project = _pyproject()["project"]
    requirements = list(project["dependencies"])
    for name, extra in project["optional-dependencies"].items():
        if name != "dev":
            requirements += extra
    unbounded = [r for r in requirements if ">=" not in r and "==" not in r and "~=" not in r]
    assert not unbounded, f"dependencies without a lower bound: {unbounded}"
