"""Publication must be gated on a ranking-quality test for the exact commit.

``publish.yml`` and ``publish-testpypi.yml`` both call ``release-validation.yml``
and only publish if it passed, so a check belongs in the release gate exactly
when it is a job of ``release-validation.yml``. These tests read the workflow
files (they do not run CI) and pin three things: the fast synthetic accuracy
smoke is in the gate, it really is the cheap network-free one, and publishing
cannot start without the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_SMOKE_TEST = "tests/benchmark/test_accuracy_floors.py"


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _run_steps(job: dict) -> list[str]:
    return [step["run"] for step in job.get("steps", []) if "run" in step]


def test_release_validation_runs_the_synthetic_accuracy_smoke():
    jobs = _load("release-validation.yml")["jobs"]
    smoke_jobs = [name for name, job in jobs.items() if any(_SMOKE_TEST in r for r in _run_steps(job))]
    assert smoke_jobs, f"no job in release-validation.yml runs {_SMOKE_TEST}"
    for name in smoke_jobs:
        runs = _run_steps(jobs[name])
        pytest_runs = [r for r in runs if _SMOKE_TEST in r]
        # -m benchmark is what selects the (default-deselected) benchmark marker.
        assert all("-m benchmark" in r for r in pytest_runs), pytest_runs
        # No continue-on-error anywhere: a failure here must fail the gate.
        assert jobs[name].get("continue-on-error") is not True
        assert all(step.get("continue-on-error") is not True for step in jobs[name]["steps"])


def test_the_smoke_test_is_the_fast_network_free_one():
    path = _ROOT / _SMOKE_TEST
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    # Synthetic datasets only; nothing network-backed or slow in the release gate.
    assert '_SYNTHETIC_DATASETS = ("synthetic_gaussian", "synthetic_highd")' in text
    for forbidden in ("kddcup", "covtype", "pytest.mark.network", "pytest.mark.slow"):
        assert forbidden not in text, f"{forbidden!r} must not appear in the release-gate smoke test"
    # It is a floor-based (tolerance-banded) ranking-quality check, not a smoke import.
    assert "_ROC_AUC_FLOORS" in text
    assert "roc_auc >= floor" in text


@pytest.mark.parametrize("workflow", ["publish.yml", "publish-testpypi.yml"])
def test_publishing_cannot_start_without_the_release_validation_gate(workflow: str):
    jobs = _load(workflow)["jobs"]
    validate = jobs["validate"]
    assert validate["uses"] == "./.github/workflows/release-validation.yml"
    publishing = [name for name, job in jobs.items() if "environment" in job]
    assert publishing, f"{workflow} has no environment-protected publishing job"
    for name in publishing:
        needs = jobs[name].get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert "validate" in needs, f"{workflow}: job {name!r} can publish without the validate gate"


# ---------------------------------------------------------------------------
# The release-state commit (CHANGELOG date, SECURITY.md, README wording,
# version agreement) is enforced at tag time, locally and in CI.
# ---------------------------------------------------------------------------


def test_publish_workflow_verifies_release_state_before_validation_and_publishing():
    jobs = _load("publish.yml")["jobs"]
    verify = jobs["verify-release-state"]
    runs = [r for r in _run_steps(verify) if "scripts/check_release_state.py" in r]
    assert runs, "verify-release-state does not run scripts/check_release_state.py"
    assert all('--version "${GITHUB_REF_NAME#v}"' in r and "--tag-date" in r for r in runs), runs
    needs = jobs["validate"]["needs"]
    needs = [needs] if isinstance(needs, str) else needs
    assert "verify-release-state" in needs, (
        "validate (and so publishing) does not wait for the release-state check"
    )


def test_release_script_runs_the_same_release_state_checker_with_a_tag_date():
    text = (_ROOT / "scripts" / "release.sh").read_text(encoding="utf-8")
    assert "scripts/check_release_state.py" in text
    assert '--version "$VERSION" --tag-date' in text
