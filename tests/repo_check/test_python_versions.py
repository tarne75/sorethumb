"""Every Python version the package admits is one CI tests, and no other.

pyproject's classifiers say which versions are supported, and ``requires-python`` says
which ones pip will install on. A version between them that CI never runs would ship
untested, so: the classifiers are exactly the release gate's fast-tests matrix,
``requires-python`` spans exactly the classifiers (its upper bound sits just above the
newest one, so a new CPython is admitted only once it is added and tested), and the
Windows wheel install runs on the oldest and the newest.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]


def _project() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _classified() -> list[tuple[int, int]]:
    versions = []
    for classifier in _project()["classifiers"]:
        match = re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", classifier)
        if match:
            versions.append((3, int(match.group(1))))
    assert versions, "pyproject lists no 'Programming Language :: Python :: 3.X' classifier"
    return sorted(versions)


def _text(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _release_validation() -> dict:
    yaml = pytest.importorskip("yaml")
    path = _ROOT / ".github" / "workflows" / "release-validation.yml"
    if not path.is_file():
        pytest.skip("workflows are not present (for example an unpacked sdist)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_classified_versions_are_contiguous() -> None:
    minors = [minor for _, minor in _classified()]
    assert minors == list(range(minors[0], minors[-1] + 1))


def test_requires_python_admits_exactly_the_classified_versions() -> None:
    classified = _classified()
    oldest, newest = classified[0], classified[-1]
    spec = _project()["requires-python"].replace(" ", "")
    assert spec == f">={_text(oldest)},<{newest[0]}.{newest[1] + 1}", (
        f"requires-python is {spec!r}; it must admit {_text(oldest)}-{_text(newest)} and nothing newer"
    )


def test_the_fast_tests_run_every_classified_version() -> None:
    matrix = _release_validation()["jobs"]["fast-tests"]["strategy"]["matrix"]
    assert {str(v) for v in matrix["python"]} == {_text(v) for v in _classified()}


def test_the_windows_wheel_install_runs_the_oldest_and_newest_version() -> None:
    smoke = _release_validation()["jobs"]["windows-install-smoke"]["strategy"]["matrix"]["include"]
    pythons = {str(entry["python"]) for entry in smoke}
    classified = _classified()
    assert {_text(classified[0]), _text(classified[-1])} <= pythons
