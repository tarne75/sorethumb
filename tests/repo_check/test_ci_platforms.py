"""CI tests every supported OS; dropping one must be a deliberate, reviewed change.

pyproject's classifiers advertise Linux, macOS and Windows. The fast and
integration lanes in release-validation.yml -- what gates a release -- must run
on all three, and Windows must also install and run the built wheel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_SUPPORTED = {"ubuntu-latest", "macos-latest", "windows-latest"}


def _release_validation() -> dict:
    yaml = pytest.importorskip("yaml")
    path = _ROOT / ".github" / "workflows" / "release-validation.yml"
    if not path.is_file():
        pytest.skip("workflows are not present (for example an unpacked sdist)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("job", ["fast-tests", "integration"])
def test_release_gating_lanes_run_on_every_supported_os(job: str) -> None:
    matrix_os = set(_release_validation()["jobs"][job]["strategy"]["matrix"]["os"])
    assert matrix_os == _SUPPORTED


def test_the_built_wheel_is_installed_and_run_on_windows() -> None:
    smoke = _release_validation()["jobs"]["windows-install-smoke"]
    assert smoke["runs-on"] == "windows-latest"
    assert smoke["needs"] == "build"
    pythons = {entry["python"] for entry in smoke["strategy"]["matrix"]["include"]}
    assert {"3.11", "3.13"} <= pythons
