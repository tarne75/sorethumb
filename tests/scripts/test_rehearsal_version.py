"""scripts/rehearsal_version.py: a TestPyPI rehearsal publishes a new pre-release of the version."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import rehearsal_version as rv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["0.1.0rc1", "0.1.0rc12", "0.1.0a1", "0.1.0b3"])
def test_a_pre_release_of_the_version_is_accepted(version: str) -> None:
    assert rv.problem(version, "0.1.0") is None


@pytest.mark.parametrize(
    ("version", "why"),
    [
        ("0.1.0", "not a pre-release"),  # the real release's version: never on a rehearsal
        ("0.1.0.dev1", "not a pre-release"),
        ("0.1.0-rc1", "not a pre-release"),
        ("v0.1.0rc1", "not a pre-release"),
        ("0.2.0rc1", "not a pre-release of 0.1.0"),
    ],
)
def test_anything_else_is_refused(version: str, why: str) -> None:
    reason = rv.problem(version, "0.1.0")
    assert reason is not None
    assert why in reason


def _pyproject(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n[tool.other]\nversion = "9"\n', encoding="utf-8"
    )
    return path


def test_apply_rewrites_only_the_project_version(tmp_path: Path) -> None:
    path = _pyproject(tmp_path)
    assert rv.main(["0.1.0rc2", "--apply", "--pyproject", str(path)]) == 0
    text = path.read_text(encoding="utf-8")
    assert 'version = "0.1.0rc2"' in text
    assert 'version = "9"' in text  # another table's key is untouched


def test_a_refused_version_changes_nothing(tmp_path: Path) -> None:
    path = _pyproject(tmp_path)
    before = path.read_text(encoding="utf-8")
    assert rv.main(["0.1.0", "--apply", "--pyproject", str(path)]) == 2
    assert path.read_text(encoding="utf-8") == before
