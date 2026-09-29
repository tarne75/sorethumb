"""Validate the built sdist/wheel in ``dist/`` (P2-2).

Run after ``uv build`` -- locally, or in CI's ``release-validation.yml``
``build`` job. Every test here skips gracefully when ``dist/`` hasn't been
built, so a plain ``pytest -m repo_check`` (no prior build step) never fails
on this; it exists to be run deliberately, after a build, not as part of the
ambient repo-check suite.

Guards against exactly the two failure modes the wheel-only pipeline missed:
a source distribution that is either broken (missing something the wheel
needs) or over-inclusive (shipping repository-only material, dev-tool
caches, or local settings that were never meant to leave this machine --
``.claude/`` and ``.hypothesis/`` both did, silently, before
``[tool.hatch.build.targets.sdist]`` was given an explicit allowlist).
"""

from __future__ import annotations

import email
import tarfile
import zipfile
from email.message import Message
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "dist"

_REQUIRED_WHEEL_PATHS = frozenset(
    {
        "sorethumb/py.typed",
        "sorethumb/store/migrations/001_initial.sql",
    }
)
_REQUIRED_SDIST_PATHS = frozenset(
    {
        "src/sorethumb/py.typed",
        "src/sorethumb/store/migrations/001_initial.sql",
        "LICENSE",
        "README.md",
    }
)

# Anything under these prefixes, or whose name contains one of these
# substrings, has no business in a source distribution: it is either
# repository-only material (dev-tool config, CI workflows, planning notes),
# a local cache/settings directory, or a build/test artefact.
_FORBIDDEN_SDIST_PREFIXES = (
    ".claude/",
    ".hypothesis/",
    ".github/",
    ".git/",
    "prompts/",
    "scripts/",
    "benchmark_results/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "htmlcov/",
)
_FORBIDDEN_NAME_SUBSTRINGS = ("__pycache__", ".pyc", ".DS_Store", ".coverage")

_METADATA_FIELDS_MUST_MATCH = (
    "Name",
    "Version",
    "Summary",
    "License-Expression",
    "Requires-Python",
    "Author-email",
)


def _wheel_path() -> Path:
    matches = sorted(_DIST_DIR.glob("*.whl"))
    if not matches:
        pytest.skip("dist/*.whl not built -- run `uv build` first")
    return matches[0]


def _sdist_path() -> Path:
    matches = sorted(_DIST_DIR.glob("*.tar.gz"))
    if not matches:
        pytest.skip("dist/*.tar.gz not built -- run `uv build` first")
    return matches[0]


def _wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def _sdist_names(path: Path) -> list[str]:
    """Return sdist member paths with the wrapping "<name>-<version>/" dir stripped."""
    with tarfile.open(path) as tf:
        names = tf.getnames()
    stripped = []
    for name in names:
        _prefix, _sep, rest = name.partition("/")
        stripped.append(rest or name)
    return stripped


def test_wheel_contains_required_paths() -> None:
    names = set(_wheel_names(_wheel_path()))
    missing = _REQUIRED_WHEEL_PATHS - names
    assert not missing, f"wheel is missing required path(s): {sorted(missing)}"


def test_sdist_contains_required_paths() -> None:
    names = set(_sdist_names(_sdist_path()))
    missing = _REQUIRED_SDIST_PATHS - names
    assert not missing, f"sdist is missing required path(s): {sorted(missing)}"


def test_sdist_excludes_repository_only_material() -> None:
    names = _sdist_names(_sdist_path())
    offenders = [
        n
        for n in names
        if n.startswith(_FORBIDDEN_SDIST_PREFIXES) or any(sub in n for sub in _FORBIDDEN_NAME_SUBSTRINGS)
    ]
    assert not offenders, f"sdist contains unintended file(s): {offenders}"


def _read_wheel_metadata(path: Path) -> Message:
    with zipfile.ZipFile(path) as zf:
        metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(zf.read(metadata_name))


def _read_sdist_metadata(path: Path) -> Message:
    with tarfile.open(path) as tf:
        pkg_info = next(m for m in tf.getmembers() if m.name.endswith("/PKG-INFO") or m.name == "PKG-INFO")
        extracted = tf.extractfile(pkg_info)
        assert extracted is not None, "PKG-INFO member has no extractable content"
        return email.message_from_bytes(extracted.read())


def test_wheel_and_sdist_report_the_same_version_and_metadata() -> None:
    """The two artifacts a release publishes must describe the same package.

    A build-backend bug, a stale cached artifact, or a race between building
    the two could otherwise ship a wheel and sdist that silently disagree.
    """
    wheel_metadata = _read_wheel_metadata(_wheel_path())
    sdist_metadata = _read_sdist_metadata(_sdist_path())

    mismatches = {
        field: (wheel_metadata.get(field), sdist_metadata.get(field))
        for field in _METADATA_FIELDS_MUST_MATCH
        if wheel_metadata.get(field) != sdist_metadata.get(field)
    }
    assert not mismatches, f"wheel/sdist metadata mismatch: {mismatches}"
