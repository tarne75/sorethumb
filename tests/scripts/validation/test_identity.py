"""Unit tests for scripts.validation.identity."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.validation.identity import code_revision, dependency_versions

pytestmark = pytest.mark.unit


def test_code_revision_returns_nonempty_string_in_this_repo():
    rev = code_revision(Path(__file__).resolve().parents[3])
    assert isinstance(rev, str)
    assert rev != ""


def test_code_revision_unknown_outside_a_git_checkout(tmp_path):
    rev = code_revision(tmp_path)
    assert rev == "unknown"


def test_dependency_versions_includes_sorethumb_and_numpy():
    deps = dependency_versions()
    assert "sorethumb-ml" in deps  # PyPI distribution name, not the "sorethumb" import name
    assert "numpy" in deps
    assert deps["sorethumb-ml"] != "unknown"


def test_dependency_versions_unknown_package_reports_unknown():
    from scripts.validation import identity as identity_mod

    original = identity_mod._TRACKED_PACKAGES
    identity_mod._TRACKED_PACKAGES = ("this-package-does-not-exist-anywhere",)
    try:
        deps = dependency_versions()
    finally:
        identity_mod._TRACKED_PACKAGES = original
    assert deps["this-package-does-not-exist-anywhere"] == "unknown"
