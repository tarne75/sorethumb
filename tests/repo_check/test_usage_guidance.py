"""The README keeps telling readers when this is, and is not, the right tool.

These are the statements a reader most needs before relying on a ranking, and the easiest to
lose in a README rewrite: it is a review aid and not a sole control, rates need labelled
validation, absolute comparison needs a fixed reference run, and a workspace must be trusted.
Each is pinned here, together with the places that repeat the trust warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]


def _text(name: str) -> str:
    path = _ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not present (for example an unpacked sdist)")
    return " ".join(path.read_text(encoding="utf-8").split())


def _section(text: str, title: str) -> str:
    marker = f"## {title} "
    assert marker in text, f"README has no '## {title}' section"
    return text.split(marker, 1)[1].split(" ## ", 1)[0]


def test_readme_says_when_it_is_the_right_tool() -> None:
    section = _section(_text("README.md"), "Is it the right tool?")
    assert "ranked shortlist" in section
    assert "Linux, macOS or Windows" in section
    assert "single machine" in section


def test_readme_says_not_to_rely_on_it_where_a_miss_is_unacceptable() -> None:
    section = _section(_text("README.md"), "Is it the right tool?")
    assert "missed anomaly is unacceptable" in section
    assert "labelled, supervised or domain-specific control" in section


def test_readme_requires_labelled_validation_and_denies_calibrated_scores() -> None:
    section = _section(_text("README.md"), "Is it the right tool?")
    assert "labelled data" in section
    assert "domain reviewer" in section
    assert "not calibrated risk probabilities" in section
    assert "not a prevalence estimate" in section


def test_readme_points_to_a_fixed_reference_run_for_absolute_comparison() -> None:
    section = _section(_text("README.md"), "Is it the right tool?")
    assert "score --from-run" in section
    assert "relative diagnostics only" in section
    for name in ("README.md", "docs/cli_reference.md", "docs/adapting-to-your-data.md"):
        assert "relative" in _text(name), f"{name} no longer calls independent-run trends relative"


def test_the_trust_warning_is_repeated_where_a_reader_would_look() -> None:
    assert "downloaded, shared or externally supplied workspace" in _section(
        _text("README.md"), "Is it the right tool?"
    )
    for name in ("README.md", "SECURITY.md", "docs/cli_reference.md"):
        assert "trust" in _text(name).lower(), f"{name} lost its workspace-trust warning"
    assert "Treat any workspace you did not create yourself as untrusted" in _text("SECURITY.md")
