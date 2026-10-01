"""Review documents stay out of the public repository.

Code reviews, audits and pre-release critiques are working material about the project, not
part of it. ``.gitignore`` keeps them from being committed by accident, and this check keeps
both halves honest: nothing review-like is tracked, and the ignore rules really match the
names such documents get. It skips outside a git checkout, such as an unpacked sdist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]

# Name + extension of a document that looks like a review, in any directory.
_REVIEW_DOCUMENT = re.compile(r"review[^/]*\.(md|docx|pdf)$", re.IGNORECASE)

# Names a review document plausibly gets; every one must be git-ignored.
_SAMPLE_REVIEW_PATHS = (
    "REVIEW.md",
    "review.md",
    "Code-Review.md",
    "code_review_2026-09.md",
    "docs/security-review.md",
    "docs/Pre-Release-Review.docx",
    "notes/REVIEW-FINDINGS.pdf",
    "reviews/anything.md",
    "reviews/data.json",
)

# Documents that must stay committable, so the rules are not wider than intended.
_MUST_NOT_BE_IGNORED = (
    "README.md",
    "CHANGELOG.md",
    "docs/stability.md",
    "tests/repo_check/test_no_review_documents.py",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("git") or "git", *args], cwd=_ROOT, capture_output=True, text=True, check=False
    )


def _tracked_files() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    result = _git("ls-files", "-z")
    if result.returncode != 0:
        pytest.skip("not a git checkout (for example an unpacked sdist)")
    return [name for name in result.stdout.split("\0") if name]


def test_no_review_document_is_tracked() -> None:
    offenders = [n for n in _tracked_files() if _REVIEW_DOCUMENT.search(n) or n.startswith("reviews/")]
    assert not offenders, f"review document(s) are tracked: {offenders}"


@pytest.mark.parametrize("path", _SAMPLE_REVIEW_PATHS)
def test_review_document_names_are_git_ignored(path: str) -> None:
    _tracked_files()  # skips outside a checkout
    assert _git("check-ignore", "-q", "--no-index", path).returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", _MUST_NOT_BE_IGNORED)
def test_the_ignore_rules_do_not_swallow_ordinary_documents(path: str) -> None:
    _tracked_files()
    assert _git("check-ignore", "-q", "--no-index", path).returncode == 1, f"{path} would be ignored"
