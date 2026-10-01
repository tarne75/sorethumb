"""Tracked, public files must not cite ignored planning material.

``prompts/`` (and ``PLAN.md``) hold maintainer planning notes that are git-ignored,
never published, and absent from every clone, sdist and wheel. A comment, docstring or
document that says "see prompts/release-launch-plan.md" sends a reader to a file they
cannot open, so any rationale worth keeping is written inline instead. The same goes
for the item labels used in those notes ("P1-3" and the like). This check fails if a
tracked file reintroduces such a citation.

It reads the repository's tracked files (``git ls-files``) and skips when run outside a
git checkout, such as from an unpacked sdist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]

# Ignored planning material, as .gitignore patterns (checked below: each must really be ignored).
_PRIVATE_PATHS = ("prompts/", "PLAN.md")

# What a citation looks like: a path under an ignored directory or one of the ignored files,
# or the name of a known planning document with or without its directory.
_CITATION = re.compile(
    r"(?<![\w./-])prompts/"
    r"|(?<![\w./-])PLAN\.md\b"
    r"|\b(?:release-launch-plan|pre-release-plan)\.md\b"
    r"|\baction-list-\d{8}\.md\b"
    # Planning-item labels such as "P1-3": they index the ignored action list, so in a public
    # file they are an unexplained reference. Say what happened instead.
    r"|(?<![\w-])P[0-3]-\d+\b"
)

# Files that legitimately name these paths: the ignore rule itself, the sdist allowlist test
# that forbids shipping them, and this check.
_ALLOWED = frozenset(
    {
        ".gitignore",
        "tests/repo_check/test_dist_artifacts.py",
        "tests/repo_check/test_no_private_references.py",
    }
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("git") or "git", *args],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _tracked_files() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    result = _git("ls-files", "-z")
    if result.returncode != 0:
        pytest.skip("not a git checkout (for example an unpacked sdist)")
    return [name for name in result.stdout.split("\0") if name]


def test_the_private_planning_paths_are_really_git_ignored() -> None:
    """Keeps ``_PRIVATE_PATHS`` honest: a path listed here that stopped being ignored would
    start being committed, and the citation check below would be guarding nothing."""
    _tracked_files()  # skips outside a checkout
    for path in _PRIVATE_PATHS:
        probe = path + "x" if path.endswith("/") else path
        assert _git("check-ignore", "-q", probe).returncode == 0, f"{path} is not git-ignored"


def test_no_tracked_file_is_itself_planning_material() -> None:
    tracked = _tracked_files()
    offenders = [name for name in tracked if name.startswith("prompts/") or name == "PLAN.md"]
    assert not offenders, f"planning material is tracked: {offenders}"


def test_the_citation_pattern_matches_what_it_should() -> None:
    for text in (
        "see prompts/plan.md",
        "fixed (P1-3)",
        "P0-10 concerns",
        "before P3-5,",
        "action-list-20260930.md",
    ):
        assert _CITATION.search(text), text
    for text in ("UTF-8", "HTTP-2", "TLS1-2 ok", "TOP-3", "x-P1-3", "P4-1", "AP1-3"):
        assert not _CITATION.search(text), text


def test_tracked_files_do_not_cite_ignored_planning_material() -> None:
    offenders: list[str] = []
    for name in _tracked_files():
        if name in _ALLOWED:
            continue
        path = _ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable (a tracked symlink target, say)
        for number, line in enumerate(text.splitlines(), start=1):
            match = _CITATION.search(line)
            if match:
                offenders.append(f"{name}:{number}: {match.group(0)!r}")
    assert not offenders, (
        "tracked files cite git-ignored planning material that no reader can open; inline the "
        "rationale instead:\n  " + "\n  ".join(offenders)
    )
