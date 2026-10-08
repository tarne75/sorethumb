"""Text files are LF on every checkout, so Windows checkouts match byte for byte.

Git for Windows defaults to core.autocrlf=true. Without .gitattributes a
Windows checkout gets CRLF files: harmless where text is read in text mode
(migration checksums and golden comparisons are), but byte-level comparisons,
hashes of tracked files and an sdist built from that checkout all change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_CHECKED_TREES = ("src", "tests/golden", "docs", "scripts")


def test_gitattributes_forces_lf_for_text() -> None:
    attributes = _ROOT / ".gitattributes"
    if not attributes.is_file():
        pytest.skip(".gitattributes is not present (for example an unpacked sdist)")
    rules = [line.split() for line in attributes.read_text(encoding="utf-8").splitlines()]
    assert ["*", "text=auto", "eol=lf"] in rules
    assert ["*.parquet", "binary"] in rules


def test_tracked_text_files_have_no_crlf() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "--eol", "--", *_CHECKED_TREES],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip("not a git checkout")
    # Columns: index eol, working-tree eol, attributes, path. Check both the
    # committed blob and this checkout.
    crlf = [line for line in proc.stdout.splitlines() if "crlf" in line.split("\t", 1)[0]]
    assert not crlf, "CRLF line endings in tracked text files:\n" + "\n".join(crlf)
