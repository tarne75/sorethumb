"""LICENSE is the verbatim Apache License 2.0, with only the appendix notice filled in.

``pyproject.toml`` declares ``license = "Apache-2.0"`` and ships ``LICENSE`` in both
distributions, so the file has to be the licence itself, not a paraphrase of it: an abridged
copy silently drops terms (the patent-termination clause, the NOTICE paragraph, the indemnity
sentence) while the package metadata claims the full grant, and GitHub's licence detection
stops recognising it.

The check undoes the one permitted edit (the appendix's ``Copyright [yyyy] [name of copyright
owner]`` placeholder), collapses all whitespace, and compares a SHA-256 against the canonical
text published at https://www.apache.org/licenses/LICENSE-2.0.txt. Collapsing whitespace keeps
the test indifferent to line endings and trailing blanks; any change to the words, including a
reflow that merges or splits a word, still fails it.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]

# SHA-256 of " ".join(canonical_text.split()), where canonical_text is the unmodified
# https://www.apache.org/licenses/LICENSE-2.0.txt (whose raw SHA-256 is
# cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30).
_CANONICAL_NORMALISED_SHA256 = "0ffddef9e48f8a09aed5caf2d44f7ba1c1be2d9b8e0a6f693b1635b2d5566645"
_PLACEHOLDER = "Copyright [yyyy] [name of copyright owner]"
_FILLED_NOTICE = re.compile(r"Copyright (?P<year>\d{4}) (?P<owner>[^\n]+?)[ \t]*$", re.MULTILINE)


def _license_text() -> str:
    path = _ROOT / "LICENSE"
    if not path.is_file():
        pytest.skip("LICENSE is not present")
    return path.read_text(encoding="utf-8")


def _normalised_sha256(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def test_license_has_exactly_one_filled_copyright_notice() -> None:
    text = _license_text()
    assert _PLACEHOLDER not in text, "the appendix copyright placeholder has not been filled in"
    notices = _FILLED_NOTICE.findall(text)
    assert len(notices) == 1, f"expected one copyright notice (in the appendix), found {len(notices)}"


def test_license_notice_sits_in_the_appendix() -> None:
    text = _license_text()
    match = _FILLED_NOTICE.search(text)
    assert match is not None
    assert match.start() > text.index("APPENDIX: How to apply the Apache License to your work.")


def test_license_notice_names_the_package_author() -> None:
    pyproject = _ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not present")
    author = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["authors"][0]["name"]
    match = _FILLED_NOTICE.search(_license_text())
    assert match is not None
    assert match["owner"] == author, f"LICENSE names {match['owner']!r}, pyproject names {author!r}"


def test_license_is_the_verbatim_apache_2_0_text() -> None:
    text = _license_text()
    restored = _FILLED_NOTICE.sub(_PLACEHOLDER, text, count=1)
    assert _normalised_sha256(restored) == _CANONICAL_NORMALISED_SHA256, (
        "LICENSE differs from the canonical Apache License 2.0 text "
        "(https://www.apache.org/licenses/LICENSE-2.0.txt) beyond the appendix copyright notice"
    )
