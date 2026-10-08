"""The advertised platforms agree everywhere they are stated.

pyproject's classifiers (what PyPI shows), the README's Supported platforms
table and SECURITY.md must name the same operating systems, and none of them
may still describe Windows as untested.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFIER_TO_NAME = {
    "Operating System :: POSIX :: Linux": "Linux",
    "Operating System :: MacOS": "macOS",
    "Operating System :: Microsoft :: Windows": "Windows",
}


def _read(name: str) -> str:
    path = _ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not present (for example an unpacked sdist)")
    return path.read_text(encoding="utf-8")


def _advertised() -> set[str]:
    classifiers = tomllib.loads(_read("pyproject.toml"))["project"]["classifiers"]
    return {name for classifier, name in _CLASSIFIER_TO_NAME.items() if classifier in classifiers}


def test_classifiers_advertise_all_three() -> None:
    assert _advertised() == {"Linux", "macOS", "Windows"}


def test_readme_table_lists_every_advertised_os() -> None:
    section = _read("README.md").split("## Supported platforms", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith("| OS")]
    for name in _advertised():
        assert any(row.startswith(f"| {name}") for row in rows), name


@pytest.mark.parametrize("name", ["README.md", "SECURITY.md"])
def test_no_stale_windows_disclaimer(name: str) -> None:
    text = " ".join(_read(name).split()).lower()
    for stale in ("windows is not currently tested", "windows is untested", "linux and macos only"):
        assert stale not in text, f"{name}: {stale!r}"
