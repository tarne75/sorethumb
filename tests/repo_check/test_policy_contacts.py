"""The security policy and the code of conduct name the same, attributed contact address.

Both files tell a reader to email a maintainer. If one address changes and the other is
forgotten, reports go to a mailbox nobody reads, so this pins three things: every address
either file asks people to write to is the same, that address appears in both, and each file
names the maintainer it belongs to (the package author in ``pyproject.toml``), rather than
leaving an unexplained address.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _read(name: str) -> str:
    path = _ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not present (for example an unpacked sdist)")
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_both_policy_files_use_one_contact_address() -> None:
    addresses = {name: set(_EMAIL.findall(_read(name))) for name in ("SECURITY.md", "CODE_OF_CONDUCT.md")}
    assert all(addresses.values()), f"a policy file names no contact address: {addresses}"
    assert addresses["SECURITY.md"] == addresses["CODE_OF_CONDUCT.md"], addresses
    assert len(addresses["SECURITY.md"]) == 1, addresses


def test_each_policy_file_names_the_maintainer_it_belongs_to() -> None:
    pyproject = _ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not present")
    author = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["authors"][0]["name"]
    for name in ("SECURITY.md", "CODE_OF_CONDUCT.md"):
        assert author in _flat(_read(name)), f"{name} does not name the maintainer {author!r}"
