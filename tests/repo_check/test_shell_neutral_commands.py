"""Commands we print or document work in cmd.exe and PowerShell, not only POSIX shells.

``pip install 'pkg[extra]'`` fails in cmd.exe, where single quotes are literal
characters, and an unquoted ``pkg[extra]`` is a glob in zsh. Double quotes work
everywhere, so every install hint and documented install command uses them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_SINGLE_QUOTED = re.compile(r"pip install '")
_UNQUOTED_EXTRA = re.compile(r"pip install (?!\")[^\s`'\"]*\[")


def _files() -> list[Path]:
    paths = [*sorted((_ROOT / "src").rglob("*.py")), *sorted((_ROOT / "docs").glob("*.md"))]
    paths += [p for p in (_ROOT / "README.md", _ROOT / "CONTRIBUTING.md") if p.is_file()]
    return paths


@pytest.mark.parametrize(
    "pattern", [_SINGLE_QUOTED, _UNQUOTED_EXTRA], ids=["single-quoted", "unquoted-extra"]
)
def test_install_commands_use_double_quotes(pattern: re.Pattern[str]) -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}:{n}: {line.strip()}"
        for path in _files()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, "use double quotes around package[extra]:\n" + "\n".join(offenders)
