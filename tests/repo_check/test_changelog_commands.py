"""Every CLI command the latest release notes name actually exists.

The 0.1.0 notes once listed a ``benchmark`` command after it had moved to
``scripts/run_benchmark.py``. Release notes are what PyPI users read first, so
this checks the newest released section of CHANGELOG.md against the Typer app:

- every backticked ``sorethumb <command> [<subcommand>]`` names a real command
  (and, for a command group, a real subcommand);
- every command in the ``**CLI**:`` bullet exists, and so does every
  subcommand listed in parentheses after a group.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import typer

from sorethumb_ml.cli import app

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\][^\n]*$", re.MULTILINE)
_INLINE_INVOCATION = re.compile(r"`sorethumb ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?")
# `name` optionally followed by " (`sub`, `sub`, ...)" -- one item of the CLI bullet.
_CLI_ITEM = re.compile(r"`([a-z][a-z-]*)`(?:\s*\(([^)]*)\))?")


def _latest_release_section() -> str:
    path = _ROOT / "CHANGELOG.md"
    if not path.is_file():
        pytest.skip("CHANGELOG.md is not present")
    text = path.read_text(encoding="utf-8")
    match = _RELEASE_HEADING.search(text)
    assert match is not None, "CHANGELOG.md has no released '## [X.Y.Z]' section"
    nxt = re.search(r"^## \[", text[match.end() :], flags=re.MULTILINE)
    return text[match.end() : match.end() + nxt.start()] if nxt else text[match.end() :]


def _subcommands(command: Any) -> dict[str, Any] | None:
    """A group's {name: command} (Typer may or may not use click's own Group class), else None."""
    commands = getattr(command, "commands", None)
    return commands if isinstance(commands, dict) else None


def _check(command: str, subcommand: str | None, problems: list[str]) -> None:
    top = _subcommands(typer.main.get_command(app))
    assert top is not None, "the sorethumb app is not a command group"
    found = top.get(command)
    if found is None:
        problems.append(f"`sorethumb {command}` is not a command")
        return
    subs = _subcommands(found)
    if subcommand is not None and subs is not None and subcommand not in subs:
        problems.append(f"`sorethumb {command} {subcommand}` is not a subcommand")


def test_inline_invocations_name_real_commands() -> None:
    problems: list[str] = []
    for command, subcommand in _INLINE_INVOCATION.findall(_latest_release_section()):
        _check(command, subcommand or None, problems)
    assert not problems, problems


def test_cli_bullet_lists_only_real_commands() -> None:
    section = _latest_release_section()
    bullet = re.search(r"^- \*\*CLI\*\*:(.*?)(?=^- |^### |\Z)", section, flags=re.MULTILINE | re.DOTALL)
    assert bullet is not None, "the latest release section has no '- **CLI**:' bullet"
    # Only the command list: the first sentence, up to the first full stop.
    command_list = bullet.group(1).split(". ", 1)[0]
    items = _CLI_ITEM.findall(command_list)
    assert items, "found no commands in the **CLI** bullet"
    problems: list[str] = []
    for command, subcommands in items:
        _check(command, None, problems)
        for subcommand in re.findall(r"`([a-z][a-z-]*)`", subcommands):
            _check(command, subcommand, problems)
    assert not problems, problems
