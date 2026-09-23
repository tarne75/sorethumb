"""Unit tests for cli.py's pure path-safety guards.

`_guard_reset_target` is tested directly (not through the full CLI) for the
cases that would be unsafe or impractical to exercise end-to-end -- there is
no safe way to point a real CLI invocation at "/" without first passing
through `_load_config`'s own filesystem side effects (it creates
`{workdir}/logs/` before `workspace_reset` ever reaches this guard). The
guard itself touches no destructive filesystem calls, so calling it in
isolation is safe; see tests/integration/test_cli.py for the
full-CLI-level `workspace reset` behavioural tests (valid reset, rejecting
a non-workspace directory, deletion failures).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from sorethumb.cli import _guard_reset_target

pytestmark = pytest.mark.unit


def test_guard_rejects_filesystem_root():
    with pytest.raises(typer.Exit):
        _guard_reset_target(Path("/"))


def test_guard_rejects_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(typer.Exit):
        _guard_reset_target(tmp_path.resolve())


def test_guard_rejects_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.Exit):
        _guard_reset_target(Path.cwd().resolve())


def test_guard_rejects_git_repository_root(tmp_path):
    repo = tmp_path / "some-project"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(typer.Exit):
        _guard_reset_target(repo.resolve())


def test_guard_rejects_shallow_path(tmp_path):
    # tmp_path itself is deep enough (pytest nests it many levels); build an
    # artificially shallow absolute path instead of relying on that depth.
    shallow = Path(tmp_path.anchor) / "x"
    with pytest.raises(typer.Exit):
        _guard_reset_target(shallow)


def test_guard_allows_a_plausible_nested_workspace_path(tmp_path):
    deep = tmp_path / "project" / "sorethumb-workspace"
    deep.mkdir(parents=True)
    _guard_reset_target(deep.resolve())  # must not raise
