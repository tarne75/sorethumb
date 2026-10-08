"""`workspace reset` on Windows: junctions, read-only files, and typed paths.

A directory junction is Windows' everyday stand-in for a directory symlink but
is not ``is_symlink()``; shutil.rmtree refuses junctions, so reset used to stop
halfway. A file with the read-only attribute can't be deleted on Windows. And
the typed confirmation compared text exactly, so a different letter case,
forward slashes, or Explorer's quoted "Copy as path" all aborted.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import _confirmation_matches, _is_link_like, app
from tests.factories.frames import write_grouped_csv

pytestmark = pytest.mark.integration

runner = CliRunner()
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows path and junction semantics")


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    """(sorethumb.toml, workdir) after one successful run."""
    write_grouped_csv(tmp_path / "data.csv", n_rows=200)
    workdir = tmp_path / "deep" / "ws"
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f"[source]\nuri = {json.dumps(str(tmp_path / 'data.csv'))}\n\n[run]\nworkdir = {json.dumps(str(workdir))}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert result.exit_code == 0, result.output
    return toml, workdir


def test_a_read_only_file_is_deleted(project: tuple[Path, Path]) -> None:
    toml, workdir = project
    stuck = workdir / "results" / "read-only.txt"
    stuck.write_text("x", encoding="utf-8")
    stuck.chmod(stat.S_IREAD)
    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml), "--yes"])
    assert result.exit_code == 0, result.output
    assert not workdir.exists()


@windows_only
def test_a_junction_is_removed_without_touching_its_target(
    project: tuple[Path, Path], tmp_path: Path
) -> None:
    import _winapi  # type: ignore[import-not-found]  # Windows-only stdlib module

    toml, workdir = project
    elsewhere = tmp_path / "elsewhere-reports"
    elsewhere.mkdir()
    (elsewhere / "keep.html").write_text("keep", encoding="utf-8")
    reports = workdir / "reports"
    for child in reports.iterdir():
        child.unlink()
    reports.rmdir()
    _winapi.CreateJunction(str(elsewhere), str(reports))
    assert _is_link_like(reports)
    assert not reports.is_symlink()

    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml), "--yes"])

    assert result.exit_code == 0, result.output
    assert not workdir.exists()
    assert (elsewhere / "keep.html").read_text(encoding="utf-8") == "keep"


def test_typed_confirmation_accepts_how_people_type_the_path(project: tuple[Path, Path]) -> None:
    _, workdir = project
    ws = workdir.resolve()
    accepted = [str(ws), f"  {ws}  ", f'"{ws}"', f"'{ws}'", str(ws) + os.sep]
    if sys.platform == "win32":
        accepted += [str(ws).upper(), str(ws).lower(), ws.as_posix(), f'"{ws.as_posix()}"']
    for typed in accepted:
        assert _confirmation_matches(typed, ws), typed


def test_typed_confirmation_still_rejects_anything_else(project: tuple[Path, Path], tmp_path: Path) -> None:
    _, workdir = project
    ws = workdir.resolve()
    for typed in ["", '""', str(ws.parent), str(tmp_path / "other"), f'"{ws}', "y", "yes"]:
        assert not _confirmation_matches(typed, ws), typed


def test_typed_confirmation_drives_the_real_prompt(project: tuple[Path, Path]) -> None:
    toml, workdir = project
    result = runner.invoke(
        app, ["workspace", "reset", "--config", str(toml)], input=f'"{workdir.resolve()}"\n'
    )
    assert result.exit_code == 0, result.output
    assert not workdir.exists()


@windows_only
def test_guards_hold_for_windows_roots_and_home() -> None:
    import typer

    from sorethumb_ml.cli import _guard_reset_target

    for target in (Path("C:\\"), Path.home().resolve(), Path(Path.home().anchor)):
        with pytest.raises(typer.Exit):
            _guard_reset_target(target)


def test_typed_confirmation_rejects_text_that_is_not_a_path(project: tuple[Path, Path]) -> None:
    _, workdir = project
    assert not _confirmation_matches(f"{workdir.resolve()}\x00", workdir.resolve())
