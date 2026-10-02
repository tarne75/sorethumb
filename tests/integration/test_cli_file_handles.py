"""Nothing the CLI opens outlives the command that opened it.

Windows refuses to delete, replace or rotate a file another handle still
holds. A log handler left attached after a command made ``workspace reset``
fail on ``logs/sorethumb.log``, kept in-process callers logging into the first
workspace they touched, and an unclosed SQLite connection from ``init`` pinned
``sorethumb.db``. These run on every platform; on Linux and macOS an open
handle doesn't block deletion, so the tests check the handles themselves.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import _WorkspaceLogHandler, app
from tests.factories.frames import write_grouped_csv

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture(autouse=True)
def _info_logging() -> Any:
    # Under pytest, logging.basicConfig in the CLI is a no-op (pytest already
    # installed root handlers), so INFO records would never reach the file.
    logger = logging.getLogger("sorethumb_ml")
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield
    logger.setLevel(previous)


def _workspace_handlers() -> list[_WorkspaceLogHandler]:
    return [h for h in logging.getLogger("sorethumb_ml").handlers if isinstance(h, _WorkspaceLogHandler)]


def _project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_grouped_csv(root / "data.csv", n_rows=200)
    toml = root / "sorethumb.toml"
    toml.write_text(
        f"[source]\nuri = {json.dumps(str(root / 'data.csv'))}\n\n[run]\nworkdir = {json.dumps(str(root / 'ws'))}\n",
        encoding="utf-8",
    )
    return toml


def _ok(*args: str) -> None:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output


def test_no_log_handler_survives_a_command(tmp_path: Path) -> None:
    _ok("run", "--config", str(_project(tmp_path / "a")), "--no-report")
    assert _workspace_handlers() == []


def test_a_second_workspace_gets_its_own_log(tmp_path: Path) -> None:
    toml_a, toml_b = _project(tmp_path / "a"), _project(tmp_path / "b")
    _ok("run", "--config", str(toml_a), "--no-report")
    log_a = tmp_path / "a" / "ws" / "logs" / "sorethumb.log"
    size_a = log_a.stat().st_size
    _ok("run", "--config", str(toml_b), "--no-report")
    assert log_a.stat().st_size == size_a, "the second workspace's run logged into the first one"
    assert (tmp_path / "b" / "ws" / "logs" / "sorethumb.log").stat().st_size > 0


def test_a_failing_command_still_closes_its_log(tmp_path: Path) -> None:
    toml = _project(tmp_path / "a")
    _ok("run", "--config", str(toml), "--no-report")
    result = runner.invoke(app, ["show", "no-such-run", "--config", str(toml)])
    assert result.exit_code != 0
    assert _workspace_handlers() == []


def test_reset_closes_the_log_before_deleting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = _project(tmp_path / "proj" / "deep")
    _ok("run", "--config", str(toml), "--no-report")
    held: list[str] = []
    real_rmtree = shutil.rmtree

    def _rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        held.extend(h.baseFilename for h in _workspace_handlers() if h.stream is not None)
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", _rmtree)
    _ok("workspace", "reset", "--config", str(toml), "--yes")
    assert held == [], f"log file still open while deleting: {held}"
    assert not (tmp_path / "proj" / "deep" / "ws").exists()


def test_init_closes_its_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[_TrackedConnection] = []
    real_connect = sqlite3.connect

    def _connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, factory=_TrackedConnection, **kwargs)
        opened.append(conn)  # type: ignore[arg-type]
        return conn

    monkeypatch.setattr("sorethumb_ml.store.db.sqlite3.connect", _connect)
    _ok("init", str(tmp_path / "proj"))
    assert opened, "init never opened the workspace database"
    assert all(c.closed for c in opened)


class _TrackedConnection(sqlite3.Connection):
    closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


def test_a_failed_store_open_closes_its_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb_ml.store import db

    opened: list[_TrackedConnection] = []
    real_connect = sqlite3.connect

    def _connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, factory=_TrackedConnection, **kwargs)
        opened.append(conn)  # type: ignore[arg-type]
        return conn

    def _boom(_self: db.Store) -> None:
        raise sqlite3.OperationalError("disk I/O error (simulated)")

    monkeypatch.setattr("sorethumb_ml.store.db.sqlite3.connect", _connect)
    monkeypatch.setattr(db.Store, "_init_schema_with_retry", _boom)
    with pytest.raises(sqlite3.OperationalError):
        db.Store(tmp_path / "sorethumb.db")
    assert len(opened) == 1
    assert opened[0].closed


def _lock_report_csvs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every report CSV behave as if a spreadsheet had it open (Windows)."""
    from sorethumb_ml import _atomic
    from sorethumb_ml.errors import FileInUseError

    real = _atomic.replace_with_retry

    def _replace(src: Path, dest: Path) -> None:
        if dest.suffix == ".csv" and dest.parent.parent.name == "reports":
            raise FileInUseError(f"Could not replace {dest}: the file is open in another program (simulated)")
        real(src, dest)

    monkeypatch.setattr(_atomic, "replace_with_retry", _replace)


def test_a_locked_report_file_is_named_in_the_run_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = _project(tmp_path / "a")
    _lock_report_csvs(monkeypatch)
    result = runner.invoke(app, ["run", "--config", str(toml)])
    assert result.exit_code == 4, result.output  # report failed, detection succeeded
    flat = " ".join(result.output.split())
    assert "open in another program" in flat
    assert ".csv:" in "".join(result.output.split())  # Rich may wrap a long path mid-name


def test_report_command_names_a_locked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = _project(tmp_path / "a")
    _ok("run", "--config", str(toml), "--no-report")
    _lock_report_csvs(monkeypatch)
    result = runner.invoke(app, ["report", "--config", str(toml)])
    assert result.exit_code == 1, result.output
    assert "open in another program" in " ".join(result.output.split())
