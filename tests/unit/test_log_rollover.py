"""A log rollover that can't rename the file keeps logging instead of failing.

On Windows, rotating ``logs/sorethumb.log`` renames an open file, which fails
while another sorethumb process has the same workspace log open. The stock
RotatingFileHandler then reports "--- Logging error ---" for every later
record and writes none of them.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

import pytest

from sorethumb_ml.cli import _add_file_handler, _detach_file_handlers, _WorkspaceLogHandler

pytestmark = pytest.mark.unit


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("sorethumb_ml.test", logging.INFO, __file__, 1, msg, None, None)


def test_a_failed_rename_keeps_writing_without_logging_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "sorethumb.log"
    handler = _WorkspaceLogHandler(log, maxBytes=200, backupCount=2, encoding="utf-8", delay=True)

    def _locked(*_args: object) -> None:
        raise PermissionError(errno.EACCES, "The process cannot access the file (simulated WinError 32)")

    monkeypatch.setattr(os, "rename", _locked)
    monkeypatch.setattr(os, "replace", _locked)
    try:
        for i in range(30):
            handler.emit(_record(f"record {i:02d} " + "x" * 40))
    finally:
        handler.close()
    text = log.read_text(encoding="utf-8")
    assert all(f"record {i:02d}" in text for i in range(30))
    assert "Logging error" not in capsys.readouterr().err


def test_rotation_still_happens_when_rename_works(tmp_path: Path) -> None:
    log = tmp_path / "sorethumb.log"
    handler = _WorkspaceLogHandler(log, maxBytes=200, backupCount=2, encoding="utf-8", delay=True)
    try:
        for i in range(30):
            handler.emit(_record(f"record {i:02d} " + "x" * 40))
    finally:
        handler.close()
    assert (tmp_path / "sorethumb.log.1").exists()


def test_the_log_file_is_not_opened_until_the_first_record(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "sorethumb.log"
    log.parent.mkdir()
    handler = _WorkspaceLogHandler(log, maxBytes=200, backupCount=2, encoding="utf-8", delay=True)
    try:
        assert not log.exists()
    finally:
        handler.close()


def test_a_log_that_cannot_be_opened_is_not_rolled_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = _WorkspaceLogHandler(
        tmp_path / "sorethumb.log", maxBytes=200, backupCount=2, encoding="utf-8", delay=True
    )

    def _locked() -> None:
        raise PermissionError(errno.EACCES, "The process cannot access the file (simulated WinError 32)")

    monkeypatch.setattr(handler, "_open", _locked)
    try:
        assert handler.shouldRollover(_record("x")) is False
    finally:
        handler.close()


def test_a_second_command_in_the_same_workspace_reuses_its_log_handler(tmp_path: Path) -> None:
    logger = logging.getLogger("sorethumb_ml")
    try:
        _add_file_handler(tmp_path, "INFO")
        first = [h for h in logger.handlers if isinstance(h, _WorkspaceLogHandler)]
        _add_file_handler(tmp_path, "DEBUG")
        second = [h for h in logger.handlers if isinstance(h, _WorkspaceLogHandler)]
    finally:
        _detach_file_handlers()
    assert len(first) == 1
    assert second == first
    assert second[0].level == logging.DEBUG
