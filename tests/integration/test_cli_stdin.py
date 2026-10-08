"""A prompt is only shown when someone can answer it.

``isatty()`` is True for Windows' NUL device, which is what ``< NUL``,
``subprocess.DEVNULL`` and Task Scheduler give a process; a confirm prompt
there reads EOF and aborts the run. Run as a real subprocess with
stdin=DEVNULL on every platform: that is the Windows failure mode.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_stdin_from_the_null_device_is_not_interactive() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "from sorethumb_ml.cli import _stdin_is_interactive as f; print(f())"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_a_closed_or_missing_stdin_is_not_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb_ml import cli

    class _Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(cli.sys, "stdin", _Closed())
    assert cli._stdin_is_interactive() is False
    monkeypatch.setattr(cli.sys, "stdin", None)
    assert cli._stdin_is_interactive() is False
