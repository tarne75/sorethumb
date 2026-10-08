"""Replacing or deleting a file another program holds open (Windows).

Windows refuses ``os.replace``/``unlink`` on a file open elsewhere without
delete sharing (WinError 5/32/33): a report CSV open in a spreadsheet,
antivirus scanning a file just written. Momentary locks are retried; a
persistent one becomes a FileInUseError naming the file. POSIX never retries.
These tests simulate Windows (sys.platform and the error's ``winerror``) so they
run on every platform; the real-Windows check is the Windows CI lane.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from sorethumb_ml import _atomic
from sorethumb_ml._atomic import atomic_write_bytes, replace_with_retry, unlink_with_retry
from sorethumb_ml.errors import FileInUseError, StoreError

pytestmark = pytest.mark.unit


def _in_use(winerror: int = 32) -> PermissionError:
    exc = PermissionError(errno.EACCES, "The process cannot access the file (simulated)")
    exc.winerror = winerror  # type: ignore[attr-defined]  # only set by CPython on Windows
    return exc


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Pretend to be on Windows and record (not perform) the retry sleeps."""
    sleeps: list[float] = []
    monkeypatch.setattr(_atomic.sys, "platform", "win32")
    monkeypatch.setattr(_atomic.time, "sleep", sleeps.append)
    return sleeps


def _flaky_replace(monkeypatch: pytest.MonkeyPatch, failures: int, winerror: int = 32) -> list[int]:
    calls: list[int] = []
    real_replace = os.replace

    def _replace(src: object, dst: object) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise _in_use(winerror)
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(_atomic.os, "replace", _replace)
    return calls


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_a_momentary_lock_is_retried_until_it_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[float], winerror: int
) -> None:
    calls = _flaky_replace(monkeypatch, failures=3, winerror=winerror)
    target = tmp_path / "report.csv"
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert len(calls) == 4
    assert len(windows) == 3


def test_a_persistent_lock_raises_file_in_use_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[float]
) -> None:
    target = tmp_path / "report.csv"
    target.write_bytes(b"old")
    _flaky_replace(monkeypatch, failures=10_000)
    with pytest.raises(FileInUseError, match=r"report\.csv") as info:
        atomic_write_bytes(target, b"new")
    assert isinstance(info.value, StoreError)
    assert isinstance(info.value.__cause__, PermissionError)
    assert target.read_bytes() == b"old"  # untouched
    assert [p.name for p in tmp_path.iterdir()] == ["report.csv"]  # temp file cleaned up
    assert 1.5 < sum(windows) < 3.0  # bounded: about two seconds in total


def test_other_permission_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[float]
) -> None:
    calls = _flaky_replace(monkeypatch, failures=10_000, winerror=1314)  # privilege not held
    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1
    assert windows == []


def test_posix_never_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_atomic.sys, "platform", "linux")
    calls = _flaky_replace(monkeypatch, failures=10_000)
    with pytest.raises(PermissionError):
        replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1


@pytest.mark.usefixtures("windows")
def test_unlink_retries_and_then_names_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "sorethumb.log"
    target.write_text("x", encoding="utf-8")
    real_unlink = Path.unlink
    attempts: list[int] = []

    def _unlink(self: Path, missing_ok: bool = False) -> None:
        attempts.append(1)
        if len(attempts) <= 2:
            raise _in_use()
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _unlink)
    unlink_with_retry(target)
    assert not target.exists()
    assert len(attempts) == 3

    def _always_locked(_self: Path, **_kwargs: object) -> None:
        raise _in_use()

    monkeypatch.setattr(Path, "unlink", _always_locked)
    other = tmp_path / "held.csv"
    other.write_text("x", encoding="utf-8")
    with pytest.raises(FileInUseError, match=r"delete .*held\.csv"):
        unlink_with_retry(other)
