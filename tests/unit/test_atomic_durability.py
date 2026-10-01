"""P2-2: an atomic write is only durable once the *directory entry* is, so the
parent directory is fsynced after the rename -- where the platform supports it --
and skipped cleanly where it does not. Temporary-file cleanup on exceptions is
retained and tested alongside.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from sorethumb_ml import _atomic
from sorethumb_ml._atomic import atomic_write, atomic_write_bytes, atomic_write_text

pytestmark = pytest.mark.unit


def _leftover_temp_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


def test_parent_directory_is_fsynced_after_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_replace, real_fsync = os.replace, os.fsync

    def _replace(src: object, dst: object) -> None:
        events.append("replace")
        real_replace(src, dst)  # type: ignore[arg-type]

    def _fsync(fd: int) -> None:
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        real_fsync(fd)

    monkeypatch.setattr(_atomic.os, "replace", _replace)
    monkeypatch.setattr(_atomic.os, "fsync", _fsync)

    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert events == ["fsync-file", "replace", "fsync-dir"]


def test_the_directory_fsynced_is_the_targets_own_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(_atomic, "_fsync_dir", synced.append)
    target = tmp_path / "nested" / "deeper" / "artifact.txt"
    atomic_write_text(target, "x")
    assert synced == [target.parent]


def test_a_platform_that_cannot_open_directories_is_skipped_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows raises PermissionError from os.open on a directory."""
    real_open = os.open

    def _open(path: str, flags: int, *args: int) -> int:
        if Path(path).is_dir():
            raise PermissionError(errno.EACCES, "Is a directory (simulated)")
        return real_open(path, flags, *args)

    monkeypatch.setattr(_atomic.os, "open", _open)
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"ok")
    assert target.read_bytes() == b"ok"
    assert _leftover_temp_files(tmp_path) == []


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP])
def test_a_filesystem_that_rejects_directory_fsync_is_skipped_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    real_fsync = os.fsync

    def _fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, "fsync unsupported on directories (simulated)")
        real_fsync(fd)

    monkeypatch.setattr(_atomic.os, "fsync", _fsync)
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"ok")
    assert target.read_bytes() == b"ok"
    assert _leftover_temp_files(tmp_path) == []


def test_a_real_directory_fsync_failure_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EIO on the directory is a genuine durability failure, not 'unsupported'."""
    real_fsync = os.fsync

    def _fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "I/O error (simulated)")
        real_fsync(fd)

    monkeypatch.setattr(_atomic.os, "fsync", _fsync)
    with pytest.raises(OSError, match="I/O error"):
        atomic_write_bytes(tmp_path / "artifact.bin", b"x")
    assert _leftover_temp_files(tmp_path) == []


def test_exception_in_the_body_removes_the_temp_file_and_keeps_the_original(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    atomic_write_text(target, "original")

    def _interrupted() -> None:
        with atomic_write(target) as tmp:
            tmp.write_text("half-written")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _interrupted()
    assert target.read_text() == "original"
    assert _leftover_temp_files(tmp_path) == []


def test_failure_while_syncing_the_file_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.txt"
    atomic_write_text(target, "original")

    def _fsync(_fd: int) -> None:
        raise OSError(errno.ENOSPC, "no space left (simulated)")

    monkeypatch.setattr(_atomic.os, "fsync", _fsync)
    with pytest.raises(OSError, match="no space"):
        atomic_write_text(target, "new")
    assert target.read_text() == "original"
    assert _leftover_temp_files(tmp_path) == []


def test_keyboard_interrupt_also_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"

    def _interrupted() -> None:
        with atomic_write(target) as tmp:
            tmp.write_text("partial")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _interrupted()
    assert not target.exists()
    assert _leftover_temp_files(tmp_path) == []


def test_download_promotion_also_syncs_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb_ml.io import source

    synced: list[Path] = []
    monkeypatch.setattr(_atomic, "_fsync_dir", synced.append)
    tmp_file = tmp_path / ".download.tmp"
    tmp_file.write_bytes(b"data")
    dest = tmp_path / "cached.csv"
    source._promote(tmp_file, dest)
    assert dest.read_bytes() == b"data"
    assert not tmp_file.exists()
    assert synced == [tmp_path]
