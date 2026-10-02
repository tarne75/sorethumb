"""Atomic file writes: sibling temp file, fsync, then rename over the target.

A reader never sees a partial file — either the previous version or the fully
written new one. Used for every artifact sorethumb writes and later reads back
(models, calibrators, manifests, result Parquet, HTML/JSON/CSV reports): a
crash or a concurrent reader mid-write must not be able to observe a truncated
or half-written file.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
import tempfile
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from sorethumb_ml.errors import FileInUseError

# Temp files are "<TEMP_PREFIX><8 random chars><suffix>", next to their target.
TEMP_PREFIX = ".st-"
TEMP_NAME_MAX_LEN = len(TEMP_PREFIX) + 8 + len(".tmp")


@contextmanager
def atomic_write(path: Path, *, suffix: str = ".tmp") -> Generator[Path, None, None]:
    """Yield a sibling temp path to write to; replace *path* with it on success.

    Usage::

        with atomic_write(out_path) as tmp:
            df.write_parquet(str(tmp))  # or tmp.write_text(...), joblib.dump(obj, tmp), ...

    The temp file lives next to *path* (same directory, so ``os.replace`` is
    guaranteed atomic on the same filesystem) and is fsynced before the
    rename, so a crash right after the rename can't expose a file whose bytes
    never actually made it to disk. The parent directory is fsynced after the
    rename (where the platform supports it), so the rename itself also survives
    a crash. On any exception, the temp file is removed and *path* is left
    untouched. The caller does its own writing (any library
    that accepts a path works); this only handles the fsync + rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A short fixed prefix rather than the target's own name: on Windows every
    # character counts against the 260-character path limit (see
    # store/workspace.check_path_length), and nothing reads these names back.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=TEMP_PREFIX, suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        promote_durably(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def promote_durably(tmp: Path, dest: Path) -> None:
    """Fsync *tmp*, atomically rename it over *dest*, then fsync *dest*'s directory.

    Syncing the file makes its bytes durable; the rename is only durable once the
    directory entry is, so without the last step a crash just after the rename can
    still bring back the old (or no) file on some filesystems.
    """
    _fsync_path(tmp)
    replace_with_retry(tmp, dest)
    _fsync_dir(dest.parent)


# Windows error codes for "another process has this file open":
# ERROR_ACCESS_DENIED (5) -- what replacing or deleting a file that is open
# without delete sharing reports -- ERROR_SHARING_VIOLATION (32) and
# ERROR_LOCK_VIOLATION (33).
_WINDOWS_IN_USE_ERRORS = frozenset({5, 32, 33})
# Most such locks are momentary (antivirus or the search indexer scanning a file
# just written); a persistent one (a spreadsheet holding a report open) outlasts
# these and is reported. Roughly two seconds in total.
_IN_USE_RETRY_DELAYS_S = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)


def _is_in_use(exc: OSError) -> bool:
    """Return True for a Windows "file is open in another process" failure; never on POSIX."""
    return (
        sys.platform == "win32"
        and isinstance(exc, PermissionError)
        and getattr(exc, "winerror", None) in _WINDOWS_IN_USE_ERRORS
    )


def _retry_while_in_use(operation: Callable[[], None], path: Path, action: str) -> None:
    for delay in (*_IN_USE_RETRY_DELAYS_S, None):
        try:
            operation()
            return
        except OSError as exc:
            if not _is_in_use(exc):
                raise
            if delay is None:
                msg = (
                    f"Could not {action} {path}: the file is open in another program (for example a "
                    "spreadsheet, a file-sync client or antivirus software), or access to it was "
                    "denied. Close it and try again."
                )
                raise FileInUseError(msg) from exc
            time.sleep(delay)


def replace_with_retry(src: Path, dest: Path) -> None:
    """``os.replace(src, dest)``, retrying briefly while Windows reports *dest* in use.

    On POSIX this is exactly one ``os.replace``. On Windows a sharing violation
    is retried for about two seconds, then raised as
    :class:`~sorethumb_ml.errors.FileInUseError` naming *dest*.
    """
    _retry_while_in_use(lambda: os.replace(src, dest), dest, "replace")  # noqa: PTH105 -- the atomic-rename primitive


def unlink_with_retry(path: Path, *, missing_ok: bool = False) -> None:
    """``path.unlink()``, with the same Windows in-use retry as :func:`replace_with_retry`."""
    _retry_while_in_use(lambda: path.unlink(missing_ok=missing_ok), path, "delete")


# fsync on a directory handle is rejected with these on filesystems that have no
# notion of it; that is "unsupported", not a durability failure.
_FSYNC_DIR_UNSUPPORTED = frozenset(
    getattr(errno, name) for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS") if hasattr(errno, name)
)


def _fsync_dir(directory: Path) -> None:
    """Fsync *directory* so a completed rename survives a crash; skip where unsupported.

    Platforms that cannot open a directory handle (Windows raises ``PermissionError``
    from ``os.open``) and filesystems that reject fsync on one (``EINVAL`` and
    friends) are skipped silently. Any other error, such as ``EIO``, is a real
    durability failure and propagates.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _FSYNC_DIR_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


# Flags for opening a file only to fsync it. Write access is required: on
# Windows os.fsync is _commit() -> FlushFileBuffers, which fails with EBADF
# ("Bad file descriptor") on a read-only handle. The temp files synced here are
# always ours (created by mkstemp, mode 0600), so write access is available.
_FSYNC_OPEN_FLAGS = os.O_RDWR | getattr(os, "O_BINARY", 0)


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), _FSYNC_OPEN_FLAGS)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically (temp file + fsync + rename)."""
    with atomic_write(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically (temp file + fsync + rename)."""
    with atomic_write(path) as tmp:
        tmp.write_text(text, encoding=encoding)
