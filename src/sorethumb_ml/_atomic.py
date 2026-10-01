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
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


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
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix)
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
    os.replace(tmp, dest)  # noqa: PTH105 — os.replace IS the atomic-rename primitive
    _fsync_dir(dest.parent)


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


def _fsync_path(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
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
