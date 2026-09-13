"""Atomic file writes: sibling temp file, fsync, then rename over the target.

A reader never sees a partial file — either the previous version or the fully
written new one. Used for every artifact sorethumb writes and later reads back
(models, calibrators, manifests, result Parquet, HTML/JSON/CSV reports): a
crash or a concurrent reader mid-write must not be able to observe a truncated
or half-written file.
"""

from __future__ import annotations

import contextlib
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
    never actually made it to disk. On any exception, the temp file is removed
    and *path* is left untouched. The caller does its own writing (any library
    that accepts a path works); this only handles the fsync + rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        _fsync_path(tmp)
        os.replace(tmp, path)  # noqa: PTH105 — os.replace IS the atomic-rename primitive
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


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
