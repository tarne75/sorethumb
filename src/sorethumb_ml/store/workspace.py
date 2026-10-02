"""Workspace directory layout and resolution.

A workspace is a single directory that must contain a ``sorethumb.db`` file
(created by ``Workspace.init``) before it can be opened with ``Workspace.open``.
Opening an arbitrary directory that happens to contain other files is refused —
sorethumb never silently colonises a directory the user did not intend as a workspace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from sorethumb_ml._atomic import TEMP_NAME_MAX_LEN, unlink_with_retry
from sorethumb_ml.errors import FileInUseError, NotFoundError, PathTooLongError, StoreError
from sorethumb_ml.store.db import Store

logger = logging.getLogger(__name__)

_MARKER_DB = "sorethumb.db"
# SQLite files that sit beside the database while it is open or after a crash.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
# Sub-directories init() creates. Their top-level names are everything else a
# workspace owns (see owned_entry_names); add new ones here, not in init().
_SUBDIRS = ("cache/datasets", "cache/features", "models", "results", "reports", "logs", "tmp")


def owned_entry_names() -> tuple[str, ...]:
    """Names, directly under a workspace root, of every entry sorethumb creates.

    The database, its SQLite sidecar files, and the top-level directories from
    ``init()``. ``sorethumb workspace reset`` deletes exactly these, so a
    workspace created inside a directory that already held the user's own files
    (``--workdir ~/projects/q3-analysis``) never loses them.
    """
    dirs = dict.fromkeys(sub.split("/", 1)[0] for sub in _SUBDIRS)
    return (_MARKER_DB, *(_MARKER_DB + suffix for suffix in _DB_SIDECAR_SUFFIXES), *dirs)


# Windows' MAX_PATH is 260 including the terminating NUL.
WINDOWS_MAX_PATH = 259
_RUN_ID_LEN = len("score_") + 32  # the longer of "run_<32 hex>" and "score_<32 hex>"
_GROUP_KEY_LEN = 32
_HASH_HEX_LEN = 64  # download-cache fingerprints and HTTP-metadata names (sha256 hex)
_LONGEST_SOURCE_SUFFIX = len(".ndjson.gz")
# The longest file sorethumb writes beside each detector's model.
_LONGEST_MODEL_FILE_SUFFIX = len(".calibrator.json")


def deepest_path_suffix_length(detector_names: Iterable[str]) -> int:
    """Return the length of the longest path sorethumb creates *below* a workspace root.

    Includes the leading separator, and the temp file an atomic write creates
    beside each target. Derived from the layout itself: models/<run>/<group>/
    <detector>.calibrator.json, results/<run>/<group>/, reports/<run>/,
    cache/datasets/<sha256>/data<ext> and cache/datasets/.http_meta/<sha256>.json.
    """
    longest_detector = max((len(n) for n in detector_names), default=0)

    def leaf(name_len: int) -> int:
        return 1 + max(name_len, TEMP_NAME_MAX_LEN)

    candidates = (
        len("/models/")
        + _RUN_ID_LEN
        + 1
        + _GROUP_KEY_LEN
        + leaf(longest_detector + _LONGEST_MODEL_FILE_SUFFIX),
        len("/results/") + _RUN_ID_LEN + 1 + _GROUP_KEY_LEN + leaf(len("anomalies.parquet")),
        len("/reports/") + _RUN_ID_LEN + leaf(_GROUP_KEY_LEN + len(".csv")),
        len("/cache/datasets/") + _HASH_HEX_LEN + leaf(len("data") + _LONGEST_SOURCE_SUFFIX),
        len("/cache/datasets/.http_meta") + leaf(_HASH_HEX_LEN + len(".json")),
    )
    return max(candidates)


def _long_paths_enabled() -> bool:
    """Return True if Windows long-path support is switched on (LongPathsEnabled=1)."""
    if sys.platform != "win32":
        return True
    import winreg  # noqa: PLC0415

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        return False
    return bool(value == 1)


def check_path_length(root: Path, detector_names: Iterable[str]) -> None:
    r"""Fail early when a Windows workspace is too deep for MAX_PATH.

    No-op except on Windows, and when long paths are enabled or *root* already
    uses the ``\\?\`` extended-length prefix.

    Raises:
        PathTooLongError: the deepest path the run would create exceeds 259 characters.

    """
    if sys.platform != "win32":
        return
    root_str = str(root.resolve())
    if root_str.startswith("\\\\?\\") or _long_paths_enabled():
        return
    deepest = len(root_str) + deepest_path_suffix_length(detector_names)
    if deepest <= WINDOWS_MAX_PATH:
        return
    over = deepest - WINDOWS_MAX_PATH
    msg = (
        f"The workspace path {root_str} ({len(root_str)} characters) is too long for Windows: "
        f"sorethumb creates paths up to {deepest - len(root_str)} characters below it, "
        f"{over} over the 259-character limit. Use a workspace path at least {over} characters "
        "shorter (--workdir), or enable long paths in Windows (Microsoft's "
        '"Maximum Path Length Limitation" page explains the LongPathsEnabled setting).'
    )
    raise PathTooLongError(msg)


def group_value_json_default(value: object) -> str:
    """``json.dumps`` fallback for group values that are not natively JSON-encodable.

    Tags the encoded string with the value's type name (e.g. a ``date`` becomes
    ``"date:2024-01-15"``) so it can never collide with a plain string that
    happens to share the same textual form.
    """
    if hasattr(value, "isoformat"):
        return f"{type(value).__name__}:{value.isoformat()}"
    return f"{type(value).__name__}:{value!s}"


def make_group_key(group_values: dict[str, object]) -> str:
    """Return a stable 32-character (128-bit) hex digest of the sorted group-values JSON.

    The digest is the only value that appears in key positions (filesystem paths,
    SQL primary keys). Raw group values live in the JSON column only.

    *group_values* must carry each column's *typed* value (``None``, ``int``,
    ``float``, ``str``, ``date``, ...), not a pre-stringified one: ``json.dumps``
    already renders ``None``, ``""``, and the literal string ``"None"`` as
    distinct tokens (``null``, ``""``, ``"None"``), which is what keeps those
    three group identities apart. Stringifying the values before calling this
    collapses that distinction and produces colliding keys.
    """
    json_str = json.dumps(group_values, sort_keys=True, ensure_ascii=False, default=group_value_json_default)
    return hashlib.sha256(json_str.encode()).hexdigest()[:32]


class Workspace:
    """Resolves and validates a sorethumb workspace directory.

    Use ``Workspace.init(path)`` to create a new workspace.
    Use ``Workspace.open(path)`` to attach to an existing one.
    """

    def __init__(self, root: Path, store: Store) -> None:
        """Construct directly only through init() or open()."""
        self._root = root
        self._store = store

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def init(cls, path: str | Path) -> Workspace:
        """Create a new workspace at *path*.

        Creates the directory and all sub-directories if they do not exist.
        Safe to call again on an already-initialised workspace (idempotent).
        """
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for sub in _SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)
        db_path = root / _MARKER_DB
        store = Store(db_path)
        logger.info("Workspace initialised at %s.", root)
        return cls(root, store)

    @classmethod
    def open(cls, path: str | Path) -> Workspace:
        """Open an existing workspace. Raises NotFoundError (a StoreError) if *path* is not a workspace."""
        root = Path(path).resolve()
        if not root.is_dir():
            msg = f"Workspace path does not exist or is not a directory: {root}"
            raise NotFoundError(msg)
        db_path = root / _MARKER_DB
        if not db_path.is_file():
            msg = (
                f"{root} exists but is not a sorethumb workspace "
                f"(no {_MARKER_DB}). Run 'sorethumb init {root}' first."
            )
            raise NotFoundError(msg)
        store = Store(db_path)
        return cls(root, store)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The resolved workspace root directory."""
        return self._root

    @property
    def store(self) -> Store:
        """The Store (database connection) for this workspace."""
        return self._store

    def db_path(self) -> Path:
        """Path to sorethumb.db."""
        return self._root / _MARKER_DB

    def results_dir(self, run_id: str, group_key: str) -> Path:
        """Return the results directory for a (run_id, group_key), creating it if needed."""
        d = self._root / "results" / run_id / group_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_dir(self, run_id: str) -> Path:
        """Return the run-level directory (parent of the per-group model dirs)."""
        d = self._root / "models" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def models_dir(self, run_id: str, group_key: str) -> Path:
        """Return the models directory for a (run_id, group_key), creating it if needed."""
        d = self._root / "models" / run_id / group_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def features_dir(self, run_id: str, group_key: str) -> Path:
        """Return the features cache directory, creating it if needed."""
        d = self._root / "cache" / "features" / run_id / group_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def logs_dir(self) -> Path:
        """Return the logs directory."""
        return self._root / "logs"

    def tmp_dir(self) -> Path:
        """Return the tmp scratch directory."""
        return self._root / "tmp"

    def close(self) -> None:
        """Close the database connection."""
        self._store.close()

    def __enter__(self) -> Self:
        """Support context-manager usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close on exit."""
        self.close()

    # ------------------------------------------------------------------
    # Retention / pruning
    # ------------------------------------------------------------------

    def list_prunable(self, retention_days: int) -> list[dict[str, object]]:
        """List artifacts eligible for pruning without deleting anything."""
        self._check_retention_days(retention_days)
        rows = self._store.artifacts_for_prune(retention_days)
        return [{"artifact_id": r["artifact_id"], "path": r["path"], "kind": r["kind"]} for r in rows]

    def prune(self, retention_days: int, *, dry_run: bool = False) -> list[str]:
        """Delete eligible artifacts from disk and the artifact index.

        Returns a list of paths that were (or would be) deleted.
        With dry_run=True, nothing is deleted.

        Refuses a negative ``retention_days`` (the underlying SQL comparison
        would treat that as "older than a negative age", i.e. everything,
        including artifacts just written) and refuses to delete any artifact
        whose recorded path resolves outside the workspace root -- the
        artifact index is trusted for normal operation but is still a
        database file on disk, not a signed record, so a corrupted or
        hand-edited row must not turn a routine prune into deleting an
        arbitrary path.
        """
        self._check_retention_days(retention_days)
        rows = self._store.artifacts_for_prune(retention_days)
        root = self._root.resolve()
        deleted: list[str] = []
        failed: list[str] = []
        for row in rows:
            path_str = str(row["path"])
            p = Path(path_str).resolve()
            if not p.is_relative_to(root):
                logger.warning(
                    "Refusing to prune artifact %s: path %s resolves outside workspace root %s.",
                    row["artifact_id"],
                    path_str,
                    root,
                )
                continue
            if not dry_run:
                # One file that can't be deleted (open in another program on
                # Windows) must not stop the rest. Its index row is kept, so it
                # stays prunable next time, and the failure is reported.
                try:
                    if p.exists():
                        unlink_with_retry(p)
                        logger.info("Pruned artifact: %s", path_str)
                except (OSError, FileInUseError) as exc:
                    failed.append(f"{path_str} ({exc})")
                    continue
                self._store.delete_artifact(str(row["artifact_id"]))
            deleted.append(path_str)
        if failed:
            msg = (
                f"Pruned {len(deleted)} artifact(s) but could not delete {len(failed)}; they are still "
                "indexed and will be retried on the next prune: " + "; ".join(failed)
            )
            raise StoreError(msg)
        return deleted

    @staticmethod
    def _check_retention_days(retention_days: int) -> None:
        if retention_days < 0:
            err = StoreError(
                f"retention_days must be >= 0, got {retention_days}. A negative value would "
                "match every artifact regardless of age, including ones just written."
            )
            err.failure_kind = "preflight"  # a rejected argument, before anything is touched
            raise err
