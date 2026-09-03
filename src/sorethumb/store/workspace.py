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
from pathlib import Path
from typing import Self

from sorethumb.errors import StoreError
from sorethumb.store.db import Store

logger = logging.getLogger(__name__)

_MARKER_DB = "sorethumb.db"


def make_group_key(group_values: dict[str, str]) -> str:
    """Return a stable 16-character hex digest of the sorted group-values JSON.

    The digest is the only value that appears in key positions (filesystem paths,
    SQL primary keys). Raw group values live in the JSON column only.
    """
    json_str = json.dumps(group_values, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


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
        for sub in ("cache/datasets", "cache/features", "models", "results", "reports", "logs", "tmp"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        db_path = root / _MARKER_DB
        store = Store(db_path)
        logger.info("Workspace initialised at %s.", root)
        return cls(root, store)

    @classmethod
    def open(cls, path: str | Path) -> Workspace:
        """Open an existing workspace. Raises StoreError if *path* is not a workspace."""
        root = Path(path).resolve()
        if not root.is_dir():
            msg = f"Workspace path does not exist or is not a directory: {root}"
            raise StoreError(msg)
        db_path = root / _MARKER_DB
        if not db_path.is_file():
            msg = (
                f"{root} exists but is not a sorethumb workspace "
                f"(no {_MARKER_DB}). Run 'sorethumb init {root}' first."
            )
            raise StoreError(msg)
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
        rows = self._store.artifacts_for_prune(retention_days)
        return [{"artifact_id": r["artifact_id"], "path": r["path"], "kind": r["kind"]} for r in rows]

    def prune(self, retention_days: int, *, dry_run: bool = False) -> list[str]:
        """Delete eligible artifacts from disk and the artifact index.

        Returns a list of paths that were (or would be) deleted.
        With dry_run=True, nothing is deleted.
        """
        rows = self._store.artifacts_for_prune(retention_days)
        deleted: list[str] = []
        for row in rows:
            path_str = str(row["path"])
            deleted.append(path_str)
            if not dry_run:
                p = Path(path_str)
                if p.exists():
                    p.unlink()
                    logger.info("Pruned artifact: %s", path_str)
                self._store.delete_artifact(str(row["artifact_id"]))
        return deleted
