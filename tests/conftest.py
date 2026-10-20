"""Shared pytest fixtures, available to every test in the tree."""

from __future__ import annotations

import gc
import logging
import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from tests.factories.workspaces import workspace as ws

__all__ = ["ws"]


@pytest.fixture(autouse=True)
def _no_leaked_workspace_log_handler() -> Iterator[None]:
    """Fail any test that leaves a CLI workspace log handler attached.

    A handler outliving its command holds ``logs/sorethumb.log`` open, which
    on Windows blocks deleting or rotating it, and sends later in-process
    commands' records to the wrong workspace. Matched by class name so this
    guard doesn't import the CLI for tests that never touch it.
    """
    yield
    logger = logging.getLogger("sorethumb_ml")
    leaked = [h for h in logger.handlers if type(h).__name__ == "_WorkspaceLogHandler"]
    for handler in leaked:
        logger.removeHandler(handler)
        handler.close()
    assert not leaked, f"workspace log handler(s) left attached: {[h.baseFilename for h in leaked]}"


class _TrackedConnection(sqlite3.Connection):
    """A connection that records, if it is dropped without ``close()``, the test that opened it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._closed = False

    def close(self) -> None:
        self._closed = True
        super().close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            _UNCLOSED_CONNECTIONS.append(str(getattr(self, "_opened_by", "?")))


_UNCLOSED_CONNECTIONS: list[str] = []
_real_connect = sqlite3.connect
_current_test = ["(collection)"]
_opened = [0]


def _tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    kwargs.setdefault("factory", _TrackedConnection)
    conn = _real_connect(*args, **kwargs)
    if isinstance(conn, _TrackedConnection):
        conn._opened_by = _current_test[0]  # type: ignore[attr-defined]
    _opened[0] += 1
    return conn


sqlite3.connect = _tracked_connect  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _no_unclosed_sqlite_connection(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail any test that drops a SQLite connection without closing it.

    An unclosed connection keeps the database file open (Windows then refuses to delete
    or replace it) and, since Python 3.13, raises a ResourceWarning whenever it is
    garbage-collected, possibly during an unrelated later test. Open a workspace with
    ``with Workspace.open(...) as ws:`` (or close the Store) instead.
    """
    _current_test[0] = request.node.nodeid
    before, opened_before = len(_UNCLOSED_CONNECTIONS), _opened[0]
    yield
    if _opened[0] != opened_before:
        # A workspace and its store reference each other, so a dropped one is freed
        # only by the cycle collector: run it now, but only after tests that opened one.
        gc.collect()
    leaked = _UNCLOSED_CONNECTIONS[before:]
    del _UNCLOSED_CONNECTIONS[before:]
    assert not leaked, f"SQLite connection(s) dropped without close(), opened by: {sorted(set(leaked))}"
