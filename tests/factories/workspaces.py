"""Lifecycle-resource fixtures for ``Workspace``.

A ``Workspace`` holds an open SQLite
connection that must be closed — otherwise a held file handle can block
cleanup of its ``tmp_path`` (notably on Windows). Fixtures that hand out a
``Workspace`` always ``yield`` and close it in a ``finally``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sorethumb_ml.store.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Workspace]:
    """A freshly initialised ``Workspace`` under ``tmp_path / "ws"``."""
    ws = Workspace.init(tmp_path / "ws")
    try:
        yield ws
    finally:
        ws.close()


def open_workspace(tmp_path: Path, name: str = "ws") -> Workspace:
    """Open a fresh ``Workspace`` for tests that need more than one instance
    per test (e.g. reopening, double-init) and so can't use the ``workspace``
    fixture. Callers own closing it (``Workspace`` supports the context-manager
    protocol via ``with open_workspace(tmp_path) as ws:``).
    """
    return Workspace.init(tmp_path / name)
