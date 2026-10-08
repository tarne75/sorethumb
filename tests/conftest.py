"""Shared pytest fixtures, available to every test in the tree."""

from __future__ import annotations

import logging
from collections.abc import Iterator

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
