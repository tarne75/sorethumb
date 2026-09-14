"""Shared pytest fixtures, available to every test in the tree."""

from __future__ import annotations

from tests.factories.workspaces import workspace as ws

__all__ = ["ws"]
