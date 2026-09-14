"""Fixtures shared across tests/unit/.

Re-exported (not just imported) so pytest's fixture discovery picks it up
without every test module needing its own import — a bare ``import`` alone
would not register it as a fixture for files that don't import it directly.
"""

from __future__ import annotations

from tests.factories.workspaces import workspace as ws

__all__ = ["ws"]
