"""Shared test factories: on-disk frames, Config builders, detector helpers,
Workspace/CLI lifecycle fixtures, run helpers, and benchmark-row builders.

Split by concern (see the individual modules) so a test file imports only
what it needs instead of redefining its own ad hoc builder.
"""

from __future__ import annotations
