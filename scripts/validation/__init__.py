"""Validation matrix: dataset × PCA × detector-combo sweep over the full sorethumb pipeline.

Split into a pure planner/result schema (``schema.py``, ``data.py``,
``identity.py``, ``planner.py`` — importable and unit-tested without running
any model fit) and a thin runner (``runner.py``, ``scripts/run_validation.py``)
that does the actual I/O: loading datasets, fitting detectors, scoring
held-out splits.
"""

from __future__ import annotations
