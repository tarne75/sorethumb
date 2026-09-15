"""Explicit Hypothesis profiles for the property lane.

CI runs a small, bounded example count so PRs stay fast and deterministic;
local developer runs default to a larger count that explores harder; the
nightly schedule job asks for a larger count still. Select with the
``HYPOTHESIS_PROFILE`` env var (``ci`` | ``dev`` | ``nightly``, default
``dev``) — CI sets it explicitly in ``.github/workflows/ci.yml``.

Each property test declares a baseline example count via
``scaled_examples(base)`` rather than a hardcoded ``max_examples=``, so the
active profile scales every test's exploration depth consistently instead of
each test picking its own fixed number.
"""

from __future__ import annotations

import os

from hypothesis import settings

_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "dev")
_EXAMPLE_MULTIPLIER = {"ci": 1, "dev": 3, "nightly": 10}.get(_PROFILE, 3)

# deadline=None: several of these tests fit real detectors/estimators inside
# a Hypothesis example, and wall-clock-based per-example deadlines are a
# common source of environment-dependent flakiness, not a real regression
# signal.
settings.register_profile("ci", deadline=None)
settings.register_profile("dev", deadline=None)
settings.register_profile("nightly", deadline=None)
settings.load_profile(_PROFILE)


def scaled_examples(base: int) -> int:
    """Scale a test's baseline example count by the active Hypothesis profile."""
    return base * _EXAMPLE_MULTIPLIER
