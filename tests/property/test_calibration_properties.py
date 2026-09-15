"""Hypothesis property tests for Calibrator: the calibrated (anomaly) score
is always bounded to [0, 1], monotone non-increasing in the raw score
(detectors follow "higher raw = more normal", so higher raw must never
calibrate to a higher anomaly score), tie-aware (identical raw scores
calibrate identically), and serialisation-idempotent (to_dict/from_dict
round-trips to the same transform output).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sorethumb.scoring.calibrate import Calibrator

pytestmark = pytest.mark.property

_FLOATS = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
_REFERENCE = st.lists(_FLOATS, min_size=5, max_size=200)


@given(reference=_REFERENCE, query=st.lists(_FLOATS, min_size=1, max_size=50))
@settings(max_examples=200)
def test_calibrated_scores_are_bounded(reference: list[float], query: list[float]) -> None:
    c = Calibrator()
    c.fit(np.array(reference))
    result = c.transform(np.array(query))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@given(reference=_REFERENCE, a=_FLOATS, b=_FLOATS)
@settings(max_examples=200)
def test_calibration_is_monotone_non_increasing_in_raw_score(
    reference: list[float], a: float, b: float
) -> None:
    """Higher raw score = more normal, so its calibrated (anomaly) score must
    never exceed a lower raw score's."""
    assume(a <= b)
    c = Calibrator()
    c.fit(np.array(reference))
    cal_a, cal_b = c.transform(np.array([a, b]))
    assert cal_a >= cal_b - 1e-9, f"raw {a} <= {b} but calibrated {cal_a} < {cal_b}"


@given(
    reference=_REFERENCE,
    value=_FLOATS,
    repeats=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100)
def test_calibration_is_tie_aware(reference: list[float], value: float, repeats: int) -> None:
    """Identical raw scores must calibrate to identical values, regardless of
    where else in the query batch they appear."""
    c = Calibrator()
    c.fit(np.array(reference))
    result = c.transform(np.array([value] * repeats))
    assert len(set(result.tolist())) == 1


@given(reference=_REFERENCE, query=st.lists(_FLOATS, min_size=1, max_size=50))
@settings(max_examples=200)
def test_calibration_serialisation_is_idempotent(reference: list[float], query: list[float]) -> None:
    c = Calibrator()
    c.fit(np.array(reference))
    restored = Calibrator.from_dict(c.to_dict())
    q = np.array(query)
    np.testing.assert_allclose(c.transform(q), restored.transform(q))
