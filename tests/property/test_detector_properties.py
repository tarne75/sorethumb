"""Hypothesis property tests: for detectors whose natural boundary is learned
at fit time (not recomputed from whatever's in the scoring batch),
natural_flag's result for a given row must not depend on what else is
scored alongside it in the same call.

kmeans_distance is deliberately excluded: its Tukey-fence boundary is
recomputed from the batch passed to natural_flag() each time, so it does not
(yet) promise batch invariance -- see P2-1, which persists a fit-time
threshold instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sorethumb.detectors.ecod import ECODDetector
from sorethumb.detectors.hbos import HBOSDetector
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.lof import LOFDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector

pytestmark = pytest.mark.property

_BATCH_INVARIANT_DETECTOR_CLASSES = [
    IsolationForestDetector,
    OneClassSVMDetector,
    LOFDetector,
    ECODDetector,
    HBOSDetector,
]


def _fit_on_fixed_training_data(cls: type) -> object:
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((150, 4))
    det = cls()
    det.fit(X_train, seed=0)
    return det


# Fit once per class (real sklearn/pyod work) and reuse across every example;
# only the *query* batch composition varies under @given.
_FITTED_BY_CLASS = {cls: _fit_on_fixed_training_data(cls) for cls in _BATCH_INVARIANT_DETECTOR_CLASSES}


@given(
    cls=st.sampled_from(_BATCH_INVARIANT_DETECTOR_CLASSES),
    seed=st.integers(min_value=0, max_value=10_000),
    n_rows=st.integers(min_value=4, max_value=40),
    split=st.integers(min_value=1, max_value=39),
)
@settings(max_examples=150)
def test_natural_flag_is_batch_invariant(cls: type, seed: int, n_rows: int, split: int) -> None:
    assume(split < n_rows)
    det = _FITTED_BY_CLASS[cls]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, 4)) * 5  # wide spread so some rows are plausible outliers

    # Score + flag every row in one batch.
    together_flags = det.natural_flag(det.score_samples(X))

    # Score + flag the same rows split across two separate calls.
    flags_a = det.natural_flag(det.score_samples(X[:split]))
    flags_b = det.natural_flag(det.score_samples(X[split:]))
    split_flags = np.concatenate([flags_a, flags_b])

    np.testing.assert_array_equal(
        together_flags, split_flags, err_msg=f"{cls.name}: natural_flag depends on what else is in the batch"
    )
