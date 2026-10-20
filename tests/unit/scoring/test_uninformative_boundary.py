"""A detector whose own boundary flags most of its fit rows does not vote at "auto".

Isolation Forest on data with few distinct values scores every row past its offset
(100% flagged), and One-Class SVM leaves every row on its boundary, where the side a
row falls on is floating-point noise. Such a boundary does not separate unusual rows
from the rest; under contamination="auto" it is left out of the flag vote (and of a
composite's median-of-rates cut), with a warning. Its scores still rank.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sorethumb_ml.errors import UninformativeBoundaryWarning, ZeroAnomalyWarning
from sorethumb_ml.scoring.combine import (
    MAX_INFORMATIVE_BOUNDARY_RATE,
    ScoreEnsemble,
    uninformative_boundaries,
)

pytestmark = pytest.mark.unit

_N = 100
_IDX = np.arange(_N)


def _inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # "everything" flags every row; "a" and "b" flag overlapping minorities.
    flags = {
        "everything": np.ones(_N, dtype=bool),
        "a": _IDX < 20,
        "b": (_IDX >= 10) & (_IDX < 30),
    }
    scores = {name: f.astype(float) + _IDX[::-1] / (10 * _N) for name, f in flags.items()}
    return scores, flags


def test_uninformative_boundaries_keeps_only_rates_above_half() -> None:
    rates = {"x": 1.0, "y": MAX_INFORMATIVE_BOUNDARY_RATE, "z": 0.5625, "w": 0.1}
    assert uninformative_boundaries(rates) == {"x": 1.0, "z": 0.5625}


@pytest.mark.parametrize("combination", ["intersection", "union"])
def test_an_uninformative_boundary_does_not_vote_in_a_set_policy(combination: str) -> None:
    scores, flags = _inputs()
    ens = ScoreEnsemble(combination=combination, contamination="auto")

    with pytest.warns(UninformativeBoundaryWarning, match="everything's own boundary flagged 100%"):
        out = ens.combine(scores, flags, uninformative={"everything": 1.0})

    a_and_b = flags["a"] & flags["b"]
    a_or_b = flags["a"] | flags["b"]
    expected = a_and_b if combination == "intersection" else a_or_b
    assert np.array_equal(out["anomaly_flag"], expected)
    assert out["abstaining_members"] == ["everything"]
    # The realised rate is still reported as it was.
    assert out["per_detector_rates"]["everything"] == 1.0


def test_union_no_longer_flags_every_row_because_one_boundary_does() -> None:
    scores, flags = _inputs()
    ens = ScoreEnsemble(combination="union", contamination="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UninformativeBoundaryWarning)
        before = ens.combine(scores, flags)
        after = ens.combine(scores, flags, uninformative={"everything": 1.0})
    assert before["anomaly_flag"].all()
    assert after["anomaly_flag"].sum() == (flags["a"] | flags["b"]).sum()


def test_a_composite_cut_ignores_the_uninformative_rate() -> None:
    scores, flags = _inputs()
    ens = ScoreEnsemble(combination="composite", contamination="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UninformativeBoundaryWarning)
        before = ens.combine(scores, flags)
        after = ens.combine(scores, flags, uninformative={"everything": 1.0})
    # Median of {1.0, 0.2, 0.2} is 0.2 either way here; of the voters alone it is 0.2.
    assert after["contamination_used"] == pytest.approx(0.2)
    assert after["abstaining_members"] == ["everything"]
    assert before["contamination_used"] == pytest.approx(0.2)


def test_a_composite_cut_moves_when_the_uninformative_rate_was_the_median() -> None:
    flags = {"everything": np.ones(_N, dtype=bool), "mid": _IDX < 60, "low": _IDX < 10}
    scores = {name: f.astype(float) + _IDX[::-1] / (10 * _N) for name, f in flags.items()}
    ens = ScoreEnsemble(combination="composite", contamination="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UninformativeBoundaryWarning)
        before = ens.combine(scores, flags, uninformative={})
        after = ens.combine(scores, flags, uninformative={"everything": 1.0, "mid": 0.6})
    assert before["contamination_used"] == pytest.approx(0.5)  # median 0.6, clamped to 0.5
    assert after["contamination_used"] == pytest.approx(0.1)  # only "low" votes


def test_with_no_voter_left_nothing_is_flagged_and_the_zero_warning_is_not_raised() -> None:
    scores, flags = _inputs()
    for combination in ("intersection", "union", "composite"):
        ens = ScoreEnsemble(combination=combination, contamination="auto")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = ens.combine(scores, flags, uninformative={"everything": 1.0, "a": 0.6, "b": 0.7})
        assert not out["anomaly_flag"].any(), combination
        assert sorted(out["abstaining_members"]) == ["a", "b", "everything"]
        kinds = {type(w.message) for w in caught}
        assert UninformativeBoundaryWarning in kinds
        assert ZeroAnomalyWarning not in kinds  # "no row passed all N detectors" would be false
        # The ranking is untouched.
        assert np.array_equal(out["combined_score"], ens.combine(scores, flags)["combined_score"])


def test_a_numeric_contamination_ignores_the_boundaries() -> None:
    scores, flags = _inputs()
    ens = ScoreEnsemble(combination="intersection", contamination=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZeroAnomalyWarning)  # this data's top-10% sets need not overlap
        warnings.simplefilter("error", UninformativeBoundaryWarning)
        with_guard = ens.combine(scores, flags, uninformative={"everything": 1.0})
        without = ens.combine(scores, flags)
    assert np.array_equal(with_guard["anomaly_flag"], without["anomaly_flag"])
    assert with_guard["abstaining_members"] == []


def test_the_warning_has_a_stable_code() -> None:
    assert UninformativeBoundaryWarning.code == "scoring.boundary_uninformative"
