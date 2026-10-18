"""The anchor run's fixed flag rule."""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb_ml.history.anchor import apply_rule, build_rule, combined_score

pytestmark = pytest.mark.unit

CAL = {"a": np.array([0.1, 0.5, 0.9, 0.95]), "b": np.array([0.2, 0.4, 0.99, 0.3])}
NAT = {"a": np.array([False, False, True, True]), "b": np.array([False, False, True, False])}


def _rule(combination: str, contamination: str | float, threshold: float | None = None) -> dict:
    combined = combined_score(
        {"combination": combination, "detectors": ["a", "b"], "weights": {"a": 0.5, "b": 0.5}}, CAL
    )
    return build_rule(
        combination=combination,
        contamination=contamination,
        weights={"a": 0.5, "b": 0.5},
        threshold=threshold,
        calibrated=CAL,
        combined=combined,
        source_row=None,
    )


def test_composite_rule_fixes_the_threshold_and_weights() -> None:
    rule = _rule("composite", "auto", threshold=0.6)
    assert rule["threshold"] == 0.6
    assert rule["weights"] == {"a": 0.5, "b": 0.5}
    assert len(rule["combined_percentiles"]) == 101
    assert apply_rule(rule, CAL, NAT).tolist() == [False, False, True, True]  # type: ignore[union-attr]


def test_composite_rule_weights_follow_the_saved_weights() -> None:
    rule = _rule("composite", "auto", threshold=0.6)
    rule["weights"] = {"a": 1.0, "b": 0.0}  # e.g. b dropped by the bad-member guard
    np.testing.assert_allclose(combined_score(rule, CAL), CAL["a"])
    rule["weights"] = {"a": 0.0, "b": 0.0}
    np.testing.assert_allclose(combined_score(rule, CAL), (CAL["a"] + CAL["b"]) / 2)


def test_a_composite_rule_without_a_threshold_cannot_flag() -> None:
    assert apply_rule(_rule("composite", 0.05, threshold=float("nan")), CAL, NAT) is None


def test_intersection_and_union_auto_use_the_models_own_boundaries() -> None:
    inter = _rule("intersection", "auto")
    union = _rule("union", "auto")
    assert "cutoffs" not in inter
    assert apply_rule(inter, CAL, NAT).tolist() == [False, False, True, False]  # type: ignore[union-attr]
    assert apply_rule(union, CAL, NAT).tolist() == [False, False, True, True]  # type: ignore[union-attr]
    np.testing.assert_allclose(
        combined_score(inter, CAL), np.median(np.column_stack([CAL["a"], CAL["b"]]), axis=1)
    )
    np.testing.assert_allclose(combined_score(union, CAL), np.maximum(CAL["a"], CAL["b"]))


def test_numeric_intersection_fixes_each_detectors_cutoff() -> None:
    rule = _rule("intersection", 0.5)  # each detector's top 2 of 4
    assert rule["cutoffs"] == {"a": 0.9, "b": 0.4}
    # On new rows the cutoffs stay put rather than re-taking the top half.
    new = {"a": np.array([0.95, 0.95, 0.1]), "b": np.array([0.5, 0.1, 0.9])}
    assert apply_rule(rule, new, NAT).tolist() == [True, False, False]  # type: ignore[union-attr]


def test_a_detector_that_flagged_nothing_leaves_no_cutoff() -> None:
    rule = _rule("union", 0.5)
    rule["cutoffs"]["a"] = None
    assert apply_rule(rule, CAL, NAT) is None
