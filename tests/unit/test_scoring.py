"""Unit tests for M3: scoring (calibration and combination)."""

import numpy as np
import pytest

from sorethumb.scoring.calibrate import Calibrator
from sorethumb.scoring.combine import ScoreEnsemble

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_scores(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


def _flag(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return scores < threshold


# ---------------------------------------------------------------------------
# Calibrator: basic fit/transform
# ---------------------------------------------------------------------------


def test_calibrator_transform_shape():
    c = Calibrator()
    scores = _uniform_scores()
    calibrated = c.fit_transform(scores)
    assert calibrated.shape == scores.shape


def test_calibrator_output_range():
    c = Calibrator()
    scores = _uniform_scores()
    calibrated = c.fit_transform(scores)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_calibrator_high_raw_score_low_anomaly():
    # Higher raw score = more normal = lower calibrated anomaly score
    c = Calibrator()
    ref = np.linspace(0.0, 1.0, 1000)
    c.fit(ref)
    # Score at 99th percentile of ref should calibrate to ~0.01 (almost normal)
    high = np.array([0.99])
    low = np.array([0.01])
    assert c.transform(high)[0] < c.transform(low)[0]


def test_calibrator_monotone_transform():
    c = Calibrator()
    ref = np.linspace(-5.0, 5.0, 500)
    c.fit(ref)
    test = np.linspace(-5.0, 5.0, 100)
    calibrated = c.transform(test)
    # Higher raw → lower calibrated (monotone decreasing)
    diffs = np.diff(calibrated)
    assert (diffs <= 1e-9).all(), "calibrated score must be monotone decreasing in raw score"


def test_calibrator_median_calibrates_near_half():
    c = Calibrator()
    scores = _uniform_scores()
    c.fit(scores)
    # After fitting on uniform, the median score should calibrate near 0.5
    median_score = np.median(scores)
    result = c.transform(np.array([median_score]))
    assert 0.3 < result[0] < 0.7


def test_calibrator_tie_aware_midrank():
    # Reference: half its mass sits at exactly 0.0, the rest spread over (0, 1].
    ref = np.concatenate([np.zeros(500), np.linspace(0.01, 1.0, 500)])
    c = Calibrator()
    c.fit(ref)

    # A query exactly at the tied value lands at the MIDPOINT of the band it
    # occupies: ~0 of the reference below it, ~half equal to it -> F ~ 0.25 ->
    # calibrated ~ 0.75. Not pinned to 0.0 or 1.0 (either end of the flat band).
    at_tie = c.transform(np.array([0.0]))[0]
    assert abs(at_tie - 0.75) < 0.03

    # Extremes still saturate.
    assert c.transform(np.array([-1.0]))[0] == pytest.approx(1.0)
    assert c.transform(np.array([2.0]))[0] == pytest.approx(0.0)


def test_calibrator_from_dict_ignores_legacy_mode_key():
    c = Calibrator()
    c.fit(np.arange(100, dtype=float))
    d = c.to_dict()
    d["mode"] = "reference"  # written by an older sorethumb version
    c2 = Calibrator.from_dict(d)
    np.testing.assert_allclose(
        c.transform(np.arange(100, dtype=float)), c2.transform(np.arange(100, dtype=float))
    )


def test_calibrator_constant_scores_returns_half():
    c = Calibrator()
    ref = np.ones(100)
    c.fit(ref)
    result = c.transform(np.ones(50))
    np.testing.assert_array_almost_equal(result, 0.5)


def test_calibrator_empty_transform():
    c = Calibrator()
    c.fit(np.arange(100, dtype=float))
    result = c.transform(np.array([]))
    assert len(result) == 0


def test_calibrator_transform_without_fit_raises():
    c = Calibrator()
    with pytest.raises(RuntimeError, match="fit"):
        c.transform(np.array([1.0, 2.0]))


def test_calibrator_fit_empty_raises():
    c = Calibrator()
    with pytest.raises(ValueError, match="empty"):
        c.fit(np.array([]))


# ---------------------------------------------------------------------------
# Calibrator: to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


def test_calibrator_to_dict_from_dict_roundtrip():
    c = Calibrator()
    scores = _uniform_scores()
    c.fit(scores)
    d = c.to_dict()
    c2 = Calibrator.from_dict(d)
    # Transform with both should give same result
    test = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(c.transform(test), c2.transform(test), rtol=1e-6)


def test_calibrator_from_dict_unfitted():
    c = Calibrator()
    d = c.to_dict()
    c2 = Calibrator.from_dict(d)
    assert c2._quantile_values is None


def test_calibrator_to_dict_serialisable():
    import json

    c = Calibrator()
    c.fit(np.arange(100, dtype=float))
    d = c.to_dict()
    json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# ScoreEnsemble: construction validation
# ---------------------------------------------------------------------------


def test_score_ensemble_bad_weighting():
    with pytest.raises(ValueError, match="weighting"):
        ScoreEnsemble(weighting="bad")


def test_score_ensemble_bad_combination():
    with pytest.raises(ValueError, match="combination"):
        ScoreEnsemble(combination="bad")


def test_score_ensemble_manual_no_weights():
    with pytest.raises(ValueError, match="manual_weights"):
        ScoreEnsemble(weighting="manual")


def test_score_ensemble_bad_contamination_value():
    with pytest.raises(ValueError, match="contamination"):
        ScoreEnsemble(contamination=1.5)


# ---------------------------------------------------------------------------
# ScoreEnsemble: combine — equal weighting
# ---------------------------------------------------------------------------


def _make_scores_flags(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    fa = a > 0.8
    fb = b > 0.8
    return (
        {"det_a": a, "det_b": b},
        {"det_a": fa, "det_b": fb},
    )


def test_score_ensemble_combine_returns_keys():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(contamination=0.1)
    result = ens.combine(scores, flags)
    for key in (
        "combined_score",
        "anomaly_flag",
        "threshold",
        "contamination_used",
        "weights",
        "is_auto_contamination",
        "dropped_members",
        "per_detector_rates",
    ):
        assert key in result


def test_per_detector_rates_are_the_realised_natural_flag_fractions():
    n = 400
    fa = np.zeros(n, dtype=bool)
    fa[:20] = True  # detector a flags 5%
    fb = np.zeros(n, dtype=bool)
    fb[:60] = True  # detector b flags 15%
    scores = {"a": np.linspace(0, 1, n), "b": np.linspace(0, 1, n)}
    result = ScoreEnsemble(contamination="auto").combine(scores, {"a": fa, "b": fb})
    assert result["per_detector_rates"] == {"a": pytest.approx(0.05), "b": pytest.approx(0.15)}
    # contamination="auto" is just their median — not a measurement.
    assert result["contamination_used"] == pytest.approx(0.10)


def test_score_ensemble_combined_score_shape():
    scores, flags = _make_scores_flags(n=300)
    ens = ScoreEnsemble(contamination=0.1)
    result = ens.combine(scores, flags)
    assert result["combined_score"].shape == (300,)


def test_score_ensemble_anomaly_flag_shape_and_dtype():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(contamination=0.1)
    result = ens.combine(scores, flags)
    assert result["anomaly_flag"].shape == (200,)
    assert result["anomaly_flag"].dtype == bool


def test_score_ensemble_equal_weights_sum_to_one():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(weighting="equal", contamination=0.1)
    result = ens.combine(scores, flags)
    weight_sum = sum(result["weights"].values())
    assert abs(weight_sum - 1.0) < 1e-9


def test_score_ensemble_contamination_respected():
    scores, flags = _make_scores_flags(n=1000)
    rate = 0.1
    ens = ScoreEnsemble(contamination=rate)
    result = ens.combine(scores, flags)
    assert result["contamination_used"] == pytest.approx(rate)
    assert result["is_auto_contamination"] is False
    # Flagged fraction should be approx contamination rate (within quantile rounding)
    flagged_rate = result["anomaly_flag"].mean()
    assert abs(flagged_rate - rate) < 0.03


# ---------------------------------------------------------------------------
# ScoreEnsemble: combination strategies
# ---------------------------------------------------------------------------


def test_composite_is_weighted_average():
    n = 100
    a = np.ones(n) * 0.3
    b = np.ones(n) * 0.7
    ens = ScoreEnsemble(weighting="equal", combination="composite", contamination=0.5)
    result = ens.combine({"a": a, "b": b}, {"a": a > 0.5, "b": b > 0.5})
    np.testing.assert_allclose(result["combined_score"], 0.5, atol=1e-9)


def test_intersection_is_min():
    n = 100
    a = np.linspace(0.1, 0.9, n)
    b = np.linspace(0.9, 0.1, n)
    ens = ScoreEnsemble(combination="intersection", contamination=0.5)
    result = ens.combine({"a": a, "b": b}, {"a": a > 0.5, "b": b > 0.5})
    expected = np.minimum(a, b)
    np.testing.assert_allclose(result["combined_score"], expected)


def test_union_is_max():
    n = 100
    a = np.linspace(0.1, 0.9, n)
    b = np.linspace(0.9, 0.1, n)
    ens = ScoreEnsemble(combination="union", contamination=0.5)
    result = ens.combine({"a": a, "b": b}, {"a": a > 0.5, "b": b > 0.5})
    expected = np.maximum(a, b)
    np.testing.assert_allclose(result["combined_score"], expected)


# ---------------------------------------------------------------------------
# ScoreEnsemble: weighting strategies
# ---------------------------------------------------------------------------


def test_manual_weights_normalised():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_a": 3.0, "det_b": 1.0})
    result = ens.combine(scores, flags)
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9
    assert result["weights"]["det_a"] == pytest.approx(0.75)
    assert result["weights"]["det_b"] == pytest.approx(0.25)


def test_manual_weights_all_zero_fallback():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_a": 0.0, "det_b": 0.0})
    result = ens.combine(scores, flags)
    # Falls back to equal weights
    assert abs(result["weights"]["det_a"] - 0.5) < 1e-9


def test_agreement_weights_normalised():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(weighting="agreement", combination="composite", contamination=0.1)
    result = ens.combine(scores, flags)
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9


def test_agreement_downweights_the_ranking_outlier():
    """Two detectors that rank alike keep most of the weight; a noise detector gets little."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1.0, 300)
    scores = {
        "det_a": base,
        "det_b": np.clip(base + rng.normal(0, 0.03, 300), 0, 1),  # agrees with det_a
        "det_noise": rng.uniform(0.0, 1.0, 300),  # independent ranking
    }
    flags = {k: v > 0.9 for k, v in scores.items()}
    ens = ScoreEnsemble(weighting="agreement", combination="composite", contamination=0.05)
    w = ens.combine(scores, flags)["weights"]

    assert w["det_noise"] < w["det_a"]
    assert w["det_noise"] < w["det_b"]
    assert w["det_a"] + w["det_b"] > 0.8


def test_agreement_gives_a_detector_that_flags_nothing_no_weight():
    """A flat/constant-score detector carries no ranking signal -> weight ~0 (was: highest)."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0.0, 1.0, 200)
    scores = {
        "det_a": a,
        "det_b": np.clip(a + rng.normal(0, 0.02, 200), 0, 1),
        "det_flat": np.full(200, 0.5),  # flags nothing, constant score
    }
    flags = {"det_a": a > 0.9, "det_b": a > 0.9, "det_flat": np.zeros(200, dtype=bool)}
    ens = ScoreEnsemble(weighting="agreement", combination="composite", contamination=0.05)
    w = ens.combine(scores, flags)["weights"]

    assert w["det_flat"] == pytest.approx(0.0, abs=1e-9)
    assert w["det_a"] > 0.3
    assert w["det_b"] > 0.3


def test_agreement_falls_back_to_equal_when_nothing_correlates():
    # Two anti-correlated detectors: each rho is negative -> clamped to 0 ->
    # total 0 -> equal fallback. (Two members, so the bad-member guard, which
    # needs >= 3, does not fire.)
    x = np.linspace(0.0, 1.0, 400)
    scores = {"a": x.copy(), "b": 1.0 - x}
    flags = {k: v > 0.98 for k, v in scores.items()}
    ens = ScoreEnsemble(weighting="agreement", combination="composite", contamination=0.05)
    w = ens.combine(scores, flags)["weights"]
    assert w == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_weighting_ignored_with_non_composite_combination_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="sorethumb.scoring.combine"):
        ScoreEnsemble(weighting="agreement", combination="intersection")
    assert any("no effect" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ScoreEnsemble: bad-member guard
# ---------------------------------------------------------------------------


def _three_dets(n=300, *, third):
    rng = np.random.default_rng(0)
    base = np.linspace(0.0, 1.0, n)
    rng.shuffle(base)
    a = base
    b = np.clip(base + rng.normal(0, 0.02, n), 0, 1)  # agrees with a
    return {"a": a, "b": b, "c": third(base, rng)}


@pytest.mark.parametrize("combination", ["composite", "intersection", "union"])
def test_guard_drops_member_anticorrelated_with_the_median(combination, caplog):
    import logging

    scores = _three_dets(third=lambda base, _rng: 1.0 - base)  # c ranks opposite a & b
    flags = {k: v > 0.9 for k, v in scores.items()}
    ens = ScoreEnsemble(weighting="agreement", combination=combination, contamination=0.1)

    with caplog.at_level(logging.WARNING, logger="sorethumb.scoring.combine"):
        result = ens.combine(scores, flags)

    assert result["dropped_members"] == ["c"]
    assert result["weights"]["c"] == 0.0
    assert set(result["weights"]) == {"a", "b", "c"}  # dropped member still listed
    assert any("consensus median" in r.message for r in caplog.records)


def test_guard_keeps_a_diverse_uncorrelated_member():
    scores = _three_dets(third=lambda base, rng: rng.uniform(0.0, 1.0, len(base)))  # independent
    flags = {k: v > 0.9 for k, v in scores.items()}
    result = ScoreEnsemble(combination="composite", contamination=0.1).combine(scores, flags)
    assert result["dropped_members"] == []  # uncorrelated != anti-correlated


def test_guard_needs_three_members():
    x = np.linspace(0.0, 1.0, 300)
    scores = {"a": x.copy(), "b": 1.0 - x}  # anti-correlated but only two members
    flags = {k: v > 0.9 for k, v in scores.items()}
    result = ScoreEnsemble(combination="composite", contamination=0.1).combine(scores, flags)
    assert result["dropped_members"] == []


def test_guard_wont_gut_the_ensemble():
    # a is anti to b&c; b is anti to a&c; c is anti to a&b -> would drop >= k-1.
    # The guard keeps everything instead (the consensus is unreliable).
    x = np.linspace(0.0, 1.0, 300)
    scores = {"a": x.copy(), "b": 1.0 - x, "c": np.r_[1.0 - x[: len(x) // 2], x[len(x) // 2 :]]}
    flags = {k: v > 0.9 for k, v in scores.items()}
    result = ScoreEnsemble(combination="composite", contamination=0.1).combine(scores, flags)
    assert result["dropped_members"] == []
    assert set(result["weights"]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# ScoreEnsemble: auto contamination
# ---------------------------------------------------------------------------


def test_auto_contamination_is_marked():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(contamination="auto")
    result = ens.combine(scores, flags)
    assert result["is_auto_contamination"] is True


def test_auto_contamination_within_range():
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(contamination="auto")
    result = ens.combine(scores, flags)
    assert 0.0 < result["contamination_used"] <= 0.5


# ---------------------------------------------------------------------------
# ScoreEnsemble: empty scores
# ---------------------------------------------------------------------------


def test_score_ensemble_empty_dict_raises():
    ens = ScoreEnsemble(contamination=0.1)
    with pytest.raises(ValueError, match="empty"):
        ens.combine({}, {})


# ---------------------------------------------------------------------------
# ScoreEnsemble: single detector
# ---------------------------------------------------------------------------


def test_single_detector_equal_weight():
    n = 100
    a = np.linspace(0.0, 1.0, n)
    ens = ScoreEnsemble(combination="composite", contamination=0.2)
    result = ens.combine({"only": a}, {"only": a > 0.8})
    np.testing.assert_allclose(result["combined_score"], a)
    assert result["weights"]["only"] == pytest.approx(1.0)
