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
    c = Calibrator(mode="self")
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


def test_calibrator_self_mode():
    c = Calibrator(mode="self")
    scores = _uniform_scores()
    c.fit(scores)
    # After fitting on uniform, the median score should calibrate near 0.5
    median_score = np.median(scores)
    result = c.transform(np.array([median_score]))
    assert 0.3 < result[0] < 0.7


def test_calibrator_reference_mode():
    c = Calibrator(mode="reference")
    train_scores = _uniform_scores(seed=0)
    ref_scores = np.linspace(0.0, 1.0, 1000)
    c.fit(train_scores, reference_scores=ref_scores)
    # Transform a train score
    result = c.transform(np.array([0.5]))
    assert 0.0 <= result[0] <= 1.0


def test_calibrator_reference_mode_fallback_when_no_reference(caplog):
    c = Calibrator(mode="reference")
    scores = _uniform_scores()
    import logging

    with caplog.at_level(logging.WARNING):
        c.fit(scores, reference_scores=None)
    assert "fallback" in caplog.text.lower() or "reference_scores" in caplog.text


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


def test_calibrator_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        Calibrator(mode="bad")


# ---------------------------------------------------------------------------
# Calibrator: to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


def test_calibrator_to_dict_from_dict_roundtrip():
    c = Calibrator(mode="self")
    scores = _uniform_scores()
    c.fit(scores)
    d = c.to_dict()
    c2 = Calibrator.from_dict(d)
    # Transform with both should give same result
    test = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(c.transform(test), c2.transform(test), rtol=1e-6)


def test_calibrator_from_dict_unfitted():
    c = Calibrator(mode="reference")
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
    assert "combined_score" in result
    assert "anomaly_flag" in result
    assert "threshold" in result
    assert "contamination_used" in result
    assert "weights" in result
    assert "is_auto_contamination" in result


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
    ens = ScoreEnsemble(weighting="agreement", contamination=0.1)
    result = ens.combine(scores, flags)
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9


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
