"""Unit tests for M3: scoring (calibration and combination)."""

import warnings

import numpy as np
import pytest

from sorethumb.errors import AntiCorrelatedMemberWarning
from sorethumb.scoring.calibrate import Calibrator
from sorethumb.scoring.combine import ScoreEnsemble

pytestmark = pytest.mark.unit

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
    # Exact, not approximate: round-tripping the (possibly compressed)
    # weighted reference through JSON-friendly lists loses no information.
    test = np.linspace(0.0, 1.0, 50)
    np.testing.assert_array_equal(c.transform(test), c2.transform(test))


def test_calibrator_to_dict_schema_version():
    c = Calibrator()
    c.fit(np.arange(10, dtype=float))
    assert c.to_dict()["schema_version"] == 2


def test_calibrator_from_dict_migrates_legacy_quantile_values_format():
    """A dict from before schema_version existed -- a flat quantile_values
    array, no schema_version key -- must still load and transform, not
    crash. Exactness isn't expected here (that's the pre-migration bug this
    schema fixes), only a working, bounded-[0, 1] calibrator."""
    legacy = {"quantile_values": list(np.linspace(-3.0, 3.0, 10_000))}
    c = Calibrator.from_dict(legacy)
    result = c.transform(np.linspace(-3.0, 3.0, 20))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_calibrator_from_dict_ignores_legacy_mode_key_and_migrates():
    """An even older dict shape: 'mode' plus quantile_values, no schema_version."""
    legacy = {"mode": "reference", "quantile_values": [1.0, 2.0, 2.0, 3.0]}
    c = Calibrator.from_dict(legacy)
    result = c.transform(np.array([2.0]))
    assert 0.0 <= result[0] <= 1.0


# ---------------------------------------------------------------------------
# Calibrator: exact tie-aware empirical mid-rank CDF (P2-2)
# ---------------------------------------------------------------------------


def _brute_force_midrank(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Independent, unoptimised computation of 1 - mid_rank_cdf, straight
    from the documented formula, to check the real implementation against."""
    ref = np.asarray(ref, dtype=np.float64)
    out = np.empty(len(query), dtype=np.float64)
    for i, s in enumerate(query):
        below = float(np.sum(ref < s))
        equal = float(np.sum(ref == s))
        out[i] = 1.0 - (below + 0.5 * equal) / len(ref)
    return out


@pytest.mark.parametrize(
    "ref",
    [
        np.array([1.0, 2.0]),  # n=2, distinct (smallest non-constant reference)
        np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]),  # heavy ties, small n
        np.concatenate([np.zeros(37), np.ones(3)]),  # extreme skew
    ],
    ids=["n2_distinct", "heavy_ties", "extreme_skew"],
)
def test_calibrator_matches_brute_force_midrank_cdf_for_small_and_tied_references(ref):
    """The whole point of this schema: fit() must no longer resample the
    reference through np.quantile interpolation, which smears ties and
    invents values that were never in the data. For any n at or under
    _EXACT_MAX_UNIQUE this must be the *exact* documented formula, not an
    approximation of it. (A fully constant reference is intentionally
    excluded here -- see test_calibrator_constant_scores_returns_half: it
    carries no ranking information, so it deliberately always reads 0.5
    rather than following the raw mid-rank formula's arbitrary direction.)"""
    c = Calibrator()
    c.fit(ref)
    query = np.concatenate([ref, [ref.min() - 1.0, ref.max() + 1.0]])
    np.testing.assert_array_equal(c.transform(query), _brute_force_midrank(ref, query))


def test_calibrator_single_value_reference_is_constant_guard():
    c = Calibrator()
    c.fit(np.array([1.0]))
    result = c.transform(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])


def test_calibrator_exact_reference_reports_zero_calibration_error():
    c = Calibrator()
    c.fit(np.random.default_rng(0).standard_normal(5_000))
    assert c._max_calibration_error == 0.0


def test_calibrator_compresses_large_distinct_reference_and_bounds_error():
    """A reference with more distinct values than _EXACT_MAX_UNIQUE must be
    compressed (never silently truncated or resampled through
    interpolation), and the reported max_calibration_error must actually
    bound the observed deviation from the true (uncompressed) mid-rank CDF."""
    from sorethumb.scoring.calibrate import _EXACT_MAX_UNIQUE

    rng = np.random.default_rng(3)
    ref = rng.standard_normal(_EXACT_MAX_UNIQUE + 20_000)  # all but certainly distinct

    c = Calibrator()
    c.fit(ref)
    assert c._max_calibration_error is not None
    assert 0.0 < c._max_calibration_error < 0.01

    query = rng.standard_normal(200)
    approx = c.transform(query)
    exact = _brute_force_midrank(ref, query)
    assert np.max(np.abs(approx - exact)) <= c._max_calibration_error + 1e-12


def test_calibrator_from_dict_unfitted():
    c = Calibrator()
    d = c.to_dict()
    c2 = Calibrator.from_dict(d)
    assert c2._ref_values is None


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
    from sorethumb.errors import ZeroAnomalyWarning

    n = 100
    a = np.linspace(0.1, 0.9, n)
    b = np.linspace(0.9, 0.1, n)
    ens = ScoreEnsemble(combination="intersection", contamination=0.5)
    # a and b are anti-correlated by construction, so their top-50% sets
    # never overlap -- a genuinely empty intersection (see the ZeroAnomalyWarning
    # tests below), not what this test is checking; it only cares about
    # combined_score.
    with pytest.warns(ZeroAnomalyWarning):
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
# ScoreEnsemble: exact-k numeric contamination under heavy ties (P2-3)
# ---------------------------------------------------------------------------


def test_exact_k_flags_breaks_ties_by_earliest_row_order():
    from sorethumb.scoring.combine import _exact_k_flags

    scores = np.array([5.0, 5.0, 5.0, 4.0, 3.0, 3.0, 2.0])
    # k=2 falls inside the 3-way tie for the top score; the two EARLIEST
    # tied rows (indices 0, 1) must win -- not any other pair.
    flags = _exact_k_flags(scores, 2 / 7)
    assert flags.tolist() == [True, True, False, False, False, False, False]


def test_exact_k_flags_flags_whole_tie_group_when_k_matches_its_size():
    from sorethumb.scoring.combine import _exact_k_flags

    scores = np.array([5.0, 5.0, 5.0, 4.0, 3.0, 3.0, 2.0])
    flags = _exact_k_flags(scores, 3 / 7)
    assert flags.tolist() == [True, True, True, False, False, False, False]


def test_exact_k_flags_zero_and_full_contamination():
    from sorethumb.scoring.combine import _exact_k_flags

    scores = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_exact_k_flags(scores, 0.0), [False, False, False])
    np.testing.assert_array_equal(_exact_k_flags(scores, 1.0), [True, True, True])


def test_composite_exact_k_with_heavy_ties_at_boundary():
    """8 rows tied at the top score; contamination targets k=5, which falls
    inside that tie group. A quantile threshold would flag all 8 (its own
    value satisfies `>= threshold`); exact-k must flag precisely 5."""
    n = 20
    scores_a = np.array([1.0] * 8 + [0.0] * 12)
    ens = ScoreEnsemble(weighting="equal", combination="composite", contamination=5 / n)
    result = ens.combine({"a": scores_a}, {"a": scores_a > 0.5})
    assert result["anomaly_flag"].sum() == 5
    assert result["anomaly_flag"][:5].all()
    assert not result["anomaly_flag"][5:].any()


def test_intersection_exact_k_with_heavy_ties_per_detector():
    n = 20
    a = np.array([1.0] * 8 + [0.0] * 12)  # 8-way tie at the top
    b = np.array([1.0] * 5 + [0.0] * 15)  # exactly 5, no tie ambiguity
    ens = ScoreEnsemble(combination="intersection", contamination=5 / n)
    result = ens.combine({"a": a, "b": b}, {"a": a > 0.5, "b": b > 0.5})
    # a's exact-5 selection (earliest of its 8 tied rows) and b's exact-5
    # selection are the same 5 rows by construction, so the intersection
    # is exactly those 5 -- not 8 (what a naive quantile threshold on a
    # would have flagged).
    assert result["anomaly_flag"].sum() == 5
    assert result["anomaly_flag"][:5].all()
    assert not result["anomaly_flag"][5:].any()


def test_union_exact_k_with_heavy_ties_per_detector():
    n = 20
    a = np.array([1.0] * 5 + [0.0] * 15)
    b = np.concatenate([np.zeros(5), np.ones(5), np.zeros(10)])  # a disjoint set of 5
    ens = ScoreEnsemble(combination="union", contamination=5 / n)
    result = ens.combine({"a": a, "b": b}, {"a": a > 0.5, "b": b > 0.5})
    assert result["anomaly_flag"].sum() == 10  # two disjoint exact-5 sets
    assert result["anomaly_flag"][:10].all()
    assert not result["anomaly_flag"][10:].any()


def test_three_way_intersection_requires_all_three_votes():
    """P0-3: a genuine three-way intersection must be the AND of all three
    detectors' votes, strictly smaller than every two-way intersection of its
    members -- proving all three actually matter, not just two of them
    happening to agree while a third is along for the ride.

    Rows are laid out as the seven regions of a three-set Venn diagram: an
    "only me" region per detector, an "exactly these two" region per pair,
    and one "all three" region. Each pairwise intersection then strictly
    contains the three-way intersection (its own "all three" region plus one
    "exactly these two" region the third detector never voted for) -- so
    dropping any one detector's vote would visibly change the result.
    """
    idx = np.arange(210)
    only_a = (idx >= 0) & (idx < 30)
    only_b = (idx >= 30) & (idx < 60)
    only_c = (idx >= 60) & (idx < 90)
    ab_only = (idx >= 90) & (idx < 120)
    ac_only = (idx >= 120) & (idx < 150)
    bc_only = (idx >= 150) & (idx < 180)
    all_three = (idx >= 180) & (idx < 210)

    flag_a = only_a | ab_only | ac_only | all_three
    flag_b = only_b | ab_only | bc_only | all_three
    flag_c = only_c | ac_only | bc_only | all_three

    scores = {"a": flag_a.astype(float), "b": flag_b.astype(float), "c": flag_c.astype(float)}
    flags = {"a": flag_a, "b": flag_b, "c": flag_c}

    ens = ScoreEnsemble(combination="intersection", contamination="auto")
    result = ens.combine(scores, flags)

    expected_ab = flag_a & flag_b  # ab_only + all_three
    expected_ac = flag_a & flag_c  # ac_only + all_three
    expected_bc = flag_b & flag_c  # bc_only + all_three

    np.testing.assert_array_equal(result["anomaly_flag"], all_three)
    # If any single vote had been dropped, the result would match a two-way
    # intersection instead -- prove it does not, for every pair.
    assert not np.array_equal(result["anomaly_flag"], expected_ab)
    assert not np.array_equal(result["anomaly_flag"], expected_ac)
    assert not np.array_equal(result["anomaly_flag"], expected_bc)
    assert int(result["anomaly_flag"].sum()) < int(expected_ab.sum())
    assert int(result["anomaly_flag"].sum()) < int(expected_ac.sum())
    assert int(result["anomaly_flag"].sum()) < int(expected_bc.sum())


# ---------------------------------------------------------------------------
# ScoreEnsemble: ZeroAnomalyWarning on an empty intersection (P2-4)
# ---------------------------------------------------------------------------


def test_zero_anomaly_warning_on_deterministic_empty_three_way_intersection():
    """Three detectors whose flagged sets are pairwise disjoint by
    construction (no 'all three' region, unlike the Venn-diagram test above)
    -- the three-way intersection is deterministically empty, a legitimate
    outcome that must be surfaced, not silently reported as zero anomalies."""
    from sorethumb.errors import ZeroAnomalyWarning

    idx = np.arange(90)
    flag_a = (idx >= 0) & (idx < 10)
    flag_b = (idx >= 10) & (idx < 20)
    flag_c = (idx >= 20) & (idx < 30)
    # No row is in more than one of these -- every pairwise intersection,
    # and therefore the three-way intersection, is empty.

    scores = {"a": flag_a.astype(float), "b": flag_b.astype(float), "c": flag_c.astype(float)}
    flags = {"a": flag_a, "b": flag_b, "c": flag_c}

    ens = ScoreEnsemble(combination="intersection", contamination="auto")
    with pytest.warns(ZeroAnomalyWarning) as record:
        result = ens.combine(scores, flags)

    assert result["anomaly_flag"].sum() == 0
    msg = str(record[0].message)
    assert "a" in msg
    assert "b" in msg
    assert "c" in msg
    assert "zero rows" in msg
    # Realised per-detector rates must be in the message, not just "it's empty".
    for name in ("a", "b", "c"):
        rate = 100.0 * flags[name].mean()
        assert f"{rate:.2f}%" in msg


def test_zero_anomaly_warning_not_raised_when_intersection_is_nonempty():
    """Sanity check: the warning is specific to the empty case, not raised on
    every intersection combine()."""
    n = 100
    a = np.linspace(0.0, 1.0, n)
    b = np.linspace(0.0, 1.0, n)  # perfectly correlated with a -- full overlap
    ens = ScoreEnsemble(combination="intersection", contamination=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ens.combine({"a": a, "b": b}, {"a": a > 0.8, "b": b > 0.8})
    assert result["anomaly_flag"].sum() > 0


def test_zero_anomaly_warning_not_raised_for_union():
    """The warning is intersection-specific; union's own docstring/design
    makes an all-empty result far less likely, and the plan scopes this
    warning to intersection."""
    n = 90
    idx = np.arange(n)
    flag_a = (idx >= 0) & (idx < 10)
    flag_b = (idx >= 10) & (idx < 20)
    scores = {"a": flag_a.astype(float), "b": flag_b.astype(float)}
    flags = {"a": flag_a, "b": flag_b}
    ens = ScoreEnsemble(combination="union", contamination="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ens.combine(scores, flags)  # must not raise
    assert result["anomaly_flag"].sum() == 20


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


def test_manual_weights_all_zero_rejected_at_construction():
    """P2-3: a non-positive-sum manual_weights dict must fail loudly at
    construction, not silently fall back to equal weights at combine time."""
    with pytest.raises(ValueError, match="positive total"):
        ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_a": 0.0, "det_b": 0.0})


def test_manual_weights_negative_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_a": -1.0, "det_b": 2.0})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_manual_weights_non_finite_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_a": bad, "det_b": 1.0})


def test_manual_weights_falls_back_when_combine_subset_sums_to_zero():
    """manual_weights is valid at construction, but combine() is called with
    detectors that aren't in it -- .get(name, 0.0) makes the *relevant*
    subset sum to zero for this call. That still falls back to equal
    weights with a warning; it's a different case from an invalid dict."""
    scores, flags = _make_scores_flags()
    ens = ScoreEnsemble(weighting="manual", contamination=0.1, manual_weights={"det_c": 1.0})
    result = ens.combine(scores, flags)  # scores has det_a/det_b, not det_c
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


def test_guard_drops_member_anticorrelated_with_the_median_in_composite(caplog):
    import logging

    scores = _three_dets(third=lambda base, _rng: 1.0 - base)  # c ranks opposite a & b
    flags = {k: v > 0.9 for k, v in scores.items()}
    ens = ScoreEnsemble(weighting="agreement", combination="composite", contamination=0.1)

    with caplog.at_level(logging.WARNING, logger="sorethumb.scoring.combine"):
        result = ens.combine(scores, flags)

    assert result["dropped_members"] == ["c"]
    assert result["weights"]["c"] == 0.0
    assert set(result["weights"]) == {"a", "b", "c"}  # dropped member still listed
    assert any("consensus median" in r.message for r in caplog.records)


@pytest.mark.parametrize("combination", ["intersection", "union"])
def test_guard_keeps_every_vote_for_set_combinations(combination):
    """P0-3: dropping a member from a vote silently changes the vote count --
    a configured three-way intersection/union must stay three-way even when
    one detector ranks anti-correlated with the others. The guard must warn,
    not drop.
    """
    scores = _three_dets(third=lambda base, _rng: 1.0 - base)  # c ranks opposite a & b
    flags = {k: v > 0.9 for k, v in scores.items()}
    ens = ScoreEnsemble(combination=combination, contamination=0.1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ens.combine(scores, flags)

    assert result["dropped_members"] == []
    assert set(result["weights"]) == {"a", "b", "c"}
    assert any(issubclass(w.category, AntiCorrelatedMemberWarning) for w in caught)
    assert any("consensus median" in str(w.message) for w in caught)


@pytest.mark.parametrize("combination", ["intersection", "union"])
def test_guard_warning_is_promoted_to_error_under_strict_semantics(combination):
    """run.strict promotes every SorethumbWarning to an error (see
    _pipeline._execute_group); AntiCorrelatedMemberWarning must be a real
    SorethumbWarning so that mechanism catches it too.
    """
    scores = _three_dets(third=lambda base, _rng: 1.0 - base)
    flags = {k: v > 0.9 for k, v in scores.items()}
    ens = ScoreEnsemble(combination=combination, contamination=0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AntiCorrelatedMemberWarning)
        with pytest.raises(AntiCorrelatedMemberWarning):
            ens.combine(scores, flags)


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
