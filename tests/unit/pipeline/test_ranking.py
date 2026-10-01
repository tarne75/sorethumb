"""Unit tests for P0-3: dense ranking must stay in lockstep with anomaly_flag.

``_flagged_idx_by_score_desc`` is the single source of truth _finalize_group
uses for both attribution ordering and the persisted ``rank`` column. See
src/sorethumb_ml/_pipeline.py for the full rationale: composite_score is not
necessarily the statistic that determined anomaly_flag (true for
combination="intersection"/"union", where the flag is a per-detector vote),
so a rank derived from a fresh global sort of composite_score can assign a
positive rank to a row the vote never flagged, and leave an actually-flagged
row at 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb_ml._pipeline import _flagged_idx_by_score_desc

pytestmark = pytest.mark.unit


def _dense_ranks(anomaly_flag: np.ndarray, composite_score: np.ndarray) -> np.ndarray:
    """Mirror _finalize_group's rank_arr construction for direct testing."""
    flagged_idx = _flagged_idx_by_score_desc(anomaly_flag, composite_score)
    rank_arr = np.zeros(len(anomaly_flag), dtype=int)
    rank_arr[flagged_idx] = np.arange(1, len(flagged_idx) + 1)
    return rank_arr


def test_no_flagged_rows_gives_empty_and_all_zero_ranks():
    flag = np.zeros(5, dtype=bool)
    score = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    assert _flagged_idx_by_score_desc(flag, score).size == 0
    np.testing.assert_array_equal(_dense_ranks(flag, score), np.zeros(5, dtype=int))


def test_rank_is_positive_iff_flagged_when_score_and_flag_disagree():
    """The adversarial case P0-3 exists for: the globally highest-scoring
    rows are NOT the flagged ones (as happens for intersection/union, where
    anomaly_flag is a per-detector vote independent of composite_score).
    """
    # Highest composite_score rows are indices 0, 4, 2 -- none of them flagged.
    # The vote instead flagged indices 1 and 3 (low composite_score).
    composite_score = np.array([0.95, 0.10, 0.80, 0.05, 0.90])
    anomaly_flag = np.array([False, True, False, True, False])

    rank = _dense_ranks(anomaly_flag, composite_score)

    np.testing.assert_array_equal(rank > 0, anomaly_flag)
    assert rank[0] == 0  # highest score, but not flagged -- must stay 0
    assert rank[4] == 0  # second-highest score, but not flagged -- must stay 0
    assert rank[2] == 0
    assert sorted(rank[rank > 0].tolist()) == [1, 2]


def test_ranks_are_dense_1_to_n_ordered_by_score_descending():
    composite_score = np.array([0.2, 0.9, 0.1, 0.7, 0.5, 0.3])
    anomaly_flag = np.array([True, True, False, True, True, False])

    rank = _dense_ranks(anomaly_flag, composite_score)

    flagged_positions = np.where(anomaly_flag)[0]
    assert sorted(rank[flagged_positions].tolist()) == [1, 2, 3, 4]
    assert (rank[~anomaly_flag] == 0).all()

    # rank 1 must be the highest composite_score among flagged rows (index 1,
    # score 0.9); rank 4 the lowest (index 0, score 0.2).
    assert rank[1] == 1
    assert rank[0] == 4
    assert rank[3] == 2  # score 0.7, second highest among flagged
    assert rank[4] == 3  # score 0.5, third highest among flagged


def test_ties_in_composite_score_still_produce_dense_unique_ranks():
    composite_score = np.array([0.5, 0.5, 0.5])
    anomaly_flag = np.array([True, True, True])

    rank = _dense_ranks(anomaly_flag, composite_score)

    assert sorted(rank.tolist()) == [1, 2, 3]


def test_all_rows_flagged_ranks_every_row():
    n = 20
    rng = np.random.default_rng(0)
    composite_score = rng.uniform(0, 1, n)
    anomaly_flag = np.ones(n, dtype=bool)

    rank = _dense_ranks(anomaly_flag, composite_score)

    assert sorted(rank.tolist()) == list(range(1, n + 1))
    # rank 1 is the single highest-scoring row.
    assert rank[int(np.argmax(composite_score))] == 1


# ---------------------------------------------------------------------------
# P1-9: stable descending order, earliest source row wins a tie
# ---------------------------------------------------------------------------


def test_descending_order_breaks_ties_by_earliest_row_not_by_sort_accident():
    from sorethumb_ml.scoring.combine import descending_order

    scores = np.array([0.5, 0.9, 0.5, 0.5, 0.1, 0.9])
    # 0.9 twice (1, 5), then 0.5 three times (0, 2, 3), then 0.1 (4) -- earliest first within a tie.
    assert descending_order(scores).tolist() == [1, 5, 0, 2, 3, 4]
    # The previous implementation, np.argsort(s)[::-1], reversed every tie group.
    assert np.argsort(scores)[::-1].tolist() != [1, 5, 0, 2, 3, 4]


def test_descending_order_uses_source_row_not_position_when_given():
    from sorethumb_ml.scoring.combine import descending_order

    scores = np.full(5, 0.7)
    source_row = np.array([40, 30, 20, 10, 50])  # e.g. a time-sorted group: position != source order
    assert descending_order(scores, source_row).tolist() == [3, 2, 1, 0, 4]


def test_descending_order_rejects_a_misaligned_tiebreak():
    from sorethumb_ml.scoring.combine import descending_order

    with pytest.raises(ValueError, match="source_row"):
        descending_order(np.ones(3), np.arange(2))


def test_flagged_order_and_rank_follow_source_row_among_equal_scores():
    composite_score = np.array([0.5, 0.5, 0.5, 0.5, 0.2])
    anomaly_flag = np.array([True, True, True, True, False])
    source_row = np.array([13, 11, 10, 12, 99])

    flagged = _flagged_idx_by_score_desc(anomaly_flag, composite_score, source_row)
    assert flagged.tolist() == [2, 1, 3, 0]  # source rows 10, 11, 12, 13

    rank = np.zeros(5, dtype=int)
    rank[flagged] = np.arange(1, 5)
    assert rank.tolist() == [4, 2, 1, 3, 0]


def test_flagged_order_without_a_source_row_is_position_order_among_ties():
    flagged = _flagged_idx_by_score_desc(np.ones(4, dtype=bool), np.full(4, 0.3))
    assert flagged.tolist() == [0, 1, 2, 3]


def test_higher_scores_still_beat_earlier_rows():
    flagged = _flagged_idx_by_score_desc(np.ones(4, dtype=bool), np.array([0.1, 0.4, 0.4, 0.9]), np.arange(4))
    assert flagged.tolist() == [3, 1, 2, 0]


def test_order_is_independent_of_how_the_rows_are_arranged():
    """Shuffling the rows (carrying their source ids along) must not change which source row ranks where."""
    rng = np.random.default_rng(0)
    score = np.repeat([0.9, 0.5, 0.1], 20)  # heavy ties
    source = np.arange(60)
    flag = np.ones(60, dtype=bool)
    expected = source[_flagged_idx_by_score_desc(flag, score, source)]
    for _ in range(5):
        perm = rng.permutation(60)
        got_positions = _flagged_idx_by_score_desc(flag, score[perm], source[perm])
        assert source[perm][got_positions].tolist() == expected.tolist()
