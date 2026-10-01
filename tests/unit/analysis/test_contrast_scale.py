"""Cohort contrast ranks numeric and categorical columns on one bounded scale."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sorethumb_ml.analysis.contrast import compute_contrast

pytestmark = pytest.mark.unit


def _frames(n_flagged: int = 60, n_rest: int = 600, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    flagged = pl.DataFrame(
        {
            "strong_num": rng.normal(6.0, 1.0, n_flagged),  # six sigma away from the rest
            "weak_num": rng.normal(0.02, 1.0, n_flagged),  # indistinguishable
            # weak categorical: 55/45 vs 50/50
            "weak_cat": ["A"] * int(n_flagged * 0.55) + ["B"] * (n_flagged - int(n_flagged * 0.55)),
        }
    )
    unflagged = pl.DataFrame(
        {
            "strong_num": rng.normal(0.0, 1.0, n_rest),
            "weak_num": rng.normal(0.0, 1.0, n_rest),
            "weak_cat": ["A"] * (n_rest // 2) + ["B"] * (n_rest - n_rest // 2),
        }
    )
    return flagged, unflagged


def _scores(df: pl.DataFrame) -> dict[str, float]:
    return dict(zip(df["feature"].to_list(), df["contrast_score"].to_list(), strict=True))


def test_strong_numeric_separation_outranks_weak_categorical_separation() -> None:
    flagged, unflagged = _frames()
    df = compute_contrast(flagged, unflagged, ["strong_num"], ["weak_cat"])
    assert df["feature"].to_list()[0] == "strong_num"
    s = _scores(df)
    assert s["strong_num"] > 0.9 > 0.2 > s["weak_cat"]


def test_strong_numeric_outranks_a_weak_categorical_even_with_a_rare_category() -> None:
    """The old raw-count lift substitute scored a category seen only in the flagged cohort as
    ``freq * n_unflagged`` -- tiny categorical evidence beat any numeric score once the data grew."""
    flagged, unflagged = _frames(n_rest=50_000)
    flagged = flagged.with_columns(
        pl.Series("rare_cat", ["RARE"] * 2 + ["common"] * (flagged.height - 2))  # 2 of 60 rows
    )
    unflagged = unflagged.with_columns(pl.lit("common").alias("rare_cat"))
    df = compute_contrast(flagged, unflagged, ["strong_num"], ["rare_cat"])
    s = _scores(df)
    assert s["strong_num"] > s["rare_cat"]
    assert s["rare_cat"] == pytest.approx(2 / 60)


def test_strong_categorical_separation_outranks_weak_numeric_separation() -> None:
    flagged, unflagged = _frames()
    flagged = flagged.with_columns(pl.lit("X").alias("sep_cat"))
    unflagged = unflagged.with_columns(pl.Series("sep_cat", ["A", "B"] * (unflagged.height // 2)))
    df = compute_contrast(flagged, unflagged, ["weak_num"], ["sep_cat"])
    s = _scores(df)
    assert df["feature"].to_list()[0] == "sep_cat"
    assert s["sep_cat"] == pytest.approx(1.0)
    assert s["weak_num"] < 0.3


def test_every_score_is_bounded_in_unit_interval_for_both_kinds() -> None:
    flagged, unflagged = _frames()
    flagged = flagged.with_columns(pl.lit("only_flagged").alias("c"))
    unflagged = unflagged.with_columns(pl.lit("only_rest").alias("c"))
    df = compute_contrast(flagged, unflagged, ["strong_num", "weak_num"], ["weak_cat", "c"], top_n=10)
    assert df.height == 4
    assert df["contrast_score"].is_between(0.0, 1.0).all()
    assert df["stat_value"].is_between(0.0, 1.0).all()


def test_categorical_score_does_not_depend_on_the_size_of_the_unflagged_cohort() -> None:
    flagged = pl.DataFrame({"c": ["new"] * 10 + ["old"] * 10})
    small = pl.DataFrame({"c": ["old"] * 100})
    large = pl.DataFrame({"c": ["old"] * 100_000})
    a = _scores(compute_contrast(flagged, small, [], ["c"]))["c"]
    b = _scores(compute_contrast(flagged, large, [], ["c"]))["c"]
    assert a == pytest.approx(b) == pytest.approx(0.5)


def test_categorical_score_is_total_variation_distance() -> None:
    flagged = pl.DataFrame({"c": ["a"] * 6 + ["b"] * 3 + ["c"] * 1})  # .6 .3 .1
    rest = pl.DataFrame({"c": ["a"] * 2 + ["b"] * 4 + ["d"] * 4})  # .2 .4 0 .4
    row = compute_contrast(flagged, rest, [], ["c"]).row(0, named=True)
    assert row["stat_name"] == "total_variation"
    assert row["contrast_score"] == pytest.approx(0.5 * (0.4 + 0.1 + 0.1 + 0.4))


def test_identical_distributions_score_zero_for_both_kinds() -> None:
    f = pl.DataFrame({"n": [1.0, 2.0, 3.0, 4.0] * 5, "c": ["a", "b"] * 10})
    df = compute_contrast(f, f, ["n"], ["c"])
    assert df["contrast_score"].to_list() == [0.0, 0.0]


def test_numeric_score_is_the_ks_statistic() -> None:
    flagged = pl.DataFrame({"n": [10.0, 11.0, 12.0]})
    rest = pl.DataFrame({"n": [0.0, 1.0, 2.0, 3.0]})
    row = compute_contrast(flagged, rest, ["n"], []).row(0, named=True)
    assert row["stat_name"] == "ks_statistic"
    assert row["contrast_score"] == pytest.approx(1.0)  # disjoint supports
