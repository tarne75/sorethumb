"""Unit tests for scripts.validation.data (pure DataFrame transforms)."""

from __future__ import annotations

import polars as pl
import pytest
from scripts.validation.data import (
    SPLIT_ROW_ID_COLUMN,
    label_to_anomaly_array,
    split_train_holdout,
    stamp_row_id,
)

pytestmark = pytest.mark.unit


def _df(n: int) -> pl.DataFrame:
    return pl.DataFrame({"x": list(range(n))})


def test_split_train_holdout_deterministic():
    df = _df(100)
    t1, h1 = split_train_holdout(df, 0.3, seed=7)
    t2, h2 = split_train_holdout(df, 0.3, seed=7)
    assert t1["x"].to_list() == t2["x"].to_list()
    assert h1["x"].to_list() == h2["x"].to_list()


def test_split_train_holdout_different_seeds_differ():
    df = _df(200)
    _, h1 = split_train_holdout(df, 0.3, seed=1)
    _, h2 = split_train_holdout(df, 0.3, seed=2)
    assert h1["x"].to_list() != h2["x"].to_list()


def test_split_train_holdout_respects_fraction():
    df = _df(100)
    train, holdout = split_train_holdout(df, 0.3, seed=0)
    assert len(holdout) == 30
    assert len(train) == 70


def test_split_train_holdout_disjoint_and_covers_all_rows():
    df = _df(50)
    train, holdout = split_train_holdout(df, 0.4, seed=3)
    train_set = set(train["x"].to_list())
    holdout_set = set(holdout["x"].to_list())
    assert train_set & holdout_set == set()
    assert train_set | holdout_set == set(range(50))


def test_split_train_holdout_minimum_two_rows_both_non_empty():
    df = _df(2)
    train, holdout = split_train_holdout(df, 0.99, seed=0)
    assert len(train) == 1
    assert len(holdout) == 1


def test_split_train_holdout_single_row_returns_empty_holdout():
    df = _df(1)
    train, holdout = split_train_holdout(df, 0.3, seed=0)
    assert len(train) == 1
    assert len(holdout) == 0


def test_stamp_row_id_adds_positional_range():
    df = pl.DataFrame({"a": ["x", "y", "z"]})
    stamped = stamp_row_id(df)
    assert stamped[SPLIT_ROW_ID_COLUMN].to_list() == [0, 1, 2]


def test_stamp_row_id_custom_column_name():
    df = pl.DataFrame({"a": ["x", "y"]})
    stamped = stamp_row_id(df, column="my_id")
    assert stamped["my_id"].to_list() == [0, 1]


def test_label_to_anomaly_array_maps_to_0_1():
    labels = pl.Series(["normal.", "attack.", "normal.", "attack."])
    arr = label_to_anomaly_array(labels, lambda s: s != "normal.")
    assert arr.tolist() == [0, 1, 0, 1]
