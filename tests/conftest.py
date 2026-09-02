"""Shared pytest fixtures."""

import polars as pl
import pytest

from tests.synth import make_frame


@pytest.fixture
def simple_frame() -> pl.DataFrame:
    """A plain numeric frame with no special columns."""
    df, _ = make_frame(n_rows=300, seed=0)
    return df


@pytest.fixture
def frame_with_anomalies() -> tuple[pl.DataFrame, list[int]]:
    """Frame with 10 injected point anomalies; returns (df, anomaly_row_indices)."""
    return make_frame(n_rows=300, seed=1, n_anomalies=10)
