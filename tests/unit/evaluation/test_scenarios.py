"""Unit tests for sorethumb.evaluate.scenarios: pure, deterministic, no model fitting."""

from __future__ import annotations

import numpy as np
import pytest

from sorethumb.evaluate.scenarios import (
    SCENARIOS,
    clustered_anomalies,
    contextual_anomalies,
    local_anomalies,
    masking_anomalies,
    point_anomalies,
    scenario_by_name,
    swamping_train_reference,
    varying_density_anomalies,
)

pytestmark = pytest.mark.unit

_GENERATORS = [
    point_anomalies,
    local_anomalies,
    contextual_anomalies,
    clustered_anomalies,
    masking_anomalies,
    varying_density_anomalies,
]


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_returns_binary_labels(generate):
    _, y = generate(0)
    assert set(np.unique(y).tolist()) <= {0, 1}


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_has_both_classes(generate):
    _, y = generate(0)
    assert y.sum() > 0
    assert y.sum() < len(y)


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_row_count_matches_label_count(generate):
    df, y = generate(0)
    assert len(df) == len(y)


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_has_numeric_and_categorical_columns(generate):
    df, _ = generate(0)
    import polars as pl

    numeric_cols = [c for c, dt in df.schema.items() if dt in (pl.Float64, pl.Int64)]
    categorical_cols = [c for c, dt in df.schema.items() if dt == pl.String]
    assert numeric_cols, "expected at least one numeric column"
    assert categorical_cols, "expected at least one categorical column"


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_deterministic_for_same_seed(generate):
    df1, y1 = generate(0)
    df2, y2 = generate(0)
    assert df1.equals(df2)
    assert np.array_equal(y1, y2)


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_differs_across_seeds(generate):
    df1, _ = generate(0)
    df2, _ = generate(1)
    assert not df1.equals(df2)


@pytest.mark.parametrize("generate", _GENERATORS)
def test_generator_has_no_nulls(generate):
    df, _ = generate(0)
    assert df.null_count().sum_horizontal().item() == 0


def test_masking_scenario_has_large_anomaly_block():
    """The masking scenario's whole premise is a contiguous block large enough
    to challenge per-point methods -- lock in that it stays > 10%."""
    _, y = masking_anomalies(0)
    assert y.mean() > 0.10


def test_contextual_anomaly_value_matches_a_different_context():
    """Every contextual anomaly's numeric signal should be far from its own
    context's mean (that's what makes it anomalous) but close to some other
    context's mean (that's what makes it a *contextual*, not point, anomaly).
    """
    df, y = contextual_anomalies(0)
    context_means = {"A": 0.0, "B": 10.0, "C": -10.0}
    anomalies = df.filter(y == 1)
    for row in anomalies.iter_rows(named=True):
        own_mean = context_means[row["context"]]
        assert abs(row["num_signal"] - own_mean) > 3.0
        other_means = [m for ctx, m in context_means.items() if ctx != row["context"]]
        assert min(abs(row["num_signal"] - m) for m in other_means) < 3.0


def test_swamping_train_reference_has_no_labelled_anomalies():
    """Swamping's contamination is unlabelled by design -- the model never
    sees ground truth for it, only a distorted 'normal' reference."""
    _, y = swamping_train_reference(0)
    assert y.sum() == 0


def test_swamping_train_reference_deterministic():
    df1, y1 = swamping_train_reference(0)
    df2, y2 = swamping_train_reference(0)
    assert df1.equals(df2)
    assert np.array_equal(y1, y2)


def test_scenarios_registry_names_are_unique():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))


def test_scenario_by_name_found():
    s = scenario_by_name("point")
    assert s is not None
    assert s.kind == "point"


def test_scenario_by_name_not_found():
    assert scenario_by_name("does_not_exist") is None


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_every_registered_scenario_review_budget_in_range(scenario):
    assert 0.0 < scenario.review_budget < 1.0
