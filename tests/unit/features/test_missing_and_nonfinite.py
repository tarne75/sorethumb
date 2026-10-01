"""Missing values are imputed before fitting/scaling; infinities fail clearly.

Contract under test:

* null and float NaN are both "missing": profiled as missing, imputed with the
  fitted centre (a documented transformation), never turned into a post-scale
  sentinel.
* +Inf / -Inf in a column that feeds the matrix raise ``PlanError`` naming the
  column -- nothing preserves their meaning, so nothing silently replaces it.
* Ranking holds across every shipped detector: a NaN row is a finite,
  ordinary-ranked row; a planted anomaly still outranks it.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from sorethumb_ml.config import (
    ColumnsConfig,
    Config,
    FeaturesConfig,
    ProfilingConfig,
    RunConfig,
    SourceConfig,
)
from sorethumb_ml.detectors import registry
from sorethumb_ml.errors import PlanError
from sorethumb_ml.features.build import apply_feature_plan, fit_features
from sorethumb_ml.features.scale import apply_scaler, fit_scaler
from sorethumb_ml.profiling.missing import float_nan_to_null
from sorethumb_ml.profiling.plan import build_feature_plan

pytestmark = pytest.mark.unit

_N = 300
_ANOMALY_ROW = 17
_NAN_ROW = 123
_SHIPPED_DETECTORS = sorted(registry)  # every built-in (third-party entry points aren't installed in CI)


def _config(*, ignore: list[str] | None = None, null_ratio_flag: float = 0.0, **features) -> Config:
    return Config(
        source=SourceConfig(uri="file://dummy"),
        columns=ColumnsConfig(ignore=ignore or []),
        profiling=ProfilingConfig(null_ratio_flag=null_ratio_flag),
        features=FeaturesConfig(correlation_reduction=False, **features),
        run=RunConfig(workdir="/tmp/test"),
    )


def _frame(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, _N)
    b = rng.normal(5.0, 2.0, _N)
    a[_ANOMALY_ROW], b[_ANOMALY_ROW] = 40.0, -30.0  # unmistakable planted anomaly
    return pl.DataFrame({"num_a": a.tolist(), "num_b": b.tolist()})


def _with(df: pl.DataFrame, column: str, row: int, value: float | None) -> pl.DataFrame:
    values = df[column].to_list()
    values[row] = value
    return df.with_columns(pl.Series(column, values, dtype=pl.Float64))


def _fit(df: pl.DataFrame, config: Config | None = None):
    config = config or _config()
    plan = build_feature_plan(df, config)
    return plan, fit_features(df, plan, config)


# ---------------------------------------------------------------------------
# NaN is missing
# ---------------------------------------------------------------------------


def test_float_nan_to_null_only_touches_float_columns():
    df = pl.DataFrame({"f": [1.0, float("nan"), float("inf")], "i": [1, 2, 3], "s": ["a", "nan", "c"]})
    out = float_nan_to_null(df)
    assert out["f"].to_list() == [1.0, None, float("inf")]  # inf is NOT normalised
    assert out["i"].to_list() == [1, 2, 3]
    assert out["s"].to_list() == ["a", "nan", "c"]  # the *string* "nan" is untouched
    assert out.schema == df.schema


def test_nan_and_null_are_equivalent_inputs():
    """A NaN must profile, impute and scale exactly like a null at the same row."""
    base = _frame()
    with_nan = _with(base, "num_a", _NAN_ROW, float("nan"))
    with_null = _with(base, "num_a", _NAN_ROW, None)
    plan_nan, space_nan = _fit(with_nan)
    plan_null, space_null = _fit(with_null)
    assert plan_nan.imputation_medians == plan_null.imputation_medians
    assert plan_nan.scaler_params == plan_null.scaler_params
    assert space_nan.feature_names == space_null.feature_names
    np.testing.assert_array_equal(space_nan.matrix, space_null.matrix)


@pytest.mark.parametrize("scaler", ["robust", "standard"])
def test_missing_value_is_imputed_with_the_fitted_centre_not_a_sentinel(scaler):
    df = _with(_frame(), "num_a", _NAN_ROW, float("nan"))
    plan, space = _fit(df, _config(scaler=scaler))
    col = space.feature_names.index("num_a")
    assert np.isfinite(space.matrix).all()
    # Imputed with the column's median -> scaled value is (median - centre) / scale,
    # i.e. the median's own scaled position, not an arbitrary constant.
    p = plan.scaler_params["num_a"]
    expected = (plan.imputation_medians["num_a"] - p["center"]) / p["scale"]
    assert space.matrix[_NAN_ROW, col] == pytest.approx(expected, abs=1e-5)


def test_fit_features_emits_no_nonfinite_warning_for_nan_input():
    df = _with(_frame(), "num_a", _NAN_ROW, float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (e.g. the old NonFiniteWarning) fails
        _fit(df)


def test_scaler_fits_on_observed_values_and_imputes_null_at_the_centre():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, None]})
    params = fit_scaler(df, ["x"], "robust")
    assert params["x"]["center"] == pytest.approx(3.0)  # median of the five observed values
    scaled = apply_scaler(df, params)["x"].to_list()
    assert scaled[-1] == 0.0  # imputed at the fitted centre
    assert all(v is not None and math.isfinite(v) for v in scaled)


def test_score_forward_imputes_new_nan_with_the_stored_median():
    df = _frame()
    plan, fitted = _fit(df)
    new = _with(df, "num_b", 5, float("nan"))
    applied = apply_feature_plan(new, plan)
    assert np.isfinite(applied.matrix).all()
    col = applied.feature_names.index("num_b")
    p = plan.scaler_params["num_b"]
    assert applied.matrix[5, col] == pytest.approx(
        (plan.imputation_medians["num_b"] - p["center"]) / p["scale"], abs=1e-5
    )


def test_nan_inside_a_list_column_becomes_missing_not_nonfinite():
    rng = np.random.default_rng(3)
    arrays = [[float(x) for x in rng.normal(0, 1, 3)] for _ in range(_N)]
    arrays[_NAN_ROW] = [float("nan"), 1.0]
    arrays[40] = []  # empty list -> null mean/min/max
    df = pl.DataFrame({"num_a": rng.normal(0, 1, _N).tolist(), "arr": arrays})
    _, space = _fit(df)
    assert np.isfinite(space.matrix).all()


# ---------------------------------------------------------------------------
# Infinity fails clearly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")], ids=["+inf", "-inf"])
def test_infinity_in_a_feature_column_raises_plan_error_naming_it(bad):
    df = _with(_frame(), "num_a", _NAN_ROW, bad)
    plan = build_feature_plan(df, _config())
    with pytest.raises(PlanError, match=r"Infinite value.*'num_a': 1 value"):
        fit_features(df, plan, _config())


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")], ids=["+inf", "-inf"])
def test_infinity_in_new_data_is_rejected_on_score_forward(bad):
    df = _frame()
    plan, _ = _fit(df)
    with pytest.raises(PlanError, match="Infinite value"):
        apply_feature_plan(_with(df, "num_b", 9, bad), plan)


def test_infinity_in_an_ignored_column_is_not_an_error():
    df = _frame().with_columns(pl.Series("junk", [float("inf")] * _N))
    _, space = _fit(df, _config(ignore=["junk"]))
    assert "junk" not in space.feature_names
    assert np.isfinite(space.matrix).all()


def test_infinity_in_a_list_column_is_rejected_via_its_derived_feature():
    rng = np.random.default_rng(4)
    arrays = [[float(x) for x in rng.normal(0, 1, 3)] for _ in range(_N)]
    arrays[7] = [1.0, float("inf")]
    df = pl.DataFrame({"num_a": rng.normal(0, 1, _N).tolist(), "arr": arrays})
    plan = build_feature_plan(df, _config())
    with pytest.raises(PlanError, match=r"'arr'.*feature 'arr__(max|mean)'"):
        fit_features(df, plan, _config())


# ---------------------------------------------------------------------------
# Ranking across every shipped detector
# ---------------------------------------------------------------------------


def _anomaly_rank(scores: np.ndarray, row: int) -> int:
    """0 = most anomalous. Scores are higher-is-more-normal."""
    return int(np.argsort(np.argsort(scores))[row])


def _scores_for(df: pl.DataFrame, name: str, config: Config) -> np.ndarray:
    _, space = _fit(df, config)
    det = registry[name]()
    det.fit(space.matrix, seed=0)
    scores = det.score_samples(space.matrix)
    assert np.isfinite(scores).all(), f"{name}: non-finite scores"
    return scores


@pytest.mark.parametrize("name", _SHIPPED_DETECTORS)
def test_imputed_nan_row_ranks_as_an_ordinary_row_below_a_planted_anomaly(name):
    """With the missing indicator off, a NaN is imputed to a typical value and the row is ordinary.

    This isolates the imputation itself: nothing about the filled-in value may
    make the row look anomalous (the old "zero after scaling" sentinel could).
    """
    df = _with(_frame(), "num_a", _NAN_ROW, float("nan"))
    scores = _scores_for(df, name, _config(null_ratio_flag=1.0))
    assert _anomaly_rank(scores, _ANOMALY_ROW) == 0, f"{name}: planted anomaly must be the most anomalous row"
    assert _anomaly_rank(scores, _NAN_ROW) >= 10, f"{name}: imputed NaN row ranked among the top anomalies"


@pytest.mark.parametrize("name", _SHIPPED_DETECTORS)
def test_nan_row_with_missing_indicator_is_finite_and_still_below_a_planted_anomaly(name):
    """Default config: an isolated missing value is itself a (rare) feature, so that row may rank
    as somewhat unusual -- by design -- but the planted anomaly must still rank first."""
    df = _with(_frame(), "num_a", _NAN_ROW, float("nan"))
    scores = _scores_for(df, name, _config())
    assert _anomaly_rank(scores, _ANOMALY_ROW) == 0, f"{name}: planted anomaly must be the most anomalous row"
    assert _anomaly_rank(scores, _NAN_ROW) > 0


@pytest.mark.parametrize("name", _SHIPPED_DETECTORS)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf")], ids=["+inf", "-inf"])
def test_infinite_row_never_reaches_any_detector(name, bad):
    """Infinity must stop at feature construction for every detector, not be ranked as a fake-normal row."""
    df = _with(_frame(), "num_a", _NAN_ROW, bad)
    plan = build_feature_plan(df, _config())
    with pytest.raises(PlanError, match="Infinite value"):
        fit_features(df, plan, _config())
    assert name in registry  # parametrised so each shipped detector is named in the report
