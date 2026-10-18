"""Per-period statistics and population drift (PSI)."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from sorethumb_ml.history.period_stats import (
    STATS_VERSION,
    compute_period_stats,
    field_summary,
    population_drift,
    psi_band,
)

pytestmark = pytest.mark.unit


def _stats(df: pl.DataFrame) -> dict:
    n = df.height
    return compute_period_stats(
        df_group=df,
        fields=list(df.columns),
        raw_scores={"d": np.linspace(-1, 0, n)},
        natural_flags={"d": np.arange(n) < 3},
        composite_score=np.linspace(0, 1, n),
        composite_threshold=0.9,
        contamination_used=0.05,
        n_flagged=3,
        combination="composite",
        ensemble_policy="composite-equal@1",
    )


# ---------------------------------------------------------------------------
# field_summary
# ---------------------------------------------------------------------------


def test_numeric_summary_has_percentiles_and_missing() -> None:
    s = field_summary(pl.Series("x", [1.0, 2.0, None, float("nan"), 4.0]))
    assert s is not None
    assert s["kind"] == "numeric"
    assert (s["n"], s["missing"]) == (5, 2)
    assert len(s["percentiles"]) == 101
    assert s["percentiles"][0] == 1.0
    assert s["percentiles"][100] == 4.0


def test_categorical_summary_keeps_top_values_and_other() -> None:
    values = ["a"] * 30 + [f"v{i}" for i in range(25)] + [None]
    s = field_summary(pl.Series("c", values))
    assert s is not None
    assert s["kind"] == "categorical"
    assert s["top"][0] == ["a", 30]
    assert len(s["top"]) == 20
    assert s["other"] == 55 - sum(c for _, c in s["top"])
    assert (s["missing"], s["distinct"]) == (1, 26)


def test_booleans_dates_and_datetimes_are_numeric() -> None:
    assert field_summary(pl.Series("b", [True, False, True]))["kind"] == "numeric"  # type: ignore[index]
    d = field_summary(pl.Series("d", [dt.date(2026, 1, 1), dt.date(2026, 1, 2)]))
    assert d is not None
    assert d["percentiles"][0] == dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp()
    t = field_summary(pl.Series("t", [dt.datetime(2026, 1, 1, 12)]))
    assert t is not None
    assert t["kind"] == "numeric"
    gap = field_summary(pl.Series("g", [dt.timedelta(seconds=5)]))
    assert gap is not None
    assert gap["kind"] == "numeric"


def test_all_missing_numeric_and_unsummarisable_types() -> None:
    s = field_summary(pl.Series("x", [None, None], dtype=pl.Float64))
    assert s == {"kind": "numeric", "n": 2, "missing": 2, "percentiles": None}
    assert field_summary(pl.Series("s", [{"a": 1}, {"a": 2}])) is None


def test_period_stats_are_versioned_and_json_safe() -> None:
    df = pl.DataFrame({"x": [float(i) for i in range(50)], "c": ["a", "b"] * 25})
    st = _stats(df)
    json.dumps(st, allow_nan=False)
    assert st["stats_version"] == STATS_VERSION
    assert st["n_rows"] == 50
    assert set(st["population"]) == {"x", "c"}
    assert st["scores"]["d"]["natural_count"] == 3
    assert st["scores"]["d"]["scale"] == "self_fitted"
    assert st["composite"]["flagged"] == 3
    assert st["composite"]["threshold"] == 0.9


def test_non_finite_threshold_is_recorded_as_none() -> None:
    st = compute_period_stats(
        df_group=pl.DataFrame({"x": [1.0, 2.0]}),
        fields=["x", "absent"],
        raw_scores={"d": np.array([np.nan, np.nan])},
        natural_flags={},
        composite_score=np.array([np.nan, np.nan]),
        composite_threshold=float("nan"),
        contamination_used=None,
        n_flagged=0,
        combination="intersection",
        ensemble_policy=None,
    )
    json.dumps(st, allow_nan=False)
    assert st["composite"]["threshold"] is None
    assert st["composite"]["percentiles"] is None
    assert st["scores"]["d"]["raw_percentiles"] is None
    assert "absent" not in st["population"]


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def _numeric(values: np.ndarray, name: str = "x") -> dict:
    return {"population": {name: field_summary(pl.Series(name, values))}}


def _exact_psi(base: np.ndarray, cur: np.ndarray) -> float:
    edges = np.quantile(base, np.linspace(0.1, 0.9, 9))
    b = np.histogram(base, np.concatenate([[-np.inf], edges, [np.inf]]))[0] / len(base)
    c = np.histogram(cur, np.concatenate([[-np.inf], edges, [np.inf]]))[0] / len(cur)
    b, c = np.clip(b, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - b) * np.log(c / b)))


def test_identical_data_has_no_drift() -> None:
    x = np.random.default_rng(0).normal(size=2000)
    out = population_drift([_numeric(x)], [_numeric(x)])
    assert out["x"]["psi"] == pytest.approx(0.0, abs=1e-6)
    assert out["x"]["band"] == "stable"


@pytest.mark.parametrize("shift", [0.1, 0.5, 1.0])
def test_numeric_psi_matches_an_exact_computation(shift: float) -> None:
    rng = np.random.default_rng(1)
    base, cur = rng.normal(size=5000), rng.normal(loc=shift, size=5000)
    approx = population_drift([_numeric(base)], [_numeric(cur)])["x"]["psi"]
    assert approx == pytest.approx(_exact_psi(base, cur), rel=0.1, abs=0.01)


def test_a_large_shift_is_large() -> None:
    rng = np.random.default_rng(2)
    out = population_drift([_numeric(rng.normal(size=1000))], [_numeric(rng.normal(loc=2, size=1000))])
    assert out["x"]["band"] == "large"


def test_summaries_combine_in_proportion_to_their_size() -> None:
    rng = np.random.default_rng(3)
    whole = rng.normal(size=4000)
    halves = [_numeric(whole[:1000]), _numeric(whole[1000:])]  # unequal sizes
    out = population_drift([_numeric(whole)], halves)
    assert out["x"]["psi"] < 0.01


def test_a_change_in_missing_rate_alone_registers() -> None:
    x = np.random.default_rng(4).normal(size=1000)
    holes = x.copy()
    holes[:300] = np.nan
    out = population_drift([_numeric(x)], [_numeric(holes)])
    assert out["x"]["psi"] > 0.1


def test_entirely_missing_on_one_side_compares_missing_rates() -> None:
    x = np.random.default_rng(5).normal(size=100)
    out = population_drift([_numeric(x)], [_numeric(np.full(100, np.nan))])
    assert out["x"]["band"] == "large"


def test_categorical_psi() -> None:
    def cat(values: list[str | None]) -> dict:
        return {"population": {"c": field_summary(pl.Series("c", values))}}

    same = population_drift([cat(["a"] * 50 + ["b"] * 50)], [cat(["b"] * 50 + ["a"] * 50)])
    assert same["c"]["psi"] == pytest.approx(0.0, abs=1e-9)
    moved = population_drift([cat(["a"] * 90 + ["b"] * 10)], [cat(["a"] * 10 + ["b"] * 80 + ["new"] * 10)])
    assert moved["c"]["band"] == "large"
    assert moved["c"]["kind"] == "categorical"


def test_fields_on_one_side_only_or_of_different_kinds_are_not_comparable() -> None:
    a = {
        "population": {
            "x": field_summary(pl.Series("x", [1.0, 2.0])),
            "only_a": field_summary(pl.Series("o", [1.0])),
        }
    }
    b = {"population": {"x": field_summary(pl.Series("x", ["p", "q"]))}}
    out = population_drift([a], [b])
    assert out["x"] == {"psi": None, "band": "not_comparable", "kind": "categorical/numeric"}
    assert out["only_a"]["band"] == "not_comparable"


def test_empty_sides_have_no_figure() -> None:
    zero = {"population": {"x": {"kind": "numeric", "n": 0, "missing": 0, "percentiles": None}}}
    out = population_drift([zero], [zero])
    assert out["x"]["psi"] is None
    cat0 = {
        "population": {
            "c": {"kind": "categorical", "n": 0, "missing": 0, "top": [], "other": 0, "distinct": 0}
        }
    }
    assert population_drift([cat0], [cat0])["c"]["psi"] is None


@pytest.mark.parametrize(
    ("psi", "band"), [(0.0, "stable"), (0.0999, "stable"), (0.1, "moderate"), (0.25, "large")]
)
def test_psi_bands(psi: float, band: str) -> None:
    assert psi_band(psi) == band
