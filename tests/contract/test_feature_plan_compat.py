"""FeaturePlan serialisation compatibility: ``to_json()``/``from_json()``
round-trip losslessly and deterministically, both right after profiling and
after ``fit_features`` populates the M2-stage fields (scaler, correlation,
imputation). A persisted plan is read back by score-forward and by a
resumed run days or releases later, so this is a public compatibility
contract, not an implementation detail of profiling or features.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sorethumb.config import Config, FeaturesConfig, RunConfig, SourceConfig
from sorethumb.features.build import fit_features
from sorethumb.profiling.plan import FeaturePlan, build_feature_plan
from tests.synth import make_frame

pytestmark = pytest.mark.contract


def _build_config(tmp_path: object, **kw: object) -> Config:
    return Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def _make_cat_df() -> pl.DataFrame:
    """DataFrame with a mix of categorical, numeric, boolean, and temporal columns."""
    import datetime

    n = 100
    return pl.DataFrame(
        {
            "cat_low": (["a", "b", "c", "d"] * 25),  # 4 categories → one_hot (≤20)
            "cat_high": [f"val_{i}" for i in range(n)],  # 100 categories → frequency
            "num": [float(i) for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
            "ts": [datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i) for i in range(n)],
            "with_nulls": [float(i) if i % 5 != 0 else None for i in range(n)],
        }
    )


def test_feature_plan_round_trip(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=100, seed=0, with_low_cardinality_string=True)
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)

    serialised = plan.to_json()
    restored = FeaturePlan.from_json(serialised)

    assert plan.output_features == restored.output_features
    assert plan.derived_to_original == restored.derived_to_original
    assert plan.one_hot_categories == restored.one_hot_categories
    assert plan.chosen_time_column == restored.chosen_time_column
    assert plan.schema_fingerprint == restored.schema_fingerprint

    # Decisions round-trip
    assert len(plan.decisions) == len(restored.decisions)
    for orig, rest in zip(plan.decisions, restored.decisions, strict=True):
        assert orig.column == rest.column
        assert orig.col_class == rest.col_class
        assert orig.treatment == rest.treatment
        assert orig.emit_missing_indicator == rest.emit_missing_indicator


def test_feature_plan_json_is_deterministic(tmp_path: Path) -> None:
    df, _ = make_frame(n_rows=50, seed=42, with_low_cardinality_string=True)
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert plan.to_json() == plan.to_json()


def test_feature_plan_imputation_medians_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, None, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    restored = FeaturePlan.from_json(plan.to_json())
    assert plan.imputation_medians == restored.imputation_medians


def test_plan_json_roundtrip_after_fit(tmp_path: Path) -> None:
    """FeaturePlan round-trips through JSON after M2 fields are populated."""
    df = _make_cat_df()
    config = _build_config(tmp_path, features=FeaturesConfig(pca=False))
    plan = build_feature_plan(df, config)
    fit_features(df, plan, config)

    json_str = plan.to_json()
    restored = type(plan).from_json(json_str)
    assert restored.scaler_type == plan.scaler_type
    assert restored.output_dtype == plan.output_dtype
    assert set(restored.scaler_params.keys()) == set(plan.scaler_params.keys())
    assert restored.correlation_drop_list == plan.correlation_drop_list
