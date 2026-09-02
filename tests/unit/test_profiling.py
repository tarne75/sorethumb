"""Unit tests for the profiling layer (M1b)."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from sorethumb.config import (
    ColumnsConfig,
    Config,
    FeaturesConfig,
    ProfilingConfig,
    RunConfig,
    SourceConfig,
)
from sorethumb.errors import ColumnDroppedWarning, PlanError
from sorethumb.profiling.classify import ColumnClass, Treatment, classify_column, treatment_for
from sorethumb.profiling.plan import FeaturePlan, build_feature_plan
from sorethumb.profiling.profile import ColumnProfile, profile_columns
from tests.synth import make_frame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_profiling() -> ProfilingConfig:
    return ProfilingConfig()


def _default_columns() -> ColumnsConfig:
    return ColumnsConfig()


def _default_features() -> FeaturesConfig:
    return FeaturesConfig()


def _classify(profile: ColumnProfile, **overrides: object) -> tuple[ColumnClass, str]:
    cfg = ProfilingConfig(**overrides)  # type: ignore[arg-type]
    return classify_column(profile, cfg, _default_columns(), set())


def _build_config(tmp_path: object, **kw: object) -> Config:
    import tempfile

    return Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# profile_columns
# ---------------------------------------------------------------------------


def test_profile_returns_one_per_column() -> None:
    df, _ = make_frame(n_rows=50, seed=0)
    profiles = profile_columns(df, _default_profiling())
    assert len(profiles) == len(df.columns)
    assert [p.name for p in profiles] == df.columns


def test_profile_null_count_accurate() -> None:
    df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    profiles = profile_columns(df, _default_profiling())
    p = profiles[0]
    assert p.null_count == 2
    assert p.non_null_count == 3
    assert abs(p.null_ratio - 0.4) < 1e-9


def test_profile_numeric_stats() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    p = profile_columns(df, _default_profiling())[0]
    assert p.min_val == pytest.approx(1.0)
    assert p.max_val == pytest.approx(5.0)
    assert p.mean_val == pytest.approx(3.0)


def test_profile_string_mean_length() -> None:
    df = pl.DataFrame({"s": ["hi", "hello", "hey"]})
    p = profile_columns(df, _default_profiling())[0]
    assert p.mean_length == pytest.approx((2 + 5 + 3) / 3)


def test_profile_string_has_examples() -> None:
    df = pl.DataFrame({"s": ["alpha", "beta", "gamma"]})
    p = profile_columns(df, _default_profiling())[0]
    assert len(p.examples) > 0
    assert all(isinstance(e, str) for e in p.examples)


# ---------------------------------------------------------------------------
# Classification: empty / constant / near-constant
# ---------------------------------------------------------------------------


def test_all_null_classified_empty_not_constant() -> None:
    df = pl.DataFrame({"x": [None, None, None]}, schema={"x": pl.Float64})
    p = profile_columns(df, _default_profiling())[0]
    assert p.is_empty
    assert not p.is_constant
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.empty


def test_constant_column_classified_constant() -> None:
    df = pl.DataFrame({"x": [42.0, 42.0, 42.0]})
    p = profile_columns(df, _default_profiling())[0]
    assert p.is_constant
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.constant


def test_near_constant_at_threshold() -> None:
    # n_unique = 3 = near_constant_distinct → near_constant
    df = pl.DataFrame({"x": ["a", "b", "c", "a", "b", "c", "a"]})
    p = profile_columns(df, _default_profiling())[0]
    col_class, reason = _classify(p, near_constant_distinct=3)
    assert col_class == ColumnClass.near_constant
    assert "3" in reason


def test_near_constant_above_threshold_is_categorical() -> None:
    # n_unique = 4 > near_constant_distinct=3 → not near_constant
    df = pl.DataFrame({"x": ["a", "b", "c", "d", "a"]})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p, near_constant_distinct=3)
    assert col_class != ColumnClass.near_constant


# ---------------------------------------------------------------------------
# Classification: high null
# ---------------------------------------------------------------------------


def test_high_null_above_drop_threshold() -> None:
    # 10 out of 14 null → null_ratio ≈ 0.714 > 0.70 default; 4 distinct values avoids near_constant
    vals: list[float | None] = [None] * 10 + [1.0, 2.0, 3.0, 4.0]
    df = pl.DataFrame({"x": vals})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.high_null


def test_high_null_at_boundary_not_flagged() -> None:
    # null_ratio = 0.70 exactly — equal to threshold, so NOT high_null (> not >=)
    vals: list[float | None] = [None] * 7 + [1.0, 2.0, 3.0]
    df = pl.DataFrame({"x": vals})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p, null_ratio_drop=0.70)
    assert col_class != ColumnClass.high_null


# ---------------------------------------------------------------------------
# Classification: identifier detection
# ---------------------------------------------------------------------------


def test_uuid_column_classified_identifier() -> None:
    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(100)]
    df = pl.DataFrame({"guid": uuids})
    p = profile_columns(df, _default_profiling())[0]
    col_class, reason = classify_column(p, _default_profiling(), _default_columns(), set())
    assert col_class == ColumnClass.identifier_like
    assert "UUID" in reason


def test_time_column_not_classified_identifier() -> None:
    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(100)]
    df = pl.DataFrame({"ts_id": uuids})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = classify_column(p, _default_profiling(), _default_columns(), {"ts_id"})
    assert col_class != ColumnClass.identifier_like


def test_group_by_column_not_classified_identifier() -> None:
    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(100)]
    df = pl.DataFrame({"region": uuids})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = classify_column(p, _default_profiling(), _default_columns(), {"region"})
    assert col_class != ColumnClass.identifier_like


def test_identifier_detection_off_skips_uuid() -> None:
    import uuid

    uuids = [str(uuid.uuid4()) for _ in range(100)]
    df = pl.DataFrame({"guid": uuids})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p, identifier_detection="off")
    assert col_class != ColumnClass.identifier_like


# ---------------------------------------------------------------------------
# Classification: free text
# ---------------------------------------------------------------------------


def test_free_text_classified() -> None:
    # Each sentence is distinct and long (mean length >> free_text_mean_length=20)
    sentences = [
        f"this is a long free text sentence number {i} used for testing the classifier" for i in range(100)
    ]
    df = pl.DataFrame({"text": sentences})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.free_text


def test_short_strings_not_free_text() -> None:
    df = pl.DataFrame({"s": ["yes", "no", "maybe"] * 33})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class != ColumnClass.free_text


# ---------------------------------------------------------------------------
# Classification: categorical / numeric / boolean / temporal
# ---------------------------------------------------------------------------


def test_low_cardinality_string_classified_categorical() -> None:
    # 4 distinct values > near_constant_distinct=3, so near_constant doesn't fire
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"] * 25})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.categorical


def test_numeric_float_classified_numeric() -> None:
    # 5 distinct values > near_constant_distinct=3 avoids near_constant
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.numeric


def test_boolean_classified_boolean() -> None:
    df = pl.DataFrame({"b": [True, False, True]})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.boolean


def test_datetime_classified_temporal() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    dts = [base + timedelta(hours=i) for i in range(10)]
    df = pl.DataFrame({"ts": dts})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.temporal


def test_list_column_classified_array_derived() -> None:
    # 5 distinct list values avoids near_constant
    df = pl.DataFrame({"arr": [[float(i), float(i + 1)] for i in range(5)]})
    p = profile_columns(df, _default_profiling())[0]
    col_class, _ = _classify(p)
    assert col_class == ColumnClass.array_derived


# ---------------------------------------------------------------------------
# Treatment mapping
# ---------------------------------------------------------------------------


def test_treatment_for_numeric_is_impute_median() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, None]})
    p = profile_columns(df, _default_profiling())[0]
    t = treatment_for(ColumnClass.numeric, p, _default_features())
    assert t == Treatment.impute_median


def test_treatment_for_boolean_is_cast_int() -> None:
    df = pl.DataFrame({"b": [True, False]})
    p = profile_columns(df, _default_profiling())[0]
    t = treatment_for(ColumnClass.boolean, p, _default_features())
    assert t == Treatment.cast_int


def test_treatment_for_low_card_categorical_is_one_hot() -> None:
    df = pl.DataFrame({"c": ["a", "b", "c"] * 10})
    p = profile_columns(df, _default_profiling())[0]
    t = treatment_for(ColumnClass.categorical, p, _default_features())
    assert t == Treatment.one_hot


def test_treatment_for_high_card_categorical_is_frequency() -> None:
    # 30 distinct values, one_hot_max_cardinality=20 → frequency
    cats = [f"cat_{i}" for i in range(30)] * 3
    df = pl.DataFrame({"c": cats})
    p = profile_columns(df, _default_profiling())[0]
    features_cfg = FeaturesConfig(one_hot_max_cardinality=20)
    t = treatment_for(ColumnClass.categorical, p, features_cfg)
    assert t == Treatment.frequency


# ---------------------------------------------------------------------------
# FeaturePlan: build, missing indicators, output features
# ---------------------------------------------------------------------------


def test_missing_indicator_emitted_for_high_null_string(tmp_path: object) -> None:
    # 8/10 null → null_ratio=0.8 → high_null → indicator_only
    vals: list[str | None] = [None] * 8 + ["a", "b"]
    df = pl.DataFrame({"s": vals})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "s__is_missing" in plan.output_features


def test_missing_indicator_emitted_for_flagged_numeric(tmp_path: object) -> None:
    # 4/10 null → null_ratio=0.4 > null_ratio_flag=0.3 → indicator emitted
    vals: list[float | None] = [None] * 4 + [float(i) for i in range(6)]
    df = pl.DataFrame({"x": vals})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "x__is_missing" in plan.output_features
    assert "x" in plan.output_features


def test_no_missing_indicator_when_disabled(tmp_path: object) -> None:
    vals: list[float | None] = [None] * 4 + [float(i) for i in range(6)]
    df = pl.DataFrame({"x": vals})
    cfg = _build_config(tmp_path, features=FeaturesConfig(missing_indicators=False))
    plan = build_feature_plan(df, cfg)
    assert "x__is_missing" not in plan.output_features


def test_dropped_column_produces_no_output(tmp_path: object) -> None:
    df = pl.DataFrame({"const": [1.0] * 50, "x": list(range(50, 100, 1))})
    df = df.with_columns(pl.col("x").cast(pl.Float64))
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    const_feats = [f for f in plan.output_features if f.startswith("const")]
    assert not const_feats


def test_one_hot_features_have_categories_and_other(tmp_path: object) -> None:
    # 4 distinct values > near_constant_distinct=3 → categorical not near_constant
    df = pl.DataFrame({"cat": ["a", "b", "c", "d"] * 20})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    oh_feats = [f for f in plan.output_features if f.startswith("cat__")]
    assert "cat__a" in oh_feats
    assert "cat__b" in oh_feats
    assert "cat__c" in oh_feats
    assert "cat__d" in oh_feats
    assert "cat____other" in oh_feats


def test_one_hot_categories_stored_in_plan(tmp_path: object) -> None:
    # 4 distinct values > near_constant_distinct=3
    df = pl.DataFrame({"cat": ["w", "x", "y", "z"] * 10})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "cat" in plan.one_hot_categories
    assert sorted(plan.one_hot_categories["cat"]) == ["w", "x", "y", "z"]


def test_frequency_map_stored_for_high_card(tmp_path: object) -> None:
    cats = [f"cat_{i}" for i in range(25)] * 4  # 25 distinct > default 20
    df = pl.DataFrame({"c": cats})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "c" in plan.frequency_maps
    assert abs(sum(plan.frequency_maps["c"].values()) - 1.0) < 1e-6


def test_imputation_median_stored(tmp_path: object) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "x" in plan.imputation_medians
    assert plan.imputation_medians["x"] == pytest.approx(3.0)


def test_temporal_derivatives_in_output(tmp_path: object) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    dts = [base + timedelta(hours=i) for i in range(50)]
    df = pl.DataFrame({"ts": dts, "x": [float(i) for i in range(50)]})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert "ts__hour" in plan.output_features
    assert "ts__dayofweek" in plan.output_features


def test_ignored_column_produces_no_output(tmp_path: object) -> None:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0] * 20,
            "internal_col": ["a", "b", "c"] * 20,
        }
    )
    cfg = _build_config(tmp_path, columns=ColumnsConfig(ignore=["internal_*"]))
    plan = build_feature_plan(df, cfg)
    assert not any(f.startswith("internal_col") for f in plan.output_features)


# ---------------------------------------------------------------------------
# FeaturePlan: derived_to_original completeness
# ---------------------------------------------------------------------------


def test_derived_to_original_complete(tmp_path: object) -> None:
    df, _ = make_frame(
        n_rows=100,
        seed=0,
        with_low_cardinality_string=True,
        with_boolean=True,
        with_timestamp=True,
        null_ratio=0.1,
    )
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    for feat in plan.output_features:
        assert feat in plan.derived_to_original, f"{feat!r} missing from derived_to_original"


def test_all_synth_flags_plan_complete(tmp_path: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        df, _ = make_frame(
            n_rows=100,
            seed=1,
            with_constant=True,
            with_all_null=True,
            with_high_cardinality_string=True,
            with_low_cardinality_string=True,
            with_free_text=True,
            with_guid=True,
            with_boolean=True,
            with_timestamp=True,
            with_array=True,
            with_struct=True,
        )
        # Unnest structs before planning (as the pipeline would)
        from sorethumb.io.nested import unnest_all

        df = unnest_all(df)

    cfg = _build_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ColumnDroppedWarning)
        plan = build_feature_plan(df, cfg)

    for feat in plan.output_features:
        assert feat in plan.derived_to_original


# ---------------------------------------------------------------------------
# FeaturePlan: JSON round-trip
# ---------------------------------------------------------------------------


def test_feature_plan_round_trip(tmp_path: object) -> None:
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


def test_feature_plan_json_is_deterministic(tmp_path: object) -> None:
    df, _ = make_frame(n_rows=50, seed=42, with_low_cardinality_string=True)
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    assert plan.to_json() == plan.to_json()


def test_feature_plan_imputation_medians_round_trip(tmp_path: object) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, None, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    cfg = _build_config(tmp_path)
    plan = build_feature_plan(df, cfg)
    restored = FeaturePlan.from_json(plan.to_json())
    assert plan.imputation_medians == restored.imputation_medians
