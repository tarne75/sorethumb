"""Tier 1 coverage tests.

Targets: config.py, detectors/__init__.py, sorethumb/__init__.py,
features/space.py — all at < 15 % because their code (class definitions,
module-level registration, re-exports) runs at import time, before
pytest-cov instruments the modules.  importlib.reload() forces
re-execution during the test phase.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ── sorethumb.__init__ ────────────────────────────────────────────────────────


def test_package_init_reexecuted_under_coverage() -> None:
    import sorethumb
    importlib.reload(sorethumb)


def test_package_version_after_reload() -> None:
    import sorethumb
    importlib.reload(sorethumb)
    assert sorethumb.__version__ == "0.1.0"


# ── config.py ─────────────────────────────────────────────────────────────────


def test_config_module_reexecuted_under_coverage() -> None:
    import sorethumb.config as m
    importlib.reload(m)


def test_source_config_defaults() -> None:
    from sorethumb.config import SourceConfig
    sc = SourceConfig(uri="/tmp/data.csv")
    assert sc.uri == "/tmp/data.csv"
    assert sc.format == "auto"
    assert sc.auth == "none"
    assert sc.auth_env_var is None
    assert sc.read_options == {}
    assert sc.cache is True
    assert sc.max_nesting_depth == 5


def test_source_config_all_formats() -> None:
    from sorethumb.config import SourceConfig
    for fmt in ("auto", "csv", "tsv", "parquet", "json", "jsonl", "tsf"):
        sc = SourceConfig(uri="/tmp/f", format=fmt)  # type: ignore[arg-type]
        assert sc.format == fmt


def test_source_config_auth_variants() -> None:
    from sorethumb.config import SourceConfig
    for auth in ("none", "bearer", "basic"):
        sc = SourceConfig(uri="https://host/data.csv", auth=auth, auth_env_var="TOKEN")  # type: ignore[arg-type]
        assert sc.auth == auth


def test_columns_config_defaults() -> None:
    from sorethumb.config import ColumnsConfig
    cc = ColumnsConfig()
    assert cc.time_column is None
    assert cc.group_by == []
    assert cc.id_column is None
    assert cc.reference_column is None
    assert cc.ignore == []
    assert cc.include_only is None


def test_columns_config_all_fields() -> None:
    from sorethumb.config import ColumnsConfig
    cc = ColumnsConfig(
        time_column="ts",
        group_by=["site", "region"],
        id_column="row_id",
        reference_column="label",
        ignore=["meta_*"],
        include_only=["amount", "country"],
    )
    assert cc.time_column == "ts"
    assert cc.group_by == ["site", "region"]
    assert cc.id_column == "row_id"
    assert cc.reference_column == "label"
    assert cc.ignore == ["meta_*"]
    assert cc.include_only == ["amount", "country"]


def test_profiling_config_defaults() -> None:
    from sorethumb.config import ProfilingConfig
    pc = ProfilingConfig()
    assert pc.null_ratio_drop == pytest.approx(0.70)
    assert pc.null_ratio_flag == pytest.approx(0.30)
    assert pc.near_constant_distinct == 3
    assert pc.identifier_cardinality_ratio == pytest.approx(0.90)
    assert pc.categorical_cardinality_ratio == pytest.approx(0.01)
    assert pc.free_text_mean_length == pytest.approx(20.0)
    assert pc.sample_rows_for_examples == 1000
    assert pc.identifier_detection == "conservative"


def test_profiling_config_identifier_detection_variants() -> None:
    from sorethumb.config import ProfilingConfig
    for mode in ("conservative", "aggressive", "off"):
        pc = ProfilingConfig(identifier_detection=mode)  # type: ignore[arg-type]
        assert pc.identifier_detection == mode


def test_features_config_defaults() -> None:
    from sorethumb.config import FeaturesConfig
    fc = FeaturesConfig()
    assert fc.one_hot_max_cardinality == 20
    assert fc.max_feature_width == 2000
    assert fc.missing_indicators is True
    assert fc.array_features is True
    assert fc.time_derivatives == ["hour", "dayofweek", "day", "month"]
    assert fc.scaler == "robust"
    assert fc.dtype == "float32"
    assert fc.correlation_reduction is True
    assert fc.correlation_threshold == pytest.approx(0.95)
    assert fc.pca is False
    assert fc.pca_max_components == 50
    assert fc.pca_min_explained_variance == pytest.approx(0.80)


def test_features_config_pca_variants() -> None:
    from sorethumb.config import FeaturesConfig
    fc = FeaturesConfig(pca=True, pca_max_components=20, pca_min_explained_variance=0.90)
    assert fc.pca is True
    assert fc.pca_max_components == 20


def test_features_config_scaler_and_dtype() -> None:
    from sorethumb.config import FeaturesConfig
    fc = FeaturesConfig(scaler="standard", dtype="float64")
    assert fc.scaler == "standard"
    assert fc.dtype == "float64"


def test_detector_config_defaults() -> None:
    from sorethumb.config import DetectorConfig
    dc = DetectorConfig(name="isolation_forest")
    assert dc.name == "isolation_forest"
    assert dc.enabled is True
    assert dc.params == {}
    assert dc.train_row_cap is None


def test_detector_config_with_params_and_cap() -> None:
    from sorethumb.config import DetectorConfig
    dc = DetectorConfig(
        name="one_class_svm",
        enabled=False,
        params={"kernel": "rbf", "nu": 0.1},
        train_row_cap=5000,
    )
    assert dc.enabled is False
    assert dc.params == {"kernel": "rbf", "nu": 0.1}
    assert dc.train_row_cap == 5000


def test_default_detectors_factory() -> None:
    from sorethumb.config import _default_detectors
    dets = _default_detectors()
    assert len(dets) == 3
    names = {d.name for d in dets}
    assert names == {"isolation_forest", "kmeans_distance", "one_class_svm"}


def test_scoring_config_defaults() -> None:
    from sorethumb.config import ScoringConfig
    sc = ScoringConfig()
    assert sc.contamination == "auto"
    assert sc.combination == "composite"
    assert sc.weighting == "equal"
    assert sc.weights == {}
    assert sc.min_records == 100


def test_scoring_config_contamination_float() -> None:
    from sorethumb.config import ScoringConfig
    sc = ScoringConfig(contamination=0.05)
    assert sc.contamination == pytest.approx(0.05)


def test_scoring_config_contamination_invalid_string() -> None:
    from pydantic import ValidationError

    from sorethumb.config import ScoringConfig
    with pytest.raises(ValidationError):
        ScoringConfig(contamination="bad")


def test_scoring_config_contamination_out_of_range() -> None:
    from pydantic import ValidationError

    from sorethumb.config import ScoringConfig
    with pytest.raises(ValidationError):
        ScoringConfig(contamination=0.6)


def test_scoring_config_combination_variants() -> None:
    from sorethumb.config import ScoringConfig
    for combo in ("composite", "intersection", "union"):
        sc = ScoringConfig(combination=combo)  # type: ignore[arg-type]
        assert sc.combination == combo


def test_scoring_config_manual_weighting() -> None:
    from sorethumb.config import ScoringConfig
    sc = ScoringConfig(weighting="manual", weights={"isolation_forest": 0.7, "kmeans_distance": 0.3})
    assert sc.weighting == "manual"
    assert sc.weights["isolation_forest"] == pytest.approx(0.7)


def test_explain_config_defaults() -> None:
    from sorethumb.config import ExplainConfig
    ec = ExplainConfig()
    assert ec.enabled is True
    assert ec.top_n == 3
    assert ec.max_rows == 5000
    assert ec.kernel_shap is False
    assert ec.permutation_importance is False


def test_explain_config_all_flags() -> None:
    from sorethumb.config import ExplainConfig
    ec = ExplainConfig(enabled=False, top_n=7, max_rows=100, kernel_shap=True, permutation_importance=True)
    assert ec.enabled is False
    assert ec.top_n == 7
    assert ec.kernel_shap is True
    assert ec.permutation_importance is True


def test_run_config_defaults() -> None:
    from sorethumb.config import RunConfig
    rc = RunConfig(workdir="/tmp/ws")
    assert rc.workdir == "/tmp/ws"
    assert rc.seed == 42
    assert rc.strict is False
    assert rc.max_memory_mb == 8192
    assert rc.max_rows is None
    assert rc.reuse_models is False
    assert rc.retention_days == 90
    assert rc.log_level == "INFO"
    assert rc.slow_stage_seconds == 300


def test_run_config_overrides() -> None:
    from sorethumb.config import RunConfig
    rc = RunConfig(
        workdir="/data/ws",
        seed=1234,
        strict=True,
        max_memory_mb=4096,
        max_rows=50_000,
        reuse_models=True,
        retention_days=30,
        log_level="DEBUG",
        slow_stage_seconds=60,
    )
    assert rc.seed == 1234
    assert rc.strict is True
    assert rc.max_memory_mb == 4096
    assert rc.max_rows == 50_000
    assert rc.reuse_models is True


def test_history_config_defaults() -> None:
    from sorethumb.config import HistoryConfig
    hc = HistoryConfig()
    assert hc.period_granularity == "day"
    assert hc.roll_non_business is True
    assert hc.lookback_periods == 28
    assert hc.bootstrap_periods == 28
    assert hc.max_backfill_periods == 30
    assert hc.low_volume_threshold == 100


def test_history_config_granularity_variants() -> None:
    from sorethumb.config import HistoryConfig
    for g in ("hour", "day", "week", "month"):
        hc = HistoryConfig(period_granularity=g)  # type: ignore[arg-type]
        assert hc.period_granularity == g


def test_report_config_defaults() -> None:
    from sorethumb.config import ReportConfig
    rc = ReportConfig()
    assert rc.formats == ["html", "csv"]
    assert rc.open_after is False
    assert rc.rolling_windows == [1, 7, 14, 28]


def test_report_config_json_only() -> None:
    from sorethumb.config import ReportConfig
    rc = ReportConfig(formats=["json"], open_after=True, rolling_windows=[7, 30])
    assert rc.formats == ["json"]
    assert rc.open_after is True


def test_top_level_config_minimal() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig
    cfg = Config(
        source=SourceConfig(uri="/tmp/data.csv"),
        run=RunConfig(workdir="/tmp/ws"),
    )
    assert cfg.source.uri == "/tmp/data.csv"
    assert cfg.run.workdir == "/tmp/ws"


def test_top_level_config_all_sections() -> None:
    from sorethumb.config import (
        ColumnsConfig,
        Config,
        DetectorConfig,
        ExplainConfig,
        FeaturesConfig,
        HistoryConfig,
        ProfilingConfig,
        ReportConfig,
        RunConfig,
        ScoringConfig,
        SourceConfig,
    )
    cfg = Config(
        source=SourceConfig(uri="/tmp/data.parquet", format="parquet"),
        columns=ColumnsConfig(id_column="id", time_column="ts"),
        profiling=ProfilingConfig(null_ratio_drop=0.8),
        features=FeaturesConfig(pca=True),
        detectors=[DetectorConfig(name="isolation_forest", train_row_cap=100_000)],
        scoring=ScoringConfig(contamination=0.02),
        explain=ExplainConfig(top_n=5),
        run=RunConfig(workdir="/tmp/ws", seed=99),
        history=HistoryConfig(period_granularity="hour"),
        report=ReportConfig(formats=["json"]),
    )
    assert cfg.source.format == "parquet"
    assert cfg.columns.id_column == "id"
    assert cfg.profiling.null_ratio_drop == pytest.approx(0.8)
    assert cfg.features.pca is True
    assert len(cfg.detectors) == 1
    assert cfg.scoring.contamination == pytest.approx(0.02)
    assert cfg.explain.top_n == 5
    assert cfg.run.seed == 99
    assert cfg.history.period_granularity == "hour"
    assert cfg.report.formats == ["json"]


def test_config_hash_is_stable() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig
    cfg = Config(
        source=SourceConfig(uri="/tmp/data.csv"),
        run=RunConfig(workdir="/tmp/ws"),
    )
    h1 = cfg.config_hash()
    h2 = cfg.config_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_excludes_cosmetic_fields() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig
    cfg1 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws"))
    cfg2 = Config(
        source=SourceConfig(uri="/tmp/data.csv"),
        run=RunConfig(workdir="/different/path", log_level="DEBUG", slow_stage_seconds=60),
    )
    assert cfg1.config_hash() == cfg2.config_hash()


def test_config_hash_changes_with_result_affecting_fields() -> None:
    from sorethumb.config import Config, RunConfig, SourceConfig
    cfg1 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws"))
    cfg2 = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws", seed=999))
    assert cfg1.config_hash() != cfg2.config_hash()


def test_any_config_alias() -> None:
    from sorethumb.config import AnyConfig, Config, RunConfig, SourceConfig
    cfg = Config(source=SourceConfig(uri="/tmp/data.csv"), run=RunConfig(workdir="/tmp/ws"))
    assert isinstance(cfg, Config)
    _ = AnyConfig  # ensure the alias is importable


# ── detectors/__init__.py ─────────────────────────────────────────────────────


def test_detectors_init_reexecuted_under_coverage() -> None:
    import sorethumb.detectors as m
    importlib.reload(m)


def test_registry_contains_builtins() -> None:
    from sorethumb.detectors import registry
    assert "isolation_forest" in registry
    assert "kmeans_distance" in registry
    assert "one_class_svm" in registry


def test_registry_values_are_detector_classes() -> None:
    from sorethumb.detectors import (
        IsolationForestDetector,
        KMeansDetector,
        OneClassSVMDetector,
        registry,
    )
    assert registry["isolation_forest"] is IsolationForestDetector
    assert registry["kmeans_distance"] is KMeansDetector
    assert registry["one_class_svm"] is OneClassSVMDetector


def test_register_function_rejects_non_detector() -> None:
    from sorethumb.detectors import register
    from sorethumb.errors import DetectorError
    with pytest.raises(DetectorError):
        register(object)  # type: ignore[arg-type]


def test_detectors_all_list() -> None:
    import sorethumb.detectors as det
    expected = {"Detector", "IsolationForestDetector", "KMeansDetector", "OneClassSVMDetector", "register", "registry"}
    assert set(det.__all__) == expected


# ── features/space.py ─────────────────────────────────────────────────────────


def test_feature_space_module_reexecuted_under_coverage() -> None:
    import sorethumb.features.space as m
    importlib.reload(m)


def test_feature_space_make_hash_deterministic() -> None:
    from sorethumb.features.space import FeatureSpace
    names = ["amount", "country__US", "country__GB", "hour_of_day"]
    h1 = FeatureSpace.make_hash(names)
    h2 = FeatureSpace.make_hash(names)
    assert h1 == h2
    assert len(h1) == 32


def test_feature_space_make_hash_changes_with_order() -> None:
    from sorethumb.features.space import FeatureSpace
    h1 = FeatureSpace.make_hash(["a", "b"])
    h2 = FeatureSpace.make_hash(["b", "a"])
    assert h1 != h2


def test_feature_space_make_hash_empty() -> None:
    from sorethumb.features.space import FeatureSpace
    h = FeatureSpace.make_hash([])
    assert isinstance(h, str)
    assert len(h) == 32


def test_feature_space_is_dataclass() -> None:
    import dataclasses

    from sorethumb.features.space import FeatureSpace
    assert dataclasses.is_dataclass(FeatureSpace)
    fields = {f.name for f in dataclasses.fields(FeatureSpace)}
    assert fields == {"matrix", "feature_names", "row_ids", "plan", "feature_schema_hash"}


def test_feature_space_instantiation() -> None:
    from unittest.mock import MagicMock

    from sorethumb.features.space import FeatureSpace
    matrix = np.zeros((10, 3), dtype=np.float32)
    row_ids = np.arange(10)
    plan = MagicMock()
    names = ["f0", "f1", "f2"]
    fs = FeatureSpace(
        matrix=matrix,
        feature_names=names,
        row_ids=row_ids,
        plan=plan,
        feature_schema_hash=FeatureSpace.make_hash(names),
    )
    assert fs.matrix.shape == (10, 3)
    assert fs.feature_names == names
    assert len(fs.feature_schema_hash) == 32
