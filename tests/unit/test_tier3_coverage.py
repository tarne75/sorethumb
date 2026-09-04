"""Tier 3 coverage tests — import-time code and uncovered method paths.

Targets:
  - detectors/_protocol.py      (57 %)  reload + check_protocol error paths
  - detectors/isolation_forest.py (42 %) reload
  - detectors/kmeans_distance.py  (73 %) reload
  - detectors/one_class_svm.py    (53 %) reload
  - store/workspace.py            (57 %) reload + accessor/ctx-manager/prune paths
  - profiling/classify.py         (63 %) reload + unsupported dtype, high_null treatment,
                                          identifier patterns, type: ignore patterns
  - profiling/plan.py             (69 %) reload + FeaturePlan round-trip, reference_column,
                                          multi-temporal drop, __post_init__ error
  - scoring/calibrate.py          (76 %) reload + reference mode, error paths, serialisation
  - explain/blend.py              (92 %) reload
  - io/fingerprint.py             (70 %) reload
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# explain/blend.py — import-time
# ---------------------------------------------------------------------------


def test_blend_reexecuted_under_coverage() -> None:
    m = importlib.import_module("sorethumb.explain.blend")
    importlib.reload(m)


# ---------------------------------------------------------------------------
# io/fingerprint.py — import-time
# ---------------------------------------------------------------------------


def test_fingerprint_reexecuted_under_coverage() -> None:
    import sorethumb.io.fingerprint as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# detectors/_protocol.py — import-time + check_protocol paths
# ---------------------------------------------------------------------------


def test_protocol_reexecuted_under_coverage() -> None:
    import sorethumb.detectors._protocol as m

    importlib.reload(m)


def test_check_protocol_accepts_valid_class() -> None:
    from sorethumb.detectors._protocol import check_protocol

    class GoodDet:
        name = "test_det"
        supports_tree_shap = False
        default_train_row_cap = 1000

        def fit(self, X, *, seed):
            pass

        def score_samples(self, X):
            return np.zeros(len(X))

        def natural_flag(self, scores):
            return scores < 0

        def get_params(self):
            return {}

    check_protocol(GoodDet)  # must not raise


def test_check_protocol_rejects_missing_class_attr() -> None:
    from sorethumb.detectors._protocol import check_protocol
    from sorethumb.errors import DetectorError

    class BadDet:
        # name missing
        supports_tree_shap = False
        default_train_row_cap = 1000

        def fit(self, X, *, seed):
            pass

        def score_samples(self, X):
            return np.zeros(len(X))

        def natural_flag(self, scores):
            return scores < 0

        def get_params(self):
            return {}

    with pytest.raises(DetectorError, match="name"):
        check_protocol(BadDet)


def test_check_protocol_rejects_non_callable_method() -> None:
    from sorethumb.detectors._protocol import check_protocol
    from sorethumb.errors import DetectorError

    class BadDet:
        name = "bad"
        supports_tree_shap = False
        default_train_row_cap = 1000
        fit = "not_a_function"  # non-callable

        def score_samples(self, X):
            return np.zeros(len(X))

        def natural_flag(self, scores):
            return scores < 0

        def get_params(self):
            return {}

    with pytest.raises(DetectorError, match="fit"):
        check_protocol(BadDet)


# ---------------------------------------------------------------------------
# detectors/isolation_forest.py — import-time
# ---------------------------------------------------------------------------


def test_isolation_forest_reexecuted_under_coverage() -> None:
    import sorethumb.detectors.isolation_forest as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# detectors/kmeans_distance.py — import-time
# ---------------------------------------------------------------------------


def test_kmeans_distance_reexecuted_under_coverage() -> None:
    import sorethumb.detectors.kmeans_distance as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# detectors/one_class_svm.py — import-time
# ---------------------------------------------------------------------------


def test_one_class_svm_reexecuted_under_coverage() -> None:
    import sorethumb.detectors.one_class_svm as m

    importlib.reload(m)


# ---------------------------------------------------------------------------
# store/workspace.py — import-time + uncovered methods
# ---------------------------------------------------------------------------


def test_workspace_reexecuted_under_coverage() -> None:
    import sorethumb.store.workspace as m

    importlib.reload(m)


def test_workspace_open_valid(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    ws.close()
    ws2 = Workspace.open(tmp_path / "ws")
    assert ws2.root == (tmp_path / "ws").resolve()
    ws2.close()


def test_workspace_open_missing_dir_raises(tmp_path: Path) -> None:
    from sorethumb.errors import StoreError
    from sorethumb.store.workspace import Workspace

    with pytest.raises(StoreError, match="not a directory"):
        Workspace.open(tmp_path / "no_such_ws")


def test_workspace_open_dir_without_marker_raises(tmp_path: Path) -> None:
    from sorethumb.errors import StoreError
    from sorethumb.store.workspace import Workspace

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    with pytest.raises(StoreError, match="not a sorethumb workspace"):
        Workspace.open(plain_dir)


def test_workspace_property_accessors(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    assert ws.root == (tmp_path / "ws").resolve()
    assert ws.store is not None
    assert ws.db_path().name == "sorethumb.db"
    ws.close()


def test_workspace_directory_methods(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    assert ws.models_dir("run1", "grp1").is_dir()
    assert ws.features_dir("run1", "grp1").is_dir()
    assert ws.logs_dir().is_dir()
    assert ws.tmp_dir().is_dir()
    ws.close()


def test_workspace_context_manager(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    with Workspace.init(tmp_path / "ws") as ws:
        assert ws.root.is_dir()


def test_workspace_list_prunable(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    prunable = ws.list_prunable(retention_days=0)
    assert isinstance(prunable, list)
    ws.close()


def test_workspace_prune_deletes_missing_file(tmp_path: Path) -> None:
    """prune() with a regenerable artifact whose file is already gone (line 162->165 branch)."""
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    ghost_path = str(tmp_path / "ws" / "ghost.parquet")
    ws.store._conn.execute(
        "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now', '-400 days'))",
        ("ghost_art", ghost_path, "results", 0, 1),
    )
    ws.store._conn.commit()
    deleted = ws.prune(retention_days=1, dry_run=False)
    assert ghost_path in deleted
    ws.close()


def test_workspace_prune_dry_run(tmp_path: Path) -> None:
    from sorethumb.store.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    f = tmp_path / "ws" / "old.parquet"
    f.write_bytes(b"x")
    ws.store._conn.execute(
        "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now', '-400 days'))",
        ("old_art", str(f), "results", 0, 1),
    )
    ws.store._conn.commit()
    deleted = ws.prune(retention_days=1, dry_run=True)
    assert str(f) in deleted
    assert f.exists()  # file was NOT deleted (dry_run=True)
    ws.close()


# ---------------------------------------------------------------------------
# profiling/classify.py — import-time + specific code paths
# ---------------------------------------------------------------------------


def test_classify_reexecuted_under_coverage() -> None:
    import sorethumb.profiling.classify as m

    importlib.reload(m)


def _make_profile(**overrides) -> object:
    """Build a minimal ColumnProfile, overriding specific fields."""
    from sorethumb.profiling.profile import ColumnProfile

    defaults: dict[str, object] = {
        "name": "col",
        "dtype_str": "String",
        "null_count": 0,
        "non_null_count": 100,
        "null_ratio": 0.0,
        "n_unique": 50,
        "cardinality_ratio": 0.5,
        "is_empty": False,
        "is_constant": False,
        "min_val": None,
        "max_val": None,
        "mean_val": None,
        "std_val": None,
        "mean_length": 10.0,
        "examples": ["a", "b"],
    }
    defaults.update(overrides)
    return ColumnProfile(**defaults)  # type: ignore[arg-type]


def test_classify_unsupported_dtype() -> None:
    """dtype != String, not numeric, not temporal, not list → unsupported."""
    from sorethumb.config import ColumnsConfig, ProfilingConfig
    from sorethumb.profiling.classify import ColumnClass, classify_column

    profile = _make_profile(dtype_str="Binary", n_unique=50)
    col_class, reason = classify_column(profile, ProfilingConfig(), ColumnsConfig(), set())
    assert col_class == ColumnClass.unsupported
    assert "unsupported" in reason


def test_classify_list_dtype() -> None:
    """List dtype → array_derived."""
    from sorethumb.config import ColumnsConfig, ProfilingConfig
    from sorethumb.profiling.classify import ColumnClass, classify_column

    profile = _make_profile(dtype_str="List(Float64)", n_unique=50)
    col_class, _ = classify_column(profile, ProfilingConfig(), ColumnsConfig(), set())
    assert col_class == ColumnClass.array_derived


def test_classify_temporal_datetime_dtype() -> None:
    """Datetime dtype → temporal."""
    from sorethumb.config import ColumnsConfig, ProfilingConfig
    from sorethumb.profiling.classify import ColumnClass, classify_column

    profile = _make_profile(dtype_str="Datetime(time_unit='us', time_zone=None)", n_unique=50)
    col_class, _ = classify_column(profile, ProfilingConfig(), ColumnsConfig(), set())
    assert col_class == ColumnClass.temporal


def test_treatment_for_high_null() -> None:
    """high_null class → indicator_only treatment."""
    from sorethumb.config import FeaturesConfig
    from sorethumb.profiling.classify import ColumnClass, Treatment, treatment_for

    profile = _make_profile()
    result = treatment_for(ColumnClass.high_null, profile, FeaturesConfig())
    assert result == Treatment.indicator_only


def test_treatment_for_unsupported() -> None:
    """unsupported class → drop treatment."""
    from sorethumb.config import FeaturesConfig
    from sorethumb.profiling.classify import ColumnClass, Treatment, treatment_for

    profile = _make_profile()
    result = treatment_for(ColumnClass.unsupported, profile, FeaturesConfig())
    assert result == Treatment.drop


def test_is_ignored_type_prefix_pattern() -> None:
    """'type: Int64 secret_*' pattern should match dtype+name."""
    from sorethumb.profiling.classify import _is_ignored

    result = _is_ignored("secret_key", "Int64", ["type:Int64 secret_*"], set())
    assert result is True


def test_is_ignored_type_prefix_no_match() -> None:
    """type: pattern with wrong dtype → not ignored."""
    from sorethumb.profiling.classify import _is_ignored

    result = _is_ignored("secret_key", "String", ["type:Int64 secret_*"], set())
    assert result is False


def test_is_ignored_protected_col_not_ignored() -> None:
    """Protected columns are never ignored even if they match a pattern."""
    from sorethumb.profiling.classify import _is_ignored

    result = _is_ignored("col", "String", ["col"], {"col"})
    assert result is False


def test_check_identifier_aggressive_mode() -> None:
    """Aggressive identifier detection uses cardinality_ratio only."""
    from sorethumb.config import ProfilingConfig
    from sorethumb.profiling.classify import ColumnClass, _check_identifier

    profile = _make_profile(cardinality_ratio=0.99, examples=["abc", "def"])
    cfg = ProfilingConfig(identifier_detection="aggressive", identifier_cardinality_ratio=0.9)
    col_class, reason = _check_identifier(profile, cfg)
    assert col_class == ColumnClass.identifier_like
    assert "aggressive" in reason


def test_check_identifier_hex_pattern() -> None:
    """Long hex strings at high cardinality → identifier_like."""
    from sorethumb.config import ProfilingConfig
    from sorethumb.profiling.classify import ColumnClass, _check_identifier

    hex_examples = ["deadbeefdeadbeef"] * 50
    profile = _make_profile(cardinality_ratio=0.99, examples=hex_examples)
    cfg = ProfilingConfig(identifier_cardinality_ratio=0.9)
    col_class, reason = _check_identifier(profile, cfg)
    assert col_class == ColumnClass.identifier_like
    assert "hex" in reason


def test_check_identifier_no_pattern_returns_none() -> None:
    """Normal strings at high cardinality but no UUID/hex → (None, '')."""
    from sorethumb.config import ProfilingConfig
    from sorethumb.profiling.classify import _check_identifier

    plain_examples = ["hello world"] * 50
    profile = _make_profile(cardinality_ratio=0.99, examples=plain_examples)
    cfg = ProfilingConfig(identifier_cardinality_ratio=0.9)
    col_class, reason = _check_identifier(profile, cfg)
    assert col_class is None
    assert reason == ""


# ---------------------------------------------------------------------------
# profiling/plan.py — import-time + FeaturePlan round-trip + paths
# ---------------------------------------------------------------------------


def test_plan_reexecuted_under_coverage() -> None:
    import sorethumb.profiling.plan as m

    importlib.reload(m)


def test_feature_plan_post_init_error() -> None:
    """FeaturePlan raises PlanError when output_features has entries missing from derived_to_original."""
    from sorethumb.errors import PlanError
    from sorethumb.profiling.plan import FeaturePlan

    with pytest.raises(PlanError, match="derived_to_original"):
        FeaturePlan(
            schema_fingerprint="abc",
            n_rows=10,
            decisions=[],
            output_features=["feature_missing"],
            derived_to_original={},  # empty — feature_missing has no entry
            one_hot_categories={},
            frequency_maps={},
            imputation_medians={},
            chosen_time_column=None,
            time_derivatives=[],
        )


def test_feature_plan_json_round_trip(tmp_path: Path) -> None:
    """FeaturePlan.to_json() / from_json() round-trips losslessly."""
    from sorethumb.config import ColumnsConfig, Config, RunConfig, SourceConfig
    from sorethumb.profiling.plan import build_feature_plan

    df = pl.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0] * 10,
            "country": ["US", "UK", "CA", "US", "UK"] * 10,
        }
    )
    config = Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),
    )
    plan = build_feature_plan(df, config)
    restored = type(plan).from_json(plan.to_json())
    assert restored.n_rows == plan.n_rows
    assert restored.output_features == plan.output_features
    assert restored.schema_fingerprint == plan.schema_fingerprint


def test_build_feature_plan_with_reference_column(tmp_path: Path) -> None:
    """_build_protected includes reference_column when set."""
    from sorethumb.config import (
        ColumnsConfig,
        Config,
        FeaturesConfig,
        RunConfig,
        SourceConfig,
    )
    from sorethumb.profiling.classify import ColumnClass
    from sorethumb.profiling.plan import build_feature_plan

    df = pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "ref_col": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
        }
    )
    config = Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),
        columns=ColumnsConfig(reference_column="ref_col"),
    )
    plan = build_feature_plan(df, config)
    # ref_col should not be classified as identifier_like (it's protected)
    ref_decision = next((d for d in plan.decisions if d.column == "ref_col"), None)
    assert ref_decision is not None
    # ref_col is numeric, so it should be numeric class (not ignored by protection—just protected from id detection)
    assert ref_decision.col_class == ColumnClass.numeric


def test_build_feature_plan_non_chosen_temporal_dropped(tmp_path: Path) -> None:
    """When two temporal columns exist and none is config.time_column, the lower-cardinality one is dropped."""
    from datetime import date, timedelta

    from sorethumb.config import Config, RunConfig, SourceConfig
    from sorethumb.profiling.classify import Treatment
    from sorethumb.profiling.plan import build_feature_plan

    dates_a = [date(2020, 1, 1) + timedelta(days=i) for i in range(50)]
    dates_b = [date(2021, 1, 1)] * 50  # constant date, 1 unique

    df = pl.DataFrame(
        {
            "date_a": dates_a,  # 50 unique values — will be chosen
            "date_b": dates_b,  # 1 unique value — will be dropped
            "value": list(range(50)),
        }
    )
    config = Config(
        source=SourceConfig(uri="/dev/null"),
        run=RunConfig(workdir=str(tmp_path)),
    )
    plan = build_feature_plan(df, config)
    dropped = [d for d in plan.decisions if d.column == "date_b" and d.treatment == Treatment.drop]
    assert dropped, "non-chosen temporal column should be dropped"


# ---------------------------------------------------------------------------
# scoring/calibrate.py — import-time + uncovered paths
# ---------------------------------------------------------------------------


def test_calibrate_reexecuted_under_coverage() -> None:
    import sorethumb.scoring.calibrate as m

    importlib.reload(m)


def test_calibrator_invalid_mode_raises() -> None:
    from sorethumb.scoring.calibrate import Calibrator

    with pytest.raises(ValueError, match="mode"):
        Calibrator(mode="bad_mode")


def test_calibrator_transform_before_fit_raises() -> None:
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator()
    with pytest.raises(RuntimeError, match="fit"):
        c.transform(np.array([0.5, 0.6]))


def test_calibrator_transform_empty_scores() -> None:
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator()
    c.fit(np.linspace(0.0, 1.0, 100))
    result = c.transform(np.empty(0))
    assert len(result) == 0


def test_calibrator_constant_distribution_returns_half() -> None:
    """When all reference scores are identical, transform returns 0.5 for all rows."""
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator()
    c.fit(np.full(100, 0.5))  # std = 0
    result = c.transform(np.array([0.1, 0.5, 0.9]))
    assert np.allclose(result, 0.5)


def test_calibrator_reference_mode_with_reference_scores() -> None:
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator(mode="reference")
    train = np.linspace(0.0, 1.0, 100)
    ref = np.linspace(0.2, 0.8, 100)
    c.fit(train, reference_scores=ref)
    result = c.transform(train)
    assert result.shape == train.shape
    assert 0.0 <= result.min() <= result.max() <= 1.0


def test_calibrator_reference_mode_no_reference_falls_back(caplog) -> None:
    """mode='reference' with reference_scores=None falls back to train with a warning."""
    import logging

    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator(mode="reference")
    train = np.linspace(0.0, 1.0, 100)
    with caplog.at_level(logging.WARNING, logger="sorethumb.scoring.calibrate"):
        c.fit(train, reference_scores=None)
    assert any("falling back" in r.message for r in caplog.records)


def test_calibrator_to_dict_from_dict_round_trip() -> None:
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator(mode="self")
    c.fit(np.linspace(0.0, 1.0, 100))
    d = c.to_dict()
    c2 = Calibrator.from_dict(d)
    assert c2.mode == "self"
    result = c2.transform(np.array([0.5]))
    assert 0.0 <= float(result[0]) <= 1.0


def test_calibrator_from_dict_with_none_quantile_values() -> None:
    """from_dict when quantile_values is None leaves _quantile_values as None."""
    from sorethumb.scoring.calibrate import Calibrator

    d = {"mode": "self", "quantile_values": None}
    c = Calibrator.from_dict(d)
    assert c._quantile_values is None
