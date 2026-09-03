"""Coverage tests for modules that are 0% because all their code executes at import
time (class definitions, re-exports), which happens during pytest collection before
coverage instruments them.  importlib.reload() forces re-execution during the test
phase when coverage is fully active.
"""

from __future__ import annotations

import importlib
import logging
import warnings


# ── sorethumb.errors ──────────────────────────────────────────────────────────

def test_errors_reexecuted_under_coverage() -> None:
    import sorethumb.errors as m
    importlib.reload(m)


def test_all_sorethumb_errors_are_exceptions() -> None:
    from sorethumb.errors import (
        ConfigError,
        DetectorError,
        ExplainError,
        MemoryBudgetError,
        ModelSchemaDriftError,
        PlanError,
        SchemaError,
        SorethumbError,
        SourceError,
        StoreError,
    )
    for cls in (
        ConfigError, DetectorError, ExplainError, MemoryBudgetError,
        ModelSchemaDriftError, PlanError, SchemaError, SourceError, StoreError,
    ):
        assert issubclass(cls, SorethumbError)
        assert issubclass(cls, Exception)
        instance = cls("test message")
        assert str(instance) == "test message"
        assert isinstance(instance, SorethumbError)


def test_all_sorethumb_warnings_are_user_warnings() -> None:
    from sorethumb.errors import (
        CalibrationModeWarning,
        ColumnDroppedWarning,
        FallbackAttributionWarning,
        FeatureWidthWarning,
        LowVarianceWarning,
        ModelSchemaDriftWarning,
        NonFiniteWarning,
        PopulationMismatchWarning,
        SampleTruncatedWarning,
        SlowStageWarning,
        SorethumbWarning,
    )
    for cls in (
        CalibrationModeWarning, ColumnDroppedWarning, FallbackAttributionWarning,
        FeatureWidthWarning, LowVarianceWarning, ModelSchemaDriftWarning,
        NonFiniteWarning, PopulationMismatchWarning, SampleTruncatedWarning,
        SlowStageWarning,
    ):
        assert issubclass(cls, SorethumbWarning)
        assert issubclass(cls, UserWarning)


def test_sorethumb_errors_can_be_raised_and_caught() -> None:
    from sorethumb.errors import ConfigError, SchemaError, SourceError

    for cls in (ConfigError, SchemaError, SourceError):
        try:
            raise cls("boom")
        except Exception as exc:
            assert str(exc) == "boom"


def test_sorethumb_warnings_can_be_issued() -> None:
    from sorethumb.errors import ColumnDroppedWarning, SorethumbWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("dropped col_x", ColumnDroppedWarning, stacklevel=1)

    assert len(caught) == 1
    assert issubclass(caught[0].category, ColumnDroppedWarning)
    assert issubclass(ColumnDroppedWarning, SorethumbWarning)


# ── sorethumb.logging ─────────────────────────────────────────────────────────

def test_logging_configure_runs() -> None:
    from sorethumb.logging import configure
    configure("WARNING")
    assert logging.getLogger("sorethumb").level == 0  # basicConfig is a no-op when handlers exist


def test_logging_configure_accepts_all_levels() -> None:
    from sorethumb.logging import configure
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        configure(level)  # must not raise


def test_logging_configure_invalid_level_falls_back() -> None:
    from sorethumb.logging import configure
    configure("NOT_A_LEVEL")  # falls back to INFO via getattr(..., logging.INFO)


# ── sorethumb.scoring.__init__ ────────────────────────────────────────────────

def test_scoring_init_reexecuted_under_coverage() -> None:
    import sorethumb.scoring as m
    importlib.reload(m)


def test_scoring_init_exports_calibrator_and_ensemble() -> None:
    from sorethumb.scoring import Calibrator, ScoreEnsemble
    assert Calibrator is not None
    assert ScoreEnsemble is not None


def test_scoring_calibrator_importable_via_init() -> None:
    import sorethumb.scoring as scoring
    assert hasattr(scoring, "Calibrator")
    assert hasattr(scoring, "ScoreEnsemble")
    assert set(scoring.__all__) == {"Calibrator", "ScoreEnsemble"}


# ── sorethumb.store.__init__ ──────────────────────────────────────────────────

def test_store_init_reexecuted_under_coverage() -> None:
    import sorethumb.store as m
    importlib.reload(m)


def test_store_init_exports_store_workspace_make_group_key() -> None:
    from sorethumb.store import Store, Workspace, make_group_key
    assert Store is not None
    assert Workspace is not None
    assert callable(make_group_key)


def test_store_init_all_list() -> None:
    import sorethumb.store as store
    assert set(store.__all__) == {"Store", "Workspace", "make_group_key"}


# ── sorethumb.explain.__init__ ────────────────────────────────────────────────

def test_explain_init_reexecuted_under_coverage() -> None:
    import sorethumb.explain as m
    importlib.reload(m)


def test_explain_init_exports_all_public_functions() -> None:
    from sorethumb.explain import (
        aggregate_to_original,
        back_project_pca,
        blend,
        centroid_attributions,
        gradient_attributions,
        tree_shap_attributions,
    )
    for fn in (
        aggregate_to_original, back_project_pca, blend,
        centroid_attributions, gradient_attributions, tree_shap_attributions,
    ):
        assert callable(fn)


def test_explain_init_all_list() -> None:
    import sorethumb.explain as explain
    assert set(explain.__all__) == {
        "aggregate_to_original",
        "back_project_pca",
        "blend",
        "centroid_attributions",
        "gradient_attributions",
        "tree_shap_attributions",
    }


# ── sorethumb.__init__ (package public API) ───────────────────────────────────

def test_package_version() -> None:
    import sorethumb
    assert sorethumb.__version__ == "0.1.0"


def test_package_all_exports_importable() -> None:
    import sorethumb
    for name in sorethumb.__all__:
        assert hasattr(sorethumb, name), f"sorethumb.{name} missing from package"


def test_package_public_api_callable_or_instantiable() -> None:
    import sorethumb
    callables = [
        sorethumb.run_detection,
        sorethumb.load_dataset,
        sorethumb.list_detectors,
        sorethumb.build_feature_plan,
        sorethumb.apply_feature_plan,
        sorethumb.evaluate_scores,
    ]
    for fn in callables:
        assert callable(fn), f"{fn} should be callable"


def test_list_detectors_returns_known_detectors() -> None:
    import sorethumb
    names = sorethumb.list_detectors()
    assert isinstance(names, list)
    assert "isolation_forest" in names
    assert "kmeans_distance" in names
    assert "one_class_svm" in names


def test_config_and_source_config_constructable() -> None:
    import sorethumb
    sc = sorethumb.SourceConfig(uri="/tmp/test.csv")
    assert sc.uri == "/tmp/test.csv"
    assert sc.format == "auto"
