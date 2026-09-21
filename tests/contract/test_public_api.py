"""Contract tests for sorethumb's public API surface: every top-level and
sub-package __all__, the error/warning hierarchy, and logging configuration.
"""

from __future__ import annotations

import logging
import warnings

import pytest

pytestmark = pytest.mark.contract

# ── sorethumb.errors ──────────────────────────────────────────────────────────


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
        ConfigError,
        DetectorError,
        ExplainError,
        MemoryBudgetError,
        ModelSchemaDriftError,
        PlanError,
        SchemaError,
        SourceError,
        StoreError,
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
        SampleTruncatedWarning,
        SlowStageWarning,
        SorethumbWarning,
    )

    for cls in (
        CalibrationModeWarning,
        ColumnDroppedWarning,
        FallbackAttributionWarning,
        FeatureWidthWarning,
        LowVarianceWarning,
        ModelSchemaDriftWarning,
        NonFiniteWarning,
        SampleTruncatedWarning,
        SlowStageWarning,
    ):
        assert issubclass(cls, SorethumbWarning)
        assert issubclass(cls, UserWarning)


def test_sorethumb_errors_can_be_raised_and_caught() -> None:
    from sorethumb.errors import ConfigError, SchemaError, SourceError

    for cls in (ConfigError, SchemaError, SourceError):
        with pytest.raises(cls, match="boom"):
            raise cls("boom")


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


def test_scoring_public_api_matches_all_list() -> None:
    import sorethumb.scoring as scoring
    from sorethumb.scoring import Calibrator, ScoreEnsemble  # proves both names resolve

    assert scoring.Calibrator is Calibrator
    assert scoring.ScoreEnsemble is ScoreEnsemble
    assert set(scoring.__all__) == {"Calibrator", "ScoreEnsemble"}


# ── sorethumb.store.__init__ ──────────────────────────────────────────────────


def test_store_public_api_matches_all_list() -> None:
    import sorethumb.store as store
    from sorethumb.store import Store, Workspace, make_group_key  # proves all three resolve

    assert store.Store is Store
    assert store.Workspace is Workspace
    assert callable(make_group_key)
    assert set(store.__all__) == {"Store", "Workspace", "make_group_key"}


# ── sorethumb.explain.__init__ ────────────────────────────────────────────────


def test_explain_public_api_matches_all_list() -> None:
    import sorethumb.explain as explain
    from sorethumb.explain import (
        aggregate_to_original,
        back_project_pca,
        blend,
        centroid_attributions,
        ecod_attributions,
        gradient_attributions,
        hbos_attributions,
        tree_shap_attributions,
    )

    for fn in (
        aggregate_to_original,
        back_project_pca,
        blend,
        centroid_attributions,
        ecod_attributions,
        gradient_attributions,
        hbos_attributions,
        tree_shap_attributions,
    ):
        assert callable(fn)
    assert set(explain.__all__) == {
        "aggregate_to_original",
        "back_project_pca",
        "blend",
        "centroid_attributions",
        "ecod_attributions",
        "gradient_attributions",
        "hbos_attributions",
        "tree_shap_attributions",
    }


# ── sorethumb.__init__ (package public API) ───────────────────────────────────


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
    assert sc.dataset_id is None


def test_source_config_dataset_id_validation() -> None:
    import pytest

    import sorethumb

    assert sorethumb.SourceConfig(uri="/tmp/x.csv", dataset_id="sales.eu-2024").dataset_id == "sales.eu-2024"
    for bad in ("has space", "slash/id", "café", "x" * 129, ""):
        with pytest.raises(ValueError, match="dataset_id"):
            sorethumb.SourceConfig(uri="/tmp/x.csv", dataset_id=bad)
