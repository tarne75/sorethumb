"""Contract tests for sorethumb's public API surface: every top-level and
sub-package __all__, the error/warning hierarchy, and logging configuration.
"""

from __future__ import annotations

import logging
import warnings

import pytest

pytestmark = pytest.mark.contract

# ── sorethumb_ml.errors ──────────────────────────────────────────────────────────


def test_all_sorethumb_errors_are_exceptions() -> None:
    from sorethumb_ml.errors import (
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
    from sorethumb_ml.errors import (
        AttributionBackendWarning,
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
        AttributionBackendWarning,
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
    from sorethumb_ml.errors import ConfigError, SchemaError, SourceError

    for cls in (ConfigError, SchemaError, SourceError):
        with pytest.raises(cls, match="boom"):
            raise cls("boom")


def test_sorethumb_warnings_can_be_issued() -> None:
    from sorethumb_ml.errors import ColumnDroppedWarning, SorethumbWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("dropped col_x", ColumnDroppedWarning, stacklevel=1)

    assert len(caught) == 1
    assert issubclass(caught[0].category, ColumnDroppedWarning)
    assert issubclass(ColumnDroppedWarning, SorethumbWarning)


# ── sorethumb_ml.logging ────────────────────────────────────────────────────


def test_logging_configure_runs() -> None:
    from sorethumb_ml.logging import configure

    configure("WARNING")
    assert logging.getLogger("sorethumb_ml").level == 0  # basicConfig is a no-op when handlers exist


def test_logging_configure_accepts_all_levels() -> None:
    from sorethumb_ml.logging import configure

    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        configure(level)  # must not raise


def test_logging_configure_invalid_level_falls_back() -> None:
    from sorethumb_ml.logging import configure

    configure("NOT_A_LEVEL")  # falls back to INFO via getattr(..., logging.INFO)


# ── sorethumb_ml.scoring.__init__ ────────────────────────────────────────────────


def test_scoring_public_api_matches_all_list() -> None:
    import sorethumb_ml.scoring as scoring
    from sorethumb_ml.scoring import Calibrator, ScoreEnsemble  # proves both names resolve

    assert scoring.Calibrator is Calibrator
    assert scoring.ScoreEnsemble is ScoreEnsemble
    assert set(scoring.__all__) == {"Calibrator", "ScoreEnsemble"}


# ── sorethumb_ml.store.__init__ ──────────────────────────────────────────────────


def test_store_public_api_matches_all_list() -> None:
    import sorethumb_ml.store as store
    from sorethumb_ml.store import Store, Workspace, make_group_key  # proves all three resolve

    assert store.Store is Store
    assert store.Workspace is Workspace
    assert callable(make_group_key)
    assert set(store.__all__) == {"Store", "Workspace", "make_group_key"}


# ── sorethumb_ml.explain.__init__ ────────────────────────────────────────────────


def test_explain_public_api_matches_all_list() -> None:
    import sorethumb_ml.explain as explain
    from sorethumb_ml.explain import (
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


# ── sorethumb_ml.__init__ (package public API) ───────────────────────────────────


# The intended top-level export set. Changing it is a public-API change: edit this list in the
# same commit, add a CHANGELOG entry, and follow docs/stability.md.
_EXPECTED_TOP_LEVEL_EXPORTS = frozenset(
    {
        "Config",
        "Detector",
        "FeaturePlan",
        "FeatureSpace",
        "GroupSummary",
        "Metrics",
        "RunResult",
        "SorethumbError",
        "SourceConfig",
        "Workspace",
        "__version__",
        "apply_feature_plan",
        "build_feature_plan",
        "evaluate_scores",
        "list_detectors",
        "load_dataset",
        "render_report_for_run",
        "run_detection",
        "score_forward",
    }
)


def test_package_all_is_exactly_the_intended_export_set() -> None:
    """Exact equality, not a subset: an accidental addition or a dropped name both fail."""
    import sorethumb_ml

    exported = list(sorethumb_ml.__all__)
    assert len(exported) == len(set(exported)), "duplicate names in sorethumb_ml.__all__"
    assert set(exported) == _EXPECTED_TOP_LEVEL_EXPORTS
    assert exported == sorted(exported), "__all__ is kept sorted"


def test_no_public_name_is_importable_from_the_top_level_without_being_exported() -> None:
    """A public-looking attribute that is not in __all__ is a leak that star-imports hide and
    attribute access exposes. Submodules imported as a side effect are not names the package
    defines, so they are ignored."""
    import types

    import sorethumb_ml

    leaked = sorted(
        name
        for name, value in vars(sorethumb_ml).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    )
    assert set(leaked) == _EXPECTED_TOP_LEVEL_EXPORTS - {"__version__"}


def test_every_exported_name_is_documented_in_the_package_docstring() -> None:
    import sorethumb_ml

    doc = sorethumb_ml.__doc__ or ""
    missing = [name for name in sorethumb_ml.__all__ if name not in doc]
    assert not missing, f"exported but not mentioned in the package docstring: {missing}"


def test_package_all_exports_importable() -> None:
    import sorethumb_ml

    for name in sorethumb_ml.__all__:
        assert hasattr(sorethumb_ml, name), f"sorethumb_ml.{name} missing from package"


def test_package_public_api_callable_or_instantiable() -> None:
    import sorethumb_ml

    callables = [
        sorethumb_ml.run_detection,
        sorethumb_ml.load_dataset,
        sorethumb_ml.list_detectors,
        sorethumb_ml.build_feature_plan,
        sorethumb_ml.apply_feature_plan,
        sorethumb_ml.evaluate_scores,
    ]
    for fn in callables:
        assert callable(fn), f"{fn} should be callable"


def test_list_detectors_returns_known_detectors() -> None:
    import sorethumb_ml

    names = sorethumb_ml.list_detectors()
    assert isinstance(names, list)
    assert "isolation_forest" in names
    assert "kmeans_distance" in names
    assert "one_class_svm" in names


def test_config_and_source_config_constructable() -> None:
    import sorethumb_ml

    sc = sorethumb_ml.SourceConfig(uri="/tmp/test.csv")
    assert sc.uri == "/tmp/test.csv"
    assert sc.format == "auto"
    assert sc.dataset_id is None


def test_source_config_dataset_id_validation() -> None:
    import pytest

    import sorethumb_ml

    assert (
        sorethumb_ml.SourceConfig(uri="/tmp/x.csv", dataset_id="sales.eu-2024").dataset_id == "sales.eu-2024"
    )
    for bad in ("has space", "slash/id", "café", "x" * 129, ""):
        with pytest.raises(ValueError, match="dataset_id"):
            sorethumb_ml.SourceConfig(uri="/tmp/x.csv", dataset_id=bad)
