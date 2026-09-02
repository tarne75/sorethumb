"""Exception and warning hierarchy for sorethumb.

Every degradation point issues a specific named warning so that strict mode
(run.strict = True) can promote exactly the right ones to exceptions, and so
that a failed strict-mode run names the exact config field that controls it.
"""


class SorethumbError(Exception):
    """Base exception. All library errors are subclasses."""


class ConfigError(SorethumbError):
    """Raised for invalid or missing configuration values."""


class SourceError(SorethumbError):
    """Raised when a data source cannot be resolved or fetched."""


class SchemaError(SorethumbError):
    """Raised when a dataset's schema is unreadable or ambiguous."""


class PlanError(SorethumbError):
    """Raised when a FeaturePlan is invalid or cannot be applied."""


class DetectorError(SorethumbError):
    """Raised when a detector fails to fit, score, or register."""


class ExplainError(SorethumbError):
    """Raised when attributions cannot be computed."""


class StoreError(SorethumbError):
    """Raised for workspace or database access failures."""


class MemoryBudgetError(SorethumbError):
    """Raised when a pre-flight memory estimate exceeds run.max_memory_mb."""


class ModelSchemaDriftError(SorethumbError):
    """Raised in strict mode when a persisted model's feature schema no longer matches."""


class SorethumbWarning(UserWarning):
    """Base warning. Promoted to an exception when run.strict = True.

    Each subclass corresponds to one degradation point. The name is what makes
    a strict-mode failure diagnosable — use the most specific subclass available.
    """


class ColumnDroppedWarning(SorethumbWarning):
    """A column was dropped from the feature plan."""


class FeatureWidthWarning(SorethumbWarning):
    """One-hot columns were demoted to frequency encoding to stay within max_feature_width."""


class NonFiniteWarning(SorethumbWarning):
    """NaN or ±Inf values were found and replaced with 0.0 during feature construction."""


class LowVarianceWarning(SorethumbWarning):
    """PCA retained components explain less than pca_min_explained_variance of total variance."""


class SampleTruncatedWarning(SorethumbWarning):
    """A sample was truncated to fit within a row cap."""


class FallbackAttributionWarning(SorethumbWarning):
    """TreeSHAP failed and the gradient method was used as a fallback attribution."""


class ModelSchemaDriftWarning(SorethumbWarning):
    """A persisted model's feature schema no longer matches; the group was refit."""


class PopulationMismatchWarning(SorethumbWarning):
    """Population frame is missing a grouping column or period; rate set to unknown."""


class CalibrationModeWarning(SorethumbWarning):
    """Calibration mode changed between runs; cross-run score comparison is invalid."""


class SlowStageWarning(SorethumbWarning):
    """A stage or group exceeded run.slow_stage_seconds."""
