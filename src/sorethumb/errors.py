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
    """Raised when the projected feature-matrix size exceeds run.max_memory_mb.

    A pre-flight check in features.build, before any model is fitted — not a live
    RSS ceiling.
    """


class ModelSchemaDriftError(SorethumbError):
    """Raised in strict mode when a persisted model's feature schema no longer matches."""


class ModelVersionMismatchError(SorethumbError):
    """Raised in strict mode when a persisted model was fitted under different library versions."""


class ModelIntegrityError(SorethumbError):
    """Raised when a persisted model artifact fails an identity or content check.

    Covers a manifest whose recorded (run_id, group_key, detector_name, plan_digest)
    doesn't match what it's being loaded for, and an estimator/calibrator file whose
    content no longer matches the digest recorded at save time -- both signal the
    on-disk bundle was corrupted, truncated, or swapped with another model's files.
    Unlike drift/version warnings, this always fails closed regardless of ``strict``:
    there is no safe degraded behaviour for an artifact that fails an integrity check.
    """


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


class ModelVersionMismatchWarning(SorethumbWarning):
    """A persisted model was fitted under different library versions; scores may not be reproducible."""


class PopulationMismatchWarning(SorethumbWarning):
    """Population frame is missing a grouping column or period; rate set to unknown."""


class CalibrationModeWarning(SorethumbWarning):
    """Calibration mode changed between runs; cross-run score comparison is invalid."""


class SlowStageWarning(SorethumbWarning):
    """A stage or group exceeded run.slow_stage_seconds."""


class AntiCorrelatedMemberWarning(SorethumbWarning):
    """A detector ranked anti-correlated with the ensemble but was kept anyway.

    Raised for combination="intersection"/"union", where dropping a member
    would silently change the vote count (a configured three-way intersection
    quietly becoming two-way). The bad-member guard still drops members for
    combination="composite", where dropping does not change the decision rule.
    """


class ZeroAnomalyWarning(SorethumbWarning):
    """A successful group's configured intersection flagged zero rows.

    Every configured detector's vote is required by definition for
    combination="intersection" (see AntiCorrelatedMemberWarning); when their
    top-scoring sets never overlap, the intersection is legitimately empty.
    This does not change the default automatically -- it surfaces the
    realised per-detector rates so the empty result can be told apart from a
    silent failure, with composite/union or an explicit contamination named
    as alternatives to consider.
    """
