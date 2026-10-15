"""Exception and warning hierarchy for sorethumb.

Every degradation point issues a specific named warning so that strict mode
(run.strict = True) can promote exactly the right ones to exceptions, and so
that a failed strict-mode run names the exact config field that controls it.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar


class SorethumbError(Exception):
    """Base exception. All library errors are subclasses.

    ``failure_kind`` classifies the failure for the CLI's documented exit codes:
    ``"runtime"`` (the default: work was attempted and failed), ``"preflight"``
    (the inputs were rejected before any work: configuration, schema, source,
    plan) or ``"not_found"`` (a requested run, workspace or persisted file does
    not exist). Subclasses override it; third-party subclasses inherit
    ``"runtime"``.
    """

    failure_kind: str = "runtime"


class ConfigError(SorethumbError):
    """Raised for invalid or missing configuration values."""

    failure_kind = "preflight"


class SourceError(SorethumbError):
    """Raised when a data source cannot be resolved or fetched."""

    failure_kind = "preflight"


class SchemaError(SorethumbError):
    """Raised when a dataset's schema is unreadable or ambiguous."""

    failure_kind = "preflight"


class PlanError(SorethumbError):
    """Raised when a FeaturePlan is invalid or cannot be applied."""

    failure_kind = "preflight"


class DetectorError(SorethumbError):
    """Raised when a detector fails to fit, score, or register."""


class ExplainError(SorethumbError):
    """Raised when attributions cannot be computed."""


class StoreError(SorethumbError):
    """Raised for workspace or database access failures."""


class NotFoundError(StoreError):
    """A requested run, workspace or persisted file does not exist.

    A :class:`StoreError` subclass, so existing ``except StoreError`` handlers
    keep working; the CLI reports it with the dedicated not-found exit code.
    """

    failure_kind = "not_found"


class FileInUseError(StoreError):
    """A file could not be replaced or deleted because another program holds it open.

    Windows refuses to replace or delete a file another process has open
    without delete sharing -- a report CSV open in a spreadsheet, a file-sync
    client mid-upload, antivirus scanning a fresh file. sorethumb retries
    briefly first (most such locks are momentary); this is raised only once
    the lock outlasts the retries, naming the file. A :class:`StoreError`
    subclass, so existing ``except StoreError`` handlers keep working.
    """


class PathTooLongError(StoreError):
    """The workspace is too deep for Windows' 260-character path limit.

    A pre-flight check (Windows only, skipped when long paths are enabled),
    raised before any model is fitted rather than as a bare FileNotFoundError
    at the first deep write minutes into a run.
    """

    failure_kind = "preflight"


class MemoryBudgetError(SorethumbError):
    """Raised when the projected feature-matrix size exceeds run.max_memory_mb.

    A pre-flight check in features.build, before any model is fitted — not a live
    RSS ceiling.
    """

    failure_kind = "preflight"


class UnevidencedPolicyError(ConfigError):
    """``run.strict`` refused a non-default ensemble policy that has no backtest evidence.

    The strict-mode form of :class:`UnevidencedPolicyWarning`: raised before
    the run is registered, so nothing is fitted or recorded.
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

    Every warning is also a structured record (see :func:`warning_record`): a
    stable ``code`` (the subclass default, or a more specific one passed in),
    plus, where they apply, the ``detector`` concerned and the number of rows
    affected. ``str(warning)`` is the human-readable message.
    """

    code: ClassVar[str] = "warning"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        detector: str | None = None,
        rows_affected: int | None = None,
    ) -> None:
        """Create a warning with *message*, optionally overriding ``code`` and naming a detector and row count."""
        super().__init__(message)
        if code is not None:
            self.code = code  # type: ignore[misc]  # an instance-specific code
        self.detector = detector
        self.rows_affected = rows_affected


class ColumnDroppedWarning(SorethumbWarning):
    """A column was dropped from the feature plan."""

    code: ClassVar[str] = "profile.column_dropped"


class FeatureWidthWarning(SorethumbWarning):
    """One-hot columns were demoted to frequency encoding to stay within max_feature_width."""

    code: ClassVar[str] = "features.width_demoted"


class LowVarianceWarning(SorethumbWarning):
    """PCA retained components explain less than pca_min_explained_variance of total variance."""

    code: ClassVar[str] = "features.pca_low_variance"


class SampleTruncatedWarning(SorethumbWarning):
    """A sample was truncated to fit within a row cap."""

    code: ClassVar[str] = "data.sample_truncated"


class FallbackAttributionWarning(SorethumbWarning):
    """TreeSHAP failed and the gradient method was used as a fallback attribution."""

    code: ClassVar[str] = "explain.fallback_used"


class AttributionBackendWarning(SorethumbWarning):
    """A detector's attribution backend raised; that detector contributes no explanation.

    Emitted once per detector and group. Other detectors' explanations for the
    group are unaffected. Like every explanation warning it never changes the
    group's detection output; under ``run.strict`` it makes the run exit
    ``partial`` instead of failing the group.
    """

    code: ClassVar[str] = "explain.backend_failed"


class ExplanationFailedWarning(SorethumbWarning):
    """No explanation could be produced for a group's flagged rows.

    The explanation stage raised, or no attribution source explained any row.
    The group's detection output (scores, flags, ranks, count) was already
    written and is unaffected; the group's ``explain_status`` is ``failed``
    and the run exits ``partial``.
    """

    code: ClassVar[str] = "explain.stage_failed"


class ExplanationCoverageWarning(SorethumbWarning):
    """Some flagged rows, or part of their score, could not be explained.

    Codes: ``explain.rows_beyond_max_rows`` (rows past ``explain.max_rows`` for
    every source that covers them), ``explain.pca_backprojection_failed`` (no
    flagged row has an original-column explanation), ``explain.detector_unsupported``
    (a detector with no vetted attribution method: its part of the score is not
    explained), ``explain.no_signal`` (every source attributed some rows as all
    zero), ``explain.reference_fallback`` (too few consensus-normal rows, so the
    less anomalous half was the reference; or a score-forward run whose fitting
    run saved no reference summary) and ``explain.reference_unavailable`` (no
    population normal enough to explain against; no row explained). Never
    changes detection output; under ``run.strict`` the run exits partial.
    """

    code: ClassVar[str] = "explain.coverage"


class ModelSchemaDriftWarning(SorethumbWarning):
    """A persisted model's feature schema no longer matches; the group was refit."""

    code: ClassVar[str] = "model.schema_drift"


class ModelVersionMismatchWarning(SorethumbWarning):
    """A persisted model was fitted under different library versions; scores may not be reproducible."""

    code: ClassVar[str] = "model.version_mismatch"


class SlowStageWarning(SorethumbWarning):
    """A stage or group exceeded run.slow_stage_seconds."""

    code: ClassVar[str] = "run.slow_stage"


class AntiCorrelatedMemberWarning(SorethumbWarning):
    """A detector ranked anti-correlated with the ensemble but was kept anyway.

    Raised for combination="intersection"/"union", where dropping a member
    would silently change the vote count (a configured three-way intersection
    quietly becoming two-way). The bad-member guard still drops members for
    combination="composite", where dropping does not change the decision rule.
    """

    code: ClassVar[str] = "scoring.anticorrelated_member"


class AmbiguousDateFormatWarning(SorethumbWarning):
    """A String column parses as a date under more than one day/month order.

    Auto-detection never guesses between the two; the column is left as text.
    Set ``columns.datetime_formats[col]`` to the correct order to resolve it.

    Deliberately never promoted by ``run.strict``: the outcome -- the column
    stays text and is encoded like any other category -- is exactly what every
    earlier version did, so failing a strict run over it would turn a
    previously passing configuration into a failing one for no new risk.
    """

    code: ClassVar[str] = "profile.ambiguous_date_format"


class UnevidencedPolicyWarning(SorethumbWarning):
    """A non-default ensemble policy is in use without backtest evidence that it outperforms the default.

    Raised when ``scoring.combination``/``scoring.weighting`` select a policy
    other than the default and the workspace holds no ``sorethumb backtest``
    record, for this dataset and this configuration, in which that policy
    outperforms the default. The run proceeds; ``run.strict`` refuses it
    instead (:class:`UnevidencedPolicyError`). See docs/ensemble-policy.md.
    """

    code: ClassVar[str] = "policy.unevidenced"


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

    code: ClassVar[str] = "scoring.zero_flagged"


class ReportGenerationWarning(SorethumbWarning):
    """Report rendering failed after detection/scoring already succeeded.

    Deliberately never promoted by ``run.strict``, unlike every other
    warning here: detection and scoring already completed and were
    persisted correctly by the time this fires, so a rendering problem is a
    presentation-layer concern layered on top of already-valid results, not
    a signal that the results themselves are suspect. RunResult.report_status
    ("success" | "failed" | "skipped") is the authoritative, structured
    record of this outcome; this warning exists for visibility alongside it.
    """

    code: ClassVar[str] = "report.failed"


def warning_record(warning: Warning | str, *, group: str | None = None) -> dict[str, Any]:
    """Return the structured form of *warning* persisted and reported by a run.

    ``{"code", "stage", "message", "group", "detector", "rows_affected"}``:
    ``stage`` is the part of ``code`` before the first dot. A warning that is
    not a :class:`SorethumbWarning` gets the code ``external``.
    """
    code = warning.code if isinstance(warning, SorethumbWarning) else "external"
    return {
        "code": code,
        "stage": code.split(".", 1)[0],
        "message": str(warning),
        "group": group,
        "detector": getattr(warning, "detector", None),
        "rows_affected": getattr(warning, "rows_affected", None),
    }


def warning_records_from_json(raw: str | None) -> list[dict[str, Any]]:
    """Parse persisted ``warnings_json`` into records; older plain-string lists become ``legacy`` records."""
    if not raw:
        return []
    records: list[dict[str, Any]] = []
    for item in json.loads(raw):
        if isinstance(item, dict):
            records.append(item)
        else:
            records.append(
                {
                    "code": "legacy",
                    "stage": "legacy",
                    "message": str(item),
                    "group": None,
                    "detector": None,
                    "rows_affected": None,
                }
            )
    return records
