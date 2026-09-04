"""Pydantic v2 configuration model for sorethumb.

One definition, used everywhere: the library receives a resolved ``Config``
object and never touches env vars or config files directly. Resolution
(TOML → env → flags) is the CLI layer's job (M8).

A change to any field that affects results invalidates cached artefacts;
``Config.config_hash()`` excludes purely cosmetic fields so trivial tweaks
don't bust the cache.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceConfig(BaseModel):
    """Where the raw data lives and how to read it."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="Local path or http(s) URL to the source file.")
    format: Literal["auto", "csv", "tsv", "parquet", "json", "jsonl", "tsf"] = Field(
        "auto",
        description=(
            "File format. 'auto' infers from the file extension. "
            "Set explicitly when the extension is misleading."
        ),
    )
    auth: Literal["none", "bearer", "basic"] = Field(
        "none",
        description="HTTP authentication scheme. Token/credentials come from auth_env_var.",
    )
    auth_env_var: str | None = Field(
        None,
        description=(
            "Name of the environment variable that holds the auth credential. "
            "The value is read at runtime and never stored in the config or logs."
        ),
    )
    read_options: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Format-specific reader overrides, e.g. {'delimiter': '|', 'null_values': ['NA']}. "
            "Passed verbatim to the polars scan_* call."
        ),
    )
    cache: bool = Field(
        True,
        description="Cache downloaded files locally. Disable only for tiny or always-fresh sources.",
    )
    max_nesting_depth: int = Field(
        5,
        ge=0,
        description="Maximum recursion depth for struct unnesting. 0 disables unnesting.",
    )


class ColumnsConfig(BaseModel):
    """Logical roles for specific columns."""

    model_config = ConfigDict(extra="forbid")

    time_column: str | None = Field(
        None,
        description=(
            "Primary timestamp column. Protected from identifier-drop heuristics "
            "and used for temporal features and history alignment."
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description=("Columns to group by before scoring (e.g. site_id, region). Protected from dropping."),
    )
    id_column: str | None = Field(
        None,
        description="Row-identifier column to carry through to the output but exclude from features.",
    )
    reference_column: str | None = Field(
        None,
        description="Optional binary reference label (0/1) used only for evaluation, not training.",
    )
    ignore: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns of columns to ignore. Prefix with 'type:<polars_dtype>' "
            "to restrict the glob to columns of that dtype, e.g. 'type:String id_*'."
        ),
    )
    include_only: list[str] | None = Field(
        None,
        description=(
            "When set, only these columns (plus protected ones) are considered. "
            "Useful for narrowing scope without touching the source schema."
        ),
    )


class ProfilingConfig(BaseModel):
    """Thresholds that control column classification during profiling."""

    model_config = ConfigDict(extra="forbid")

    null_ratio_drop: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Columns with null_ratio > this are classified 'high_null' and dropped. "
            "A missing-indicator feature is still emitted when missing_indicators=True."
        ),
    )
    null_ratio_flag: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Columns with null_ratio > this (and <= null_ratio_drop) get a "
            "'<col>__is_missing' indicator feature appended."
        ),
    )
    near_constant_distinct: int = Field(
        3,
        ge=2,
        description=(
            "A non-empty column with n_unique <= this is classified 'near_constant' "
            "and dropped (too low variance to be useful)."
        ),
    )
    identifier_cardinality_ratio: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description=(
            "String columns with cardinality_ratio > this are candidates for "
            "identifier_like classification (subject to pattern checks)."
        ),
    )
    categorical_cardinality_ratio: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description=(
            "String columns with cardinality_ratio <= this are classified 'categorical'. "
            "Above this threshold they go through the identifier / free-text path."
        ),
    )
    free_text_mean_length: float = Field(
        20.0,
        ge=1.0,
        description=(
            "String columns with mean value length > this (after failing other checks) "
            "are classified 'free_text' and dropped."
        ),
    )
    sample_rows_for_examples: int = Field(
        1000,
        ge=10,
        description="Number of non-null rows sampled for example values and mean-length estimation.",
    )
    identifier_detection: Literal["conservative", "aggressive", "off"] = Field(
        "conservative",
        description=(
            "'conservative' only flags UUID/hex patterns at high cardinality. "
            "'aggressive' also flags any high-cardinality string. "
            "'off' never classifies as identifier_like (use when IDs carry signal)."
        ),
    )


class FeaturesConfig(BaseModel):
    """Feature engineering knobs."""

    model_config = ConfigDict(extra="forbid")

    one_hot_max_cardinality: int = Field(
        20,
        ge=2,
        description=(
            "Categorical columns with n_unique <= this get one-hot encoding. "
            "Above this threshold frequency encoding is used instead."
        ),
    )
    max_feature_width: int = Field(
        2000,
        ge=1,
        description=(
            "If the encoded feature matrix would exceed this many columns a "
            "FeatureWidthWarning is raised (or error in strict mode)."
        ),
    )
    missing_indicators: bool = Field(
        True,
        description="Emit '<col>__is_missing' boolean features for high-null columns.",
    )
    array_features: bool = Field(
        True,
        description="Derive __len/__mean/__min/__max features from List columns.",
    )
    time_derivatives: list[str] = Field(
        default_factory=lambda: ["hour", "dayofweek", "day", "month"],
        description=(
            "Temporal derivatives to extract from the chosen time column. "
            "Supported: hour, dayofweek, day, month, year, quarter."
        ),
    )
    scaler: Literal["standard", "robust"] = Field(
        "robust",
        description=(
            "'robust' uses median+IQR (less sensitive to extreme outliers). "
            "'standard' uses mean+std (required if downstream models assume z-scores)."
        ),
    )
    dtype: Literal["float32", "float64"] = Field(
        "float32",
        description="Output dtype of the feature matrix. float32 halves memory vs float64.",
    )
    correlation_reduction: bool = Field(
        True,
        description=(
            "Drop one column from each pair with Pearson |r| > correlation_threshold. "
            "Reduces redundancy for distance-based detectors."
        ),
    )
    correlation_threshold: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Correlation magnitude above which one of the pair is dropped.",
    )
    pca: bool = Field(
        False,
        description=(
            "Compress features with PCA after scaling. Useful when the feature "
            "matrix is very wide; adds latency and reduces explainability."
        ),
    )
    pca_max_components: int = Field(
        50,
        ge=1,
        description="Maximum number of PCA components to retain.",
    )
    pca_min_explained_variance: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Stop adding PCA components once cumulative explained variance exceeds this.",
    )


class DetectorConfig(BaseModel):
    """Configuration for a single anomaly detector."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Detector identifier, e.g. 'isolation_forest'.")
    enabled: bool = Field(True, description="Skip this detector entirely when False.")
    params: dict[str, object] = Field(
        default_factory=dict,
        description="Detector-specific hyperparameters, forwarded verbatim to the constructor.",
    )
    train_row_cap: int | None = Field(
        None,
        ge=1,
        description=(
            "Subsample training data to at most this many rows. "
            "None means use the full training set. "
            "Detectors with quadratic complexity (SVM) need a low cap."
        ),
    )


def _default_detectors() -> list[DetectorConfig]:
    return [
        DetectorConfig(name="isolation_forest", train_row_cap=250_000),
        DetectorConfig(name="kmeans_distance", train_row_cap=200_000),
        DetectorConfig(name="one_class_svm", train_row_cap=20_000),
    ]


class ScoringConfig(BaseModel):
    """How per-detector scores are combined into a single anomaly score."""

    model_config = ConfigDict(extra="forbid")

    contamination: str | float = Field(
        "auto",
        description=(
            "Expected fraction of anomalies. 'auto' uses each detector's natural boundary "
            "(OCSVM zero-hyperplane, KMeans Tukey fence, IsolationForest offset). "
            "A float in (0, 0.5] makes each detector flag exactly that fraction of rows "
            "and also sets the OneClassSVM nu training parameter to match."
        ),
    )
    combination: Literal["composite", "intersection", "union"] = Field(
        "intersection",
        description=(
            "'composite' averages normalised detector scores and applies a single global threshold. "
            "'intersection' thresholds each detector independently then flags rows where ALL detectors agree (high precision). "
            "'union' thresholds each detector independently then flags rows where ANY detector agrees (high recall)."
        ),
    )
    weighting: Literal["equal", "manual", "agreement"] = Field(
        "equal",
        description="How detector weights are determined when combination='composite'.",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-detector weights, used only when weighting='manual'.",
    )
    min_records: int = Field(
        100,
        ge=1,
        description=(
            "Minimum rows needed to run scoring. Fewer rows raise a CalibrationModeWarning "
            "(or error in strict mode)."
        ),
    )

    @field_validator("contamination")
    @classmethod
    def _validate_contamination(cls, v: str | float) -> str | float:
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
        else:
            v = float(v)
            if not (0.0 < v <= 0.5):
                raise ValueError("contamination float must be in (0, 0.5]")
        return v


class ExplainConfig(BaseModel):
    """Controls SHAP-based anomaly explanations."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Disable to skip the explanation stage entirely.")
    top_n: int = Field(
        3,
        ge=1,
        description="Number of top contributing features to surface per anomalous row.",
    )
    max_rows: int = Field(
        5000,
        ge=1,
        description="Explain at most this many rows; rows beyond the cap are skipped.",
    )
    kernel_shap: bool = Field(
        False,
        description=(
            "Fall back to KernelSHAP for models where TreeSHAP is unavailable. "
            "Much slower; sets a FallbackAttributionWarning."
        ),
    )
    permutation_importance: bool = Field(
        False,
        description=(
            "Also compute permutation importance as a cross-check. Roughly doubles explanation runtime."
        ),
    )


class RunConfig(BaseModel):
    """Execution-level settings. workdir is required; everything else has a default."""

    model_config = ConfigDict(extra="forbid")

    workdir: str = Field(description="Workspace root directory where all run artefacts are stored.")
    seed: int = Field(42, description="Global random seed for reproducible results.")
    strict: bool = Field(
        False,
        description="Promote all SorethumbWarnings to errors. Always active in the test suite.",
    )
    max_memory_mb: int = Field(
        8192,
        ge=256,
        description=(
            "Approximate RSS budget. Triggers MemoryBudgetError if exceeded mid-run. "
            "Set generously; the check is coarse."
        ),
    )
    max_rows: int | None = Field(
        None,
        ge=1,
        description=(
            "Truncate the input to at most this many rows (after filtering). "
            "Triggers SampleTruncatedWarning. None uses all rows."
        ),
    )
    reuse_models: bool = Field(
        False,
        description=(
            "If a matching model artefact exists in workdir, skip retraining. Useful for score-forward runs."
        ),
    )
    retention_days: int = Field(
        90,
        ge=1,
        description="Prune run artefacts older than this many days from workdir.",
    )
    log_level: str = Field(
        "INFO",
        description=(
            "Python logging level for the sorethumb logger. "
            "Logs are written to both the console and {workdir}/logs/sorethumb.log "
            "(rotating, 10 MB limit, 5 backups). Does not affect the config hash."
        ),
    )
    slow_stage_seconds: int = Field(
        300,
        ge=1,
        description=(
            "Emit a SlowStageWarning if any pipeline stage exceeds this many seconds. "
            "Purely diagnostic; does not affect results."
        ),
    )


class HistoryConfig(BaseModel):
    """Controls baseline comparison and trend detection."""

    model_config = ConfigDict(extra="forbid")

    period_granularity: Literal["hour", "day", "week", "month"] = Field(
        "day",
        description="Bucket size for period-over-period comparisons.",
    )
    roll_non_business: bool = Field(
        True,
        description="Include weekends and holidays when computing rolling baselines.",
    )
    lookback_periods: int = Field(
        28,
        ge=1,
        description="Number of historical periods used to build the baseline.",
    )
    bootstrap_periods: int = Field(
        28,
        ge=1,
        description="Minimum periods of history required before history scoring activates.",
    )
    max_backfill_periods: int = Field(
        30,
        ge=0,
        description="Maximum periods to backfill when catch-up runs are requested.",
    )
    low_volume_threshold: int = Field(
        100,
        ge=1,
        description=(
            "Periods with fewer rows than this emit a PopulationMismatchWarning "
            "and are excluded from the baseline."
        ),
    )


class ReportConfig(BaseModel):
    """Output report settings. Excluded from the config hash (cosmetic only)."""

    model_config = ConfigDict(extra="forbid")

    formats: list[str] = Field(
        default_factory=lambda: ["html", "csv"],
        description="Report formats to generate. Supported: html, csv, json.",
    )
    open_after: bool = Field(
        False,
        description="Open the HTML report in the default browser after generation.",
    )
    rolling_windows: list[int] = Field(
        default_factory=lambda: [1, 7, 14, 28],
        description="Rolling window sizes (in periods) shown in trend charts.",
    )


class Config(BaseModel):
    """Top-level sorethumb configuration.

    The library always receives a fully-resolved ``Config`` object.
    Construct it programmatically in tests; use the CLI layer for TOML/env/flag
    resolution in production.
    """

    model_config = ConfigDict(extra="forbid")

    source: SourceConfig
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    detectors: list[DetectorConfig] = Field(default_factory=_default_detectors)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    run: RunConfig
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    def config_hash(self) -> str:
        """16-char hex hash covering only result-affecting fields.

        Excludes run.workdir, run.log_level, run.slow_stage_seconds, and the
        entire report section so purely cosmetic changes don't bust artefact caches.
        """
        d = self.model_dump()
        run = d["run"]
        for key in ("workdir", "log_level", "slow_stage_seconds"):
            run.pop(key, None)
        d.pop("report", None)
        serialised = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]


# Convenience type alias used across the codebase
AnyConfig = Annotated[Config, Field(description="Resolved sorethumb configuration.")]
