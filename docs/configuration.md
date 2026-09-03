# Configuration reference

> **Auto-generated** from `src/sorethumb/config.py` by `docs/generate_config_docs.py`.
> Do not edit manually — run `python docs/generate_config_docs.py` to regenerate.

sorethumb is configured through a single TOML file (default: `sorethumb.toml`).
Run `sorethumb init` to create a fully commented starter file.
Run `sorethumb config schema` to emit the JSON schema.
For scenario-based TOML snippets see [configuration-examples.md](configuration-examples.md).

## Resolution order

1. `sorethumb.toml` (or `--config PATH`)
2. Environment variables with the `SORETHUMB_` prefix
3. Command-line flags (highest priority)

## Config hash

`Config.config_hash()` is a 16-character hex digest that covers all
result-affecting fields. Cosmetic fields (`run.log_level`,
`run.slow_stage_seconds`, and the entire `[report]` section) are excluded
so trivial changes do not invalidate cached artefacts.

## `[source]` — Where the raw data lives and how to fetch it.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source.uri` | str | **required** | Local path or http(s) URL to the source file. |
| `source.format` | "auto" \| "csv" \| "tsv" \| "parquet" \| "json" \| "jsonl" \| "tsf" | "auto" | File format. 'auto' infers from the file extension. Set explicitly when the extension is misleading. |
| `source.auth` | "none" \| "bearer" \| "basic" | "none" | HTTP authentication scheme. Token/credentials come from auth_env_var. |
| `source.auth_env_var` | str \| null | null | Name of the environment variable that holds the auth credential. The value is read at runtime and never stored in the config or logs. |
| `source.read_options` | dict[str, object] | {} | Format-specific reader overrides, e.g. {'delimiter': '\|', 'null_values': ['NA']}. Passed verbatim to the polars scan_* call. |
| `source.cache` | bool | true | Cache downloaded files locally. Disable only for tiny or always-fresh sources. |
| `source.max_nesting_depth` | int | 5 | Maximum recursion depth for struct unnesting. 0 disables unnesting. |

## `[columns]` — Logical roles for specific columns.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `columns.time_column` | str \| null | null | Primary timestamp column. Protected from identifier-drop heuristics and used for temporal features and history alignment. |
| `columns.group_by` | list[str] | [] | Columns to group by before scoring (e.g. site_id, region). Protected from dropping. |
| `columns.id_column` | str \| null | null | Row-identifier column to carry through to the output but exclude from features. |
| `columns.reference_column` | str \| null | null | Optional binary reference label (0/1) used only for evaluation, not training. |
| `columns.ignore` | list[str] | [] | Glob patterns of columns to ignore. Prefix with 'type:<polars_dtype>' to restrict the glob to columns of that dtype, e.g. 'type:String id_*'. |
| `columns.include_only` | list[str] \| null | null | When set, only these columns (plus protected ones) are considered. Useful for narrowing scope without touching the source schema. |

## `[profiling]` — Thresholds that control column classification.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `profiling.null_ratio_drop` | float | 0.7 | Columns with null_ratio > this are classified 'high_null' and dropped. A missing-indicator feature is still emitted when missing_indicators=True. |
| `profiling.null_ratio_flag` | float | 0.3 | Columns with null_ratio > this (and <= null_ratio_drop) get a '<col>__is_missing' indicator feature appended. |
| `profiling.near_constant_distinct` | int | 3 | A non-empty column with n_unique <= this is classified 'near_constant' and dropped (too low variance to be useful). |
| `profiling.identifier_cardinality_ratio` | float | 0.9 | String columns with cardinality_ratio > this are candidates for identifier_like classification (subject to pattern checks). |
| `profiling.categorical_cardinality_ratio` | float | 0.01 | String columns with cardinality_ratio <= this are classified 'categorical'. Above this threshold they go through the identifier / free-text path. |
| `profiling.free_text_mean_length` | float | 20.0 | String columns with mean value length > this (after failing other checks) are classified 'free_text' and dropped. |
| `profiling.sample_rows_for_examples` | int | 1000 | Number of non-null rows sampled for example values and mean-length estimation. |
| `profiling.identifier_detection` | "conservative" \| "aggressive" \| "off" | "conservative" | 'conservative' only flags UUID/hex patterns at high cardinality. 'aggressive' also flags any high-cardinality string. 'off' never classifies as identifier_like (use when IDs carry signal). |

## `[features]` — Feature engineering options.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `features.one_hot_max_cardinality` | int | 20 | Categorical columns with n_unique <= this get one-hot encoding. Above this threshold frequency encoding is used instead. |
| `features.max_feature_width` | int | 2000 | If the encoded feature matrix would exceed this many columns a FeatureWidthWarning is raised (or error in strict mode). |
| `features.missing_indicators` | bool | true | Emit '<col>__is_missing' boolean features for high-null columns. |
| `features.array_features` | bool | true | Derive __len/__mean/__min/__max features from List columns. |
| `features.time_derivatives` | list[str] | ['hour', 'dayofweek', 'day', 'month'] | Temporal derivatives to extract from the chosen time column. Supported: hour, dayofweek, day, month, year, quarter. |
| `features.scaler` | "standard" \| "robust" | "robust" | 'robust' uses median+IQR (less sensitive to extreme outliers). 'standard' uses mean+std (required if downstream models assume z-scores). |
| `features.dtype` | "float32" \| "float64" | "float32" | Output dtype of the feature matrix. float32 halves memory vs float64. |
| `features.correlation_reduction` | bool | true | Drop one column from each pair with Pearson \|r\| > correlation_threshold. Reduces redundancy for distance-based detectors. |
| `features.correlation_threshold` | float | 0.95 | Correlation magnitude above which one of the pair is dropped. |
| `features.pca` | bool | false | Compress features with PCA after scaling. Useful when the feature matrix is very wide; adds latency and reduces explainability. |
| `features.pca_max_components` | int | 50 | Maximum number of PCA components to retain. |
| `features.pca_min_explained_variance` | float | 0.8 | Stop adding PCA components once cumulative explained variance exceeds this. |

## `[detectors]` — Per-detector block (repeatable `[[detectors]]`).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `detectors.name` | str | **required** | Detector identifier, e.g. 'isolation_forest'. |
| `detectors.enabled` | bool | true | Skip this detector entirely when False. |
| `detectors.params` | dict[str, object] | {} | Detector-specific hyperparameters, forwarded verbatim to the constructor. |
| `detectors.train_row_cap` | int \| null | null | Subsample training data to at most this many rows. None means use the full training set. Detectors with quadratic complexity (SVM) need a low cap. |

## `[scoring]` — How per-detector scores are combined.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `scoring.contamination` | str \| float | "auto" | Expected fraction of anomalies. 'auto' estimates from the score distribution. A float in (0, 0.5] sets the threshold directly. |
| `scoring.combination` | "composite" \| "intersection" \| "union" | "composite" | 'composite' averages normalised detector scores. 'intersection'/'union' use voting across detectors. |
| `scoring.weighting` | "equal" \| "manual" \| "agreement" | "equal" | How detector weights are determined when combination='composite'. |
| `scoring.weights` | dict[str, float] | {} | Per-detector weights, used only when weighting='manual'. |
| `scoring.min_records` | int | 100 | Minimum rows needed to run scoring. Fewer rows raise a CalibrationModeWarning (or error in strict mode). |

## `[explain]` — SHAP-based anomaly explanation controls.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `explain.enabled` | bool | true | Disable to skip the explanation stage entirely. |
| `explain.top_n` | int | 3 | Number of top contributing features to surface per anomalous row. |
| `explain.max_rows` | int | 5000 | Explain at most this many rows; rows beyond the cap are skipped. |
| `explain.kernel_shap` | bool | false | Fall back to KernelSHAP for models where TreeSHAP is unavailable. Much slower; sets a FallbackAttributionWarning. |
| `explain.permutation_importance` | bool | false | Also compute permutation importance as a cross-check. Roughly doubles explanation runtime. |

## `[run]` — Execution-level settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `run.workdir` | str | **required** | Workspace root directory where all run artefacts are stored. |
| `run.seed` | int | 42 | Global random seed for reproducible results. |
| `run.strict` | bool | false | Promote all SorethumbWarnings to errors. Always active in the test suite. |
| `run.max_memory_mb` | int | 8192 | Approximate RSS budget. Triggers MemoryBudgetError if exceeded mid-run. Set generously; the check is coarse. |
| `run.max_rows` | int \| null | null | Truncate the input to at most this many rows (after filtering). Triggers SampleTruncatedWarning. None uses all rows. |
| `run.reuse_models` | bool | false | If a matching model artefact exists in workdir, skip retraining. Useful for score-forward runs. |
| `run.retention_days` | int | 90 | Prune run artefacts older than this many days from workdir. |
| `run.log_level` | str | "INFO" | Python logging level for the sorethumb logger. Does not affect the config hash. |
| `run.slow_stage_seconds` | int | 300 | Emit a SlowStageWarning if any pipeline stage exceeds this many seconds. Purely diagnostic; does not affect results. |

## `[history]` — Period-over-period baseline comparison.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `history.period_granularity` | "hour" \| "day" \| "week" \| "month" | "day" | Bucket size for period-over-period comparisons. |
| `history.roll_non_business` | bool | true | Include weekends and holidays when computing rolling baselines. |
| `history.lookback_periods` | int | 28 | Number of historical periods used to build the baseline. |
| `history.bootstrap_periods` | int | 28 | Minimum periods of history required before history scoring activates. |
| `history.max_backfill_periods` | int | 30 | Maximum periods to backfill when catch-up runs are requested. |
| `history.low_volume_threshold` | int | 100 | Periods with fewer rows than this emit a PopulationMismatchWarning and are excluded from the baseline. |

## `[report]` — Output report settings (cosmetic; excluded from config hash).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `report.formats` | list[str] | ['html', 'csv'] | Report formats to generate. Supported: html, csv, json. |
| `report.open_after` | bool | false | Open the HTML report in the default browser after generation. |
| `report.rolling_windows` | list[int] | [1, 7, 14, 28] | Rolling window sizes (in periods) shown in trend charts. |
