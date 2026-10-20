# Configuration reference

> **Auto-generated** from `src/sorethumb_ml/config.py` by `docs/generate_config_docs.py`.
> Do not edit manually — run `python docs/generate_config_docs.py` to regenerate.

sorethumb is configured through a single TOML file (default: `sorethumb.toml`).
A config file is optional — passing a data file directly to `sorethumb run`
uses all defaults with `workdir = "sorethumb-workspace"` (a directory in the current
directory, not next to the data file) and prompts to save a config on first run:

```bash
sorethumb run /path/to/data.parquet
```

Run `sorethumb init` to create a fully commented starter file with every option documented.
Run `sorethumb config schema` to emit the JSON schema.
For scenario-based TOML snippets see [configuration-examples.md](configuration-examples.md).

## Resolution order

1. `sorethumb.toml` (or `--config PATH`)
2. Environment variables with the `SORETHUMB_` prefix
3. Command-line flags (highest priority)

## Config hash

`Config.config_hash()` is a 32-character hex digest that covers all
result-affecting fields. Cosmetic or execution-only fields (`run.workdir`,
`run.log_level`, `run.slow_stage_seconds`, `run.reuse_models`,
`source.dataset_id`, and the entire `[report]` section) are excluded
so trivial changes do not invalidate cached artefacts.

## `[source]` — Where the raw data lives and how to fetch it.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source.uri` | str | **required** | Local path or http(s) URL to the source file. |
| `source.dataset_id` | str \| null | null | Stable logical identity for this dataset, kept constant across snapshots (appended rows, corrections, the file moving to a new path). All history -- periods, per-group totals, runs -- is keyed on it. When unset it is derived from 'uri'; set it explicitly so a change of path does not orphan prior history. Spellings of one Windows path that differ only in letter case, separators or a file:// prefix derive the same id. The query string is ignored when deriving, so sources that differ only in query values share one dataset; if the set of query keys changes between runs without a dataset_id, the run is refused -- set one to say whether they are the same dataset. Allowed characters: letters, digits, '.', '_', '-' (max 128). |
| `source.format` | "auto" \| "csv" \| "tsv" \| "parquet" \| "json" \| "jsonl" \| "tsf" | "auto" | File format. 'auto' infers from the file extension. Set explicitly when the extension is misleading. |
| `source.auth` | "none" \| "bearer" \| "basic" | "none" | HTTP authentication scheme. Token/credentials come from auth_env_var. Credentials are only ever sent over https: a plaintext http:// uri combined with auth other than 'none' is rejected. |
| `source.auth_env_var` | str \| null | null | Name of the environment variable that holds the auth credential. For auth='bearer', the token value. For auth='basic', 'user:password' in plain text -- it is base64-encoded automatically to build the header; do not pre-encode it. Read at runtime and never stored in the config or logs. |
| `source.read_options` | dict[str, object] | {} | Format-specific reader overrides, e.g. {'delimiter': '\|', 'null_values': ['NA']}. Passed verbatim to the polars scan_* call. |
| `source.cache` | bool | true | Cache downloaded files locally. Disable only for tiny or always-fresh sources. |
| `source.max_nesting_depth` | int | 5 | Maximum recursion depth for struct unnesting. 0 disables unnesting. |
| `source.max_download_bytes` | int | 2000000000 | Reject an http(s) download whose declared (Content-Length) or actual streamed size exceeds this many bytes. The streamed count is always enforced; a missing or malformed Content-Length is logged and ignored. Guards against an unbounded or misconfigured remote response. |

## `[columns]` — Logical roles for specific columns.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `columns.time_column` | str \| null | null | Primary timestamp column. Protected from identifier-drop heuristics and used for temporal features and history alignment. |
| `columns.group_by` | list[str] | [] | Columns to group by before scoring (e.g. site_id, region). Protected from dropping. |
| `columns.id_column` | str \| null | null | Row-identifier column to carry through to the output but exclude from features. |
| `columns.id_scope` | "auto" \| "dataset" \| "group" | "auto" | What 'unique' means for id_column, validated once over the full source population before any period/group/anomaly filtering: 'dataset' requires id_column unique across every row; 'group' requires it unique only within each group_by combination (a row_id can then repeat across groups, e.g. an order_id that resets per store_id); 'auto' picks 'group' when group_by is set, else 'dataset'. Ignored when id_column is unset. A violation raises SchemaError -- a duplicate or null id_column would otherwise make an output row_id ambiguous between multiple source records, undetectable downstream since only the flagged subset is ever persisted. |
| `columns.reference_column` | str \| null | null | Optional binary reference label (0/1) used only for evaluation, not training. |
| `columns.ignore` | list[str] | [] | Glob patterns of columns to ignore. Prefix with 'type:<polars_dtype>' to restrict the glob to columns of that dtype, e.g. 'type:String id_*'. |
| `columns.include_only` | list[str] \| null | null | When set, only these columns (exact names) plus protected ones -- time_column, group_by, id_column, reference_column -- are considered; every other column is ignored. A name not in the dataset is an error. Useful for narrowing scope without touching the source schema. |
| `columns.datetime_formats` | dict[str, str] | {} | Force a column to be parsed as a date/time, overriding auto-detection. The value is either a strptime-style format (e.g. '%d/%m/%Y') for a String column, or 'epoch_s' / 'epoch_ms' for an integer or float column holding Unix seconds or milliseconds. Auto-detection never guesses an ambiguous day/month order (e.g. both <= 12), and never treats an integer column as time -- this is the only way to resolve either case. A value that fails to parse becomes null, like any other missing value. |

## `[profiling]` — Thresholds that control column classification.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `profiling.null_ratio_drop` | float | 0.7 | Columns with null_ratio > this are classified 'high_null' and dropped. A missing-indicator feature is still emitted when missing_indicators=True. |
| `profiling.null_ratio_flag` | float | 0.0 | Columns with null_ratio > this (and <= null_ratio_drop) get a '<col>__is_missing' indicator feature appended. Default 0.0 means any column with at least one null emits an indicator. |
| `profiling.near_constant_distinct` | int | 2 | Drop a non-empty column with 2 < n_unique <= this as 'near_constant'. The default (2) drops none: a column is never dropped just for having few distinct values, since a rare level of a low-cardinality column can be exactly the anomaly. Constant (single-valued) columns are always dropped. Raise it only after measuring that it helps on your data. |
| `profiling.identifier_cardinality_ratio` | float | 0.9 | String columns with cardinality_ratio > this are candidates for identifier_like classification (subject to pattern checks). |
| `profiling.categorical_cardinality_ratio` | float | 0.01 | String columns with cardinality_ratio <= this are classified 'categorical'. Above this threshold they go through the identifier / free-text path. |
| `profiling.free_text_mean_length` | float | 20.0 | String columns with mean value length > this (after failing other checks) are classified 'free_text' and dropped. |
| `profiling.sample_rows_for_examples` | int | 1000 | Number of non-null rows sampled for example values and mean-length estimation. |
| `profiling.identifier_detection` | "conservative" \| "aggressive" \| "off" | "conservative" | 'conservative' only flags UUID/hex patterns at high cardinality. 'aggressive' also flags any high-cardinality string. 'off' never classifies as identifier_like (use when IDs carry signal). |

## `[features]` — Feature engineering options.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `features.one_hot_max_cardinality` | int | 20 | Categorical columns with n_unique <= this get one-hot encoding. Above this threshold frequency encoding is used instead. |
| `features.categorical_distance` | float \| null | 0.5 | How far apart two rows that differ only in one one-hot-encoded categorical column are, in the units numeric features are scaled to (robust scaling: interquartile ranges). The default 0.5 counts a change of category as about half a typical numeric spread, so uninformative categorical columns cannot swamp the distance-based detectors; Isolation Forest is unaffected. null scales one-hot dummies like numeric columns. See docs/approximations.md. |
| `features.max_feature_width` | int | 2000 | If the encoded feature matrix would exceed this many columns a FeatureWidthWarning is raised (or error in strict mode). |
| `features.missing_indicators` | bool | true | Emit '<col>__is_missing' features for columns whose null ratio exceeds profiling.null_ratio_flag (by default, any column with a null), including high-null columns that are otherwise dropped. |
| `features.array_features` | bool | true | Derive __len/__mean/__min/__max features from List columns. |
| `features.time_derivatives` | list[str] | ['hour', 'dayofweek', 'day', 'month'] | Derived features to extract from each date/time column (not only time_column). Supported: hour, dayofweek, day, month, year, quarter. |
| `features.scaler` | "standard" \| "robust" | "robust" | 'robust' uses median+IQR (less sensitive to extreme outliers). 'standard' uses mean+std (required if downstream models assume z-scores). |
| `features.dtype` | "float32" \| "float64" | "float32" | Output dtype of the feature matrix. float32 halves memory vs float64. |
| `features.correlation_reduction` | bool | true | Drop one column from each pair with Pearson \|r\| >= correlation_threshold. Reduces redundancy for distance-based detectors. |
| `features.correlation_threshold` | float | 0.95 | Correlation magnitude above which one of the pair is dropped. |
| `features.pca` | bool | false | Compress features with PCA after scaling. Useful when the feature matrix is very wide; adds latency and reduces explainability. |
| `features.pca_max_components` | int | 50 | Maximum number of PCA components to retain. |
| `features.pca_min_explained_variance` | float | 0.8 | Stop adding PCA components once cumulative explained variance exceeds this. |

## `[detectors]` — Per-detector block (repeatable `[[detectors]]`).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `detectors.name` | str | **required** | Detector identifier, e.g. 'isolation_forest'. |
| `detectors.enabled` | bool | true | Skip this detector entirely when False. |
| `detectors.params` | dict[str, object] | {} | Detector-specific hyperparameters, forwarded verbatim to the wrapper constructor. Nest an 'extra_params' table to pass arbitrary kwargs straight to the underlying sklearn estimator, e.g. params = { n_estimators = 300, extra_params = { n_jobs = 4 } }. Keys are validated at construction; reserved keys (random_state, contamination) and keys already exposed as wrapper arguments are rejected. |
| `detectors.train_row_cap` | int \| null | null | Subsample the training data to at most this many rows before fitting this detector. Unset uses the detector's built-in cap (isolation_forest 250000, kmeans_distance 200000, one_class_svm 25000, lof 50000, ecod/hbos 500000). Set a value larger than your dataset to train on every row. |

## `[scoring]` — How per-detector scores are combined.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `scoring.contamination` | str \| float | "auto" | Review-budget knob. NOT an estimate of how many anomalies the data contains -- it sizes the shortlist you are willing to review. A float in (0, 0.5] makes each detector flag exactly that fraction and sets the OneClassSVM nu to match. How many rows the run then flags depends on scoring.combination: exactly that fraction for 'composite'; only rows every detector flags for 'intersection' (the default -- usually far fewer); rows any detector flags for 'union' (usually more). 'auto' lets each detector use its own heuristic boundary (OCSVM zero-hyperplane, KMeans Tukey fence, IsolationForest offset); these rates differ per detector and the run output reports the realised rate of each. A detector whose boundary flagged more than half the rows it was fitted on does not vote (UninformativeBoundaryWarning). Pair a composite combination with a float: under 'auto' a composite thresholds at the median of those rates and has flagged several times a 5% shortlist on the benchmarks -- see docs/ensemble-policy.md. |
| `scoring.combination` | "composite" \| "intersection" \| "union" | "intersection" | 'composite' averages normalised detector scores and applies a single global threshold. 'intersection' thresholds each detector independently then flags rows where ALL detectors agree. 'union' thresholds each detector independently then flags rows where ANY detector agrees. The default, 'intersection', is a deliberate design decision that accepts known limitations (it can flag few or no rows on local, varying-density and contextual anomalies, and at 'auto' you do not control how many rows it flags); with a fixed review budget the evidenced choice is composite with weighting 'agreement' and a numeric contamination. Choose another policy on measured evidence: `sorethumb backtest` measures it on your own labelled data, and a run with a non-default policy that has no such evidence warns (run.strict refuses it) -- see docs/ensemble-policy.md. |
| `scoring.weighting` | "equal" \| "manual" \| "agreement" | "equal" | How detector weights are set for combination='composite' (ignored by 'intersection' / 'union'). 'equal': 1/n each. 'manual': from `weights`. 'agreement': by each detector's rank correlation (leave-one-out Spearman) with the consensus ranking; anti-correlated or flat detectors get weight 0. |
| `scoring.weights` | dict[str, float] | {} | Per-detector weights, used only when weighting='manual'. |
| `scoring.min_records` | int | 100 | Minimum rows a group needs to be scored. Smaller groups are skipped and reported with status too_few_records (not an error, even in strict mode). |

## `[explain]` — SHAP-based anomaly explanation controls.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `explain.enabled` | bool | true | Disable to skip the explanation stage entirely. |
| `explain.top_n` | int | 3 | Number of top contributing features to surface per anomalous row. |
| `explain.max_rows` | int | 5000 | Rows (most anomalous first) the gradient and KernelSHAP methods explain per group. Beyond it those detectors contribute nothing to a row's explanation; the other methods explain every flagged row. |
| `explain.kernel_shap` | bool | false | Explain One-Class SVM and LOF with KernelSHAP (needs the explain extra) instead of finite-difference gradients. Much slower; keep explain.max_rows small. Other detectors are unaffected. |
| `explain.permutation_importance` | bool | false | Also compute permutation importance per detector as a cross-check, written to the log only. Roughly doubles explanation runtime. |

## `[run]` — Execution-level settings.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `run.workdir` | str | **required** | Workspace root directory where all run artefacts are stored. Required when using the library; the CLI fills in ./sorethumb-workspace/ (relative to the current directory) when neither --workdir nor this field is given. |
| `run.seed` | int | 42 | Global random seed for reproducible results. |
| `run.strict` | bool | false | Promote every sorethumb warning to an error. Explanation warnings make the run exit partial instead of failing the group, so detection output is never lost. |
| `run.max_memory_mb` | int | 8192 | Pre-flight cap on the projected feature matrix's memory, sized for how many copies of it (rows x encoded columns x dtype bytes) can realistically be alive at once -- 2x the bare matrix size (one group's matrix plus polars encode/scale headroom), or 4x when explain.enabled (the default) and features.dtype='float32' (explain needs an additional float64 copy of the full matrix). A run whose estimate exceeds this aborts with MemoryBudgetError before any model is fitted. It is not a live RSS ceiling, and does not include detector-internal memory (e.g. a kernel-based fit's O(train_rows^2) usage) -- actual peak memory can still exceed the budget. |
| `run.max_rows` | int \| null | null | Truncate the input to at most this many rows (after filtering). Triggers SampleTruncatedWarning. None uses all rows. |
| `run.reuse_models` | bool | false | When this run already has a saved model for a group and detector (an earlier attempt of the same run, or a --force re-run), score with it instead of fitting again. score --from-run always reuses its source run's models, whatever this says. |
| `run.retention_days` | int | 90 | Default retention, in days, for sorethumb workspace prune (its --days overrides it). Nothing is pruned automatically. |
| `run.log_level` | str | "INFO" | Python logging level for the sorethumb_ml logger. Logs are written to both the console and {workdir}/logs/sorethumb.log (rotating, 10 MB limit, 5 backups). Does not affect the config hash. |
| `run.slow_stage_seconds` | int | 300 | Emit a SlowStageWarning if any pipeline stage exceeds this many seconds. Purely diagnostic; does not affect results. |

## `[history]` — Period-over-period baseline comparison.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `history.period_granularity` | "hour" \| "day" \| "week" \| "month" | "day" | Bucket size for period-over-period comparisons. |
| `history.roll_non_business` | bool | true | Treat only Monday-Friday as periods (day granularity): a run on a weekend is filed under the preceding Friday, backfill skips weekend days, and a trend window does not expect them. False makes every calendar day a period. There is no holiday calendar. |
| `history.lookback_periods` | int | 28 | How far back sorethumb backfill looks for missing periods once history exists, in periods before the current one. |
| `history.bootstrap_periods` | int | 28 | How many periods the first sorethumb backfill fills, when no period has been completed yet under this configuration. |
| `history.max_backfill_periods` | int | 30 | The most periods one sorethumb backfill runs (the most recent ones); --max-periods overrides it. |
| `history.low_volume_threshold` | int | 100 | A rolling window is flagged low-volume (in sorethumb history and the HTML report) when its current or prior population is below this -- informational only; low-volume periods are still included in every aggregate. |
| `history.anchor_run` | str \| null | null | Run id of a completed fitting run of this dataset whose models, calibrators and flag rule score every period run as well as that period's own fit. Gives a fixed scale, so sorethumb history can split a change in anomaly count into volume, score drift and threshold drift. Not part of the config hash, so setting it keeps the trend's history; periods already run are scored against it only when re-run. |

## `[report]` — Output report settings (cosmetic; excluded from config hash).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `report.formats` | list[str] | ['html', 'csv'] | Report formats to generate. Supported: html, csv, json. |
| `report.open_after` | bool | false | Open the HTML report in the default browser after generation. |
| `report.csv_bom` | bool | false | Start each CSV report with a UTF-8 byte-order mark. Excel on Windows needs it to read non-ASCII column names and values correctly; other tools don't. |
| `report.rolling_windows` | list[int] | [1, 7, 14, 28] | Rolling window sizes (in periods) shown by `sorethumb history`. |

## Detector `extra_params`

A pass-through to the underlying scikit-learn estimator.

`isolation_forest`, `lof`, `kmeans_distance` and `one_class_svm` each wrap a
scikit-learn estimator. Any constructor argument the wrapper does not expose
directly can be handed to it through a nested `extra_params` table:

```toml
[[detectors]]
name = "isolation_forest"
params = { n_estimators = 300, extra_params = { n_jobs = 4, max_features = 0.8 } }
```

Keys are validated when the detector is constructed, before any training runs:

- Keys sorethumb manages itself — `random_state`, `contamination`, `novelty`,
  `n_clusters` — are rejected. Use `run.seed`, `scoring.contamination`, or the
  detector's own `k` instead.
- Keys already exposed as a wrapper argument (`n_estimators`, `nu`, `n_neighbors`,
  `n_init`, …) are rejected, so there is one unambiguous source.
- Any other key the estimator does not accept is rejected up front, with the
  estimator's full parameter list in the error message.

`ecod` and `hbos` have no underlying estimator and reject any non-empty
`extra_params`.

### Accepted keys

Generated from the installed scikit-learn; the exact set may shift between
scikit-learn releases.

#### `isolation_forest`

| Key | scikit-learn default |
| --- | --- |
| `bootstrap` | `False` |
| `max_features` | `1.0` |
| `n_jobs` | `None` |
| `verbose` | `0` |
| `warm_start` | `False` |

#### `lof`

| Key | scikit-learn default |
| --- | --- |
| `algorithm` | `'auto'` |
| `leaf_size` | `30` |
| `metric` | `'minkowski'` |
| `metric_params` | `None` |
| `n_jobs` | `None` |
| `p` | `2` |

#### `kmeans_distance`

| Key | scikit-learn default |
| --- | --- |
| `algorithm` | `'lloyd'` |
| `copy_x` | `True` |
| `init` | `'k-means++'` |
| `max_iter` | `300` |
| `tol` | `0.0001` |
| `verbose` | `0` |

#### `one_class_svm`

| Key | scikit-learn default |
| --- | --- |
| `cache_size` | `200` |
| `coef0` | `0.0` |
| `degree` | `3` |
| `max_iter` | `-1` |
| `shrinking` | `True` |
| `tol` | `0.001` |
| `verbose` | `False` |
