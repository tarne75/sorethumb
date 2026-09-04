# Configuration examples

Scenario-based TOML snippets. Every example shows only the fields being changed
from their defaults; unset optional fields keep their documented defaults.
See [configuration.md](configuration.md) for the full field reference.

---

## `[source]`

### Local CSV with a non-standard delimiter

```toml
[source]
uri = "/data/exports/transactions.psv"
format = "csv"

[source.read_options]
separator = "|"
null_values = ["NA", "N/A", ""]

[run]
workdir = "./workspace"
```

### Remote Parquet behind bearer auth

The token is never written to the config file — it is read from the named
environment variable at runtime.

```toml
[source]
uri = "https://data.internal/exports/events.parquet"
auth = "bearer"
auth_env_var = "DATA_API_TOKEN"

[run]
workdir = "./workspace"
```

### Monash TSF time series archive

Each series in the file becomes one row; `@attribute` columns are preserved and
observations expand to `value_0`, `value_1`, ….

```toml
[source]
uri = "/data/electricity_hourly_dataset.tsf"

[run]
workdir = "./workspace"
```

### Nested JSON with struct flattening disabled

```toml
[source]
uri = "/data/events.json"
max_nesting_depth = 0   # keep nested structs as-is; do not unnest

[run]
workdir = "./workspace"
```

---

## `[columns]`

### E-commerce transactions

Exclude the surrogate key, declare the event timestamp, and group by store so
each store gets its own anomaly model.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[columns]
id_column     = "transaction_id"
time_column   = "created_at"
group_by      = ["store_id"]
```

### IoT sensor readings

Sensors share a device ID and emit many metadata string columns that are not
useful for anomaly scoring. Glob them away.

```toml
[source]
uri = "/data/sensor_readings.parquet"

[run]
workdir = "./workspace"

[columns]
id_column   = "reading_id"
time_column = "timestamp"
group_by    = ["device_id", "site_id"]
ignore      = ["firmware_version", "meta_*", "type:String *_tag"]
```

### Narrow scope with `include_only`

When you only care about a handful of columns and want everything else ignored
without listing it explicitly.

```toml
[source]
uri = "/data/wide_survey.csv"

[run]
workdir = "./workspace"

[columns]
id_column    = "respondent_id"
include_only = ["age", "income_band", "region", "score_q1", "score_q2", "score_q3"]
```

### Labelled evaluation dataset

`reference_column` is a known 0/1 ground-truth label used only to compute
precision/recall after scoring — it is never fed to the detectors.

```toml
[source]
uri = "/data/kdd_labelled.csv"

[run]
workdir = "./workspace"

[columns]
reference_column = "is_attack"
```

---

## `[profiling]`

### Noisy external feed with many missing values

Lower the drop threshold so columns with up to 85 % nulls are kept, and raise
the flag threshold so missing-indicator features are only added when truly
warranted.

```toml
[source]
uri = "/data/external_feed.csv"

[run]
workdir = "./workspace"

[profiling]
null_ratio_drop = 0.85
null_ratio_flag = 0.50
```

### IDs that carry real signal

The default `conservative` identifier detection would drop high-cardinality
string columns that follow a UUID-like pattern. Setting `off` forces them
through the feature pipeline instead.

```toml
[source]
uri = "/data/network_events.parquet"

[run]
workdir = "./workspace"

[profiling]
identifier_detection = "off"
```

### Very fine-grained categorical columns

The default `categorical_cardinality_ratio` of 0.01 treats columns with more
than 1 % distinct values as high-cardinality. For small datasets this is too
tight and useful categories get missed.

```toml
[source]
uri = "/data/products.csv"

[run]
workdir = "./workspace"

[profiling]
categorical_cardinality_ratio = 0.05   # allow up to 5 % distinct values
```

### Aggressive identifier pruning

Useful when you have many internal reference codes that truly carry no signal
and you want them dropped without having to list each one in `columns.ignore`.

```toml
[source]
uri = "/data/crm_export.csv"

[run]
workdir = "./workspace"

[profiling]
identifier_detection           = "aggressive"
identifier_cardinality_ratio   = 0.80   # flag at 80 % unique rather than 90 %
```

---

## `[features]`

### Wide sensor matrix — use PCA to compress

Sensors produce hundreds of correlated channels. PCA keeps 90 % of variance in
at most 40 components.

```toml
[source]
uri = "/data/factory_sensors.parquet"

[run]
workdir = "./workspace"

[features]
pca                       = true
pca_max_components        = 40
pca_min_explained_variance = 0.90
```

### Financial time series — custom temporal derivatives

Daily trading data. Hour-of-day and day-of-week carry no meaning; year and
quarter matter.

```toml
[source]
uri = "/data/trade_log.parquet"

[run]
workdir = "./workspace"

[columns]
time_column = "trade_date"

[features]
time_derivatives = ["day", "month", "quarter", "year"]
```

### Preserve all features for interpretability

Disable correlation reduction so every original feature is visible in SHAP
explanations — useful when you need to explain decisions to stakeholders.

```toml
[source]
uri = "/data/loan_applications.csv"

[run]
workdir = "./workspace"

[features]
correlation_reduction = false
```

### High-cardinality categoricals

Allow one-hot encoding for columns with up to 50 distinct values and use
`float64` throughout for a downstream model that requires it.

```toml
[source]
uri = "/data/events.csv"

[run]
workdir = "./workspace"

[features]
one_hot_max_cardinality = 50
dtype                   = "float64"
```

---

## `[[detectors]]`

The `[[detectors]]` block is a TOML array of tables — each entry is one detector.
Omitting the block entirely uses the built-in defaults
(`isolation_forest`, `kmeans_distance`, `one_class_svm`).

### Large dataset — disable SVM, cap the others

`one_class_svm` has quadratic complexity; drop it and reduce the row caps for
faster training on 1 M+ row datasets.

```toml
[source]
uri = "/data/large_events.parquet"

[run]
workdir = "./workspace"

[[detectors]]
name          = "isolation_forest"
train_row_cap = 500_000

[[detectors]]
name          = "kmeans_distance"
train_row_cap = 300_000
```

### Tune Isolation Forest hyperparameters

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[[detectors]]
name = "isolation_forest"
[detectors.params]
n_estimators  = 300
max_samples   = 0.8
random_state  = 42

[[detectors]]
name = "kmeans_distance"

[[detectors]]
name          = "one_class_svm"
train_row_cap = 20_000
```

### Lightweight single-detector run

Good for a fast first-pass or CI smoke test.

```toml
[source]
uri = "/data/sample.csv"

[run]
workdir = "./workspace"

[[detectors]]
name = "isolation_forest"
```

### Disable one detector without removing it

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[[detectors]]
name = "isolation_forest"

[[detectors]]
name    = "kmeans_distance"
enabled = false   # skip without deleting the block

[[detectors]]
name          = "one_class_svm"
train_row_cap = 20_000
```

---

## `[scoring]`

### Known contamination rate

If domain knowledge says roughly 2 % of records are anomalous, set it directly
instead of letting sorethumb estimate from the score distribution.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[scoring]
contamination = 0.02
```

### Conservative flagging with intersection

Each detector independently flags its top `contamination` fraction of rows.
A row is only marked anomalous if **all three detectors** agree — not just the
combined score. This maximises precision at the cost of recall and also sets
the OneClassSVM `nu` training parameter to match the contamination rate.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[scoring]
combination   = "intersection"
contamination = 0.05
```

Use `combination = "union"` for the opposite: flag a row if *any* detector
considers it anomalous (maximises recall).

Use `combination = "composite"` (the default) to blend scores into a single
ranked list and apply one global threshold — better when detectors disagree
often and you want a smooth ranking rather than a hard vote.

### Manual detector weighting

Weight Isolation Forest more heavily because it performs best on this dataset
(from a prior benchmark run).

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[scoring]
weighting = "manual"

[scoring.weights]
isolation_forest = 0.6
kmeans_distance  = 0.3
one_class_svm    = 0.1
```

---

## `[explain]`

### Disable explanations for speed

SHAP attribution is the slowest stage. Skip it when you only need ranked scores.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[explain]
enabled = false
```

### Surface more contributing features per row

Show the top 7 features per anomalous row and explain up to 10 000 rows.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[explain]
top_n    = 7
max_rows = 10_000
```

### KernelSHAP fallback for non-tree models

Required if you add a custom detector that is not tree-based. Much slower than
TreeSHAP — keep `max_rows` small.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[explain]
kernel_shap = true
max_rows    = 500
```

---

## `[run]`

### Reproducible research run

Pin the seed and reuse trained models when re-scoring with the same config.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir       = "./research/run-2026-09"
seed          = 1234
reuse_models  = true
```

### Memory-constrained environment

Hard-cap at 4 GB and subsample to 200 000 rows if the dataset is larger.

```toml
[source]
uri = "/data/large.parquet"

[run]
workdir        = "./workspace"
max_memory_mb  = 4096
max_rows       = 200_000
```

### Verbose logging with a slow-stage alert

Emit a warning if any pipeline stage takes more than 2 minutes.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir              = "./workspace"
log_level            = "DEBUG"
slow_stage_seconds   = 120
```

---

## `[history]`

### Hourly monitoring pipeline

Compare each hour's batch against the same hour in previous days.

```toml
[source]
uri = "/data/hourly_events.parquet"

[run]
workdir = "./workspace"

[columns]
time_column = "event_time"

[history]
period_granularity  = "hour"
lookback_periods    = 168    # one week of hours
bootstrap_periods   = 72     # require three days before activating
```

### Monthly business reporting

Exclude weekends from the baseline to avoid weekend-vs-weekday drift.

```toml
[source]
uri = "/data/monthly_sales.parquet"

[run]
workdir = "./workspace"

[columns]
time_column = "sale_date"

[history]
period_granularity    = "month"
roll_non_business     = false
lookback_periods      = 12
low_volume_threshold  = 500
```

---

## `[report]`

### JSON output for a downstream pipeline

Disable HTML and CSV; emit only machine-readable JSON.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[report]
formats = ["json"]
```

### Open report in browser automatically

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[report]
open_after = true
```

### Custom trend-chart windows

Show 3-day, 7-day, and 30-day rolling windows in the HTML trend charts.

```toml
[source]
uri = "/data/transactions.csv"

[run]
workdir = "./workspace"

[report]
rolling_windows = [3, 7, 30]
```
