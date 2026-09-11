<div align="center">
  <img src="docs/banner.svg" alt="sorethumb — unsupervised anomaly detection for tabular data" width="860"/>
</div>

<div align="center">

[![CI](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml/badge.svg)](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/tarne75/sorethumb/graph/badge.svg?token=UJLJS8GHDV)](https://codecov.io/github/tarne75/sorethumb)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?color=123A66)](https://github.com/tarne75/sorethumb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?color=2F80ED)](LICENSE)

</div>

`sorethumb` takes a dataset it knows nothing about, works out how to treat every
column, fits an ensemble of detectors, ranks the records that stand out, and
explains *why* each one stands out in terms of the original columns — so the
result is actionable without a background in machine learning.

**What it is:** a *ranker* — an ordering of records by statistical oddity — and,
if you set a review budget, a *selector* that hands back the top slice of that
ranking. **What it is not:** a prevalence estimator. The flagged count is the
size of the shortlist you asked for, not a measurement of how many anomalies
your data contains. `contamination = "auto"` is the median of three heuristic
per-detector cut-offs; the run output reports each detector's realised rate so
the number is never mistaken for ground truth.

It is built to run on a single machine — a laptop, a CI runner, one VM — for
datasets from roughly ten thousand to a few million rows. At that size a
distributed engine adds cost, operational surface, and non-determinism without
buying meaningful speed, so there is no Spark, Dask, or cloud-service dependency
anywhere. The detectors whose training cost grows faster than linearly (LOF at
roughly `O(n·log²n)`, one-class SVM at `O(n²)`) are each fitted on a bounded row
sample rather than the whole dataset; every row is still scored. The
[Scale guide](#scale-guide) and [Honest limitations](#honest-limitations) spell
out where these choices bite.

> **Status: pre-release (0.1.0).** The API and on-disk formats may still change
> between minor versions, and the package is not yet published to PyPI — install
> from source (see below). See [Honest limitations](#honest-limitations) for what
> it does not do.

---

## Why sorethumb?

There are already good anomaly-detection libraries (PyOD ships far more detectors).
`sorethumb` is not competing on detector count. Its contribution is the surrounding
machinery those libraries leave to the user:

1. **Zero-configuration column handling** — profile, classify, encode, impute, derive.
2. **Ensemble scoring with percentile-calibrated scores.** Each run maps its raw
   detector scores onto [0, 1] against its own training distribution. Scores are
   comparable *across* runs only when a fitted calibrator is reused —
   `sorethumb score --from-run RUN_ID` — since a plain run self-calibrates.
3. **Per-record explanations** in original feature terms, labelled exact or heuristic.
4. **Run history you can reason about** — models and calibrators are persisted, so a
   `score --from-run` trend line moves with the data, not with a refitted model. A
   plain `backfill` refits each period; its trend shows relative movement, not an
   absolute drift level (see limitations).
5. **Idempotent, resumable execution** with a completion ledger.

It runs on a single machine, uses Polars throughout, and has no dependency on Spark,
Databricks, or any cloud vendor.

---

## Installation

Not yet on PyPI. Install from a clone:

```bash
git clone https://github.com/tarne75/sorethumb
cd sorethumb
pip install .
```

For the benchmark harness:

```bash
pip install ".[benchmark]"
```

---

## Supported file formats

| Format | Extensions | Notes |
| --- | --- | --- |
| Parquet | `.parquet` | Recommended for large datasets; column-oriented, fast |
| CSV | `.csv`, `.csv.gz` | Auto-detects delimiter; override via `read_options.separator` |
| TSV | `.tsv`, `.tsv.gz` | Tab-separated; `\t` separator injected automatically |
| JSON | `.json`, `.json.gz` | Full document loaded eagerly (not streamed) |
| JSONL / NDJSON | `.jsonl`, `.ndjson`, `.jsonl.gz`, `.ndjson.gz` | Streamed line-by-line |
| TSF | `.tsf` | [Monash Time Series Forecasting](https://github.com/rakshitha123/TSForecasting/tree/master/utils) format — each series becomes one row; `@attribute` columns preserved, series observations expand to `value_0`, `value_1`, … |

Format is auto-detected from the file extension. Set `source.format` explicitly when the extension is ambiguous:

```toml
[source]
uri = "/data/my_file.dat"
format = "csv"
```

---

## 60-second quickstart

```python
import polars as pl
from sklearn.datasets import fetch_kddcup99

from sorethumb import Config, SourceConfig, run_detection

# Fetch the KDDCup99 10% subset and write a small CSV for the demo.
# as_frame=True keeps real column names and dtypes (protocol_type / service /
# flag are categorical); decode those bytes columns to text and take 20k rows
# so the run finishes in about a minute.
bunch = fetch_kddcup99(percent10=True, shuffle=True, random_state=0, as_frame=True)
df = pl.from_pandas(bunch.data).with_columns(pl.col(pl.Binary).cast(pl.String))
df.head(20_000).write_csv("/tmp/kdd.csv")

config = Config(
    source=SourceConfig(uri="/tmp/kdd.csv"),
    run={"workdir": "/tmp/sorethumb_demo"},
)
result = run_detection(config, no_report=True)
print(f"Flagged {result.n_anomalies} rows for review across {result.n_succeeded} group(s)")
```

---

## CLI quickstart — local CSV, watching live output

### Zero-config: point at a file and go

No config file needed. Pass the data file directly and sorethumb runs with
sensible defaults, writing artefacts to the current directory:

```bash
cd /path/to/my-analysis
sorethumb run --log-level INFO /path/to/data.csv
```

On first run you'll be prompted to save a `sorethumb.toml` for future runs.
Re-running the same command always uses the file you pass — overriding whatever
`uri` is in the config — so iterating across datasets is frictionless.

### Config-based workflow (full control)

**1. Create a workspace next to your data**

```bash
sorethumb init /path/to/my-analysis
cd /path/to/my-analysis
```

`init` writes a fully-commented `sorethumb.toml`. Open it and set the one required field:

```toml
[source]
uri = "/absolute/path/to/your/data.csv"    # also accepts .parquet, .json, and http(s):// URLs

[run]
workdir = "."   # where models, results and the SQLite ledger are stored
```

**2. Check how your columns will be treated (no models trained)**

```bash
sorethumb inspect
```

This profiles every column and prints the classification table:

```
┌─────────────────┬──────────────┬─────────┬───────────────────────────────┐
│ column          │ treatment    │ missing │ reason                        │
├─────────────────┼──────────────┼─────────┼───────────────────────────────┤
│ age             │ numeric      │ 2.1 %   │ continuous, 847 unique values │
│ country         │ one_hot      │ 0.0 %   │ categorical, 31 categories    │
│ session_id      │ drop         │ 0.0 %   │ high cardinality (99.8 %)     │
│ created_at      │ derive_time  │ 0.0 %   │ datetime → hour, dow, month   │
│ …               │ …            │ …       │ …                             │
└─────────────────┴──────────────┴─────────┴───────────────────────────────┘
```

Edit `sorethumb.toml` to override any column treatment before running:

```toml
[columns]
id_column = "session_id"      # excluded from features, used as row label
```

**3. Run detection — watch progress live**

```bash
sorethumb run --log-level INFO
```

You'll see each stage scroll by in real time:

```
INFO  Loaded 84 231 rows × 22 columns from data.csv
INFO  Profiling: 22 columns → 9 numeric, 6 one_hot, 3 derive_time, 4 drop
INFO  Feature space: 38 output features after encoding
INFO  [group=ALL] fitting isolation_forest on 84 231 rows × 38 features …
INFO  [group=ALL] fitting kmeans_distance on 84 231 rows × 38 features …
INFO  [group=ALL] fitting one_class_svm on 20 000 rows × 38 features (capped)
INFO  [group=ALL] scoring + calibration …
INFO  [group=ALL] SHAP explanations (TreeSHAP) for 421 flagged rows …
INFO  Results written: results/<run_id>/ALL/anomalies.parquet

Run abc12345  succeeded=1  skipped=0  failed=0
  Flagged for review: 421 (0.50% of 84 231 rows) — the review shortlist at the
  chosen budget, not an estimate of true prevalence

  Realised detector flag rates (own natural boundary):
    isolation_forest=1.83%, kmeans_distance=0.74%, one_class_svm=2.10%

  Report: ./runs/abc12345/report.html
```

The three per-detector rates are heuristic boundaries, not measurements, and
they disagree; `contamination = "auto"` is just their median. Treat the flagged
count as *"records worth a look given this budget"*, never *"this is how many
anomalies the data has"*.

**4. Print the top anomalies with their SHAP reasons**

```bash
sorethumb anomalies           # latest run, top reasons for every flagged row
sorethumb anomalies --top 20  # just the 20 most anomalous rows
sorethumb anomalies --reasons 5 --top 50   # five reason columns
```

```
                    Anomalies — run abc12345678
┌────┬────────┬───────────┬──────────────────────┬────────────────────────┐
│  # │  score │ kind      │ reason 1             │ reason 2               │
├────┼────────┼───────────┼──────────────────────┼────────────────────────┤
│  1 │ 0.9821 │ exact     │ amount=14 500.00     │ country=NG             │
│  2 │ 0.9714 │ exact     │ hour_of_day=3        │ failed_attempts=12     │
│  3 │ 0.9601 │ heuristic │ session_duration=0.1 │ amount=9 200.00        │
│  …                                                                      │
└────┴────────┴───────────┴──────────────────────┴────────────────────────┘
  421 flagged row(s)   run=abc12345678   workspace=.
```

`kind=exact` means TreeSHAP (Isolation Forest); `kind=heuristic` means centroid
or gradient attribution. See [docs/explanations.md](docs/explanations.md).

For a machine-readable result pipe `--json`:

```bash
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
```

---

## Benchmark results

Detector × dataset × metric, measured on a MacBook Pro M3, Python 3.12.
Average precision (AP) is the headline metric — it accounts for class imbalance
in a way ROC-AUC does not.

<!-- benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `sorethumb benchmark` to regenerate. -->

| dataset | detector | n_rows | n_features | roc_auc | average_precision | precision_at_k | recall_at_k | f1_at_contamination | fit_seconds | score_seconds | peak_rss_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_gaussian | isolation_forest | 1050 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.092 | 0.010 | 4.5 |
| synthetic_gaussian | kmeans_distance | 1050 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.172 | 0.000 | 18.7 |
| synthetic_gaussian | one_class_svm | 1050 | 8 | 0.9127 | 0.2054 | 0.1200 | 0.1200 | 0.1200 | 0.003 | 0.003 | 1.3 |
| synthetic_highd | isolation_forest | 2100 | 32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.089 | 0.016 | 0.8 |
| synthetic_highd | kmeans_distance | 2100 | 32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.451 | 0.000 | 27.1 |
| synthetic_highd | one_class_svm | 2100 | 32 | 0.9205 | 0.2204 | 0.0800 | 0.0800 | 0.0800 | 0.013 | 0.012 | 4.4 |
| kddcup99_sa | isolation_forest | 100655 | 38 | 0.9448 | 0.4652 | 0.2852 | 0.2852 | 0.2852 | 0.157 | 0.221 | 3.7 |
| kddcup99_sa | kmeans_distance | 100655 | 38 | 0.8024 | 0.0971 | 0.0086 | 0.0086 | 0.0086 | 18.231 | 0.005 | 381.0 |
| kddcup99_sa | one_class_svm | 100655 | 38 | 0.7695 | 0.0799 | 0.0086 | 0.0086 | 0.0086 | 42.186 | 34.526 | 0.0 |
| covtype | isolation_forest | 581012 | 54 | 0.9346 | 0.0365 | 0.0346 | 0.0346 | 0.0346 | 0.297 | 1.135 | 44.6 |
| covtype | kmeans_distance | 581012 | 54 | 0.5366 | 0.0047 | 0.0000 | 0.0000 | 0.0000 | 26.066 | 0.131 | 1040.9 |
| covtype | one_class_svm | 581012 | 54 | 0.7490 | 0.0087 | 0.0000 | 0.0000 | 0.0000 | 1623.567 | 1212.133 | 0.0 |
<!-- benchmark-results-end -->

---

## Detectors

### Default ensemble

Three detectors run by default, each from a different algorithmic family so
that no single blind spot dominates the ensemble:

| Detector | Key | Algorithm | Strength | Weakness |
|----------|-----|-----------|----------|----------|
| `isolation_forest` | ★ default | Random tree partitioning | Fast, scales to millions of rows, exact TreeSHAP attributions | Struggles with very high-dimensional sparse data |
| `kmeans_distance` | ★ default | CBLOF: negative distance to nearest *large*-cluster centroid | Interpretable; a tight anomaly cluster can't hide by capturing its own centroid | Assumes anomalies are ≲ 10 % of rows (tune `large_cluster_coverage`); spherical clusters; sensitive to `k` |
| `one_class_svm` | ★ default | RBF kernel boundary | Genuinely different family, useful in ensemble | Quadratic training cost; capped at 25 k rows by default |

### Additional built-in detectors

These are registered and available in config but **not included in the default
ensemble**. Add any of them to `[[detectors]]` to use them:

| Detector | Algorithm | Best suited for | Limitation |
|----------|-----------|-----------------|------------|
| `ecod` | Empirical CDF (two-tailed) | High-dimensional data with marginal outliers; parameter-free, near-linear | Assumes marginal independence — misses joint-distribution anomalies |
| `lof` | Local Outlier Factor (novelty) | Datasets with varying-density clusters; catches what global methods miss | Stores full kNN graph; capped at 50 k rows by default |
| `hbos` | Histogram-based density | Speed-critical pipelines; useful as a sanity baseline | Feature-independence assumption; ignores feature interactions |

### Swapping or extending the ensemble

**Remove a detector** — omit its block or set `enabled = false`:

```toml
[[detectors]]
name = "isolation_forest"
train_row_cap = 250_000

[[detectors]]
name = "ecod"   # replaces one_class_svm
```

**Add ECOD + LOF alongside the defaults:**

```toml
[[detectors]]
name = "isolation_forest"
train_row_cap = 250_000

[[detectors]]
name = "kmeans_distance"
train_row_cap = 200_000

[[detectors]]
name = "ecod"

[[detectors]]
name = "lof"
train_row_cap = 50_000
```

**Register a third-party detector** without modifying sorethumb — implement the
[Detector protocol](src/sorethumb/detectors/_protocol.py) and add an entry point
to your package's `pyproject.toml`:

```toml
[project.entry-points."sorethumb.detectors"]
my_detector = "my_package.detectors:MyDetector"
```

sorethumb picks it up automatically at startup; use it in config by its `name`.

---

## Scale guide

| Dataset size | Recommended setup |
| --- | --- |
| < 100 k rows | All three detectors, default config |
| 100 k – 1 M rows | Disable `one_class_svm`, or set its `train_row_cap = 10000` (default 25 000) |
| > 1 M rows | Set `run.max_rows` to subsample; enable `features.pca` |
| > 2000 features | Enable `features.pca`; raises `FeatureWidthWarning` by default |

Memory footprint is dominated by the feature matrix: `n_rows × n_features × 4 bytes`
(float32). An 8 GB budget handles roughly 500 M cells.

---

## Honest limitations

- `run.max_memory_mb` caps the *projected* feature-matrix size (rows × encoded
  columns × dtype bytes): a run whose estimate exceeds it aborts with
  `MemoryBudgetError` before any model is fitted. It is not a live RSS ceiling —
  Python cannot impose one — so actual peak memory can still exceed the budget.
- The composite score is an interpretable ranking score, not a calibrated probability.
- `contamination` is a review budget, not a prevalence estimate. The flagged
  count is *"the top N% of the ranking"*, chosen by you (or, for `"auto"`, the
  median of three heuristic per-detector cut-offs that themselves disagree —
  see the realised rates in the run output). It says nothing about how many
  genuine anomalies the data holds; only labelled data can tell you that.
- Only Isolation Forest yields exact (TreeSHAP) attributions; all others are heuristic
  (centroid distance or input gradient). See [docs/explanations.md](docs/explanations.md).
- Self-calibration maps every run's scores to roughly uniform on [0, 1] by
  construction, so two independently-fitted runs — including the per-period runs
  `sorethumb backfill` produces — are not on a common scale. A `sorethumb history`
  trend surfaces *relative* change (which groups/records move, period to period),
  not an absolute level of "how anomalous is this period". For a trend on one
  fixed scale, reuse a fitted run: `sorethumb score --from-run RUN_ID`.
- Unsupervised anomaly ≠ the thing you care about. The library ranks statistical oddity;
  whether an odd record is *interesting* is a domain judgement it cannot make.

---

## Documentation

- [Documentation index](docs/index.md)
- [CLI reference](docs/cli_reference.md)
- [Configuration reference](docs/configuration.md)
- [Configuration examples](docs/configuration-examples.md)
- [Detector models](docs/models.md)
- [Example runs](docs/example-runs.md)
- [Adapting to your data](docs/adapting-to-your-data.md)
- [Explanations: exact vs heuristic](docs/explanations.md)
- [Approximations and error characteristics](docs/approximations.md)
- [Contributing](CONTRIBUTING.md)
