<div align="center">
  <img src="https://raw.githubusercontent.com/tarne75/sorethumb/main/docs/banner.svg" alt="sorethumb — unsupervised anomaly detection for tabular data" width="860"/>
</div>

<div align="center">

[![CI](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml/badge.svg)](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/tarne75/sorethumb/graph/badge.svg?token=UJLJS8GHDV)](https://codecov.io/github/tarne75/sorethumb)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?color=123A66)](https://github.com/tarne75/sorethumb)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?color=2F80ED)](https://github.com/tarne75/sorethumb/blob/main/LICENSE)

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
3. **Per-record explanations** in original feature terms, labelled model_specific or heuristic.
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

  Report: ./reports/abc12345/index.html
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
┌────┬────────┬────────────────┬──────────────────────┬────────────────────────┐
│  # │  score │ kind           │ reason 1             │ reason 2               │
├────┼────────┼────────────────┼──────────────────────┼────────────────────────┤
│  1 │ 0.9821 │ model_specific │ amount=14 500.00     │ country=NG             │
│  2 │ 0.9714 │ model_specific │ hour_of_day=3        │ failed_attempts=12     │
│  3 │ 0.9601 │ heuristic      │ session_duration=0.1 │ amount=9 200.00        │
│  …                                                                           │
└────┴────────┴────────────────┴──────────────────────┴────────────────────────┘
  421 flagged row(s)   run=abc12345678   workspace=.
```

`kind=exact` means ECOD/HBOS's own additive score decomposition (summing it
recovers the score with zero error, by construction); `kind=model_specific`
means TreeSHAP (Isolation Forest) — it uses the fitted trees' actual
structure, but (`check_additivity=False`) isn't a verified exact
decomposition of the score; `kind=heuristic` means centroid distance
(KMeans) or finite-difference gradient (One-Class SVM, LOF). See
[docs/explanations.md](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md).

For a machine-readable result pipe `--json`:

```bash
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
```

---

## Benchmark results

One command regenerates both tables below and injects them into this file:

```bash
pip install 'sorethumb[benchmark]'
sorethumb benchmark
```

`sorethumb benchmark` runs two independent suites (each can be disabled with
`--no-pipeline` / `--no-legacy`) and writes `benchmark_results/*.md` / `*.csv`
alongside the injected tables. Average precision (AP) is the headline metric
in both — it accounts for class imbalance in a way ROC-AUC does not.

### Full-pipeline scenario benchmark

`src/sorethumb/evaluate/pipeline_benchmark.py` fits a named taxonomy of
synthetic anomaly types — point, local, contextual, clustered, masking,
swamping, varying-density (see `src/sorethumb/evaluate/scenarios.py`) — with
mixed numeric **and categorical** columns through the real feature pipeline
(encoding, scaling, optional PCA, detectors, calibration, ensemble), never
training and evaluating on the same rows. Every (scenario, ablation) cell
runs over several seeds and reports mean ± a 95% confidence interval, not a
bare mean. `peak_memory_mb` is a real OS-tracked high-water mark from an
isolated subprocess per cell, not a before/after snapshot. `sklearn:*` rows
are the same scenario fit with bare sklearn detectors and no sorethumb
pipeline at all, for comparison (PyOD is deliberately not used here — see
`docs/approximations.md`).

`tests/benchmark/test_pipeline_accuracy_floors.py` locks in a ROC-AUC floor
per guarded (scenario, ablation) cell — this is what makes the harness
capable of rejecting a default configuration: a real accuracy regression
fails that suite. It also documents where the *shipped default* combination
mode (`intersection`) genuinely underperforms (`local`, `varying_density` —
see `docs/approximations.md`) rather than asserting something the data shows
is false.

<!-- pipeline-benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `sorethumb benchmark` to regenerate. -->

| scenario | kind | ablation | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | 3 | 588 | 252 | 0.9731 ± 0.0141 | 0.6847 ± 0.1391 | 0.5128 | 0.5580 | 0.810 | 0.008 | 244.9 |
| point | point | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | 0.162 | 0.006 | 230.6 |
| point | point | sklearn:lof | 3 | 588 | 252 | 0.3239 ± 0.0298 | 0.0377 ± 0.0100 | 0.0000 | 0.0000 | 0.059 | 0.002 | 220.8 |
| point | point | sklearn:one_class_svm | 3 | 588 | 252 | 0.9251 ± 0.0304 | 0.3431 ± 0.0486 | 0.2308 | 0.2504 | 0.047 | 0.000 | 218.1 |
| local | local | default | 3 | 588 | 252 | 0.0443 ± 0.0073 | 0.0299 ± 0.0081 | 0.0000 | 0.0000 | 0.599 | 0.009 | 245.9 |
| local | local | sklearn:isolation_forest | 3 | 588 | 252 | 0.9024 ± 0.0583 | 0.4356 ± 0.1883 | 0.3846 | 0.3973 | 0.165 | 0.006 | 230.8 |
| local | local | sklearn:lof | 3 | 588 | 252 | 0.8213 ± 0.0719 | 0.6270 ± 0.1236 | 0.5641 | 0.5777 | 0.059 | 0.001 | 221.0 |
| local | local | sklearn:one_class_svm | 3 | 588 | 252 | 0.0073 ± 0.0109 | 0.0269 ± 0.0074 | 0.0000 | 0.0000 | 0.046 | 0.000 | 217.7 |
| contextual | contextual | default | 3 | 840 | 360 | 0.4718 ± 0.0861 | 0.0532 ± 0.0107 | 0.0370 | 0.0417 | 0.676 | 0.010 | 249.5 |
| contextual | contextual | sklearn:isolation_forest | 3 | 840 | 360 | 0.7495 ± 0.0652 | 0.1524 ± 0.0577 | 0.1667 | 0.1838 | 0.165 | 0.007 | 231.4 |
| contextual | contextual | sklearn:lof | 3 | 840 | 360 | 0.5295 ± 0.0950 | 0.1012 ± 0.0607 | 0.1111 | 0.1225 | 0.063 | 0.003 | 221.5 |
| contextual | contextual | sklearn:one_class_svm | 3 | 840 | 360 | 0.6000 ± 0.0267 | 0.0901 ± 0.0262 | 0.0926 | 0.1017 | 0.045 | 0.001 | 218.6 |
| clustered | clustered | default | 3 | 588 | 252 | 0.9754 ± 0.0082 | 0.6933 ± 0.0527 | 0.5641 | 0.6027 | 0.633 | 0.008 | 244.5 |
| clustered | clustered | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | 0.163 | 0.006 | 230.6 |
| clustered | clustered | sklearn:lof | 3 | 588 | 252 | 0.3503 ± 0.1922 | 0.0418 ± 0.0157 | 0.0000 | 0.0000 | 0.059 | 0.002 | 220.9 |
| clustered | clustered | sklearn:one_class_svm | 3 | 588 | 252 | 0.9174 ± 0.0337 | 0.2915 ± 0.0327 | 0.2564 | 0.2867 | 0.045 | 0.001 | 218.0 |
| masking | masking | default | 3 | 595 | 255 | 0.5867 ± 0.0228 | 0.2203 ± 0.0054 | 0.2105 | 0.1874 | 0.614 | 0.008 | 245.3 |
| masking | masking | sklearn:isolation_forest | 3 | 595 | 255 | 0.9749 ± 0.0208 | 0.8911 ± 0.0878 | 0.8684 | 0.7674 | 0.162 | 0.006 | 230.6 |
| masking | masking | sklearn:lof | 3 | 595 | 255 | 0.6479 ± 0.0817 | 0.3159 ± 0.0725 | 0.4123 | 0.3723 | 0.058 | 0.001 | 220.7 |
| masking | masking | sklearn:one_class_svm | 3 | 595 | 255 | 0.8448 ± 0.0466 | 0.4534 ± 0.1458 | 0.4123 | 0.3603 | 0.046 | 0.001 | 218.2 |
| varying_density | varying_density | default | 3 | 728 | 312 | 0.2437 ± 0.0817 | 0.0259 ± 0.0013 | 0.0000 | 0.0000 | 0.601 | 0.009 | 247.5 |
| varying_density | varying_density | sklearn:isolation_forest | 3 | 728 | 312 | 0.9386 ± 0.0155 | 0.3351 ± 0.1728 | 0.2292 | 0.3158 | 0.165 | 0.007 | 230.7 |
| varying_density | varying_density | sklearn:lof | 3 | 728 | 312 | 0.8671 ± 0.0970 | 0.4416 ± 0.4803 | 0.4375 | 0.5966 | 0.060 | 0.002 | 220.7 |
| varying_density | varying_density | sklearn:one_class_svm | 3 | 728 | 312 | 0.1297 ± 0.0348 | 0.0233 ± 0.0037 | 0.0000 | 0.0000 | 0.045 | 0.000 | 217.9 |
| swamping | swamping | default | 3 | 880 | 252 | 0.9758 ± 0.0094 | 0.6914 ± 0.1138 | 0.5128 | 0.5549 | 0.649 | 0.008 | 249.9 |
<!-- pipeline-benchmark-results-end -->

### Real-dataset / legacy synthetic benchmark

`src/sorethumb/evaluate/benchmark.py` fits bare detectors directly (no
feature pipeline) against KDDCup99, Covtype, and two synthetic Gaussian-plus-
point-anomaly datasets. `k` comes from a fixed, label-independent 5% review
budget shared across every dataset (not derived from the labels being
scored, which would collapse precision@k/recall@k/F1 into one number by
construction); ROC-AUC/AP are `n/a` (not a fabricated 0.0) when a sample is
single-class; every run records its generation timestamp, platform, and
Python/numpy/scipy/scikit-learn/sorethumb versions.

<!-- benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `sorethumb benchmark` to regenerate. -->

_No benchmark results._
<!-- benchmark-results-end -->

---

## Detectors

### Default ensemble

Three detectors run by default, each from a different algorithmic family so
that no single blind spot dominates the ensemble:

| Detector | Key | Algorithm | Strength | Weakness |
|----------|-----|-----------|----------|----------|
| `isolation_forest` | ★ default | Random tree partitioning | Fast, scales to millions of rows, model-specific TreeSHAP attributions | Struggles with very high-dimensional sparse data |
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
[Detector protocol](https://github.com/tarne75/sorethumb/blob/main/src/sorethumb/detectors/_protocol.py) and add an entry point
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
- Detectors fit on the same (unlabelled) data they then score — there is no
  held-out "known normal" reference set, since that is what makes this
  unsupervised in the first place. A detector whose row cap is smaller than
  the group (e.g. `one_class_svm`'s default 25,000) fits on a random
  subsample but still scores every row, so a rare pattern absent from that
  subsample by chance may never shape the fitted boundary. See
  [docs/approximations.md](https://github.com/tarne75/sorethumb/blob/main/docs/approximations.md).
- ECOD and HBOS are the only detectors with a truly exact attribution — their
  score is already an additive sum of per-feature terms, so decomposing it is
  zero-error by construction. Isolation Forest yields model-specific
  (TreeSHAP) attributions, and even those aren't a verified exact
  decomposition of the score (`check_additivity=False`); KMeans and
  One-Class SVM/LOF are heuristic (centroid distance or finite-difference
  gradient, the latter restricted to the two detectors whose score responds
  continuously to a small perturbation). See
  [docs/explanations.md](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md).
- Feature reduction can discard exactly the signal that matters. Correlation
  pruning (`features.correlation_threshold`, default 0.95) drops one column
  from each highly-correlated pair entirely — invisible to every detector if
  the anomaly is specifically a *broken* relationship between that pair.
  PCA (opt-in, off by default) selects components by variance explained
  across the whole dataset, which is a property of the normal bulk, not of
  where a specific anomaly deviates — a run can clear
  `pca_min_explained_variance` with no warning and still have discarded the
  one dimension that would have flagged a particular row. See
  [docs/approximations.md](https://github.com/tarne75/sorethumb/blob/main/docs/approximations.md).
- Robust scaling (the default) can leave a zero-inflated column — mostly
  zeros, with a real signal in the nonzero minority — effectively unscaled:
  its median and IQR are both 0, so the degenerate-column guard sets its
  scale to 1.0 instead of dividing by zero, leaving its raw magnitude to
  dominate or be drowned out in distance-based detectors regardless of how
  informative it actually is. See
  [docs/approximations.md](https://github.com/tarne75/sorethumb/blob/main/docs/approximations.md).
- Self-calibration maps every run's scores to roughly uniform on [0, 1] by
  construction, so two independently-fitted runs — including the per-period runs
  `sorethumb backfill` produces — are not on a common scale. A `sorethumb history`
  trend surfaces *relative* change (which groups/records move, period to period),
  not an absolute level of "how anomalous is this period". For a trend on one
  fixed scale, reuse a fitted run: `sorethumb score --from-run RUN_ID`.
- Unsupervised anomaly ≠ the thing you care about. The library ranks statistical oddity;
  whether an odd record is *interesting* is a domain judgement it cannot make.
- **A workspace is executable, not just data.** Persisted runs store fitted
  estimators and calibrators as `joblib`/pickle files, which execute arbitrary
  code on load. `sorethumb score --from-run` (and `run.reuse_models`, which
  reloads a persisted model within the same run) unpickle those files with no
  sandboxing. SHA-256 file digests catch accidental corruption or a
  misplaced/swapped file — they do **not** make an untrusted pickle safe,
  since a deliberately crafted malicious file carries its own matching
  digest. Only load a workspace you created yourself or that came from a
  source you fully trust; never point these at a workspace received from
  someone else without inspecting it first. See [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).
- Fetching `source.uri` over http(s) is hardened against the obvious cases
  (a redirect pivoting to a cloud metadata endpoint or another internal
  host, an unbounded response), not a general-purpose sandbox for an
  untrusted remote server — see [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

---

## Documentation

- [Documentation index](https://github.com/tarne75/sorethumb/blob/main/docs/index.md)
- [CLI reference](https://github.com/tarne75/sorethumb/blob/main/docs/cli_reference.md)
- [Configuration reference](https://github.com/tarne75/sorethumb/blob/main/docs/configuration.md)
- [Configuration examples](https://github.com/tarne75/sorethumb/blob/main/docs/configuration-examples.md)
- [Detector models](https://github.com/tarne75/sorethumb/blob/main/docs/models.md)
- [Example runs](https://github.com/tarne75/sorethumb/blob/main/docs/example-runs.md)
- [Adapting to your data](https://github.com/tarne75/sorethumb/blob/main/docs/adapting-to-your-data.md)
- [Explanations: model-specific vs heuristic](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md)
- [Approximations and error characteristics](https://github.com/tarne75/sorethumb/blob/main/docs/approximations.md)
- [Contributing](https://github.com/tarne75/sorethumb/blob/main/CONTRIBUTING.md)
