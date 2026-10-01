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

## Supported platforms

Tested and supported: **Linux and macOS**, Python 3.11–3.13. CI runs the full
suite — including workspace/SQLite, CLI subprocess, and report-rendering
integration tests, not just unit tests — on both `ubuntu-latest` and
`macos-latest` for every change (`.github/workflows/release-validation.yml`).

**Windows is not currently tested or supported.** A one-off run of the
integration suite on `windows-latest` failed broadly (most `sorethumb run`/
`backfill`/`report` CLI paths hit `OSError: [Errno 9] Bad file descriptor`,
plus a console-encoding mismatch mangling non-ASCII output) — real,
unresolved compatibility work, not a formality away from working. Tracked as
a deliberate gap, not an oversight — see
[SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md#supported-platforms).

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

No network access and nothing beyond sorethumb's own core dependencies — the
dataset is a small synthetic table generated in-process, with 30 of its 2,000
rows nudged far outside the normal range so there's something for the
detectors to actually find:

```python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from sorethumb_ml import Config, SourceConfig, run_detection

# A fresh directory every run -- sorethumb's runs are idempotent/resumable
# (it skips a group it's already completed for a given config+data), so a
# fixed path would make a second copy-paste of this snippet report 0 newly
# succeeded groups instead of re-running the demo.
tmp_dir = Path(tempfile.mkdtemp(prefix="sorethumb_quickstart_"))

rng = np.random.default_rng(0)
n_rows, n_anomalies = 2_000, 30
amount = rng.normal(loc=50.0, scale=10.0, size=n_rows)
latency_ms = rng.normal(loc=200.0, scale=30.0, size=n_rows)
region = rng.choice(["us", "eu", "apac"], size=n_rows)

# Push a random slice of rows far outside the normal range.
anomaly_idx = rng.choice(n_rows, size=n_anomalies, replace=False)
amount[anomaly_idx] = rng.uniform(500.0, 1000.0, size=n_anomalies)
latency_ms[anomaly_idx] = rng.uniform(2000.0, 5000.0, size=n_anomalies)

df = pl.DataFrame({"amount": amount, "latency_ms": latency_ms, "region": region})
df.write_csv(tmp_dir / "quickstart.csv")

config = Config(
    source=SourceConfig(uri=str(tmp_dir / "quickstart.csv")),
    run={"workdir": str(tmp_dir / "workspace")},
)
result = run_detection(config, no_report=True)
print(f"Flagged {result.n_anomalies} rows for review across {result.n_succeeded} group(s)")
```

---

## CLI quickstart — local CSV, watching live output

### Zero-config: point at a file and go

No config file needed. Pass the data file directly and sorethumb runs with
sensible defaults, writing every artefact under one dedicated
`./sorethumb-workspace/` directory rather than beside your data or scattered
loose in the current directory:

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

`init` writes a fully-commented `sorethumb.toml` and creates a
`sorethumb-workspace/` directory next to it. Open the file and set the one
field it leaves for you — `workdir` is already filled in, pointing at the
workspace `init` just created:

```toml
[source]
uri = "/absolute/path/to/your/data.csv"    # also accepts .parquet, .json, and http(s):// URLs

[run]
workdir = "/path/to/my-analysis/sorethumb-workspace"   # where models, results and the SQLite ledger are stored
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

One maintainer script regenerates both tables below and, when you point it at
this file, injects them (from a clone — see [Installation](#installation); not
yet on PyPI). It is a repo script, not part of the installed `sorethumb`
command, and it only writes where you tell it to:

```bash
pip install ".[benchmark]"
python scripts/run_benchmark.py --output-dir benchmark_results --readme README.md
```

`scripts/run_benchmark.py` runs two independent suites (each can be disabled
with `--no-pipeline` / `--no-legacy`) and writes `*.md` / `*.csv` into the
required `--output-dir`; the README is edited only if `--readme` is given.
Average precision (AP) is the headline metric
in both — it accounts for class imbalance in a way ROC-AUC does not.

### Full-pipeline scenario benchmark

`src/sorethumb_ml/evaluate/pipeline_benchmark.py` fits a named taxonomy of
synthetic anomaly types — point, local, contextual, clustered, masking,
swamping, varying-density (see `src/sorethumb_ml/evaluate/scenarios.py`) — with
mixed numeric **and categorical** columns through the real feature pipeline
(encoding, scaling, optional PCA, detectors, calibration, ensemble), never
training and evaluating on the same rows. Every (scenario, ablation) cell
runs over several seeds and reports mean ± a 95% confidence interval, not a
bare mean. `peak_memory_mb` is a real OS-tracked high-water mark from an
isolated subprocess per cell, not a before/after snapshot. `sklearn:*` rows
are the same scenario fit with bare sklearn detectors and no sorethumb
pipeline at all, for comparison (PyOD is deliberately not used here — see
`docs/approximations.md`).

**Matrix shape**: `scripts/run_benchmark.py` runs every scenario under the shipped
*default* ablation only (plus its `sklearn:*` baselines) — 10 pipeline
ablations exist in total (`AblationSpec`s: PCA on, standard scaler, each
detector alone, all six detectors, composite/union combination), but only
`default` is committed here to keep regeneration time bounded; run
`sorethumb_ml.evaluate.pipeline_benchmark.run_pipeline_benchmark` directly with
other `ablation_names` for the rest. `flag_precision`/`flag_recall`/
`flag_f1`/`flag_false_positive_rate`/`flag_count` describe the ensemble's
*actual* `anomaly_flag` decision — distinct from `roc_auc`/`average_precision`/
`precision_at_k`/`recall_at_k`, which describe the continuous ranking (or a
hypothetical top-k cut on it) and can disagree with the real flag
substantially, especially for `intersection`'s naturally conservative
unanimous-agreement rule (see `docs/approximations.md`). `swamping_clean`/
`swamping_contaminated` are a matched pair (identical normal training
population, differing only by injected contamination) scored against the
same genuinely-normal "at-risk" holdout — a single-class population by
design, so `roc_auc`/`average_precision` are `n/a` there; `flag_false_positive_rate`
is the metric that actually demonstrates the mechanism (see
`docs/approximations.md` for what it shows).

`tests/benchmark/test_pipeline_accuracy_floors.py` locks in a ROC-AUC floor
per guarded (scenario, ablation) cell — this is what makes the harness
capable of rejecting a default configuration: a real accuracy regression
fails that suite. It also documents where the *shipped default* combination
mode (`intersection`) genuinely underperforms (`local`, `varying_density` —
see `docs/approximations.md`) rather than asserting something the data shows
is false. `scripts/run_benchmark.py` itself refuses to publish a
matrix with a missing or errored cell (`assert_complete_and_error_free`).

<!-- pipeline-benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme README.md` to regenerate. -->

| scenario | kind | ablation | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | flag_precision | flag_recall | flag_f1 | flag_false_positive_rate | flag_count | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | 0.8889 | 0.1178 | 0.1973 | 0.0014 | 1.7 | 0.823 | 0.008 | 244.1 |
| point | point | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | 0.158 | 0.006 | 230.0 |
| point | point | sklearn:lof | 3 | 588 | 252 | 0.3239 ± 0.0298 | 0.0377 ± 0.0100 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.054 | 0.002 | 220.5 |
| point | point | sklearn:one_class_svm | 3 | 588 | 252 | 0.9251 ± 0.0304 | 0.3431 ± 0.0486 | 0.2308 | 0.2504 | n/a | n/a | n/a | n/a | n/a | 0.042 | 0.000 | 217.6 |
| local | local | default | 3 | 588 | 252 | 0.7085 ± 0.0335 | 0.1075 ± 0.0460 | 0.1026 | 0.0958 | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 3.0 | 0.576 | 0.008 | 245.0 |
| local | local | sklearn:isolation_forest | 3 | 588 | 252 | 0.9024 ± 0.0583 | 0.4356 ± 0.1883 | 0.3846 | 0.3973 | n/a | n/a | n/a | n/a | n/a | 0.160 | 0.006 | 230.4 |
| local | local | sklearn:lof | 3 | 588 | 252 | 0.8213 ± 0.0719 | 0.6270 ± 0.1236 | 0.5641 | 0.5777 | n/a | n/a | n/a | n/a | n/a | 0.059 | 0.001 | 220.2 |
| local | local | sklearn:one_class_svm | 3 | 588 | 252 | 0.0073 ± 0.0109 | 0.0269 ± 0.0074 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.000 | 217.5 |
| contextual | contextual | default | 3 | 840 | 360 | 0.4965 ± 0.0842 | 0.0579 ± 0.0091 | 0.0370 | 0.0417 | 0.0333 | 0.0208 | 0.0256 | 0.0204 | 7.3 | 0.674 | 0.010 | 249.0 |
| contextual | contextual | sklearn:isolation_forest | 3 | 840 | 360 | 0.7495 ± 0.0652 | 0.1524 ± 0.0577 | 0.1667 | 0.1838 | n/a | n/a | n/a | n/a | n/a | 0.164 | 0.007 | 230.8 |
| contextual | contextual | sklearn:lof | 3 | 840 | 360 | 0.5295 ± 0.0950 | 0.1012 ± 0.0607 | 0.1111 | 0.1225 | n/a | n/a | n/a | n/a | n/a | 0.064 | 0.003 | 220.7 |
| contextual | contextual | sklearn:one_class_svm | 3 | 840 | 360 | 0.6000 ± 0.0267 | 0.0901 ± 0.0262 | 0.0926 | 0.1017 | n/a | n/a | n/a | n/a | n/a | 0.044 | 0.001 | 218.2 |
| clustered | clustered | default | 3 | 588 | 252 | 1.0000 ± 0.0001 | 0.9988 ± 0.0024 | 0.8718 | 0.9375 | 0.4222 | 0.1606 | 0.2286 | 0.0041 | 2.7 | 0.626 | 0.008 | 244.3 |
| clustered | clustered | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | 0.160 | 0.006 | 229.9 |
| clustered | clustered | sklearn:lof | 3 | 588 | 252 | 0.3503 ± 0.1922 | 0.0418 ± 0.0157 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.060 | 0.002 | 220.3 |
| clustered | clustered | sklearn:one_class_svm | 3 | 588 | 252 | 0.9174 ± 0.0337 | 0.2915 ± 0.0327 | 0.2564 | 0.2867 | n/a | n/a | n/a | n/a | n/a | 0.040 | 0.000 | 217.5 |
| masking | masking | default | 3 | 595 | 255 | 0.8813 ± 0.0328 | 0.4686 ± 0.1070 | 0.4825 | 0.4236 | 0.3185 | 0.0394 | 0.0684 | 0.0187 | 5.7 | 0.641 | 0.008 | 244.5 |
| masking | masking | sklearn:isolation_forest | 3 | 595 | 255 | 0.9749 ± 0.0208 | 0.8911 ± 0.0878 | 0.8684 | 0.7674 | n/a | n/a | n/a | n/a | n/a | 0.162 | 0.006 | 230.2 |
| masking | masking | sklearn:lof | 3 | 595 | 255 | 0.6479 ± 0.0817 | 0.3159 ± 0.0725 | 0.4123 | 0.3723 | n/a | n/a | n/a | n/a | n/a | 0.058 | 0.002 | 220.4 |
| masking | masking | sklearn:one_class_svm | 3 | 595 | 255 | 0.8448 ± 0.0466 | 0.4534 ± 0.1458 | 0.4123 | 0.3603 | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.001 | 217.8 |
| varying_density | varying_density | default | 3 | 728 | 312 | 0.6824 ± 0.0341 | 0.0844 ± 0.0552 | 0.0208 | 0.0256 | 0.0000 | 0.0000 | 0.0000 | 0.0155 | 4.7 | 0.552 | 0.009 | 246.5 |
| varying_density | varying_density | sklearn:isolation_forest | 3 | 728 | 312 | 0.9386 ± 0.0155 | 0.3351 ± 0.1728 | 0.2292 | 0.3158 | n/a | n/a | n/a | n/a | n/a | 0.161 | 0.007 | 230.4 |
| varying_density | varying_density | sklearn:lof | 3 | 728 | 312 | 0.8671 ± 0.0970 | 0.4416 ± 0.4803 | 0.4375 | 0.5966 | n/a | n/a | n/a | n/a | n/a | 0.057 | 0.001 | 220.2 |
| varying_density | varying_density | sklearn:one_class_svm | 3 | 728 | 312 | 0.1297 ± 0.0348 | 0.0233 ± 0.0037 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.000 | 217.5 |
| swamping_clean | swamping | default | 3 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0111 | 1.7 | 0.696 | 0.007 | 249.7 |
| swamping_contaminated | swamping | default | 3 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.7 | 0.655 | 0.007 | 251.0 |
<!-- pipeline-benchmark-results-end -->

### Real-dataset / legacy synthetic benchmark

`src/sorethumb_ml/evaluate/benchmark.py` fits bare detectors directly (no
feature pipeline) against KDDCup99, Covtype, and two synthetic Gaussian-plus-
point-anomaly datasets. `k` comes from a fixed, label-independent 5% review
budget shared across every dataset (not derived from the labels being
scored, which would collapse precision@k/recall@k/F1 into one number by
construction); ROC-AUC/AP are `n/a` (not a fabricated 0.0) when a sample is
single-class; every run records its generation timestamp, platform, and
Python/numpy/scipy/scikit-learn/sorethumb versions (see the "Generated ..."
line above the table) so a published number can always be tied back to the
environment that produced it.

The committed table below caps KDDCup99/Covtype at a seeded random 20,000-row
sample (`BenchmarkConfig(max_rows=20_000)`) and the shipped default three
detectors (`isolation_forest`, `kmeans_distance`, `one_class_svm`) — the full,
uncapped datasets (Covtype has 581k rows) take on the order of an hour to
regenerate, dominated by `one_class_svm`'s O(n²) training cost; run without
`max_rows` for full-dataset numbers. `scripts/run_benchmark.py`'s `--seeds` only
controls this suite (default 5; the table below was regenerated with 3 for a
faster turnaround — still real mean ± standard deviation across independent
seeds, not a single draw).

<!-- benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme README.md` to regenerate. -->

Generated 2026-09-29T01:58:13+00:00 on macOS-26.6.2-arm64-arm-64bit — Python 3.12.9, sorethumb 0.1.0, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0.

| dataset | detector | n_rows | n_features | n_seeds | roc_auc | average_precision | precision_at_k | recall_at_k | f1_at_contamination | fit_seconds | score_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_gaussian | isolation_forest | 1050 | 8 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9615 ± 0.0000 | 1.0000 ± 0.0000 | 0.9804 ± 0.0000 | 0.085 | 0.010 |
| synthetic_gaussian | kmeans_distance | 1050 | 8 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9615 ± 0.0000 | 1.0000 ± 0.0000 | 0.9804 ± 0.0000 | 0.170 | 0.000 |
| synthetic_gaussian | one_class_svm | 1050 | 8 | 3 | 0.9113 ± 0.0016 | 0.2038 ± 0.0037 | 0.1218 ± 0.0480 | 0.1267 ± 0.0499 | 0.1242 ± 0.0489 | 0.003 | 0.003 |
| synthetic_highd | isolation_forest | 2100 | 32 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9524 ± 0.0000 | 1.0000 ± 0.0000 | 0.9756 ± 0.0000 | 0.088 | 0.016 |
| synthetic_highd | kmeans_distance | 2100 | 32 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9524 ± 0.0000 | 1.0000 ± 0.0000 | 0.9756 ± 0.0000 | 0.433 | 0.001 |
| synthetic_highd | one_class_svm | 2100 | 32 | 3 | 0.9231 ± 0.0018 | 0.2243 ± 0.0028 | 0.0952 ± 0.0078 | 0.1000 ± 0.0082 | 0.0976 ± 0.0080 | 0.013 | 0.012 |
| kddcup99_sa | isolation_forest | 20000 | 38 | 3 | 0.9386 ± 0.0061 | 0.4443 ± 0.0143 | 0.1807 ± 0.0026 | 0.2722 ± 0.0049 | 0.2171 ± 0.0028 | 0.116 | 0.067 |
| kddcup99_sa | kmeans_distance | 20000 | 38 | 3 | 0.8053 ± 0.0090 | 0.0984 ± 0.0017 | 0.0067 ± 0.0017 | 0.0101 ± 0.0027 | 0.0080 ± 0.0021 | 17.013 | 0.001 |
| kddcup99_sa | one_class_svm | 20000 | 38 | 3 | 0.7670 ± 0.0084 | 0.0793 ± 0.0016 | 0.0067 ± 0.0017 | 0.0101 ± 0.0027 | 0.0080 ± 0.0021 | 1.374 | 1.291 |
| covtype | isolation_forest | 20000 | 54 | 3 | 0.9837 ± 0.0030 | 0.3486 ± 0.0320 | 0.1667 ± 0.0162 | 0.8976 ± 0.0510 | 0.2809 ± 0.0249 | 0.116 | 0.082 |
| covtype | kmeans_distance | 20000 | 54 | 3 | 0.5745 ± 0.0142 | 0.0103 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 14.014 | 0.002 |
| covtype | one_class_svm | 20000 | 54 | 3 | 0.7902 ± 0.0091 | 0.0224 ± 0.0024 | 0.0247 ± 0.0033 | 0.1339 ± 0.0211 | 0.0416 ± 0.0056 | 1.465 | 1.449 |
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
[Detector protocol](https://github.com/tarne75/sorethumb/blob/main/src/sorethumb_ml/detectors/_protocol.py) and add an entry point
to your package's `pyproject.toml`:

```toml
[project.entry-points."sorethumb_ml.detectors"]
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
  untrusted remote server; a configured `Authorization` header is only ever
  sent to the exact origin `source.uri` names, and a redirect that would
  downgrade HTTPS to HTTP is refused outright — see
  [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

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
