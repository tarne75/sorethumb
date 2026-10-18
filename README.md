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

**What it is:** a *ranker*, an ordering of records by statistical oddity, plus a
*flag decision* on top of it. **What it is not:** a prevalence estimator. The
flagged count is never a measurement of how many anomalies your data contains.
What decides the count depends on two settings:
- **By default** (`scoring.contamination = "auto"`, `combination = "intersection"`),
  each detector flags by its own heuristic boundary, and a row is flagged only if all
  of them flag it. The count is whatever they agree on, and you don't choose it. The
  run output reports each detector's realised rate.
- **To get exactly the top N%** (a fixed review budget), set `contamination` to N/100
  *and* use a composite policy; the recommended one is `composite-agreement@1`.

See [docs/ensemble-policy.md](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md#where-things-stand-2026-10-15)
for what each combination does and the evidence behind it.

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

## Installation

Not yet on PyPI. Install from a clone:

```bash
git clone https://github.com/tarne75/sorethumb
cd sorethumb
pip install .
```

The core install is everything the quickstart below, the `sorethumb` command, and
the CSV/HTML reports need. Optional extras add to it:

- `pip install ".[explain]"` adds the SHAP package for TreeSHAP and KernelSHAP
  attributions. Without it those explanations fall back to a finite-difference
  method and say so with a warning.
- `pip install ".[benchmark]"` adds what the benchmark harness needs.

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

## What it does not do

The short version; every point is expanded under [Honest limitations](#honest-limitations).

- **It ranks; it does not count.** The flagged number is a review shortlist, not an
  estimate of how many anomalies your data holds. By default its size is set by the
  detectors' own boundaries, not by you.
- **It is unsupervised.** Detectors fit on the same data they score, with no
  held-out normal reference, and the slower ones fit on a row sample.
- **Scores are rankings, not probabilities,** and two independently fitted runs
  are not on a common scale. Reuse a fitted run (`sorethumb score --from-run`), or
  set an anchor run for `sorethumb history`, to compare on one scale.
- **Explanations are exact only for ECOD and HBOS.** The rest are model-specific
  or heuristic, labelled as such, and a missing `explain` extra downgrades some.
- **A workspace is executable, not just data.** Persisted models are pickles; open
  only a workspace you created or fully trust.
- **One machine, local storage.** It runs on Linux, macOS and Windows, keeps its
  workspace on a local drive (not a network share or a synced folder), and is
  built for roughly ten thousand to a few million rows.

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
3. **Per-record explanations** in original feature terms, labelled `exact`,
   `model_specific` or `heuristic`, each with the record of how it was produced.
4. **Run history you can reason about.** Models and calibrators are persisted,
   and `sorethumb history` reports four separate signals: population drift,
   score drift, threshold drift and anomaly counts, each with a verdict. A plain
   `backfill` refits each period, so score and threshold drift need a fixed
   anchor run (`history.anchor_run`); see the
   [trends guide](https://github.com/tarne75/sorethumb/blob/main/docs/trends.md).
5. **Idempotent, resumable execution** with a completion ledger.

It runs on a single machine, uses Polars throughout, and has no dependency on Spark,
Databricks, or any cloud vendor.

---

## Is it the right tool?

Use it where a ranked shortlist for a person to review is the goal: on a
single machine (Linux, macOS or Windows, see [Supported platforms](#supported-platforms)),
over data that fits [the scale guide](#scale-guide), with someone able to judge
whether a flagged record is actually interesting.

- **Do not use it as the only control where a missed anomaly is unacceptable.**
  It is unsupervised and ranks statistical oddity, so it can miss exactly the
  record you care about. Where missing one has serious consequences, use a
  labelled, supervised or domain-specific control and treat this as an extra
  signal at most.
- **Validate before you rely on error rates.** If a decision depends on how
  many flagged records are false positives, or how many true anomalies are
  missed, measure that on labelled data and have a domain reviewer check the
  flagged records. Scores and flagged counts are not calibrated risk
  probabilities and are not a prevalence estimate, and should not be presented
  as either.
- **Compare over time on one scale.** For an absolute comparison across periods,
  fit one accepted reference run and score later data against it with
  `sorethumb score --from-run RUN_ID`, or set it as `history.anchor_run` so
  `sorethumb history` measures every period against it. Without one, treat
  `backfill` and independent-run history as relative diagnostics only.
- **Only reuse models from a workspace you made or fully trust.** A workspace is
  executable, not just data: do not accept a downloaded, shared or externally
  supplied workspace. See [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md#trust-boundary-workspaces-are-executable-not-just-data).

---

## Supported platforms

| OS | Python | Tested in CI on every change |
| --- | --- | --- |
| Linux (x86-64) | 3.11–3.13 | unit, contract, integration, property, packaging (`ubuntu-latest`) |
| macOS (Apple silicon) | 3.11–3.13 | unit, contract, integration (`macos-latest`) |
| Windows 10/11 (x64) | 3.11–3.13 | unit, contract, integration, and installing the built wheel and sdist and running them from PowerShell (`windows-latest`) |

The integration suite exercises a real workspace, SQLite, the CLI as a
subprocess and report rendering on all three
(`.github/workflows/release-validation.yml`). On Windows, the CLI works the
same in PowerShell, Windows PowerShell 5.1 and cmd.exe.

Not supported:

- **A workspace on a network share or in a synced folder** (SMB/NFS, a mapped
  drive, OneDrive, Dropbox, Google Drive, iCloud) on any OS. Its SQLite
  database and atomic file replacement need a local drive; sorethumb warns
  when it detects one. Your data can be anywhere.
- **Native Windows on ARM.** pyarrow publishes no Windows ARM64 wheel; an x64
  Python under emulation works.
- **Very deep workspace paths on Windows** without long-path support enabled:
  sorethumb stops before running and says how much shorter the path must be.

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

**Text encoding.** CSV, TSV and JSON sources must be UTF-8 (a byte-order mark
is fine). Excel's plain "CSV" format on Windows writes cp1252 instead, which
fails with a message pointing here. Re-save it as "CSV UTF-8 (Comma delimited)",
or convert it once, in PowerShell 7 with
`Get-Content data.csv -Encoding windows-1252 | Set-Content data-utf8.csv -Encoding utf8`
or with Python:
`python -c "open('data-utf8.csv', 'w', encoding='utf-8').write(open('data.csv', encoding='cp1252').read())"`.
As a last resort, `read_options = { encoding = "utf8-lossy" }` reads the file
with undecodable characters replaced.

Format is auto-detected from the file extension. Set `source.format` explicitly when the extension is ambiguous:

```toml
[source]
uri = "/data/my_file.dat"
format = "csv"
```

---

## CLI quickstart — local CSV, watching live output

### Zero-config: point at a file and go

No config file needed. Pass the data file directly and sorethumb runs with
sensible defaults, writing every artefact under one dedicated
`./sorethumb-workspace/` directory — relative to the directory you run the
command from, not beside your data file, and not scattered loose in the current
directory:

```bash
cd /path/to/my-analysis
sorethumb run --log-level INFO /path/to/data.csv
```

Then read the results from the same directory; no config file is needed for
that either:

```bash
sorethumb anomalies --top 10
sorethumb runs
sorethumb report
```

On an interactive terminal, the first run offers to save a `sorethumb.toml` for
future runs. Under cron, CI or a pipe it doesn't ask (it says how to save one on
stderr); `--save-config` / `--no-save-config` decide without asking. Re-running the same command always uses the file you pass — overriding whatever
`uri` is in the config — so iterating across datasets is frictionless. The same
`./sorethumb-workspace/` default applies to a config file that does not set
`run.workdir`; `--workdir` and `run.workdir` override it (see
[Where the workspace lives](https://github.com/tarne75/sorethumb/blob/main/docs/cli_reference.md#where-the-workspace-lives)).

> **The workspace is executable, not just data.** Its `models/` directory holds
> fitted estimators as `joblib`/pickle files, which run arbitrary code when
> loaded. Only open a workspace you created or fully trust — see
> [Honest limitations](#honest-limitations) and `SECURITY.md`.

### Config-based workflow (full control)

**1. Create a workspace next to your data**

```bash
sorethumb init /path/to/my-analysis
cd /path/to/my-analysis
```

`init` writes a fully-commented `sorethumb.toml` and creates a
`sorethumb-workspace/` directory next to it (it does nothing if a
`sorethumb.toml` is already there). Open the file and set the one
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

This profiles every column and prints the feature plan (abridged):

```
                                 Feature plan
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Column     ┃ Class           ┃ Treatment     ┃ Reason                                 ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ amount     │ numeric         │ impute_median │ numeric dtype (Float64)                │
│ country    │ categorical     │ one_hot       │ String with cardinality_ratio=0.0004   │
│ session_id │ identifier_like │ drop          │ UUID pattern detected in sample (n=1000)│
│ created_at │ temporal        │ derive_time   │ temporal dtype (Datetime(...))         │
│ region     │ near_constant   │ drop          │ only 3 distinct value(s) (threshold=3) │
└────────────┴─────────────────┴───────────────┴────────────────────────────────────────┘

Projected feature width: 38 columns
```

A column with exactly three distinct values is dropped by default (as `region`
is here); set `profiling.near_constant_distinct = 2` to keep such columns.

Edit `sorethumb.toml` to override any column treatment before running:

```toml
[columns]
id_column = "session_id"      # excluded from features, used as row label
```

**3. Run detection — watch progress live**

```bash
sorethumb run --log-level INFO
```

Each stage logs as it runs, and the run ends with a summary (abridged; the
numbers are illustrative):

```
… sorethumb_ml.detectors.one_class_svm INFO OneClassSVM: fitting nu=0.100 kernel=rbf on 25000 rows x 38 features.
… sorethumb_ml.scoring.combine INFO ScoreEnsemble: weighting=equal combination=intersection contamination=auto [heuristic] per-detector rates {'isolation_forest': 1.8, 'kmeans_distance': 0.7, 'one_class_svm': 2.1} → intersection: 421/84231 (0.5%).
… sorethumb_ml.store.results INFO Results written: run=run_5c72… group=4413… rows=421 path=…/anomalies.parquet

Run run_5c720f703718c39e3c5b55b171083f30  succeeded=1  skipped=0  failed=0
  Ensemble policy: intersection@1
  Flagged for review: 421 (0.50% of 84,231 rows) — a review shortlist, not an
estimate of true prevalence

  Realised detector flag rates (own natural boundary):
    isolation_forest=1.83%, kmeans_distance=0.74%, one_class_svm=2.10%

  Group timings (slowest first):
    __all__                           51.0s  anomalies=421

Report: ./sorethumb-workspace/reports/run_5c720f703718c39e3c5b55b171083f30/index.html
```

The three per-detector rates are heuristic boundaries, not measurements, and
they disagree. Under the default (`"auto"`, `intersection`), a row is flagged only
when all three boundaries flag it, so the flagged count is what they agree on. It is
not a budget you set. Treat the count as *"records worth a look"*, never *"this is
how many anomalies the data has"*.

**4. Print the top anomalies with their reasons**

```bash
sorethumb anomalies           # latest run, top reasons for every flagged row
sorethumb anomalies --top 20  # just the 20 most anomalous rows
sorethumb anomalies --reasons 5 --top 50   # five reason columns
sorethumb anomalies --detail --top 5       # the record behind each reason
```

```
                      Anomalies — run run_5c720f70
┏━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃  # ┃  score ┃ kind      ┃ reason 1             ┃ reason 2           ┃
┡━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│  1 │ 0.9821 │ heuristic │ amount=14500.0       │ country=NG         │
├────┼────────┼───────────┼──────────────────────┼────────────────────┤
│  2 │ 0.9714 │ heuristic │ hour_of_day=3        │ failed_attempts=12 │
├────┼────────┼───────────┼──────────────────────┼────────────────────┤
│  … │        │           │                      │                    │
└────┴────────┴───────────┴──────────────────────┴────────────────────┘
  421 anomaly row(s)   run=run_5c720f703718c39e3c5b55b171083f30   workspace=sorethumb-workspace
```

`kind=exact` means ECOD/HBOS's own additive score decomposition (summing it
recovers the score with zero error, by construction); `kind=model_specific`
means TreeSHAP (Isolation Forest), which uses the fitted trees' actual
structure but decomposes path length, of which the score is a nonlinear
transform; `kind=heuristic` means centroid distance (KMeans), a
finite-difference gradient or KernelSHAP (One-Class SVM, LOF), or a
reference-scaled deviation where a score is flat. With several detectors, a row
gets the weakest kind among them, so the default ensemble's rows are usually
`heuristic`. See
[docs/explanations.md](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md).

For a machine-readable result pipe `--json`:

```bash
sorethumb anomalies --top 100 --json | jq '.[] | {rank, score: .composite_score, r1: .reason_1}'
```

---

## Benchmark results (summary)

Two suites, both reproducible from a clone. Full matrices, method and every
metric are in the
[benchmark results page](https://github.com/tarne75/sorethumb/blob/main/docs/benchmarks.md).
These are the headline cells from the committed run (regenerated 2026-10-14, 3 seeds,
mean ± spread across seeds):

| Setting | ROC-AUC | Average precision |
| --- | --- | --- |
| Synthetic, point anomalies, default pipeline | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Synthetic, clustered anomalies, default pipeline | 1.0000 ± 0.0001 | 0.9988 ± 0.0024 |
| Synthetic, masking, default pipeline | 0.8813 ± 0.0328 | 0.4686 ± 0.1070 |
| Synthetic, local anomalies, default pipeline | 0.7085 ± 0.0335 | 0.1075 ± 0.0460 |
| Synthetic, varying density, default pipeline | 0.6824 ± 0.0341 | 0.0844 ± 0.0552 |
| Synthetic, contextual anomalies, default pipeline | 0.4965 ± 0.0842 | 0.0579 ± 0.0091 |
| Synthetic, rare categorical value, default pipeline | 0.4649 ± 0.0920 | 0.0381 ± 0.0168 |
| Synthetic, off-hours text timestamps, default pipeline | 0.9110 ± 0.0035 | 0.3268 ± 0.0626 |
| Synthetic, 20 irrelevant columns, default pipeline | 0.9557 ± 0.0132 | 0.6758 ± 0.0770 |
| Synthetic, all-categorical combinations, default pipeline | 0.8740 ± 0.0491 | 0.1872 ± 0.0569 |
| KDDCup99 (SA), bare Isolation Forest, 20,000-row sample | 0.9386 ± 0.0061 | 0.4443 ± 0.0143 |
| Covtype, bare Isolation Forest, 20,000-row sample | 0.9837 ± 0.0030 | 0.3486 ± 0.0320 |

Read the weak rows as seriously as the strong ones: the default ensemble is close to
chance on contextual and rare-categorical anomalies and weak on local and varying-density ones, and average
precision is the more telling column when anomalies are rare. These two columns
measure the ranking, which is the same whatever `scoring.contamination` is. The page
above also shows the bare-detector baselines and the flag-level precision and recall.
Those were measured with each detector at the scenario's review budget, not at the
shipped `"auto"`.

The same page has the ensemble-policy matrix: every scenario under every built-in
policy over 10 seeds, with flag precision/recall, false-positive rate, over-flag
ratio and refit stability checked against versioned budgets. Every policy there runs
at the scenario's review budget. The default policy (`intersection@1`) misses some of
those budgets by design; each accepted miss is
listed with its reason in [`tests/benchmark/budgets.toml`](https://github.com/tarne75/sorethumb/blob/main/tests/benchmark/budgets.toml)
(rationale: [`docs/ensemble-policy.md`](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md)).

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
The exact output contract is in [Writing a detector plugin](https://github.com/tarne75/sorethumb/blob/main/docs/models.md#writing-a-detector-plugin).

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
- `contamination` is a review budget, not a prevalence estimate. What it does depends
  on the policy. Under a composite policy a number N means *exactly* the top N% of the
  ranking. Under the default `intersection` it means each detector's top N%, and only
  rows all of them flag are kept, typically far fewer than N%. At `"auto"` (the
  default), each detector uses its own heuristic boundary (see the realised rates in
  the run output); a composite then cuts at the median of those rates, which flagged
  several times a review budget in the backtests. None of this says how many genuine
  anomalies the data holds; only labelled data can tell you that.
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
  (interventional TreeSHAP) attributions, which decompose path length rather
  than the score itself; KMeans and One-Class SVM/LOF are heuristic (centroid
  distance, or a finite-difference gradient restricted to the two detectors
  whose score responds continuously to a small perturbation). See
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
  `sorethumb backfill` produces — are not on a common scale. A raw rate trend
  mixes real change with each period's moving threshold. `sorethumb history`
  reports population drift without a model, and with `history.anchor_run` scores
  every period against one fixed run to separate score drift from threshold
  drift. For one-off comparisons, reuse a fitted run:
  `sorethumb score --from-run RUN_ID`.
- The default ensemble policy (`intersection@1`: every detector must flag a row, each at
  its own boundary under the default `scoring.contamination = "auto"`) is a deliberate
  design decision. It can be precise, but how many rows it flags is unpredictable: on the
  real datasets tested it ranged from 0.18x to 8.4x a review budget.
  Some anomaly types (local, varying-density, contextual) can go unflagged. If you have a
  review budget, the evidence favours `composite-agreement@1` with `scoring.contamination`
  set to that budget; never pair a composite with `"auto"`. Choose on measured evidence,
  not intuition: `sorethumb backtest` measures every policy against the default on your own
  labelled data, and a run with a non-default policy that lacks that evidence warns
  (`run.strict` refuses it). The default is not changing for now, by decision; the
  evidence, what it does and does not cover, and the reasons are in
  [docs/ensemble-policy.md](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md#where-things-stand-2026-10-15).
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
  sent over HTTPS to the exact origin `source.uri` names (credentials with
  an `http://` URI are rejected), and a redirect that would downgrade
  HTTPS to HTTP is refused outright — see
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
- [Explanations: exact, model-specific and heuristic](https://github.com/tarne75/sorethumb/blob/main/docs/explanations.md)
- [Approximations and error characteristics](https://github.com/tarne75/sorethumb/blob/main/docs/approximations.md)
- [Ensemble policy: the default and the evidence](https://github.com/tarne75/sorethumb/blob/main/docs/ensemble-policy.md)
- [Trends: what `sorethumb history` can and cannot tell you](https://github.com/tarne75/sorethumb/blob/main/docs/trends.md)
- [Benchmark results](https://github.com/tarne75/sorethumb/blob/main/docs/benchmarks.md)
- [API and stability policy](https://github.com/tarne75/sorethumb/blob/main/docs/stability.md)
- [Security policy](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md)
- [Contributing](https://github.com/tarne75/sorethumb/blob/main/CONTRIBUTING.md)
