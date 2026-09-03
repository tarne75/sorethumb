# sorethumb

**Unsupervised anomaly detection for tabular data.**

`sorethumb` takes a dataset it knows nothing about, works out how to treat every
column, fits an ensemble of detectors, ranks the records that stand out, and explains
*why* each one stands out in terms of the original columns.

[![CI](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml/badge.svg)](https://github.com/tarne75/sorethumb/actions/workflows/ci.yml)

---

## Why sorethumb?

There are already good anomaly-detection libraries (PyOD ships far more detectors).
`sorethumb` is not competing on detector count. Its contribution is the surrounding
machinery those libraries leave to the user:

1. **Zero-configuration column handling** — profile, classify, encode, impute, derive.
2. **Ensemble scoring with calibrated, comparable scores** across runs.
3. **Per-record explanations** in original feature terms, labelled exact or heuristic.
4. **Run history that is actually valid** — persisted models and score calibration so
   trend lines don't conflate model change with data change.
5. **Idempotent, resumable execution** with a completion ledger.

It runs on a single machine, uses Polars throughout, and has no dependency on Spark,
Databricks, or any cloud vendor.

---

## Installation

```bash
pip install sorethumb
```

For the benchmark harness:

```bash
pip install "sorethumb[benchmark]"
```

---

## 60-second quickstart

```python
import polars as pl
from sklearn.datasets import fetch_kddcup99
from sorethumb import Config, SourceConfig, run_detection

# Fetch KDDCup99 and save a CSV for the demo
bunch = fetch_kddcup99(subset=None, shuffle=True, random_state=0, percent10=True)
df = pl.from_numpy(bunch.data, schema={f"f{i}": pl.Float64 for i in range(bunch.data.shape[1])})
df.write_csv("/tmp/kdd.csv")

config = Config(
    source=SourceConfig(uri="/tmp/kdd.csv"),
    run={"workdir": "/tmp/sorethumb_demo"},
)
result = run_detection(config, no_report=True)
print(f"Found {result.n_anomalies} anomalies across {result.n_succeeded} groups")
```

Or with the CLI:

```bash
sorethumb init .
# edit sorethumb.toml to point at your data
sorethumb inspect          # profile columns, no model fitting
sorethumb run              # full pipeline
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
| synthetic_gaussian | isolation_forest | 1050 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.439 | 0.011 | 82.5 |
| synthetic_gaussian | kmeans_distance | 1050 | 8 | 0.0828 | 0.0258 | 0.0000 | 0.0000 | 0.0000 | 0.335 | 0.000 | 22.5 |
| synthetic_gaussian | one_class_svm | 1050 | 8 | 0.9127 | 0.2054 | 0.1200 | 0.1200 | 0.1200 | 0.004 | 0.003 | 1.3 |
| synthetic_highd | isolation_forest | 2100 | 32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.090 | 0.017 | 1.0 |
| synthetic_highd | kmeans_distance | 2100 | 32 | 0.0002 | 0.0244 | 0.0000 | 0.0000 | 0.0000 | 0.569 | 0.000 | 27.2 |
| synthetic_highd | one_class_svm | 2100 | 32 | 0.9205 | 0.2204 | 0.0800 | 0.0800 | 0.0800 | 0.014 | 0.014 | 0.1 |
| kddcup99_sa | isolation_forest | 100655 | 38 | 0.9448 | 0.4652 | 0.2852 | 0.2852 | 0.2852 | 0.164 | 0.225 | 7.6 |
| kddcup99_sa | kmeans_distance | 100655 | 38 | 0.8024 | 0.0969 | 0.0086 | 0.0086 | 0.0086 | 23.832 | 0.007 | 544.4 |
| kddcup99_sa | one_class_svm | 100655 | 38 | 0.7695 | 0.0799 | 0.0086 | 0.0086 | 0.0086 | 42.464 | 34.022 | 0.0 |
| covtype | isolation_forest | 581012 | 54 | 0.9346 | 0.0365 | 0.0346 | 0.0346 | 0.0346 | 0.299 | 1.114 | 0.9 |
| covtype | kmeans_distance | 581012 | 54 | 0.4669 | 0.0040 | 0.0000 | 0.0000 | 0.0000 | 35.619 | 0.033 | 467.6 |
| covtype | one_class_svm | 581012 | 54 | 0.7490 | 0.0087 | 0.0000 | 0.0000 | 0.0000 | 1685.427 | 1247.062 | 0.0 |
<!-- benchmark-results-end -->

---

## Scale guide

| Dataset size | Recommended setup |
| --- | --- |
| < 100 k rows | All three detectors, default config |
| 100 k – 1 M rows | Disable `one_class_svm` or set `train_row_cap = 20000` |
| > 1 M rows | Set `run.max_rows` to subsample; enable `features.pca` |
| > 2000 features | Enable `features.pca`; raises `FeatureWidthWarning` by default |

Memory footprint is dominated by the feature matrix: `n_rows × n_features × 4 bytes`
(float32). An 8 GB budget handles roughly 500 M cells.

---

## Honest limitations

- `run.max_memory_mb` is advisory, not enforced — Python cannot impose a hard RSS ceiling.
- The composite score is an interpretable ranking score, not a calibrated probability.
- Only Isolation Forest yields exact (TreeSHAP) attributions; all others are heuristic
  (centroid distance or input gradient). See [docs/explanations.md](docs/explanations.md).
- Self-calibrated runs are not strictly comparable to each other — use
  `sorethumb score --from-run RUN_ID` for cross-run trends.
- Unsupervised anomaly ≠ the thing you care about. The library ranks statistical oddity;
  whether an odd record is *interesting* is a domain judgement it cannot make.

---

## Documentation

- [Configuration reference](docs/configuration.md)
- [Adapting to your data](docs/adapting-to-your-data.md)
- [Explanations: exact vs heuristic](docs/explanations.md)
- [Approximations and error characteristics](docs/approximations.md)
- [Contributing](CONTRIBUTING.md)
