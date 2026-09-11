# Adapting sorethumb to your data

This document walks you from an unfamiliar CSV to a working, well-configured
anomaly detection run. It follows the same path as `sorethumb init && sorethumb
inspect && sorethumb run`.

---

## Step 0: Create a starter config

```bash
# In your project directory
sorethumb init .
```

This writes a fully commented `sorethumb.toml` with every field and its default.
Open it in an editor — you only need to fill in the `[source]` section to get
started.

```toml
[source]
uri = "data/transactions.csv"   # local path or https:// URL
format = "auto"                  # inferred from extension
```

---

## Step 1: Inspect your data

```bash
sorethumb inspect --config sorethumb.toml
```

This loads the dataset, profiles every column, and prints a classification table:

| column | class | treatment | reason |
| --- | --- | --- | --- |
| id | identifier_like | drop | cardinality_ratio=1.00, uuid_pattern |
| timestamp | datetime | time_derivatives | protected: time_column |
| amount | numeric | scale | numeric, n_unique=8420 |
| category | categorical | one_hot | cardinality_ratio=0.002, n_unique=4 |
| description | free_text | drop | mean_length=87.4 |

Use this table to decide:

1. **If a column is classified `identifier_like` but carries signal** (e.g. a store
   ID that is not globally unique), add it to `columns.group_by` or
   `columns.include_only`.

2. **If a column is classified `categorical` but has too many values** for one-hot
   encoding, it will automatically switch to frequency encoding. Adjust
   `features.one_hot_max_cardinality` if the default (20) is wrong for your domain.

3. **If a column is dropped but you want it**, add it to `columns.include_only`
   (to allow only those columns) or remove it from the ignore list.

4. **If the dataset is grouped** (e.g. one file contains multiple sites), add
   the grouping column to `columns.group_by`:
   ```toml
   [columns]
   group_by = ["site_id"]
   ```
   sorethumb will fit a separate detector ensemble per group, which is almost
   always the right choice for multi-entity datasets.

5. **If the dataset has a time column**, tell sorethumb so it can derive temporal
   features and build a history ledger:
   ```toml
   [columns]
   time_column = "timestamp"
   ```

---

## Step 2: Tune the profiling thresholds

The defaults work well for most datasets. Common reasons to change them:

| Situation | Setting to adjust |
| --- | --- |
| Many columns dropped for high nulls when you want to keep them | Raise `profiling.null_ratio_drop` (default 0.70) |
| Missing-indicator columns appearing for columns that are only rarely null | Raise `profiling.null_ratio_flag` (default 0.0 — any null emits an indicator) |
| Low-cardinality IDs being used as features | Set `profiling.identifier_detection = "conservative"` (default) or `"off"` |
| Free-text columns kept when they should be dropped | Lower `profiling.free_text_mean_length` (default 20.0) |

---

## Step 3: Choose your detectors

sorethumb ships three detectors and enables all three by default:

| Detector | Best for | Limitation |
| --- | --- | --- |
| `isolation_forest` | High-dimensional tabular data; model-specific SHAP explanations | Quadratic memory in n_features |
| `kmeans_distance` | Cluster-structured data; fast on large datasets | Assumes cluster structure exists |
| `one_class_svm` | Compact, non-linear decision boundaries | Very slow without a row cap |

For a first run, the defaults are fine. If scoring is slow, reduce
`detectors[2].train_row_cap` for the SVM or disable it entirely:

```toml
[[detectors]]
name = "one_class_svm"
enabled = false
```

---

## Step 4: Set the review budget (`contamination`)

`contamination` is how much of the ranking you want back, **not** an estimate of
how many anomalies exist. `scoring.contamination = "auto"` (the default) lets
each detector fall back to its own heuristic boundary; the flagged count is
whatever those boundaries produce, and the run output prints each detector's
realised rate so you can see they disagree.

Set it explicitly to a fixed shortlist size — e.g. "I have capacity to review
the top 2% each day":
```toml
[scoring]
contamination = 0.02  # flag the top 2% of the ranking for review
```

A labelled validation set can tell you which budget gives acceptable
precision/recall; `contamination` itself never measures prevalence.

---

## Step 5: Run and inspect results

```bash
sorethumb run --config sorethumb.toml
```

Results are written per group to the workspace as Parquet files. An HTML report
is generated at the workspace root. Open it to see:

- Per-group anomaly counts and rates
- A trend chart if you have a time column
- A per-anomaly explanation table: which features drove the score, labelled
  model_specific or heuristic

The run is idempotent — running it again for the same dataset and period skips
groups already completed. Use `--force` to re-run everything.

---

## Step 6: Iterate

Common tuning loops:

| Observation | Action |
| --- | --- |
| Too many anomalies flagged | Lower `scoring.contamination` or raise `profiling.null_ratio_drop` |
| No interesting patterns in the explanation table | Check if the relevant columns are being kept — run `sorethumb inspect` |
| Explanations mention one-hot columns (`cat__A`) instead of the original | This is normal; the output aggregates them into `cat` |
| Run is very slow | Set `detectors[2].enabled = false` for the SVM, or lower `train_row_cap` values |
| `MemoryBudgetError` (projected feature matrix too large) | Lower `run.max_rows` to subsample, lower `features.one_hot_max_cardinality`, enable `features.pca`, or raise `run.max_memory_mb` |

---

## Step 7: Backfill history

If you have historical data and a time column, backfill the ledger:

```bash
sorethumb backfill --config sorethumb.toml --max-periods 90
```

This fills missing periods up to the configured maximum. Each period is fitted
and self-calibrated independently, so the trend chart shows relative movement
period to period rather than an absolute anomaly level on one scale. For a
single-scale trend, score each period against one fixed run with
`sorethumb score --from-run RUN_ID`.

After backfill, the trend chart in the HTML report will show the full history.
