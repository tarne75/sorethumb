# Adapting sorethumb to your data

This document walks you from an unfamiliar CSV to a working, well-configured
anomaly detection run. It follows the same path as `sorethumb init`, then `sorethumb
inspect`, then `sorethumb run`.

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

This loads the dataset, profiles every column, and prints the feature plan: each
column's class, its treatment and the reason. For example:

| Column | Class | Treatment | Reason |
| --- | --- | --- | --- |
| id | identifier_like | drop | UUID pattern detected in sample (n=1000) |
| timestamp | temporal | derive_time | temporal dtype (Datetime(time_unit='us', time_zone=None)) |
| amount | numeric | impute_median | numeric dtype (Float64) |
| category | categorical | one_hot | String with cardinality_ratio=0.0004 |
| region | categorical | one_hot | String with cardinality_ratio=0.0002 |
| description | free_text | drop | mean length=87.4 > 20.0 |

A categorical column is one-hot encoded however few levels it has; only a
constant column is dropped for lack of variety. A change of category counts as
half a typical numeric spread (`features.categorical_distance`, see
[Approximations](approximations.md)), so a column of noise categories does not
drown out the numeric signal.

Use this table to decide:

1. **If a column is classified `identifier_like` but carries signal** (e.g. a store
   ID that is not globally unique), add it to `columns.group_by` (grouping
   columns are exempt from identifier detection), or set
   `profiling.identifier_detection = "off"`.

2. **If a column is classified `categorical` but has too many values** for one-hot
   encoding, it will automatically switch to frequency encoding. Adjust
   `features.one_hot_max_cardinality` if the default (20) is wrong for your domain.

3. **If a column is dropped but you want it**, check the reason column: remove
   it from `columns.ignore` if a pattern matched it, add it to
   `columns.include_only` if that is set and does not list it, or adjust the
   profiling threshold the reason names. `include_only` only narrows which
   columns are considered -- listing a column there does not override a
   classification such as `identifier_like` or `high_null`.

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
   A timestamp stored as text (`"2026-01-15"`, `"2026-01-15T10:30:00"`, a
   bare time of day, or a day/month/year date) is detected and parsed
   automatically -- it does not need to already be a native date/time
   column, and every such column gets temporal features, not only
   `time_column`. Columns that identify rows or groups (`group_by`,
   `id_column`, `reference_column`) and columns an `ignore` pattern matches
   are left as text, so their keys and your ignore rules don't change. An
   ambiguous day/month order (every value's day and month
   both ≤ 12, so either reading would parse) is never guessed; set
   `columns.datetime_formats` to resolve it, or to force-parse a column auto-detection
   misses, including Unix epoch seconds/milliseconds:
   ```toml
   [columns.datetime_formats]
   "order_date" = "%m/%d/%Y"   # force month-first where it would otherwise be ambiguous
   "created_at" = "epoch_s"    # an integer column holding Unix seconds
   ```

---

## Step 2: Tune the profiling thresholds

The defaults work well for most datasets. Common reasons to change them:

| Situation | Setting to adjust |
| --- | --- |
| Many columns dropped for high nulls when you want to keep them | Raise `profiling.null_ratio_drop` (default 0.70) |
| Missing-indicator columns appearing for columns that are only rarely null | Raise `profiling.null_ratio_flag` (default 0.0 — any null emits an indicator) |
| A high-cardinality text ID being used as a feature | Set `profiling.identifier_detection = "aggressive"` (flags any high-cardinality text column), or list it in `columns.ignore` |
| An ID that carries signal being dropped | Set `profiling.identifier_detection = "off"`, or add it to `columns.group_by` |
| Free-text columns kept when they should be dropped | Lower `profiling.free_text_mean_length` (default 20.0) |

---

## Step 3: Choose your detectors

sorethumb ships six detectors and runs three by default (`ecod`, `hbos` and `lof`
are opt-in; see [Detector models](models.md)):

| Detector | Best for | Limitation |
| --- | --- | --- |
| `isolation_forest` | General-purpose tabular data; model-specific TreeSHAP explanations | Struggles with very high-dimensional sparse data |
| `kmeans_distance` | Cluster-structured data; fast on large datasets | Assumes cluster structure exists |
| `one_class_svm` | Compact, non-linear decision boundaries | Very slow without a row cap |

For a first run, the defaults are fine. A `[[detectors]]` list **replaces** the
default ensemble, so to drop or tune one detector, list every detector you want.
For example, if training is slow, keep the other two and leave out the SVM:

```toml
[[detectors]]
name = "isolation_forest"

[[detectors]]
name = "kmeans_distance"
```

(or keep it with a lower `train_row_cap`). A list naming only `one_class_svm`
with `enabled = false` would leave no detector at all.

---

## Step 4: Set the review budget (`contamination`)

`contamination` is how much of the ranking you want back, **not** an estimate of
how many anomalies exist. `scoring.contamination = "auto"` (the default) lets
each detector fall back to its own heuristic boundary; the flagged count is
whatever those boundaries produce, and the run output prints each detector's
realised rate so you can see they disagree.

Set it explicitly to a fixed shortlist size, e.g. "I have capacity to review
the top 2% each day". To get exactly that share, use a composite policy as well:
under the default `intersection`, each detector flags its top 2% and only rows all
of them flag are kept, usually far fewer.
```toml
[scoring]
contamination = 0.02      # flag the top 2% of the ranking for review
combination   = "composite"
weighting     = "agreement"  # composite-agreement@1: the evidenced choice at a fixed budget
```
Never pair a composite with `"auto"`; see
[ensemble-policy.md](ensemble-policy.md#where-things-stand-2026-10-15). A run with
this non-default policy warns until `sorethumb backtest` has produced evidence for it
on your data.

A labelled validation set can tell you which budget gives acceptable
precision/recall; `contamination` itself never measures prevalence.

---

## Step 5: Run and inspect results

```bash
sorethumb run --config sorethumb.toml
```

Results are written per group to the workspace as Parquet files, and an HTML
report to `reports/<run_id>/index.html` inside the workspace (the run summary
prints its path). Open it to see:

- Per-group anomaly counts and rates
- A per-anomaly explanation table: which features drove the score, labelled
  exact, model_specific, or heuristic, with the record behind each reason
- For a run with a period, the trend sections of [Trends](trends.md)

The run is idempotent — running it again for the same dataset and period skips
groups already completed. Use `--force` to re-run everything.

---

## Step 6: Iterate

Common tuning loops:

| Observation | Action |
| --- | --- |
| Too many (or too few) anomalies flagged | At the default `"auto"` you don't set the count. Set a numeric `scoring.contamination` with a composite policy (Step 4) to choose it |
| No interesting patterns in the explanation table | Check if the relevant columns are being kept — run `sorethumb inspect` |
| Explanations mention one-hot columns (`cat__A`) instead of the original | This is normal; the output aggregates them into `cat` |
| Run is very slow | List only the detectors you want under `[[detectors]]` (leaving out `one_class_svm`), or lower `train_row_cap` values |
| `MemoryBudgetError` (projected feature matrix too large) | Lower `run.max_rows` to subsample, lower `features.one_hot_max_cardinality`, enable `features.pca`, or raise `run.max_memory_mb` |

---

## Step 7: Backfill history

If you have historical data and a time column, backfill the ledger:

```bash
sorethumb backfill --config sorethumb.toml --max-periods 90
```

This fills missing periods up to the configured maximum. Then
`sorethumb history --config sorethumb.toml` shows how the data is changing:
population drift, and the anomaly rate per window. Each period is fitted and
self-calibrated independently, so its rate is relative to that period's own
threshold, not an absolute anomaly level. To tell a real change in scores from a
moved threshold, set `history.anchor_run` to one accepted run, which every
period is then also scored against. See [Trends](trends.md).
