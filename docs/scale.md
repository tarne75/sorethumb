# Scale guide

sorethumb is built to run on a single machine — a laptop, a CI runner, one VM. It
is measured end to end from 10,000 to 1,000,000 rows: a default run on a
million-row table takes about 7 minutes and 5 GB of memory on an 8-core, 16 GB
laptop. Larger datasets are untested. At that size a
distributed engine adds cost, operational surface, and non-determinism without
buying meaningful speed, so there is no Spark, Dask, or cloud-service dependency
anywhere. The detectors whose training cost grows faster than linearly (LOF at
roughly `O(n·log²n)`, one-class SVM at `O(n²)`) are each fitted on a bounded row
sample rather than the whole dataset; every row is still scored. The
[limitations](limitations.md) spell out where these choices bite.

Measured with the defaults (`sorethumb run data.csv`) on a 14-column mixed table, on an
8-core, 16 GB laptop; one run per size
([docs/benchmarks.md](benchmarks.md#end-to-end-scale)
has the full breakdown and the command to reproduce it):

| Rows | Run | Peak memory | Score a new file of that size |
| --- | --- | --- | --- |
| 10,000 | 13 s | 0.9 GB | 7 s |
| 100,000 | 68 s | 1.8 GB | 47 s |
| 1,000,000 | 6.9 min | 4.9 GB | 6.5 min |

Above 1,000,000 rows sorethumb is untested. What drives the cost:

- **Explanations**, three quarters of the run at 1,000,000 rows. Their cost follows
  the rows flagged (about 6 to 8 ms each), and `scoring.contamination = "auto"`
  flagged about 5% of rows. A numeric review budget flags fewer rows; `explain.enabled
  = false` drops the stage.
- **Scoring every row**, which grows linearly. One-Class SVM is most of it (67 s at
  1,000,000 rows); disabling it, or lowering its `train_row_cap` (default 25,000),
  shortens it.
- **Training is bounded** by each detector's row cap (Isolation Forest 250,000, KMeans
  200,000, One-Class SVM 25,000): about 25 s at 100,000 and at 1,000,000 rows alike.
- **`run.max_rows`** caps the rows read, if you would rather sample a larger table than
  run all of it.
- **More than 2000 features**: enable `features.pca`; a wider run raises
  `FeatureWidthWarning` by default.

Peak memory is not the feature matrix (27 float32 features on 1,000,000 rows is about
100 MB): loading and profiling the data and the explanations dominate it.
`run.max_memory_mb` checks only the projected matrix, so leave room above it.
