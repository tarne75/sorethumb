# Detector models — what they do and when to use them

sorethumb runs an *ensemble* of detectors: several independent methods each
score every record, and their votes are combined. Using detectors from
different algorithmic families means no single blind spot dominates. A record
has to look strange to multiple independently-reasoning methods before it gets
flagged.

This page explains what each detector actually does — in plain terms, without
equations — and helps you choose which ones belong in your ensemble.

---

## How anomaly scoring works

Every detector answers the same question for every row: *how unusual does this
look, compared to everything the model was trained on?*

Each detector's answer is put on a common 0–1 scale and the detectors are
combined into a single `composite_score`, where **1 = extremely unusual** and
**0 = looks completely normal**. This is the score shown in the results table,
the HTML report and `--json` output; rows are listed from the highest score
(most anomalous) down.

A record is flagged as anomalous when its score crosses a threshold learned
from your data — the top `contamination` fraction of rows, or each detector's
own natural boundary when `contamination = "auto"`. Nothing is hard-coded.

---

## The default ensemble

Three detectors run unless you change anything. They were chosen to complement
each other — each catches a different *kind* of oddness.

---

### `isolation_forest` — the random interrogator

**Analogy.** Imagine you have a crowd of people and you want to find the
unusual ones. You pick random questions — "are you taller than 5'10"?", "do
you earn more than £50k?" — and keep splitting the crowd based on the answers.
Normal people blend into large groups and take many questions to isolate.
Unusual people end up alone after just a few questions, because they sit in
sparse, unpopulated corners of the space.

Isolation Forest does exactly this, automatically, across all your numeric
columns at once. It builds 200 random decision trees and measures how quickly
each record gets isolated. Records that are isolated quickly are anomalies.

**What it catches well.**
- Outliers in any combination of features (not just individual columns)
- Very large datasets — it's fast and memory-efficient
- Cases where anomalies are genuinely rare and isolated

**Where it struggles.**
- Dense clusters of anomalies (if your anomalies tend to cluster together, the
  method may not isolate them quickly)
- Very high-dimensional sparse data where every record looks isolated

**Explanations.** This is the only detector that produces *model-specific*
explanations via TreeSHAP, using the fitted trees' actual structure rather
than a generic gradient/centroid heuristic. It is not a verified *exact*
decomposition of the score, though: TreeSHAP is run with
`check_additivity=False` because IsolationForest's score is a nonlinear
transform of the path length SHAP explains, so nothing guarantees the
attributions sum to the model output. All other detectors use heuristic
approximations for explanations.

**Training cap.** 250,000 rows by default. Above that, a random sample is
taken for training, but all rows are scored.

**Config.**
```toml
[[detectors]]
name = "isolation_forest"
train_row_cap = 250_000   # optional: lower if memory is tight

[detectors.params]
n_estimators = 200        # more trees = more stable, slower
# extra_params = { n_jobs = -1 }   # any other sklearn IsolationForest argument
```

---

### `kmeans_distance` — the neighbourhood watch

**Analogy.** Imagine sorting your records into natural groups — customers who
buy frequently and spend a little, customers who buy rarely and spend a lot,
and so on. The *big* groups are what "normal" looks like. Every record is then
scored by how far it sits from the nearest **big** group's centre — not its own
group's centre. A small clump of records that formed its own little group still
scores as anomalous, because it is nowhere near a big one.

This is CBLOF-style scoring, and it matters: score every record against *its
own* centre and a tight cluster of anomalies gets handed its own centroid and
is rated perfectly normal. That was a real bug — `kmeans_distance` used to
score planted anomalies as the *most* normal records (benchmark ROC-AUC
~0.0002). Measuring against the big clusters only removes that failure mode.

"Big" is controlled by **`large_cluster_coverage`** (default `0.90`): the
largest groups that together hold 90% of the training rows are the reference
set; every other cluster is a candidate for anomalies. A consequence of that
default: the detector assumes anomalies are no more than about **10%** of your
data. If your true anomaly rate is higher, lower `large_cluster_coverage` so
more groups count as "big"; raise it toward `1.0` to be stricter, at the cost
of re-exposing the own-centroid failure for a larger anomaly cluster.

**What it catches well.**
- Straightforward numerical outliers — records that don't fit any natural
  grouping
- Large datasets (fast to train and score)
- Situations where normal data has clear structure (e.g. customers who reliably
  fall into behavioural segments)

**Where it struggles.**
- Contamination above ~10% at the default `large_cluster_coverage` — an anomaly
  group big enough to count as a reference cluster is scored normal. Lower
  `large_cluster_coverage` for higher anomaly rates.
- Unusual-combination anomalies: a record with individually normal values that
  combine in an impossible way (e.g. age=5, income=£200k). It only cares about
  distance, not plausibility.
- Datasets where the number of natural groups is hard to guess (it picks *k*
  automatically but that heuristic isn't perfect)
- Non-spherical clusters — if your groups are elongated or ring-shaped, the
  distance measure is misleading

**Training cap.** 200,000 rows by default.

**Config.**
```toml
[[detectors]]
name = "kmeans_distance"
train_row_cap = 200_000
[detectors.params]
large_cluster_coverage = 0.90   # lower it if anomalies are more than ~10% of rows
# extra_params = { max_iter = 500, tol = 1e-5 }   # any other sklearn KMeans argument
```

---

### `one_class_svm` — the boundary drawer

**Analogy.** Imagine you're drawing a fence around all the normal records on a
map. You want the tightest fence that still encloses all (or most) of the
normal data. Once the fence is drawn, any new record that lands outside it is
flagged as anomalous.

One-Class SVM draws that fence. It doesn't care about individual distances or
clusters — it finds the smoothest boundary that separates "this is the normal
territory" from "this is outside normal territory".

**What it catches well.**
- A genuinely different algorithmic view of the data — its concept of "outside
  normal" is different from what Isolation Forest and KMeans see, so it catches
  things those methods miss
- Datasets where normal data occupies a compact, well-defined region

**Where it struggles.**
- Speed — it scales poorly with the number of rows (training cost grows
  quadratically). It is capped at 25,000 training rows by default; above that,
  a random sample is used
- Very high-dimensional data — the boundary becomes harder to draw reliably
- Large datasets where you want all rows trained on

**Training cap.** 25,000 rows by default. This is the tightest cap of the
three default detectors, by design.

**Config.**
```toml
[[detectors]]
name = "one_class_svm"
train_row_cap = 25_000   # raising this makes training much slower

[detectors.params]
nu = "auto"   # fraction of training data allowed to be anomalous; "auto" uses 0.1
# extra_params = { tol = 1e-4, shrinking = false }   # any other sklearn OneClassSVM argument
```

---

## Additional detectors

These are built in and ready to use, but not included in the default ensemble.
Add any of them to your config to extend the ensemble.

---

### `ecod` — the percentile inspector

**Analogy.** Imagine you're looking at salary data. You know what the
distribution looks like: most people earn between £25k and £80k, very few earn
under £10k or over £500k. ECOD formalises this intuition across every column
simultaneously. For each column, it works out where each value sits in the
distribution — is it in the bottom 1%? The top 0.5%? — and flags records that
are extreme on one or more dimensions.

The "two-tailed" part means it looks for values that are unusually *low* and
values that are unusually *high*. A salary of £1 and a salary of £10,000,000
would both be flagged.

**What it catches well.**
- Marginal outliers — records with a single column value that is extreme (age
  = 150, transaction = £0.00, session_duration = 0.001 seconds)
- High-dimensional data — it evaluates each column independently so it scales
  well even with hundreds of features
- Situations where you have no tuning budget — it has no meaningful
  hyperparameters

**Where it struggles.**
- Combination anomalies — a person aged 25 earning £200k might not be extreme
  on either dimension alone, but together they're unusual. ECOD would miss
  this. (Isolation Forest and One-Class SVM would catch it.)
- Datasets where the distribution is wildly non-standard (highly multimodal)

**Training cap.** 500,000 rows.

**Config.**
```toml
[[detectors]]
name = "ecod"
```

---

### `lof` — the local density detective

**Analogy.** Normal neighbourhoods vary in density. In a packed city centre,
everyone is close to many others. In a rural village, everyone is close to a
few neighbours but far from the city. LOF doesn't ask "are you far from the
average?" — it asks "are you *less densely surrounded than the records around
you*?"

Think of it this way: if your 20 nearest neighbours are all very close
together, but you're somewhat distant from them, you're a local anomaly — even
if you're not a global one. This is what LOF detects.

**What it catches well.**
- Anomalies that are only unusual *in their local context* — records that
  aren't global outliers but are sparse relative to their nearest neighbours
- Datasets with multiple clusters at different densities (some groups are
  naturally closer together than others)
- Subtle outliers that sit just outside the edge of a local cluster

**Where it struggles.**
- Large datasets — it stores the full neighbourhood graph, so memory and
  training time grow with the number of rows. It is capped at 50,000 training
  rows by default.
- Very high-dimensional data — measuring "nearest neighbours" becomes
  unreliable when there are hundreds of columns (the curse of dimensionality)
- Cases where every record is locally dense (LOF has nothing to contrast
  against)

**Training cap.** 50,000 rows — the tightest cap of all detectors. Keep this
in mind on large datasets; a representative sample is still used for fitting,
but LOF is best suited to smaller datasets.

**Config.**
```toml
[[detectors]]
name = "lof"
train_row_cap = 50_000

[detectors.params]
n_neighbors = 20   # how many neighbours to consult; higher = smoother, slower
# extra_params = { n_jobs = -1, leaf_size = 40 }   # any other sklearn LocalOutlierFactor argument
```

---

### `hbos` — the frequency counter

**Analogy.** Imagine building a histogram for every column in your data — a
bar chart that shows how many records fall into each range of values. A record
that lands in a very short bar (a low-frequency bucket) is unusual. HBOS adds
up the "how unusual is this value?" scores across all columns and produces a
total oddness score for each record.

It's the simplest detector conceptually and the fastest computationally.

**What it catches well.**
- Pure marginal outliers — records whose individual column values fall in sparse
  histogram buckets
- Speed-critical pipelines — fit and score are both extremely fast, even on
  very large datasets
- A quick sanity baseline: if Isolation Forest and HBOS agree a record is
  anomalous, it's almost certainly worth investigating

**Where it struggles.**
- Combination anomalies — like ECOD, it treats each column independently and
  cannot detect records that are unusual only in how their values *combine*
- Columns with complex distributions — the histogram binning (even with
  automatic bin selection) can miss nuance in highly skewed or multimodal data
- It is the least sensitive detector overall; use it for breadth, not depth

**Training cap.** 500,000 rows.

**Config.**
```toml
[[detectors]]
name = "hbos"

[detectors.params]
n_bins = "auto"   # "auto" uses Freedman-Diaconis bin selection per column
                  # or pass an integer (e.g. 50) to fix the bin count
```

---

## Quick reference

| Detector | Default | Train cap | Best for | Main weakness | Explanations |
|---|---|---|---|---|---|
| `isolation_forest` | ★ | 250k | General purpose; combination anomalies | Dense anomaly clusters | Exact (TreeSHAP) |
| `kmeans_distance` | ★ | 200k | Datasets with clear natural groupings | Non-spherical clusters; anomalies > ~10% of rows; unusual combos | Heuristic |
| `one_class_svm` | ★ | 25k | Ensemble diversity; compact normal regions | Slow on large data | Heuristic |
| `ecod` | — | 500k | Marginal (single-column) outliers; many features | Combination anomalies | Heuristic |
| `lof` | — | 50k | Local density anomalies; mixed-density clusters | Large datasets; high dimensionality | Heuristic |
| `hbos` | — | 500k | Speed; sanity baseline | Combination anomalies; least sensitive | Heuristic |

**Advanced tuning.** `isolation_forest`, `kmeans_distance`, `one_class_svm` and
`lof` wrap a scikit-learn estimator; pass any of its other constructor arguments
through a nested `extra_params` table (`params = { extra_params = { n_jobs = -1 } }`).
The [configuration reference](configuration.md#detector-extra_params) lists every
accepted key per detector. `sorethumb init` writes them all, commented out, into
the starter config.

---

## Choosing your ensemble

**Stick with the defaults** if you're not sure. The three-detector default was
chosen to cover complementary failure modes: Isolation Forest handles
combination anomalies, KMeans handles cluster-distance outliers, and One-Class
SVM draws an independent boundary. Together they catch a wide range of real
anomaly patterns.

**Add ECOD or HBOS** if you have very high-dimensional data (many columns) and
want extra signal on marginal (per-column) outliers, or if you want a
fast, parameter-free cross-check.

**Add LOF** if your data has multiple natural subpopulations with different
densities — for example, transaction data where enterprise customers behave
very differently from individual consumers, and you want to catch outliers
*within* each population rather than just globally.

**Drop `one_class_svm`** if your dataset is large (> 100k rows) and training
time matters. Replace it with `ecod` for a faster, parameter-free alternative
that still provides a different view from Isolation Forest.

```toml
# Example: speed-optimised ensemble for large datasets
[[detectors]]
name = "isolation_forest"
train_row_cap = 250_000

[[detectors]]
name = "kmeans_distance"

[[detectors]]
name = "ecod"
```

**Use the CLI shorthand** to try combinations without editing your config file:

```bash
sorethumb run -d if,km,ecod data.parquet       # swap in ECOD for speed
sorethumb run -d if,lof data.parquet           # focus on local density
sorethumb run -d if,km,oc,ecod,lof,hbos data.parquet  # run everything
```

Each run with `-d` automatically saves the detector list to your config file.
