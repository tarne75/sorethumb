# Explanations: exact, model-specific, and heuristic

sorethumb attributes each anomalous row's score to the original features that
contributed most to it. Not all attribution methods are equally trustworthy —
this file documents which methods are `exact`, which are `model_specific`,
and which are `heuristic`, and what each one actually measures. When more
than one detector contributes to a row's explanation, the blended result
keeps only the *weakest* of the contributing tags (`exact` > `model_specific`
> `heuristic` — see "Blending and aggregation" below).

---

## Detection first, explanation second

Explanation never changes detection. Each group runs in two stages:

1. **Detection** combines the detectors' scores, flags rows and ranks them, then
   writes `results/<run>/<group>/anomalies.parquet` (scores, `flagged`, `rank`,
   per-detector scores) and records its SHA-256 as the group's
   `detection_digest`. The flagged rows and their count are final at this point.
   Detection always runs on the feature matrix at `features.dtype`, whatever the
   explain settings.
2. **Explanation** then explains those flagged rows and writes
   `explanations.parquet`, one row per flagged row. It works on a read-only view
   of the flagged set, in its own error boundary. The file's columns:

   | column | meaning |
   | --- | --- |
   | `row_id` | the flagged row |
   | `explain_status` | `ok`; `partial` (some detector's share of the score is unexplained); or `unavailable` (no explanation for this row) |
   | `explain_issue` | null when `ok`; otherwise `detectors_unexplained:<names>`, `beyond_max_rows` or `pca_backprojection_failed` |
   | `attribution_kind` | `exact`, `model_specific`, `heuristic`, or `unavailable` |
   | `reason_<k>` | `field=value`, most important first. Null for an unavailable row: these columns hold reasons only |

Readers (`sorethumb anomalies`, the report) join the two files by `row_id`.

**Status.** Each group records an `explain_status`:

| status | meaning |
| --- | --- |
| `ok` | every flagged row explained, no explainer warned |
| `partial` | some flagged rows unexplained (e.g. beyond `explain.max_rows`), or an explainer warned |
| `failed` | no flagged row has an explanation: the stage raised, or no attribution source explained anything |
| `not_requested` | `explain.enabled = false` |
| `no_flagged_rows` | nothing to explain |

It also records `n_explained` and `n_explain_unavailable`. All of these appear in
`run --json` (per group), on the group row in the workspace and in `sorethumb show`.

**Every warning is a structured record.** Each one carries a stable `code`, its
`stage` (the part before the dot), the `message`, the `group`, and, where they
apply, the `detector` and `rows_affected`. Records are stored on the group row
(`warnings_json`) and appear in `run --json` (`warnings`, top level and per group),
the run summary, `sorethumb show` and the report. The explanation codes:

| code | when |
| --- | --- |
| `explain.stage_failed` | the explanation stage raised; no flagged row in the group is explained |
| `explain.no_source` | no attribution source explained any flagged row |
| `explain.backend_failed` | one detector's explainer raised; its share of the score is unexplained (rows `partial`) |
| `explain.detector_unsupported` | a detector has no vetted attribution method; its share is unexplained (rows `partial`) |
| `explain.fallback_used` | TreeSHAP or KernelSHAP was unavailable or failed; a heuristic method was used instead |
| `explain.rows_beyond_max_rows` | some flagged rows are beyond `explain.max_rows` for every source that covers them (rows `unavailable`) |
| `explain.pca_backprojection_failed` | PCA-space attributions could not be mapped to original columns (all rows `unavailable`) |
| `explain.reference_fallback` | fewer than 10 unflagged rows, so the reference for perturbation scales and SHAP backgrounds is the whole group, flagged rows included |

**Failures are reported, never hidden.**
- An explanation-stage failure raises `ExplanationFailedWarning`.
- A single detector's explainer failing raises `AttributionBackendWarning`.
- Neither ever fails the group or alters its detection output.
- A `failed` explanation stage makes the run exit `partial` (code 4).
- Under `run.strict`, any explanation warning makes the run exit `partial`; it no
  longer fails the group, which would have discarded its detection output.

---

## Exact attributions

### ECOD and HBOS

**Method:** Both detectors define their score, by construction, as an
unweighted average of independent per-feature terms — an empirical-CDF tail
probability for ECOD, a histogram bin log-density for HBOS. The per-feature
term *is* the per-feature contribution; nothing is inferred or approximated.

**What it measures:** How much each feature's own tail rarity (ECOD) or
bin rarity (HBOS) contributes to the row's total outlier score.

**Note:** summing the returned contribution vector recovers the row's outlier
score with zero error — not "high fidelity", zero error, because the score
was already defined as that sum. This is a stronger guarantee than
TreeSHAP's `model_specific` tag below: there is no unverified additivity
assumption here, because there was never an assumption to verify.

**Label in output:** `exact`.

**Applicable when:** detector is `ecod` or `hbos`.

**Why not finite-difference gradients:** both scores are literally step
functions of the input — a rank position (ECOD) or a histogram bin index
(HBOS), with no sub-resolution structure at all. A `step_factor`-sized
perturbation (default 1% of a feature's std) of a genuinely anomalous,
far-tail row routinely lands in the *exact same* rank or bin as the
unperturbed row, producing an exact zero finite difference for precisely the
row an explanation matters most for. That is why ECOD and HBOS get this
dedicated exact decomposition instead of ever reaching the gradient method
below.

---

## Model-specific attributions

### TreeSHAP (Isolation Forest only)

**Method:** SHAP values computed by path enumeration over all trees in the
Isolation Forest ensemble, via `shap.TreeExplainer` with
`check_additivity=False`.

**What it measures:** Each feature's contribution to the row's path length
relative to the expected path length across the training set. A short path
means the record was isolated early, which is what anomaly means for IF.

**Note:** this is *not* tagged `exact`, for two separate reasons.
`check_additivity=False` is required because the path-length trick breaks the
strict SHAP additivity assumption (the sum of SHAP values does not equal the
model output in the usual sense), so additivity is unverified at runtime. And
even where the decomposition holds exactly, what it decomposes is path
length — IsolationForest's score is a nonlinear transform of that, so an
exact accounting of path length is still not an exact accounting of the score
itself. `model_specific` reflects that this uses the fitted trees' actual
structure (more principled than a generic gradient/centroid heuristic)
without claiming the guarantee `exact` would imply. The values are still
directionally correct and comparable across features within a row.

**Label in output:** `model_specific`.

**Applicable when:** detector is `isolation_forest`, `explain.enabled = true`,
and the `explain` extra (shap) is installed. Without shap, or if TreeSHAP
fails for a group, Isolation Forest falls back to the gradient method below
with the out-of-range fallback switched on, and the label becomes `heuristic`.

---

## Heuristic attributions

### Centroid attributions (KMeans)

**Method:** For each anomalous row, compute the vector distance from the row to
its nearest *large-cluster* centroid (the same CBLOF reference set the detector
scores against — small clusters are excluded). The contribution of feature *i*
is the absolute difference between the row's value and the centroid's value on
feature *i*, scaled by that feature's importance in the distance measure.

**What it measures:** Which features are most responsible for the row being far
from any large cluster's centre. This is not a Shapley value — it does not have the
theoretical guarantees (efficiency, symmetry, dummy) that SHAP values carry.
In practice it is a reliable proxy for "why this row is a KMeans outlier."

**Label in output:** `heuristic` (the mechanism — centroid vs gradient — is
not distinguished in the output tag; see below).

**Applicable when:** detector is `kmeans_distance`.

### Gradient attributions (One-Class SVM, LOF)

**Method:** Central finite-difference of the decision function with respect
to each input feature, evaluated at the anomalous row (`explain/gradient.py`).

**What it measures:** How much the anomaly score would change per unit change in
each feature. This is a local linearisation — it is accurate for features with
near-linear influence on the score, and less accurate near decision boundary
non-linearities.

**Cost:** Two forward/backward passes per row (approximately
`2 × n_features` operations). For large `explain.max_rows` values this can be
slow. The default cap (`explain.max_rows = 5000`) keeps total explanation time
bounded.

**Label in output:** `heuristic`.

**Rows outside the data (Isolation Forest fallback, RBF/sigmoid One-Class
SVM):** these scores go flat far from the data. Isolation Forest's score is
piecewise constant, and an RBF kernel decays to nothing away from every support
vector, so for a row outside the reference population's range a small step
changes nothing in exactly the column that put it there. The gradient then
ranks whatever residual slope the other columns have: a row with
`amount=900` against a reference around 50 would be explained as `lat` or
`region`. For these detectors, a row outside the reference range in any
column, or whose gradient is exactly zero, is attributed instead by its
reference-scaled marginal deviation, `|x − median| / scale` per column, where
scale is the reference's IQR/1.349 (falling back to the scaled MAD, then the
standard deviation, then 1 for a constant column). That keeps growing with
distance, so the column the row is furthest out in ranks first. It ignores
interactions between columns, which is why it is used only where the gradient
has already lost the signal. LOF and linear/polynomial One-Class SVMs keep the
plain gradient: their scores keep changing outside the data. The label stays
`heuristic` either way.

**Applicable when:** detector is `one_class_svm` or `lof` — the two detectors
whose score responds continuously to a small perturbation. OneClassSVM's
decision function is a differentiable kernel expansion; LOF's score is built
from continuous distances (its only discreteness is *which* points count as
neighbours, which a small perturbation essentially never flips for a genuine
outlier). Finite-difference gradients are restricted to these two
specifically because ECOD and HBOS are the opposite case — see "Exact
attributions" above.

---

## Blending and aggregation

When multiple detectors contribute, their per-feature attributions are blended
using the same weights as the composite score. This means a detector that
contributes more to the final anomaly score also contributes proportionally
more to the explanation. The blended tag is the *weakest* of the contributing
tags (`exact` > `model_specific` > `heuristic`): averaging an exact ECOD
decomposition with a heuristic gradient result doesn't un-corrupt the
heuristic part, so the blend can only be as trustworthy as its least
trustworthy input.

Each source is L2-normalised per row before blending, so a detector with a
larger native magnitude cannot dominate. A source row whose norm is negligible
(at most 1/1000 of that source's median non-zero row norm, or exactly zero)
carries no information for that row, whether from a flat score or because the
row is beyond that source's `explain.max_rows` cap. It is left out of that
row's blend rather than normalised, since normalising would scale numerical
noise up to the same weight as a real signal, and the remaining sources'
weights are rescaled for that row.

A detector `combination="composite"` assigned zero weight — dropped by the
bad-member guard (see the main scoring docs) as anti-correlated with the
consensus, or independently zeroed by `weighting="agreement"` — is excluded
before attribution even runs, not merely weighted to 0 in the blend: it never
supplies a source, is never counted as covering a row, and can never set the
blended tag or appear in `reason_1`/`reason_2`/etc, even when it is the only
detector whose attribution method would otherwise have succeeded. If that leaves
no source at all, the group's `explain_status` is `failed` with an
`ExplanationFailedWarning`; the rows are never shown as explained with
nothing. A detector's per-row score
columns (`score_raw_<name>`, `score_cal_<name>`) are unaffected and always
recorded for inspection regardless of its weight — only the *blended,
causal* explanation excludes a zero-weight member. `combination="intersection"`/
`"union"` never drop anyone (every configured vote is required), so this
does not apply there.

After blending, `derived → original` aggregation maps one-hot encoded columns
back to their source column. A row that triggers `cat__A = 1, cat__B = 0` does
not produce two separate explanation entries for `cat__A` and `cat__B`; it
produces one entry for `cat` with the combined attribution weight.

`top_n_reasons` then returns the `explain.top_n` features with the highest
absolute attribution weights. These are the features that most distinguish the
anomalous row from the rest of the population.

---

## What the attributions do NOT tell you

- **They are not probabilities.** A high attribution on `revenue` does not mean
  there is a 90% chance the anomaly is caused by unusual revenue.
- **They are local.** The attribution is computed for the specific row, not for
  the class of anomalies that share its pattern.
- **Only `exact` (ECOD, HBOS) is guaranteed to sum to the model output**, and
  only because the score was already defined that way. Heuristic methods make
  no such claim to begin with; `model_specific` (TreeSHAP) runs with
  additivity checking disabled, so it isn't verified there either. Both are
  directionally correct, not numerically exact.
- **They do not imply causation.** An anomalous revenue figure may be caused by
  an anomalous quantity, not by revenue itself, if the two are correlated.
