# Explanations: exact, model-specific, and heuristic

sorethumb attributes each anomalous row's score to the original features that
contributed most to it. Not all attribution methods are equally trustworthy —
this file documents which methods are `exact`, which are `model_specific`,
and which are `heuristic`, and what each one actually measures. When more
than one detector contributes to a row's explanation, the blended result
keeps only the *weakest* of the contributing tags (`exact` > `model_specific`
> `heuristic` — see "Blending and aggregation" below).

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

**Applicable when:** detector is `isolation_forest` and `explain.enabled = true`.

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
