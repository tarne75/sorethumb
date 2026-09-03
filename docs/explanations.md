# Explanations: exact vs heuristic

sorethumb attributes each anomalous row's score to the original features that
contributed most to it. Not all attribution methods are equally trustworthy.
This file documents exactly which methods are exact and which are heuristic,
and what each heuristic actually measures.

---

## Exact attributions

### TreeSHAP (Isolation Forest only)

**Method:** SHAP values computed by exact path enumeration over all trees in
the Isolation Forest ensemble.

**What it measures:** Each feature's contribution to the row's path length
relative to the expected path length across the training set. A short path
means the record was isolated early, which is what anomaly means for IF.

**Note:** TreeSHAP requires `check_additivity=False` for Isolation Forest because
the path-length trick breaks the strict SHAP additivity assumption (the sum of
SHAP values does not equal the model output in the usual sense). The values are
still directionally correct and comparable across features within a row.

**Applicable when:** detector is `isolation_forest` and `explain.enabled = true`.

---

## Heuristic attributions

### Centroid attributions (KMeans)

**Method:** For each anomalous row, compute the vector distance from the row to
its nearest cluster centroid. The contribution of feature *i* is the absolute
difference between the row's value and the centroid's value on feature *i*,
scaled by that feature's importance in the distance measure.

**What it measures:** Which features are most responsible for the row being far
from any cluster centre. This is not a Shapley value — it does not have the
theoretical guarantees (efficiency, symmetry, dummy) that SHAP values carry.
In practice it is a reliable proxy for "why this row is a KMeans outlier."

**Label in output:** `heuristic:centroid`

**Applicable when:** detector is `kmeans_distance`.

### Gradient attributions (One-Class SVM and others)

**Method:** Input gradient of the decision function with respect to each input
feature, evaluated at the anomalous row. Integrated gradients (a single forward
pass minus a single backward pass) are used when the model exposes a
differentiable score function.

**What it measures:** How much the anomaly score would change per unit change in
each feature. This is a local linearisation — it is accurate for features with
near-linear influence on the score, and less accurate near decision boundary
non-linearities.

**Cost:** Two forward/backward passes per row (approximately
`2 × n_features` operations). For large `explain.max_rows` values this can be
slow. The default cap (`explain.max_rows = 5000`) keeps total explanation time
bounded.

**Label in output:** `heuristic:gradient`

**Applicable when:** detector supports neither TreeSHAP nor centroid attribution.

---

## Blending and aggregation

When multiple detectors contribute, their per-feature attributions are blended
using the same weights as the composite score. This means a detector that
contributes more to the final anomaly score also contributes proportionally
more to the explanation.

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
- **For heuristic methods, they are not guaranteed to sum to the model output.**
  They are directionally correct, not numerically exact.
- **They do not imply causation.** An anomalous revenue figure may be caused by
  an anomalous quantity, not by revenue itself, if the two are correlated.
