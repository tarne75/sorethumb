# Approximations and error characteristics

This file documents every step where `sorethumb` uses an approximation rather than an
exact computation, along with its practical error characteristics.

## Correlation reduction — sampled Pearson matrix

When the feature matrix exceeds 200,000 rows, the Pearson correlation matrix is computed
on a random subsample of 200,000 rows rather than the full dataset. Pearson correlation
coefficients stabilise quickly with sample size; at 200,000 rows the standard error of
the estimate is ~0.002, which is negligible relative to the 0.95 threshold used for
column dropping. The subsample is drawn with a fixed seed for reproducibility.

## Silhouette score for KMeans `k` selection — sampled

The silhouette score is near-quadratic in the number of rows. When evaluating candidate
`k` values, silhouette is computed on a subsample capped at 20,000 rows. This subsample
is seeded and deterministic. The elbow criterion (computed on the full training data) is
the primary selection criterion; silhouette breaks ties. No accuracy guarantee is made for
the selected `k` when the true cluster structure requires >20,000 rows to distinguish.

## TreeSHAP for Isolation Forest — additivity unverified, and not of the final score

`shap.TreeExplainer` is run with `check_additivity=False`, because IsolationForest's
`score_samples` is not exactly the SHAP sum of its base value and per-feature
contributions — the path-length trick breaks strict additivity, and additivity
checking would otherwise raise on every call. Two consequences follow, and both
are why the result is labelled `model_specific`, not `exact`: (1) with the check
disabled, there is no runtime guarantee the returned values actually sum to the
model output; (2) even where TreeSHAP's decomposition is exact, what it
decomposes is path length, which IsolationForest's score is a nonlinear
transform of — so an exact accounting of path length is still not an exact
accounting of the score the user sees. `model_specific` reflects that this is
more principled than a generic gradient/centroid heuristic (it uses the fitted
trees' actual structure via the standard TreeSHAP algorithm), while stopping
short of claiming the additivity guarantee "exact" would imply.

## KernelSHAP — Monte Carlo approximation

When `explain.kernel_shap = true`, attributions for non-tree detectors are computed via
`shap.KernelExplainer`. This uses Monte Carlo sampling of the feature space to estimate
Shapley values. The result is labelled `heuristic`, not `model_specific` — it never touches
the detector's internal structure, unlike TreeSHAP. Accuracy increases with
`nsamples` but so does runtime. The default `nsamples` is documented in `explain/`.

## Feature matrix dtype — float32 default

The feature matrix uses `float32` by default (`features.dtype = "float32"`), halving
memory footprint relative to `float64`. All three default detectors (Isolation Forest,
KMeans, One-Class SVM) produce materially identical results at both precisions on typical
tabular anomaly detection tasks. If your use case requires `float64` precision, set
`features.dtype = "float64"` in config.

## `auto` contamination — median of natural flag rates

When `scoring.contamination = "auto"`, the review budget is derived as the median
of each enabled detector's natural flag rate (the fraction of rows the detector's
own boundary flags as anomalous). This is a heuristic — it does not produce a
calibrated estimate of the true anomaly rate, and the per-detector rates it takes
the median of routinely disagree by 2–3×. The run summary and `sorethumb run --json`
report each detector's realised rate so the resulting flag count is read as a
shortlist size, not a measurement. Validation against labelled benchmark datasets
is ongoing; see the benchmark table in the README.
