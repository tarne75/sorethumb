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

## ECOD / HBOS — exact by construction, not an approximation

Both detectors already define their score as an unweighted average of
independent per-feature terms (an empirical-CDF tail probability for ECOD, a
histogram bin log-density for HBOS). `explain/native.py` returns exactly
those terms as the per-feature attribution — summing them recovers the score
with zero error, because that sum *is* the score's definition. This is the
one attribution method in `sorethumb` that involves no approximation step at
all, hence the `exact` tag (see `docs/explanations.md`).

## Finite-difference gradients — restricted to smooth-enough scores

`explain/gradient.py`'s central finite-difference is only applied to
OneClassSVM and LOF (`_pipeline.py::_compute_attributions`) — the two
detectors whose score responds continuously to a small perturbation.
ECOD and HBOS are excluded on purpose: their score is a discrete rank/bin
lookup with no sub-resolution structure, so a `step_factor`-sized
perturbation (default 1% of a feature's std) of a genuinely anomalous,
far-tail row routinely lands in the exact same rank or bin as the
unperturbed row — an exact zero finite difference for precisely the row an
explanation matters most for. They get the exact decomposition above
instead. A detector type the dispatch doesn't recognise is skipped (logged,
not attributed) rather than defaulting to gradient — silently guessing at a
score's smoothness is exactly the failure mode this restriction exists to
prevent.

## KernelSHAP — Monte Carlo approximation

When `explain.kernel_shap = true`, attributions for OneClassSVM/LOF are computed via
`shap.KernelExplainer` instead of finite-difference gradients (ECOD and HBOS always use
their exact decomposition above, regardless of this setting). This uses Monte Carlo
sampling of the feature space to estimate Shapley values. The result is labelled
`heuristic`, not `model_specific` — it never touches the detector's internal structure,
unlike TreeSHAP. Accuracy increases with `nsamples` but so does runtime. The default
`nsamples` is documented in `explain/`.

## Feature matrix dtype — float32 default

The feature matrix uses `float32` by default (`features.dtype = "float32"`), halving
memory footprint relative to `float64`. All three default detectors (Isolation Forest,
KMeans, One-Class SVM) produce materially identical results at both precisions on typical
tabular anomaly detection tasks. If your use case requires `float64` precision, set
`features.dtype = "float64"` in config.

## Time derivatives — ordinal, not cyclical

`features.time_derivatives` (hour, dayofweek, day, month, year, quarter) are emitted as
plain integers, not a sin/cos cyclical encoding. This means the natural adjacency of, e.g.,
`hour=23` and `hour=0` is invisible to every detector: they are 23 apart in feature space,
not 1. For detectors sensitive to Euclidean/Mahalanobis distance (KMeans, OneClassSVM),
midnight-adjacent anomalies can be scored as more unusual than they are. This is a
deliberate scope decision, not an oversight: cyclical encoding doubles the derivative's
width (one feature becomes two: `sin`, `cos`), interacts with scaling and correlation
reduction (the two components are related, not independent, and would need to bypass
robust/standard scaling to stay a unit circle), and is a real new feature rather than a
fix to an invalid state. If cyclical adjacency matters for a given dataset, derive
`sin(2*pi*hour/24)`/`cos(2*pi*hour/24)`-style columns upstream before feeding data in.

## `auto` contamination — median of natural flag rates

When `scoring.contamination = "auto"`, the review budget is derived as the median
of each enabled detector's natural flag rate (the fraction of rows the detector's
own boundary flags as anomalous). This is a heuristic — it does not produce a
calibrated estimate of the true anomaly rate, and the per-detector rates it takes
the median of routinely disagree by 2–3×. The run summary and `sorethumb run --json`
report each detector's realised rate so the resulting flag count is read as a
shortlist size, not a measurement. Validation against labelled benchmark datasets
is ongoing; see the benchmark table in the README.

## Full-pipeline scenario benchmark — sklearn baselines only, no PyOD

`src/sorethumb/evaluate/pipeline_benchmark.py`'s `sklearn:*` comparison rows use
bare `sklearn.ensemble.IsolationForest` / `sklearn.neighbors.LocalOutlierFactor` /
`sklearn.svm.OneClassSVM` (already a sorethumb dependency), not PyOD. This is a
deliberate scope decision, not an oversight: sorethumb's `ecod`/`hbos`/`lof`
detectors are already independent sklearn-based reimplementations, not PyOD
wrappers, and PyOD would add real dependency weight (numba/torch-adjacent
transitive dependencies, similar to the friction already documented for SHAP)
purely for a comparison baseline. If PyOD-specific comparisons become valuable
later, add it as its own optional extra rather than a core/benchmark dependency.

## Full-pipeline scenario benchmark — `intersection` combination is not robust to one bad member

`tests/benchmark/test_pipeline_accuracy_floors.py` measured that the pipeline's
*shipped default* (`scoring.combination = "intersection"`, the default three-
detector ensemble) scores **worse than random** (ROC-AUC as low as ~0.04) on the
`local` and `varying_density` scenarios (see `evaluate/scenarios.py`). Root
cause: `intersection`'s combined score is `min()` across each detector's
calibrated score (`scoring/combine.py::ScoreEnsemble._combine`) — a principled
choice for the *flag* decision (only flag when every detector agrees), but it
means the combined *continuous ranking* is only as good as its single worst
member. On both scenarios, `OneClassSVM`'s RBF-kernel boundary badly misranks
points in a low-density gap between clusters or a mixed-density regime
(individually measured ROC-AUC as low as 0.002), and `min()` lets that one
detector's bad ranking dominate the ensemble's, even though `isolation_forest`
alone scores 0.90+ on the same data. `union` (`max()` across detectors) is
measured to be far more robust on these same two scenarios (ROC-AUC ~0.85) —
it takes the *best*-performing detector's opinion per row instead of the
worst's. This is real, measured behaviour of the shipped default on a
plausible data regime, not a synthetic-data artifact tuned to fail; it is
surfaced here rather than silently worked around, matching the project's
practice of documenting known limitations (see `lof`'s clustered-anomaly
weakness above) instead of asserting something known to be false.

## Full-pipeline scenario benchmark — `contextual` anomalies are near-chance for every combination mode

The `contextual` scenario (a categorical "context" column changes what
"normal" means for a numeric feature; see `evaluate/scenarios.py`) measures
consistently at ROC-AUC 0.45–0.49 — chance, not occasionally by seed luck —
under every combination mode tried (`intersection`, `composite`, `union`,
and the six-detector `all_detectors` ablation). None of sorethumb's current
detectors/ensemble strategies learn the conditional "normal depends on
context" structure from an ordinary categorical feature column (the context
is deliberately not modelled via `group_by`, which would let a group-aware
fit trivially solve it — see the scenario's own docstring for why). This is
a genuine, currently-unaddressed structural gap, not a benchmark bug;
`tests/benchmark/test_pipeline_accuracy_floors.py` checks only for a
catastrophic regression here (a real inversion), not for beating random,
since beating random is not something any current configuration achieves.
