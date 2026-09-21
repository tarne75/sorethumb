# Approximations and error characteristics

This file documents every step where `sorethumb` uses an approximation rather than an
exact computation, along with its practical error characteristics.

## Correlation reduction — sampled Pearson matrix

When the feature matrix exceeds 200,000 rows, the Pearson correlation matrix is computed
on a random subsample of 200,000 rows rather than the full dataset. Pearson correlation
coefficients stabilise quickly with sample size; at 200,000 rows the standard error of
the estimate is ~0.002, which is negligible relative to the 0.95 threshold used for
column dropping. The subsample is drawn with a fixed seed for reproducibility.

## Correlation pruning — can hide relational anomalies

`features/correlate.py::drop_correlated` groups columns into connected components
wherever |Pearson r| exceeds `features.correlation_threshold` (default 0.95, computed
on the sampled matrix above) and keeps only the first member of each component,
dropping the rest entirely — not just down-weighting them. This assumes redundant
columns carry no information beyond what the survivor already captures, which holds
for the *typical* row. It does not hold for a row where the correlation itself breaks:
two sensors that normally track together (temperature and pressure on the same
system, a metric and its trailing average) and decouple specifically on an anomalous
record are a classic case where the *relationship*, not either value alone, is the
anomaly signal. Once one of the pair is dropped, that decoupling is invisible to every
detector — the surviving column's value in isolation may look unremarkable. This is a
structural blind spot of correlation-based reduction, not a bug: raise
`correlation_threshold` (bounded to `[0.0, 1.0]`; the comparison is `|r| >= threshold`,
so `1.0` only ever catches a literal duplicate/linearly-dependent column and is the
practical way to disable pruning) if your data's anomalies are expected to show up as
broken relationships between otherwise-correlated columns.

## Silhouette score for KMeans `k` selection — sampled

The silhouette score is near-quadratic in the number of rows. When evaluating candidate
`k` values, silhouette is computed on a subsample capped at 20,000 rows. This subsample
is seeded and deterministic. The elbow criterion (computed on the full training data) is
the primary selection criterion; silhouette breaks ties. No accuracy guarantee is made for
the selected `k` when the true cluster structure requires >20,000 rows to distinguish.

## PCA — retained variance is not the same as retained anomaly signal

`features/reduce.py::fit_pca` (used only when `features.pca = true`, off by default)
selects components by cumulative *explained variance* and warns
(`LowVarianceWarning`) when that cumulative figure falls below
`pca_min_explained_variance` — but explained variance is a property of the *normal*
bulk of the data, and PCA is not told anything about which rows are anomalous. A
component can be dropped for explaining very little of the dataset's overall variance
while still being exactly the direction a specific anomaly deviates along — a rare
event is, by definition, rare, so it rarely contributes much to a variance ranking
computed across the whole sample. This means a run can clear the
`pca_min_explained_variance` threshold (say, 95%) with no warning at all, and still
have silently discarded the one dimension that would have flagged a particular
anomalous record. This is a known, general limitation of variance-based dimensionality
reduction for outlier detection, not specific to this implementation; if PCA is enabled,
treat `pca_min_explained_variance` as a floor on how much of the *typical* data is
preserved, not a guarantee about anomaly sensitivity. Leaving `features.pca = false`
(the default) avoids this risk entirely, at the cost of not compressing wide feature
spaces.

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

## Contaminated fitting — no clean reference set

Every detector's `.fit()` call (`_pipeline.py`, the per-group fitting loop) is given
the group's own feature matrix — the same data (or a subsample of it, see below) that
is then scored for anomalies. There is no held-out "known normal" reference set: this
is what makes the workflow unsupervised, but it also means each detector's notion of
"normal" is itself shaped by whatever anomalies happen to be present in that data.
Robust scaling (`features/scale.py`) is deliberately designed to resist this — median
and IQR barely move when a small fraction of rows are extreme — but the *detectors*
themselves have no equivalent protection built in here: IsolationForest, KMeans, and
OneClassSVM all fit their notion of the data's shape on the contaminated input as
supplied. In practice this is self-limiting for a low anomaly fraction (the reason
outlier detection works at all), but it degrades gracefully rather than being
guaranteed robust — a large enough or sufficiently clustered contaminated subset can
shift a detector's boundary enough to under-flag exactly the anomalies it is meant to
catch. This is a structural property of unsupervised anomaly detection in general, not
a defect specific to `sorethumb`.

## Capped detectors train on a subsample, score the full group

`_pipeline.py`'s per-group fitting loop (`train_row_cap` / `default_train_row_cap`,
e.g. 25,000 rows for OneClassSVM, 50,000 for LOF) draws a seeded random subsample of
at most the cap and fits *only* on that subsample when the group exceeds it — but
`score_samples` is always called against the *full* group afterwards. This keeps
runtime bounded for detectors whose training cost grows faster than linearly, and the
random sample is representative of the bulk distribution by construction — but two
consequences follow directly from training and scoring on different row sets: (1) a
rare anomalous pattern that exists in the full group may be entirely absent from the
capped training sample purely by chance, so the fitted model never "saw" that
region of feature space when it formed its boundary; (2) for data with meaningful
temporal or other structure (not i.i.d. across rows), a uniform random subsample may
not represent the tail of that structure as well as the full data would. Every row is
still scored — this does not silently drop rows from results — but a detector's
sensitivity to a specific rare pattern is not guaranteed to be as strong as if it had
been fit on the complete group. Raising the relevant detector's `train_row_cap` in
config (or setting it above the group's row count) removes the asymmetry entirely, at
the runtime cost the cap exists to bound in the first place — see the
[Scale guide](../README.md#scale-guide).

## KernelSHAP — Monte Carlo approximation

When `explain.kernel_shap = true`, attributions for OneClassSVM/LOF are computed via
`shap.KernelExplainer` instead of finite-difference gradients (ECOD and HBOS always use
their exact decomposition above, regardless of this setting). This uses Monte Carlo
sampling of the feature space to estimate Shapley values. The result is labelled
`heuristic`, not `model_specific` — it never touches the detector's internal structure,
unlike TreeSHAP. Accuracy increases with `nsamples` but so does runtime. The default
`nsamples` is documented in `explain/`.

## Zero-inflated columns — robust scaling can leave them effectively unscaled

`features/scale.py::fit_scaler`'s robust mode (the default) centres on the median and
scales by the IQR (`q75 - q25`). For a zero-inflated column — a large majority of rows
at exactly 0, with a real, information-carrying minority of nonzero values (fees,
error counts, discount amounts, and similar "mostly absent" measures are typical) — a
sufficiently zero-heavy column has median = q25 = q75 = 0, so the IQR is exactly zero.
`_spread_or_unit` (the degenerate-column guard, `scale.py`'s `_ZERO_SPREAD` threshold)
then sets `scale = 1.0` rather than dividing by zero — correct for a genuinely constant
column, but for a zero-inflated one this leaves its real, information-carrying nonzero
values effectively unscaled (divided by `1.0`) while every other column has been scaled
to a comparable range. For distance-based detectors (KMeans, OneClassSVM), a column
left at its raw natural magnitude this way can dominate — or, if its nonzero values are
small in absolute terms, be drowned out by — the Euclidean/RBF distance calculation
regardless of how informative it actually is. Standard-mode scaling (`features.scaler =
"standard"`) has the same failure shape (trimmed mean/std over the central 98%, which
is also 0/near-0 for a sufficiently zero-heavy column). There is no automatic
zero-inflation-aware scaling mode; if a known zero-inflated column matters for anomaly
detection, consider deriving an explicit "is nonzero" indicator column alongside it
upstream, before feeding data in.

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
surfaced here rather than silently worked around, matching the rest of this
file's practice of documenting known limitations instead of asserting
something known to be false.

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
