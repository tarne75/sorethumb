# Benchmark results

This page holds the full benchmark matrices. The [README](../README.md#benchmark-results-summary)
carries a short summary of the same run.

One maintainer script regenerates both tables below and, when you point it at
this file, injects them (from a clone, see [Installation](../README.md#installation)).
It is a repo script, not part of the installed `sorethumb` command, and it only
writes where you tell it to:

```bash
pip install ".[benchmark]"
python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md
```

`scripts/run_benchmark.py` runs two independent suites (each can be disabled
with `--no-pipeline` / `--no-legacy`) and writes `*.md` / `*.csv` into the
required `--output-dir`; the document passed to `--readme` is the only file edited, and only if the flag is given.
Average precision (AP) is the headline metric
in both — it accounts for class imbalance in a way ROC-AUC does not.

## Full-pipeline scenario benchmark

`src/sorethumb_ml/evaluate/pipeline_benchmark.py` fits a named taxonomy of
synthetic anomaly types — point, local, contextual, clustered, masking,
swamping, varying-density (see `src/sorethumb_ml/evaluate/scenarios.py`) — with
mixed numeric **and categorical** columns through the real feature pipeline
(encoding, scaling, optional PCA, detectors, calibration, ensemble), never
training and evaluating on the same rows. Every (scenario, ablation) cell
runs over several seeds and reports mean ± a 95% confidence interval, not a
bare mean. `peak_memory_mb` is a real OS-tracked high-water mark from an
isolated subprocess per cell, not a before/after snapshot. `sklearn:*` rows
are the same scenario fit with bare sklearn detectors and no sorethumb
pipeline at all, for comparison (PyOD is deliberately not used here — see
[`approximations.md`](approximations.md)).

**Matrix shape**: `scripts/run_benchmark.py` runs every scenario under the shipped
*default* ablation only (plus its `sklearn:*` baselines) — 12 pipeline
ablations exist in total (`AblationSpec`s: PCA on, standard scaler, each
detector alone, all six detectors, each combination policy, and
`near_constant_2`). The comparison across ensemble policies is published
separately, in the [ensemble policy matrix](#ensemble-policy-matrix) below; run
`sorethumb_ml.evaluate.pipeline_benchmark.run_pipeline_benchmark` directly with
other `ablation_names` for the rest. `flag_precision`/`flag_recall`/
`flag_f1`/`flag_false_positive_rate`/`flag_count` describe the ensemble's
*actual* `anomaly_flag` decision — distinct from `roc_auc`/`average_precision`/
`precision_at_k`/`recall_at_k`, which describe the continuous ranking (or a
hypothetical top-k cut on it) and can disagree with the real flag
substantially, especially for `intersection`'s naturally conservative
unanimous-agreement rule (see [`approximations.md`](approximations.md)). `swamping_clean`/
`swamping_contaminated` are a matched pair (identical normal training
population, differing only by injected contamination) scored against the
same genuinely-normal "at-risk" holdout — a single-class population by
design, so `roc_auc`/`average_precision` are `n/a` there; `flag_false_positive_rate`
is the metric that actually demonstrates the mechanism (see
[`approximations.md`](approximations.md) for what it shows).

`tests/benchmark/test_pipeline_accuracy_floors.py` locks in a ROC-AUC floor
per guarded (scenario, ablation) cell — this is what makes the harness
capable of rejecting a default configuration: a real accuracy regression
fails that suite. It runs on every pull request (the required
`pipeline-accuracy-floors` check), not only on a schedule. The same check guards the
one alternative policy with published backtest evidence, `composite-agreement@1`, on
what that evidence rests on: its flag F1 on the hypothesis scenarios, and never more
wrong flags than the review budget. It also documents where the *shipped default* combination
mode (`intersection`) genuinely underperforms (`local`, `varying_density` —
see [`approximations.md`](approximations.md)) rather than asserting something the data shows
is false. `scripts/run_benchmark.py` itself refuses to publish a
matrix with a missing or errored cell (`assert_complete_and_error_free`).

<!-- pipeline-benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md` to regenerate. -->

| scenario | kind | ablation | policy | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | flag_precision | flag_recall | flag_f1 | flag_false_positive_rate | flag_count | over_flag_ratio | false_positive_budget_ratio | flag_jaccard_stability | rank_stability | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | intersection@1 | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | 0.8889 | 0.1178 | 0.1973 | 0.0014 | 1.7 | 0.13 | 0.03 | n/a | n/a | 0.738 | 0.010 | 282.9 |
| point | point | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.193 | 0.007 | 249.4 |
| point | point | sklearn:lof | n/a | 3 | 588 | 252 | 0.3239 ± 0.0298 | 0.0377 ± 0.0100 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.076 | 0.002 | 237.6 |
| point | point | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.9251 ± 0.0304 | 0.3431 ± 0.0486 | 0.2308 | 0.2504 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.051 | 0.000 | 234.4 |
| local | local | default | intersection@1 | 3 | 588 | 252 | 0.7085 ± 0.0335 | 0.1075 ± 0.0460 | 0.1026 | 0.0958 | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 3.0 | 0.24 | 0.23 | n/a | n/a | 0.720 | 0.010 | 279.8 |
| local | local | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.9024 ± 0.0583 | 0.4356 ± 0.1883 | 0.3846 | 0.3973 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.196 | 0.008 | 250.0 |
| local | local | sklearn:lof | n/a | 3 | 588 | 252 | 0.8213 ± 0.0719 | 0.6270 ± 0.1236 | 0.5641 | 0.5777 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.073 | 0.001 | 236.6 |
| local | local | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.0073 ± 0.0109 | 0.0269 ± 0.0074 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.049 | 0.000 | 234.1 |
| contextual | contextual | default | intersection@1 | 3 | 840 | 360 | 0.4965 ± 0.0842 | 0.0579 ± 0.0091 | 0.0370 | 0.0417 | 0.0333 | 0.0208 | 0.0256 | 0.0204 | 7.3 | 0.41 | 0.39 | n/a | n/a | 0.797 | 0.011 | 305.0 |
| contextual | contextual | sklearn:isolation_forest | n/a | 3 | 840 | 360 | 0.7495 ± 0.0652 | 0.1524 ± 0.0577 | 0.1667 | 0.1838 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.190 | 0.008 | 251.6 |
| contextual | contextual | sklearn:lof | n/a | 3 | 840 | 360 | 0.5295 ± 0.0950 | 0.1012 ± 0.0607 | 0.1111 | 0.1225 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.077 | 0.003 | 238.2 |
| contextual | contextual | sklearn:one_class_svm | n/a | 3 | 840 | 360 | 0.6000 ± 0.0267 | 0.0901 ± 0.0262 | 0.0926 | 0.1017 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.051 | 0.001 | 234.9 |
| clustered | clustered | default | intersection@1 | 3 | 588 | 252 | 1.0000 ± 0.0001 | 0.9988 ± 0.0024 | 0.8718 | 0.9375 | 0.4222 | 0.1606 | 0.2286 | 0.0041 | 2.7 | 0.21 | 0.08 | n/a | n/a | 0.742 | 0.010 | 275.0 |
| clustered | clustered | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.191 | 0.007 | 251.5 |
| clustered | clustered | sklearn:lof | n/a | 3 | 588 | 252 | 0.3503 ± 0.1922 | 0.0418 ± 0.0157 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.072 | 0.002 | 237.1 |
| clustered | clustered | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.9174 ± 0.0337 | 0.2915 ± 0.0327 | 0.2564 | 0.2867 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.051 | 0.000 | 234.8 |
| masking | masking | default | intersection@1 | 3 | 595 | 255 | 0.8813 ± 0.0328 | 0.4686 ± 0.1070 | 0.4825 | 0.4236 | 0.3185 | 0.0394 | 0.0684 | 0.0187 | 5.7 | 0.15 | 0.11 | n/a | n/a | 0.721 | 0.010 | 285.8 |
| masking | masking | sklearn:isolation_forest | n/a | 3 | 595 | 255 | 0.9749 ± 0.0208 | 0.8911 ± 0.0878 | 0.8684 | 0.7674 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.190 | 0.007 | 254.6 |
| masking | masking | sklearn:lof | n/a | 3 | 595 | 255 | 0.6479 ± 0.0817 | 0.3159 ± 0.0725 | 0.4123 | 0.3723 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.071 | 0.002 | 238.4 |
| masking | masking | sklearn:one_class_svm | n/a | 3 | 595 | 255 | 0.8448 ± 0.0466 | 0.4534 ± 0.1458 | 0.4123 | 0.3603 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.048 | 0.001 | 234.7 |
| varying_density | varying_density | default | intersection@1 | 3 | 728 | 312 | 0.6824 ± 0.0341 | 0.0844 ± 0.0552 | 0.0208 | 0.0256 | 0.0000 | 0.0000 | 0.0000 | 0.0155 | 4.7 | 0.30 | 0.29 | n/a | n/a | 0.716 | 0.011 | 291.5 |
| varying_density | varying_density | sklearn:isolation_forest | n/a | 3 | 728 | 312 | 0.9386 ± 0.0155 | 0.3351 ± 0.1728 | 0.2292 | 0.3158 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.194 | 0.008 | 255.3 |
| varying_density | varying_density | sklearn:lof | n/a | 3 | 728 | 312 | 0.8671 ± 0.0970 | 0.4416 ± 0.4803 | 0.4375 | 0.5966 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.074 | 0.002 | 237.9 |
| varying_density | varying_density | sklearn:one_class_svm | n/a | 3 | 728 | 312 | 0.1297 ± 0.0348 | 0.0233 ± 0.0037 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.050 | 0.000 | 233.6 |
| categorical_rare | categorical_rare | default | intersection@1 | 3 | 577 | 248 | 0.4649 ± 0.0920 | 0.0381 ± 0.0168 | 0.0278 | 0.0333 | 0.0000 | 0.0000 | 0.0000 | 0.0139 | 3.3 | 0.27 | 0.28 | n/a | n/a | 0.738 | 0.010 | 276.5 |
| categorical_rare | categorical_rare | sklearn:isolation_forest | n/a | 3 | 577 | 248 | 0.9887 ± 0.0141 | 0.7668 ± 0.1961 | 0.5833 | 0.8333 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.190 | 0.007 | 251.3 |
| categorical_rare | categorical_rare | sklearn:lof | n/a | 3 | 577 | 248 | 0.8741 ± 0.1819 | 0.3491 ± 0.3109 | 0.2222 | 0.2667 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.072 | 0.002 | 237.2 |
| categorical_rare | categorical_rare | sklearn:one_class_svm | n/a | 3 | 577 | 248 | 0.8983 ± 0.0371 | 0.3003 ± 0.0962 | 0.2778 | 0.4000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.050 | 0.001 | 233.7 |
| text_timestamp | text_timestamp | default | intersection@1 | 3 | 581 | 249 | 0.9110 ± 0.0035 | 0.3268 ± 0.0626 | 0.3333 | 0.4048 | 0.5556 | 0.2262 | 0.3185 | 0.0070 | 3.7 | 0.29 | 0.14 | n/a | n/a | 0.740 | 0.011 | 284.6 |
| text_timestamp | text_timestamp | sklearn:isolation_forest | n/a | 3 | 581 | 249 | 0.4841 ± 0.1170 | 0.0406 ± 0.0103 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.188 | 0.006 | 278.8 |
| text_timestamp | text_timestamp | sklearn:lof | n/a | 3 | 581 | 249 | 0.4520 ± 0.1401 | 0.0408 ± 0.0175 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.086 | 0.003 | 294.6 |
| text_timestamp | text_timestamp | sklearn:one_class_svm | n/a | 3 | 581 | 249 | 0.4096 ± 0.1207 | 0.0338 ± 0.0095 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.090 | 0.034 | 276.4 |
| irrelevant_columns | irrelevant_columns | default | intersection@1 | 3 | 588 | 252 | 0.9557 ± 0.0132 | 0.6758 ± 0.0770 | 0.6667 | 0.7174 | 0.6889 | 0.2920 | 0.4045 | 0.0069 | 5.3 | 0.42 | 0.13 | n/a | n/a | 0.739 | 0.011 | 278.5 |
| irrelevant_columns | irrelevant_columns | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.8160 ± 0.0556 | 0.2932 ± 0.0875 | 0.1795 | 0.1992 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.190 | 0.007 | 251.5 |
| irrelevant_columns | irrelevant_columns | sklearn:lof | n/a | 3 | 588 | 252 | 0.9629 ± 0.0041 | 0.6200 ± 0.0157 | 0.5128 | 0.5485 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.081 | 0.001 | 246.1 |
| irrelevant_columns | irrelevant_columns | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.8826 ± 0.0260 | 0.3559 ± 0.0377 | 0.3590 | 0.3890 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.051 | 0.001 | 235.8 |
| all_categorical | all_categorical | default | intersection@1 | 3 | 588 | 252 | 0.8740 ± 0.0491 | 0.1872 ± 0.0569 | 0.2051 | 0.2136 | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 4.0 | 0.32 | 0.31 | n/a | n/a | 0.715 | 0.011 | 279.0 |
| all_categorical | all_categorical | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.8568 ± 0.0546 | 0.1566 ± 0.0411 | 0.0256 | 0.0333 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.191 | 0.007 | 247.9 |
| all_categorical | all_categorical | sklearn:lof | n/a | 3 | 588 | 252 | 0.5097 ± 0.0924 | 0.0527 ± 0.0261 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.081 | 0.001 | 243.5 |
| all_categorical | all_categorical | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.6236 ± 0.0202 | 0.0656 ± 0.0184 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.050 | 0.001 | 233.6 |
| swamping_clean | swamping | default | intersection@1 | 3 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0111 | 1.7 | 0.22 | 0.21 | n/a | n/a | 0.770 | 0.009 | 297.9 |
| swamping_contaminated | swamping | default | intersection@1 | 3 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.7 | 0.09 | 0.08 | n/a | n/a | 0.775 | 0.009 | 306.6 |
<!-- pipeline-benchmark-results-end -->

## Ensemble policy matrix

Every scenario above, plus four that exercise recent input handling --
`categorical_rare` (the anomaly is a rare level of a 3-value category),
`text_timestamp` (unusual-hour activity in a text timestamp column),
`irrelevant_columns` (two signal columns among twenty noise columns) and
`all_categorical` (no numeric column; anomalies are unseen combinations of
individually common levels) -- under
every registered [ensemble policy](ensemble-policy.md), each identified by its
versioned id. The default policy, `intersection@1`, is a deliberate design
decision; this matrix is the evidence any other choice should be made on.

Per cell, over every seed (mean, 95% CI where shown):

- **Precision and recall** of the actual flag decision (`flag_precision`,
  `flag_recall`), and at the fixed review budget (`precision_at_k`, `recall_at_k`).
- **Ranking**: `average_precision` (headline) and `roc_auc`.
- **Stability**: the pipeline is refitted on random 80% subsamples of the
  training rows; `flag_jaccard_stability` is how much of the flagged set the
  refits keep, `rank_stability` the Spearman correlation of their scores.
- **False-positive budget**: `flag_false_positive_rate` on normal rows;
  `over_flag_ratio`, the flagged count divided by the review budget (1.0 flags
  exactly the budget); and `false_positive_budget_ratio`, the wrongly flagged
  rows divided by the review budget in whole rows (at most 1.0: a policy may
  spend its review budget on wrong rows, never more). The budget is checked on
  the last of these; a rate over normal rows alone could be breached by a
  policy flagging exactly its budget.

Each cell is checked against explicit per-anomaly-type budgets in
[`tests/benchmark/budgets.toml`](../tests/benchmark/budgets.toml), set from
stated principles and versioned. Where the default policy misses a budget, the
miss is listed there under `accepted_limitations` with its reason; it is a
documented trade-off, not a hidden one, and `tests/benchmark/test_policy_budgets.py`
fails if the default misses a budget that is not accepted, or meets one that
is still listed as accepted. The `near_constant_2` row is the evidence for the
`profiling.near_constant_distinct` default (see the conclusion below the table).

<!-- policy-matrix-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --no-legacy --no-pipeline --policy-matrix --readme docs/benchmarks.md` to regenerate. -->

Suite version `8068b9d70a987a63`, budgets v3, 10 seeds, 3 stability refits per seed, sorethumb 0.1.0 at `6d6bfd6854b5`. Full results with provenance: [`benchmark-results/policy-matrix.json`](benchmark-results/policy-matrix.json).

| scenario | kind | ablation | policy | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | flag_precision | flag_recall | flag_f1 | flag_false_positive_rate | flag_count | over_flag_ratio | false_positive_budget_ratio | flag_jaccard_stability | rank_stability | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0001 | 0.9992 ± 0.0016 | 0.8000 | 0.9679 | 0.6033 | 0.1278 | 0.1903 | 0.0037 | 2.2 | 0.17 | 0.07 | 0.563 | 0.980 | 0.449 | 0.012 | 271.9 |
| point | point | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9883 ± 0.0038 | 0.8172 ± 0.0552 | 0.6385 | 0.7858 | 0.6385 | 0.7858 | 0.6966 | 0.0195 | 13.0 | 1.03 | 0.36 | 0.796 | 0.988 | 0.382 | 0.011 | 297.5 |
| point | point | combination_union | union@1 | 10 | 588 | 252 | 0.9727 ± 0.0050 | 0.5289 ± 0.0673 | 0.4769 | 0.5911 | 0.4467 | 0.9938 | 0.6130 | 0.0547 | 24.0 | 1.90 | 1.02 | 0.877 | 0.982 | 0.373 | 0.010 | 298.4 |
| point | point | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9870 ± 0.0040 | 0.8027 ± 0.0580 | 0.6308 | 0.7758 | 0.6308 | 0.7758 | 0.6880 | 0.0199 | 13.0 | 1.03 | 0.37 | 0.791 | 0.988 | 0.375 | 0.011 | 305.2 |
| point | point | near_constant_2 | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0001 | 0.9983 ± 0.0023 | 0.8000 | 0.9679 | 0.5917 | 0.1249 | 0.1962 | 0.0025 | 1.9 | 0.15 | 0.05 | 0.626 | 0.980 | 0.377 | 0.011 | 309.8 |
| local | local | default | intersection@1 | 10 | 588 | 252 | 0.6452 ± 0.0598 | 0.0859 ± 0.0193 | 0.0615 | 0.0665 | 0.0000 | 0.0000 | 0.0000 | 0.0145 | 3.5 | 0.28 | 0.27 | 0.877 | 0.979 | 0.351 | 0.011 | 317.4 |
| local | local | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.5390 ± 0.0207 | 0.0482 ± 0.0075 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0539 | 13.0 | 1.03 | 1.00 | 0.870 | 0.990 | 0.345 | 0.011 | 311.4 |
| local | local | combination_union | union@1 | 10 | 588 | 252 | 0.8481 ± 0.0213 | 0.1793 ± 0.0379 | 0.1538 | 0.1938 | 0.1439 | 0.3513 | 0.2008 | 0.0925 | 26.0 | 2.06 | 1.72 | 0.775 | 0.987 | 0.339 | 0.011 | 312.0 |
| local | local | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.6773 ± 0.0427 | 0.0731 ± 0.0182 | 0.0077 | 0.0111 | 0.0077 | 0.0111 | 0.0091 | 0.0535 | 13.0 | 1.03 | 0.99 | 0.783 | 0.989 | 0.336 | 0.011 | 316.2 |
| local | local | near_constant_2 | intersection@1 | 10 | 588 | 252 | 0.3111 ± 0.0485 | 0.0375 ± 0.0064 | 0.0154 | 0.0174 | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 2.6 | 0.21 | 0.20 | 0.414 | 0.964 | 0.341 | 0.011 | 310.2 |
| contextual | contextual | default | intersection@1 | 10 | 840 | 360 | 0.4722 ± 0.0339 | 0.0623 ± 0.0182 | 0.0389 | 0.0462 | 0.0425 | 0.0223 | 0.0290 | 0.0201 | 7.2 | 0.40 | 0.38 | 0.594 | 0.952 | 0.419 | 0.012 | 349.5 |
| contextual | contextual | combination_composite | composite-equal@1 | 10 | 840 | 360 | 0.4695 ± 0.0317 | 0.0540 ± 0.0072 | 0.0389 | 0.0434 | 0.0389 | 0.0434 | 0.0406 | 0.0504 | 18.0 | 1.00 | 0.96 | 0.748 | 0.969 | 0.415 | 0.012 | 347.7 |
| contextual | contextual | combination_union | union@1 | 10 | 840 | 360 | 0.4770 ± 0.0334 | 0.0649 ± 0.0159 | 0.0444 | 0.0539 | 0.0444 | 0.0920 | 0.0590 | 0.0860 | 30.9 | 1.72 | 1.64 | 0.778 | 0.946 | 0.420 | 0.012 | 354.1 |
| contextual | contextual | combination_composite_agreement | composite-agreement@1 | 10 | 840 | 360 | 0.4694 ± 0.0318 | 0.0540 ± 0.0073 | 0.0389 | 0.0434 | 0.0389 | 0.0434 | 0.0406 | 0.0504 | 18.0 | 1.00 | 0.96 | 0.748 | 0.970 | 0.425 | 0.013 | 344.7 |
| contextual | contextual | near_constant_2 | intersection@1 | 10 | 840 | 360 | 0.5999 ± 0.0296 | 0.0862 ± 0.0206 | 0.0611 | 0.0716 | 0.0762 | 0.0383 | 0.0508 | 0.0160 | 6.0 | 0.33 | 0.31 | 0.702 | 0.986 | 0.418 | 0.013 | 347.3 |
| clustered | clustered | default | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0000 | 0.9996 ± 0.0007 | 0.8000 | 0.9679 | 0.6017 | 0.1597 | 0.2432 | 0.0029 | 2.3 | 0.18 | 0.05 | 0.645 | 0.981 | 0.405 | 0.013 | 300.6 |
| clustered | clustered | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9854 ± 0.0052 | 0.7774 ± 0.0682 | 0.6000 | 0.7270 | 0.6000 | 0.7270 | 0.6505 | 0.0215 | 13.0 | 1.03 | 0.40 | 0.800 | 0.988 | 0.412 | 0.014 | 304.5 |
| clustered | clustered | combination_union | union@1 | 10 | 588 | 252 | 0.9702 ± 0.0068 | 0.4915 ± 0.0727 | 0.4154 | 0.5265 | 0.4569 | 0.9933 | 0.6228 | 0.0527 | 23.5 | 1.87 | 0.98 | 0.877 | 0.982 | 0.410 | 0.014 | 302.5 |
| clustered | clustered | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9844 ± 0.0054 | 0.7654 ± 0.0680 | 0.5923 | 0.7170 | 0.5923 | 0.7170 | 0.6418 | 0.0219 | 13.0 | 1.03 | 0.41 | 0.791 | 0.988 | 0.409 | 0.013 | 302.5 |
| clustered | clustered | near_constant_2 | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0001 | 0.9993 ± 0.0014 | 0.8000 | 0.9679 | 0.6917 | 0.1564 | 0.2457 | 0.0016 | 2.0 | 0.16 | 0.03 | 0.537 | 0.975 | 0.401 | 0.014 | 311.0 |
| masking | masking | default | intersection@1 | 10 | 595 | 255 | 0.8548 ± 0.0261 | 0.4355 ± 0.0606 | 0.4447 | 0.3877 | 0.2765 | 0.0243 | 0.0435 | 0.0145 | 4.2 | 0.11 | 0.08 | 0.655 | 0.992 | 0.403 | 0.014 | 300.1 |
| masking | masking | combination_composite | composite-equal@1 | 10 | 595 | 255 | 0.7911 ± 0.0263 | 0.3563 ± 0.0391 | 0.3395 | 0.2942 | 0.3395 | 0.2942 | 0.3136 | 0.1190 | 38.0 | 0.99 | 0.66 | 0.857 | 0.994 | 0.391 | 0.013 | 301.3 |
| masking | masking | combination_union | union@1 | 10 | 595 | 255 | 0.8921 ± 0.0205 | 0.5447 ± 0.0689 | 0.5421 | 0.4663 | 0.4721 | 0.7744 | 0.5846 | 0.1794 | 71.7 | 1.87 | 0.99 | 0.909 | 0.994 | 0.385 | 0.013 | 306.2 |
| masking | masking | combination_composite_agreement | composite-agreement@1 | 10 | 595 | 255 | 0.8109 ± 0.0262 | 0.3843 ± 0.0445 | 0.3632 | 0.3139 | 0.3632 | 0.3139 | 0.3350 | 0.1147 | 38.0 | 0.99 | 0.64 | 0.859 | 0.994 | 0.383 | 0.013 | 301.8 |
| masking | masking | near_constant_2 | intersection@1 | 10 | 595 | 255 | 0.8490 ± 0.0236 | 0.4252 ± 0.0622 | 0.4211 | 0.3609 | 0.3552 | 0.0355 | 0.0640 | 0.0146 | 4.7 | 0.12 | 0.08 | 0.627 | 0.984 | 0.385 | 0.013 | 303.7 |
| varying_density | varying_density | default | intersection@1 | 10 | 728 | 312 | 0.6507 ± 0.0215 | 0.0627 ± 0.0197 | 0.0125 | 0.0154 | 0.0000 | 0.0000 | 0.0000 | 0.0176 | 5.3 | 0.34 | 0.33 | 0.704 | 0.985 | 0.382 | 0.014 | 330.0 |
| varying_density | varying_density | combination_composite | composite-equal@1 | 10 | 728 | 312 | 0.6207 ± 0.0191 | 0.0488 ± 0.0070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0532 | 16.0 | 1.03 | 1.00 | 0.818 | 0.991 | 0.384 | 0.014 | 336.0 |
| varying_density | varying_density | combination_union | union@1 | 10 | 728 | 312 | 0.8341 ± 0.0264 | 0.1357 ± 0.0312 | 0.0875 | 0.1267 | 0.1111 | 0.3209 | 0.1623 | 0.0937 | 31.8 | 2.04 | 1.76 | 0.803 | 0.984 | 0.381 | 0.014 | 337.9 |
| varying_density | varying_density | combination_composite_agreement | composite-agreement@1 | 10 | 728 | 312 | 0.7619 ± 0.0319 | 0.0806 ± 0.0180 | 0.0125 | 0.0154 | 0.0125 | 0.0154 | 0.0138 | 0.0525 | 16.0 | 1.03 | 0.99 | 0.770 | 0.991 | 0.379 | 0.014 | 333.2 |
| varying_density | varying_density | near_constant_2 | intersection@1 | 10 | 728 | 312 | 0.3667 ± 0.0405 | 0.0319 ± 0.0051 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0163 | 4.9 | 0.31 | 0.31 | 0.641 | 0.953 | 0.389 | 0.014 | 333.9 |
| categorical_rare | categorical_rare | default | intersection@1 | 10 | 577 | 248 | 0.5058 ± 0.0478 | 0.0501 ± 0.0143 | 0.0500 | 0.0768 | 0.0250 | 0.0200 | 0.0222 | 0.0166 | 4.1 | 0.33 | 0.33 | 0.645 | 0.959 | 0.401 | 0.012 | 308.7 |
| categorical_rare | categorical_rare | combination_composite | composite-equal@1 | 10 | 577 | 248 | 0.4967 ± 0.0452 | 0.0636 ± 0.0358 | 0.0333 | 0.0525 | 0.0333 | 0.0525 | 0.0399 | 0.0482 | 12.0 | 0.97 | 0.97 | 0.718 | 0.969 | 0.408 | 0.013 | 305.0 |
| categorical_rare | categorical_rare | combination_union | union@1 | 10 | 577 | 248 | 0.4912 ± 0.0543 | 0.0479 ± 0.0137 | 0.0417 | 0.0650 | 0.0427 | 0.1286 | 0.0630 | 0.0856 | 21.5 | 1.73 | 1.72 | 0.729 | 0.942 | 0.401 | 0.013 | 304.0 |
| categorical_rare | categorical_rare | combination_composite_agreement | composite-agreement@1 | 10 | 577 | 248 | 0.4968 ± 0.0456 | 0.0636 ± 0.0358 | 0.0333 | 0.0525 | 0.0333 | 0.0525 | 0.0399 | 0.0482 | 12.0 | 0.97 | 0.97 | 0.723 | 0.970 | 0.409 | 0.013 | 306.2 |
| categorical_rare | categorical_rare | near_constant_2 | intersection@1 | 10 | 577 | 248 | 0.8789 ± 0.0154 | 0.2001 ± 0.0650 | 0.1583 | 0.2646 | 0.4167 | 0.1368 | 0.1947 | 0.0037 | 1.9 | 0.15 | 0.07 | 0.572 | 0.945 | 0.409 | 0.014 | 302.9 |
| text_timestamp | text_timestamp | default | intersection@1 | 10 | 581 | 249 | 0.9392 ± 0.0171 | 0.3759 ± 0.0908 | 0.3250 | 0.5678 | 0.5317 | 0.2998 | 0.3669 | 0.0074 | 3.8 | 0.31 | 0.15 | 0.574 | 0.956 | 0.417 | 0.013 | 306.3 |
| text_timestamp | text_timestamp | combination_composite | composite-equal@1 | 10 | 581 | 249 | 0.9476 ± 0.0148 | 0.4670 ± 0.0765 | 0.3417 | 0.5995 | 0.3417 | 0.5995 | 0.4165 | 0.0326 | 12.0 | 0.96 | 0.66 | 0.719 | 0.968 | 0.427 | 0.013 | 309.0 |
| text_timestamp | text_timestamp | combination_union | union@1 | 10 | 581 | 249 | 0.9407 ± 0.0143 | 0.3252 ± 0.0677 | 0.2583 | 0.4438 | 0.2600 | 0.8025 | 0.3789 | 0.0677 | 22.1 | 1.78 | 1.37 | 0.712 | 0.945 | 0.420 | 0.014 | 302.8 |
| text_timestamp | text_timestamp | combination_composite_agreement | composite-agreement@1 | 10 | 581 | 249 | 0.9468 ± 0.0151 | 0.4645 ± 0.0748 | 0.3417 | 0.5995 | 0.3417 | 0.5995 | 0.4165 | 0.0326 | 12.0 | 0.96 | 0.66 | 0.725 | 0.969 | 0.437 | 0.013 | 308.9 |
| text_timestamp | text_timestamp | near_constant_2 | intersection@1 | 10 | 581 | 249 | 0.9289 ± 0.0262 | 0.4025 ± 0.1145 | 0.2917 | 0.5121 | 0.5233 | 0.2619 | 0.3230 | 0.0062 | 3.1 | 0.25 | 0.12 | 0.598 | 0.943 | 0.426 | 0.014 | 307.0 |
| irrelevant_columns | irrelevant_columns | default | intersection@1 | 10 | 588 | 252 | 0.9728 ± 0.0108 | 0.7219 ± 0.0366 | 0.6154 | 0.7514 | 0.7079 | 0.3882 | 0.4861 | 0.0070 | 5.8 | 0.46 | 0.13 | 0.646 | 0.979 | 0.419 | 0.014 | 296.2 |
| irrelevant_columns | irrelevant_columns | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9662 ± 0.0104 | 0.6989 ± 0.0575 | 0.5385 | 0.6595 | 0.5385 | 0.6595 | 0.5854 | 0.0249 | 13.0 | 1.03 | 0.46 | 0.720 | 0.984 | 0.423 | 0.015 | 298.8 |
| irrelevant_columns | irrelevant_columns | combination_union | union@1 | 10 | 588 | 252 | 0.9606 ± 0.0105 | 0.4647 ± 0.0743 | 0.4538 | 0.5562 | 0.4671 | 0.9617 | 0.6224 | 0.0489 | 22.1 | 1.75 | 0.91 | 0.744 | 0.967 | 0.423 | 0.014 | 301.8 |
| irrelevant_columns | irrelevant_columns | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9674 ± 0.0101 | 0.7061 ± 0.0578 | 0.5385 | 0.6595 | 0.5385 | 0.6595 | 0.5854 | 0.0249 | 13.0 | 1.03 | 0.46 | 0.729 | 0.985 | 0.424 | 0.015 | 293.0 |
| irrelevant_columns | irrelevant_columns | near_constant_2 | intersection@1 | 10 | 588 | 252 | 0.9711 ± 0.0123 | 0.6956 ± 0.0560 | 0.6231 | 0.7649 | 0.6958 | 0.3534 | 0.4468 | 0.0074 | 5.5 | 0.44 | 0.14 | 0.691 | 0.979 | 0.424 | 0.015 | 304.1 |
| all_categorical | all_categorical | default | intersection@1 | 10 | 588 | 252 | 0.8626 ± 0.0150 | 0.1656 ± 0.0265 | 0.1538 | 0.1845 | 0.0000 | 0.0000 | 0.0000 | 0.0220 | 5.3 | 0.42 | 0.41 | 0.742 | 0.955 | 0.372 | 0.014 | 297.6 |
| all_categorical | all_categorical | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.8514 ± 0.0141 | 0.1493 ± 0.0181 | 0.1077 | 0.1367 | 0.1077 | 0.1367 | 0.1194 | 0.0481 | 13.0 | 1.03 | 0.89 | 0.717 | 0.958 | 0.375 | 0.013 | 286.0 |
| all_categorical | all_categorical | combination_union | union@1 | 10 | 588 | 252 | 0.8401 ± 0.0155 | 0.1334 ± 0.0159 | 0.0308 | 0.0422 | 0.1699 | 0.3530 | 0.2278 | 0.0768 | 22.3 | 1.77 | 1.42 | 0.820 | 0.959 | 0.398 | 0.014 | 295.7 |
| all_categorical | all_categorical | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.8513 ± 0.0148 | 0.1499 ± 0.0188 | 0.1231 | 0.1558 | 0.1231 | 0.1558 | 0.1364 | 0.0473 | 13.0 | 1.03 | 0.88 | 0.724 | 0.961 | 0.384 | 0.014 | 295.7 |
| all_categorical | all_categorical | near_constant_2 | intersection@1 | 10 | 588 | 252 | 0.8626 ± 0.0150 | 0.1656 ± 0.0265 | 0.1538 | 0.1845 | 0.0000 | 0.0000 | 0.0000 | 0.0220 | 5.3 | 0.42 | 0.41 | 0.742 | 0.955 | 0.380 | 0.014 | 294.4 |
| swamping_clean | swamping | default | intersection@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0107 | 1.6 | 0.21 | 0.20 | 0.361 | 0.906 | 0.464 | 0.012 | 340.2 |
| swamping_clean | swamping | combination_composite | composite-equal@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.672 | 0.947 | 0.479 | 0.011 | 330.7 |
| swamping_clean | swamping | combination_union | union@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1107 | 16.6 | 2.21 | 2.08 | 0.619 | 0.961 | 0.465 | 0.012 | 330.1 |
| swamping_clean | swamping | combination_composite_agreement | composite-agreement@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.656 | 0.946 | 0.483 | 0.011 | 332.4 |
| swamping_clean | swamping | near_constant_2 | intersection@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0107 | 1.6 | 0.21 | 0.20 | 0.611 | 0.911 | 0.490 | 0.012 | 333.3 |
| swamping_contaminated | swamping | default | intersection@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0093 | 1.4 | 0.19 | 0.17 | 0.414 | 0.943 | 0.458 | 0.011 | 345.1 |
| swamping_contaminated | swamping | combination_composite | composite-equal@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.622 | 0.967 | 0.461 | 0.011 | 349.2 |
| swamping_contaminated | swamping | combination_union | union@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1073 | 16.1 | 2.15 | 2.01 | 0.671 | 0.984 | 0.461 | 0.012 | 359.3 |
| swamping_contaminated | swamping | combination_composite_agreement | composite-agreement@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.615 | 0.971 | 0.454 | 0.011 | 351.3 |
| swamping_contaminated | swamping | near_constant_2 | intersection@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 | 1.5 | 0.20 | 0.19 | 0.472 | 0.966 | 0.462 | 0.012 | 348.8 |
<!-- policy-matrix-results-end -->

### What the matrix shows (2026-10-14, 10 seeds)

- **`intersection@1` (the default)** is the most conservative policy. Its
  false-positive rate stays at or below 2%, and it flags 11-46% of the review
  budget. Even where its ranking is excellent (`point` and `clustered`, AP 1.00),
  its flags catch only 13-16% of the anomalies. Recall is what the
  precision-first default gives up.
- **`composite-equal@1` / `composite-agreement@1`** flag about the full review
  budget (`over_flag_ratio` ≈ 1.0), with recall of 0.6-0.8 on point,
  clustered, text-timestamp and irrelevant-column anomalies. Neither reaches
  `local` or `varying_density`. The two weightings differ only slightly, by up
  to 0.14 ROC-AUC on `local` and `varying_density`.
- **`union@1`** is the only policy that flags any `local` or `varying_density`
  anomalies (recall 0.32-0.35), but it flags 1.7-2.2x the review budget,
  breaching the false-positive budget.
- **`contextual`** is near chance under every policy: no ensemble rule fixes it.
- **`all_categorical`** is ranked reasonably by every policy (ROC-AUC 0.84-0.86),
  but `intersection@1` flags none of its anomalies; the composites catch 14-16% at
  the review budget and `union@1` 35% at 1.8x it.

Whether any of these differences justifies a different policy is decided by a
pre-registered backtest, not by this table: see
[the backtest results](ensemble-policy.md#backtests-when-a-candidate-may-be-said-to-outperform).

### `profiling.near_constant_distinct`: why 3 stays the default

The `near_constant_2` row keeps 3-value columns. Against the default (3):
`categorical_rare` improves from AP 0.05 to 0.20 (ROC-AUC 0.51 to 0.88) and
`contextual` slightly, but `local` falls from ROC-AUC 0.65 to 0.31 and
`varying_density` from 0.65 to 0.37, both *below chance*, because the
scenarios' 3-level noise column is kept. Everything else is unchanged. The
default stays 3. Its failure mode is visible: `sorethumb inspect` shows the
column dropped as near-constant, with the reason. Setting it to 2 risks a
silent below-chance ranking. Set it to 2 where you have measured that a 3-value
column carries your signal.

## Real-dataset / legacy synthetic benchmark

`src/sorethumb_ml/evaluate/benchmark.py` fits bare detectors directly (no
feature pipeline) against KDDCup99, Covtype, and two synthetic Gaussian-plus-
point-anomaly datasets. `k` comes from a fixed, label-independent 5% review
budget shared across every dataset (not derived from the labels being
scored, which would collapse precision@k/recall@k/F1 into one number by
construction); ROC-AUC/AP are `n/a` (not a fabricated 0.0) when a sample is
single-class; every run records its generation timestamp, platform, and
Python/numpy/scipy/scikit-learn/sorethumb versions (see the "Generated ..."
line above the table) so a published number can always be tied back to the
environment that produced it.

The committed table below caps KDDCup99/Covtype at a seeded random 20,000-row
sample (`BenchmarkConfig(max_rows=20_000)`) and the shipped default three
detectors (`isolation_forest`, `kmeans_distance`, `one_class_svm`) — the full,
uncapped datasets (Covtype has 581k rows) take on the order of an hour to
regenerate, dominated by `one_class_svm`'s O(n²) training cost; run without
`max_rows` for full-dataset numbers. `scripts/run_benchmark.py`'s `--seeds` only
controls this suite (default 5; the table below was regenerated with 3 for a
faster turnaround — still real mean ± standard deviation across independent
seeds, not a single draw).

<!-- benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md` to regenerate. -->

Generated 2026-09-29T01:58:13+00:00 on macOS-26.6.2-arm64-arm-64bit — Python 3.12.9, sorethumb 0.1.0, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0.

| dataset | detector | n_rows | n_features | n_seeds | roc_auc | average_precision | precision_at_k | recall_at_k | f1_at_contamination | fit_seconds | score_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_gaussian | isolation_forest | 1050 | 8 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9615 ± 0.0000 | 1.0000 ± 0.0000 | 0.9804 ± 0.0000 | 0.085 | 0.010 |
| synthetic_gaussian | kmeans_distance | 1050 | 8 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9615 ± 0.0000 | 1.0000 ± 0.0000 | 0.9804 ± 0.0000 | 0.170 | 0.000 |
| synthetic_gaussian | one_class_svm | 1050 | 8 | 3 | 0.9113 ± 0.0016 | 0.2038 ± 0.0037 | 0.1218 ± 0.0480 | 0.1267 ± 0.0499 | 0.1242 ± 0.0489 | 0.003 | 0.003 |
| synthetic_highd | isolation_forest | 2100 | 32 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9524 ± 0.0000 | 1.0000 ± 0.0000 | 0.9756 ± 0.0000 | 0.088 | 0.016 |
| synthetic_highd | kmeans_distance | 2100 | 32 | 3 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9524 ± 0.0000 | 1.0000 ± 0.0000 | 0.9756 ± 0.0000 | 0.433 | 0.001 |
| synthetic_highd | one_class_svm | 2100 | 32 | 3 | 0.9231 ± 0.0018 | 0.2243 ± 0.0028 | 0.0952 ± 0.0078 | 0.1000 ± 0.0082 | 0.0976 ± 0.0080 | 0.013 | 0.012 |
| kddcup99_sa | isolation_forest | 20000 | 38 | 3 | 0.9386 ± 0.0061 | 0.4443 ± 0.0143 | 0.1807 ± 0.0026 | 0.2722 ± 0.0049 | 0.2171 ± 0.0028 | 0.116 | 0.067 |
| kddcup99_sa | kmeans_distance | 20000 | 38 | 3 | 0.8053 ± 0.0090 | 0.0984 ± 0.0017 | 0.0067 ± 0.0017 | 0.0101 ± 0.0027 | 0.0080 ± 0.0021 | 17.013 | 0.001 |
| kddcup99_sa | one_class_svm | 20000 | 38 | 3 | 0.7670 ± 0.0084 | 0.0793 ± 0.0016 | 0.0067 ± 0.0017 | 0.0101 ± 0.0027 | 0.0080 ± 0.0021 | 1.374 | 1.291 |
| covtype | isolation_forest | 20000 | 54 | 3 | 0.9837 ± 0.0030 | 0.3486 ± 0.0320 | 0.1667 ± 0.0162 | 0.8976 ± 0.0510 | 0.2809 ± 0.0249 | 0.116 | 0.082 |
| covtype | kmeans_distance | 20000 | 54 | 3 | 0.5745 ± 0.0142 | 0.0103 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 14.014 | 0.002 |
| covtype | one_class_svm | 20000 | 54 | 3 | 0.7902 ± 0.0091 | 0.0224 ± 0.0024 | 0.0247 ± 0.0033 | 0.1339 ± 0.0211 | 0.0416 ± 0.0056 | 1.465 | 1.449 |
<!-- benchmark-results-end -->
