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

**Which contamination.** Every ablation here, `default` included, runs the ensemble
with `scoring.contamination` set to the scenario's review budget (5%, or 15% for
`masking`), not the shipped `"auto"`. So the `default` rows are the default *policy*
(`intersection@1`) at a fixed budget, not the default *configuration*. The ranking
columns (`roc_auc`, `average_precision`, `precision_at_k`, `recall_at_k`) do not depend
on contamination, so they hold for the shipped default too. The `flag_*` columns,
`over_flag_ratio` and `false_positive_budget_ratio` do depend on it. How the
default behaves at `"auto"` is reported in
[ensemble-policy.md](ensemble-policy.md#where-things-stand-2026-10-15).

**Matrix shape**: `scripts/run_benchmark.py` runs every scenario under the shipped
*default* ablation only (plus its `sklearn:*` baselines) — 18 pipeline
ablations exist in total (`AblationSpec`s: PCA on, standard scaler, each
detector alone, all six detectors, each combination policy, each discrete
rule, the shipped `"auto"` contamination, and the two categorical-handling
alternatives `near_constant_3` and `categorical_unscaled`). The comparison across ensemble policies is published
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
| point | point | default | intersection@1 | 3 | 588 | 252 | 0.9999 ± 0.0001 | 0.9970 ± 0.0059 | 0.8718 | 0.9375 | 1.0000 | 0.0845 | 0.1554 | 0.0000 | 1.0 | 0.08 | 0.00 | n/a | n/a | 0.750 | 0.010 | 285.9 |
| point | point | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.177 | 0.007 | 252.3 |
| point | point | sklearn:lof | n/a | 3 | 588 | 252 | 0.3239 ± 0.0298 | 0.0377 ± 0.0100 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.069 | 0.002 | 237.3 |
| point | point | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.9251 ± 0.0304 | 0.3431 ± 0.0486 | 0.2308 | 0.2504 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.000 | 234.8 |
| local | local | default | intersection@1 | 3 | 588 | 252 | 0.9446 ± 0.0104 | 0.3823 ± 0.1325 | 0.3333 | 0.3462 | 0.3333 | 0.0208 | 0.0392 | 0.0041 | 1.3 | 0.11 | 0.08 | n/a | n/a | 0.657 | 0.010 | 288.3 |
| local | local | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.9024 ± 0.0583 | 0.4356 ± 0.1883 | 0.3846 | 0.3973 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.183 | 0.007 | 253.3 |
| local | local | sklearn:lof | n/a | 3 | 588 | 252 | 0.8213 ± 0.0719 | 0.6270 ± 0.1236 | 0.5641 | 0.5777 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.066 | 0.001 | 237.0 |
| local | local | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.0073 ± 0.0109 | 0.0269 ± 0.0074 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.000 | 234.5 |
| contextual | contextual | default | intersection@1 | 3 | 840 | 360 | 0.6451 ± 0.0800 | 0.0863 ± 0.0080 | 0.0556 | 0.0625 | 0.0833 | 0.0208 | 0.0333 | 0.0126 | 4.7 | 0.26 | 0.24 | n/a | n/a | 0.743 | 0.012 | 312.9 |
| contextual | contextual | sklearn:isolation_forest | n/a | 3 | 840 | 360 | 0.7495 ± 0.0652 | 0.1524 ± 0.0577 | 0.1667 | 0.1838 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.180 | 0.008 | 254.4 |
| contextual | contextual | sklearn:lof | n/a | 3 | 840 | 360 | 0.5295 ± 0.0950 | 0.1012 ± 0.0607 | 0.1111 | 0.1225 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.072 | 0.003 | 240.6 |
| contextual | contextual | sklearn:one_class_svm | n/a | 3 | 840 | 360 | 0.6000 ± 0.0267 | 0.0901 ± 0.0262 | 0.0926 | 0.1017 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.047 | 0.001 | 234.5 |
| clustered | clustered | default | intersection@1 | 3 | 588 | 252 | 0.9998 ± 0.0003 | 0.9958 ± 0.0082 | 0.8718 | 0.9375 | 0.4722 | 0.1606 | 0.2381 | 0.0028 | 2.3 | 0.19 | 0.05 | n/a | n/a | 0.692 | 0.010 | 284.0 |
| clustered | clustered | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.190 | 0.007 | 251.9 |
| clustered | clustered | sklearn:lof | n/a | 3 | 588 | 252 | 0.3503 ± 0.1922 | 0.0418 ± 0.0157 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.074 | 0.002 | 237.8 |
| clustered | clustered | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.9174 ± 0.0337 | 0.2915 ± 0.0327 | 0.2564 | 0.2867 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.049 | 0.000 | 234.6 |
| masking | masking | default | intersection@1 | 3 | 595 | 255 | 0.8784 ± 0.0340 | 0.4524 ± 0.1061 | 0.4649 | 0.4059 | 0.2444 | 0.0320 | 0.0564 | 0.0109 | 3.7 | 0.10 | 0.06 | n/a | n/a | 0.708 | 0.010 | 282.1 |
| masking | masking | sklearn:isolation_forest | n/a | 3 | 595 | 255 | 0.9749 ± 0.0208 | 0.8911 ± 0.0878 | 0.8684 | 0.7674 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.182 | 0.007 | 252.5 |
| masking | masking | sklearn:lof | n/a | 3 | 595 | 255 | 0.6479 ± 0.0817 | 0.3159 ± 0.0725 | 0.4123 | 0.3723 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.067 | 0.001 | 237.6 |
| masking | masking | sklearn:one_class_svm | n/a | 3 | 595 | 255 | 0.8448 ± 0.0466 | 0.4534 ± 0.1458 | 0.4123 | 0.3603 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.001 | 235.3 |
| varying_density | varying_density | default | intersection@1 | 3 | 728 | 312 | 0.6894 ± 0.0320 | 0.0822 ± 0.0301 | 0.0625 | 0.0812 | 0.0000 | 0.0000 | 0.0000 | 0.0100 | 3.0 | 0.19 | 0.19 | n/a | n/a | 0.683 | 0.011 | 298.6 |
| varying_density | varying_density | sklearn:isolation_forest | n/a | 3 | 728 | 312 | 0.9386 ± 0.0155 | 0.3351 ± 0.1728 | 0.2292 | 0.3158 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.183 | 0.008 | 251.3 |
| varying_density | varying_density | sklearn:lof | n/a | 3 | 728 | 312 | 0.8671 ± 0.0970 | 0.4416 ± 0.4803 | 0.4375 | 0.5966 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.067 | 0.001 | 237.8 |
| varying_density | varying_density | sklearn:one_class_svm | n/a | 3 | 728 | 312 | 0.1297 ± 0.0348 | 0.0233 ± 0.0037 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.000 | 235.6 |
| categorical_rare | categorical_rare | default | intersection@1 | 3 | 577 | 248 | 0.8240 ± 0.0188 | 0.1088 ± 0.0437 | 0.0556 | 0.0667 | 0.1667 | 0.0333 | 0.0556 | 0.0028 | 1.0 | 0.08 | 0.06 | n/a | n/a | 0.711 | 0.011 | 282.9 |
| categorical_rare | categorical_rare | sklearn:isolation_forest | n/a | 3 | 577 | 248 | 0.9887 ± 0.0141 | 0.7668 ± 0.1961 | 0.5833 | 0.8333 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.185 | 0.007 | 251.8 |
| categorical_rare | categorical_rare | sklearn:lof | n/a | 3 | 577 | 248 | 0.8741 ± 0.1819 | 0.3491 ± 0.3109 | 0.2222 | 0.2667 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.071 | 0.002 | 237.2 |
| categorical_rare | categorical_rare | sklearn:one_class_svm | n/a | 3 | 577 | 248 | 0.8983 ± 0.0371 | 0.3003 ± 0.0962 | 0.2778 | 0.4000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.000 | 236.8 |
| text_timestamp | text_timestamp | default | intersection@1 | 3 | 581 | 249 | 0.9117 ± 0.0442 | 0.3799 ± 0.0834 | 0.3889 | 0.5179 | 0.5000 | 0.0893 | 0.1481 | 0.0042 | 1.7 | 0.13 | 0.08 | n/a | n/a | 0.706 | 0.011 | 287.0 |
| text_timestamp | text_timestamp | sklearn:isolation_forest | n/a | 3 | 581 | 249 | 0.4841 ± 0.1170 | 0.0406 ± 0.0103 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.178 | 0.006 | 269.3 |
| text_timestamp | text_timestamp | sklearn:lof | n/a | 3 | 581 | 249 | 0.4520 ± 0.1401 | 0.0408 ± 0.0175 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.077 | 0.003 | 294.5 |
| text_timestamp | text_timestamp | sklearn:one_class_svm | n/a | 3 | 581 | 249 | 0.4096 ± 0.1207 | 0.0338 ± 0.0095 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.084 | 0.033 | 273.5 |
| irrelevant_columns | irrelevant_columns | default | intersection@1 | 3 | 588 | 252 | 0.9430 ± 0.0153 | 0.6234 ± 0.0174 | 0.5897 | 0.6360 | 0.7833 | 0.3170 | 0.4444 | 0.0042 | 4.7 | 0.37 | 0.08 | n/a | n/a | 0.714 | 0.011 | 288.1 |
| irrelevant_columns | irrelevant_columns | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.8160 ± 0.0556 | 0.2932 ± 0.0875 | 0.1795 | 0.1992 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.180 | 0.007 | 251.1 |
| irrelevant_columns | irrelevant_columns | sklearn:lof | n/a | 3 | 588 | 252 | 0.9629 ± 0.0041 | 0.6200 ± 0.0157 | 0.5128 | 0.5485 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.073 | 0.002 | 247.5 |
| irrelevant_columns | irrelevant_columns | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.8826 ± 0.0260 | 0.3559 ± 0.0377 | 0.3590 | 0.3890 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.001 | 234.1 |
| all_categorical | all_categorical | default | intersection@1 | 3 | 588 | 252 | 0.8744 ± 0.0486 | 0.1882 ± 0.0555 | 0.2051 | 0.2136 | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 4.0 | 0.32 | 0.31 | n/a | n/a | 0.672 | 0.010 | 273.7 |
| all_categorical | all_categorical | sklearn:isolation_forest | n/a | 3 | 588 | 252 | 0.8568 ± 0.0546 | 0.1566 ± 0.0411 | 0.0256 | 0.0333 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.181 | 0.006 | 248.8 |
| all_categorical | all_categorical | sklearn:lof | n/a | 3 | 588 | 252 | 0.5097 ± 0.0924 | 0.0527 ± 0.0261 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.072 | 0.001 | 245.3 |
| all_categorical | all_categorical | sklearn:one_class_svm | n/a | 3 | 588 | 252 | 0.6236 ± 0.0202 | 0.0656 ± 0.0184 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.001 | 234.9 |
| swamping_clean | swamping | default | intersection@1 | 3 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0067 | 1.0 | 0.13 | 0.12 | n/a | n/a | 0.757 | 0.009 | 310.7 |
| swamping_contaminated | swamping | default | intersection@1 | 3 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0067 | 1.0 | 0.13 | 0.12 | n/a | n/a | 0.749 | 0.009 | 322.7 |
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

Every policy here runs at the scenario's review budget (`scoring.contamination` set to
it, so each detector flags exactly that share), not at the shipped `"auto"`. The
shipped default, `intersection@1` at `"auto"`, flags what every detector's own
boundary agrees on. It is compared separately; composite policies at `"auto"` are not
a configuration to use: see
[ensemble-policy.md](ensemble-policy.md#recommended-setup-if-you-have-a-review-budget).

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
is still listed as accepted. The `near_constant_3` and `categorical_unscaled` rows
are the evidence for the categorical-handling defaults (see the conclusion below
the table).

<!-- policy-matrix-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --no-legacy --no-pipeline --policy-matrix --readme docs/benchmarks.md` to regenerate. -->

Suite version `8f3a4461eb7ff2dd`, budgets v3, 10 seeds, 3 stability refits per seed, sorethumb 0.1.0 at `c40cc874aa5d`. Full results with provenance: [`benchmark-results/policy-matrix.json`](benchmark-results/policy-matrix.json).

| scenario | kind | ablation | policy | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | flag_precision | flag_recall | flag_f1 | flag_false_positive_rate | flag_count | over_flag_ratio | false_positive_budget_ratio | flag_jaccard_stability | rank_stability | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | intersection@1 | 10 | 588 | 252 | 0.9999 ± 0.0001 | 0.9975 ± 0.0034 | 0.8000 | 0.9679 | 0.6500 | 0.0967 | 0.1599 | 0.0016 | 1.4 | 0.11 | 0.03 | 0.711 | 0.976 | 0.341 | 0.010 | 307.8 |
| point | point | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9908 ± 0.0025 | 0.8398 ± 0.0390 | 0.6615 | 0.8117 | 0.6615 | 0.8117 | 0.7201 | 0.0182 | 13.0 | 1.03 | 0.34 | 0.830 | 0.985 | 0.339 | 0.010 | 307.4 |
| point | point | combination_union | union@1 | 10 | 588 | 252 | 0.9692 ± 0.0060 | 0.4962 ± 0.0878 | 0.4154 | 0.5240 | 0.4321 | 0.9804 | 0.5946 | 0.0576 | 24.5 | 1.94 | 1.07 | 0.874 | 0.975 | 0.337 | 0.010 | 303.6 |
| point | point | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9872 ± 0.0037 | 0.7876 ± 0.0623 | 0.6154 | 0.7552 | 0.6154 | 0.7552 | 0.6697 | 0.0207 | 13.0 | 1.03 | 0.38 | 0.795 | 0.984 | 0.339 | 0.010 | 311.2 |
| point | point | near_constant_3 | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8000 | 0.9679 | 0.5817 | 0.0967 | 0.1511 | 0.0037 | 1.9 | 0.15 | 0.07 | 0.687 | 0.980 | 0.338 | 0.009 | 306.1 |
| point | point | categorical_unscaled | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0001 | 0.9983 ± 0.0023 | 0.8000 | 0.9679 | 0.5917 | 0.1249 | 0.1962 | 0.0025 | 1.9 | 0.15 | 0.05 | 0.626 | 0.980 | 0.343 | 0.010 | 316.7 |
| local | local | default | intersection@1 | 10 | 588 | 252 | 0.9307 ± 0.0127 | 0.3456 ± 0.0803 | 0.2692 | 0.3219 | 0.3333 | 0.0340 | 0.0596 | 0.0041 | 1.4 | 0.11 | 0.08 | 0.489 | 0.948 | 0.320 | 0.010 | 310.3 |
| local | local | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9388 ± 0.0118 | 0.3880 ± 0.0936 | 0.3231 | 0.3959 | 0.3231 | 0.3959 | 0.3521 | 0.0365 | 13.0 | 1.03 | 0.68 | 0.650 | 0.963 | 0.325 | 0.010 | 312.0 |
| local | local | combination_union | union@1 | 10 | 588 | 252 | 0.9348 ± 0.0161 | 0.3689 ± 0.1083 | 0.2923 | 0.3593 | 0.3021 | 0.8267 | 0.4376 | 0.0854 | 29.5 | 2.34 | 1.58 | 0.779 | 0.954 | 0.323 | 0.010 | 312.8 |
| local | local | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9354 ± 0.0128 | 0.3906 ± 0.0970 | 0.3231 | 0.3959 | 0.3231 | 0.3959 | 0.3521 | 0.0365 | 13.0 | 1.03 | 0.68 | 0.643 | 0.964 | 0.322 | 0.010 | 316.2 |
| local | local | near_constant_3 | intersection@1 | 10 | 588 | 252 | 0.9516 ± 0.0094 | 0.4180 ± 0.0713 | 0.3308 | 0.4124 | 0.3417 | 0.1054 | 0.1604 | 0.0070 | 2.9 | 0.23 | 0.13 | 0.601 | 0.973 | 0.322 | 0.010 | 319.8 |
| local | local | categorical_unscaled | intersection@1 | 10 | 588 | 252 | 0.3111 ± 0.0485 | 0.0375 ± 0.0064 | 0.0154 | 0.0174 | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 2.6 | 0.21 | 0.20 | 0.414 | 0.964 | 0.327 | 0.010 | 315.5 |
| contextual | contextual | default | intersection@1 | 10 | 840 | 360 | 0.6097 ± 0.0396 | 0.0866 ± 0.0217 | 0.0556 | 0.0651 | 0.0736 | 0.0306 | 0.0422 | 0.0105 | 4.0 | 0.22 | 0.20 | 0.655 | 0.936 | 0.404 | 0.012 | 340.2 |
| contextual | contextual | combination_composite | composite-equal@1 | 10 | 840 | 360 | 0.6118 ± 0.0383 | 0.0885 ± 0.0241 | 0.0722 | 0.0806 | 0.0722 | 0.0806 | 0.0753 | 0.0487 | 18.0 | 1.00 | 0.93 | 0.747 | 0.964 | 0.403 | 0.013 | 348.9 |
| contextual | contextual | combination_union | union@1 | 10 | 840 | 360 | 0.6372 ± 0.0408 | 0.0943 ± 0.0200 | 0.0833 | 0.0934 | 0.0881 | 0.1891 | 0.1192 | 0.0930 | 35.0 | 1.94 | 1.77 | 0.747 | 0.937 | 0.403 | 0.012 | 350.5 |
| contextual | contextual | combination_composite_agreement | composite-agreement@1 | 10 | 840 | 360 | 0.5991 ± 0.0395 | 0.0859 ± 0.0244 | 0.0722 | 0.0806 | 0.0722 | 0.0806 | 0.0753 | 0.0487 | 18.0 | 1.00 | 0.93 | 0.724 | 0.961 | 0.403 | 0.012 | 351.1 |
| contextual | contextual | near_constant_3 | intersection@1 | 10 | 840 | 360 | 0.4951 ± 0.0434 | 0.0669 ± 0.0249 | 0.0389 | 0.0469 | 0.0389 | 0.0223 | 0.0282 | 0.0157 | 5.7 | 0.32 | 0.30 | 0.683 | 0.948 | 0.402 | 0.011 | 345.2 |
| contextual | contextual | categorical_unscaled | intersection@1 | 10 | 840 | 360 | 0.5999 ± 0.0296 | 0.0862 ± 0.0206 | 0.0611 | 0.0716 | 0.0762 | 0.0383 | 0.0508 | 0.0160 | 6.0 | 0.33 | 0.31 | 0.702 | 0.986 | 0.395 | 0.012 | 333.4 |
| clustered | clustered | default | intersection@1 | 10 | 588 | 252 | 0.9999 ± 0.0001 | 0.9987 ± 0.0024 | 0.8000 | 0.9679 | 0.6417 | 0.1486 | 0.2362 | 0.0012 | 1.8 | 0.14 | 0.02 | 0.722 | 0.978 | 0.349 | 0.010 | 301.3 |
| clustered | clustered | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9900 ± 0.0044 | 0.8249 ± 0.0655 | 0.6538 | 0.7996 | 0.6538 | 0.7996 | 0.7117 | 0.0186 | 13.0 | 1.03 | 0.35 | 0.828 | 0.987 | 0.345 | 0.010 | 308.0 |
| clustered | clustered | combination_union | union@1 | 10 | 588 | 252 | 0.9717 ± 0.0061 | 0.5008 ± 0.0660 | 0.4231 | 0.5361 | 0.4494 | 1.0000 | 0.6159 | 0.0551 | 24.2 | 1.92 | 1.02 | 0.877 | 0.978 | 0.348 | 0.010 | 304.0 |
| clustered | clustered | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9868 ± 0.0053 | 0.7859 ± 0.0728 | 0.5923 | 0.7199 | 0.5923 | 0.7199 | 0.6430 | 0.0219 | 13.0 | 1.03 | 0.41 | 0.844 | 0.987 | 0.350 | 0.010 | 309.3 |
| clustered | clustered | near_constant_3 | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0000 | 0.9996 ± 0.0007 | 0.8000 | 0.9679 | 0.5933 | 0.1386 | 0.2152 | 0.0025 | 2.0 | 0.16 | 0.05 | 0.695 | 0.979 | 0.348 | 0.010 | 303.3 |
| clustered | clustered | categorical_unscaled | intersection@1 | 10 | 588 | 252 | 1.0000 ± 0.0001 | 0.9993 ± 0.0014 | 0.8000 | 0.9679 | 0.6917 | 0.1564 | 0.2457 | 0.0016 | 2.0 | 0.16 | 0.03 | 0.537 | 0.975 | 0.350 | 0.010 | 306.9 |
| masking | masking | default | intersection@1 | 10 | 595 | 255 | 0.8529 ± 0.0221 | 0.4186 ± 0.0553 | 0.4263 | 0.3669 | 0.1883 | 0.0193 | 0.0347 | 0.0127 | 3.6 | 0.09 | 0.07 | 0.593 | 0.980 | 0.350 | 0.010 | 310.0 |
| masking | masking | combination_composite | composite-equal@1 | 10 | 595 | 255 | 0.7650 ± 0.0211 | 0.3318 ± 0.0383 | 0.3079 | 0.2657 | 0.3079 | 0.2663 | 0.2841 | 0.1247 | 38.0 | 0.99 | 0.69 | 0.854 | 0.986 | 0.347 | 0.011 | 307.2 |
| masking | masking | combination_union | union@1 | 10 | 595 | 255 | 0.8764 ± 0.0152 | 0.5154 ± 0.0632 | 0.5368 | 0.4607 | 0.4626 | 0.8032 | 0.5851 | 0.1936 | 76.0 | 1.99 | 1.07 | 0.832 | 0.966 | 0.344 | 0.010 | 307.7 |
| masking | masking | combination_composite_agreement | composite-agreement@1 | 10 | 595 | 255 | 0.8093 ± 0.0251 | 0.3835 ± 0.0549 | 0.3658 | 0.3158 | 0.3658 | 0.3158 | 0.3372 | 0.1142 | 38.0 | 0.99 | 0.63 | 0.854 | 0.990 | 0.347 | 0.011 | 306.1 |
| masking | masking | near_constant_3 | intersection@1 | 10 | 595 | 255 | 0.8626 ± 0.0228 | 0.4414 ± 0.0595 | 0.4474 | 0.3880 | 0.2817 | 0.0218 | 0.0396 | 0.0108 | 3.3 | 0.09 | 0.06 | 0.653 | 0.982 | 0.345 | 0.010 | 304.5 |
| masking | masking | categorical_unscaled | intersection@1 | 10 | 595 | 255 | 0.8490 ± 0.0236 | 0.4252 ± 0.0622 | 0.4211 | 0.3609 | 0.3552 | 0.0355 | 0.0640 | 0.0146 | 4.7 | 0.12 | 0.08 | 0.627 | 0.984 | 0.336 | 0.011 | 309.9 |
| varying_density | varying_density | default | intersection@1 | 10 | 728 | 312 | 0.6245 ± 0.0758 | 0.0663 ± 0.0158 | 0.0375 | 0.0646 | 0.0000 | 0.0000 | 0.0000 | 0.0119 | 3.6 | 0.23 | 0.23 | 0.640 | 0.960 | 0.339 | 0.012 | 328.3 |
| varying_density | varying_density | combination_composite | composite-equal@1 | 10 | 728 | 312 | 0.6112 ± 0.0505 | 0.0485 ± 0.0061 | 0.0063 | 0.0077 | 0.0063 | 0.0077 | 0.0069 | 0.0529 | 16.0 | 1.03 | 0.99 | 0.718 | 0.975 | 0.340 | 0.012 | 327.7 |
| varying_density | varying_density | combination_union | union@1 | 10 | 728 | 312 | 0.6909 ± 0.0439 | 0.0717 ± 0.0111 | 0.0625 | 0.0851 | 0.0548 | 0.1600 | 0.0803 | 0.0971 | 30.9 | 1.98 | 1.82 | 0.755 | 0.963 | 0.339 | 0.012 | 335.8 |
| varying_density | varying_density | combination_composite_agreement | composite-agreement@1 | 10 | 728 | 312 | 0.6449 ± 0.0504 | 0.0538 ± 0.0072 | 0.0063 | 0.0077 | 0.0063 | 0.0077 | 0.0069 | 0.0529 | 16.0 | 1.03 | 0.99 | 0.712 | 0.975 | 0.339 | 0.012 | 332.1 |
| varying_density | varying_density | near_constant_3 | intersection@1 | 10 | 728 | 312 | 0.8054 ± 0.0216 | 0.0982 ± 0.0197 | 0.0500 | 0.0736 | 0.0000 | 0.0000 | 0.0000 | 0.0156 | 4.7 | 0.30 | 0.29 | 0.696 | 0.981 | 0.336 | 0.011 | 333.3 |
| varying_density | varying_density | categorical_unscaled | intersection@1 | 10 | 728 | 312 | 0.3667 ± 0.0405 | 0.0319 ± 0.0051 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0163 | 4.9 | 0.31 | 0.31 | 0.641 | 0.953 | 0.344 | 0.012 | 329.1 |
| categorical_rare | categorical_rare | default | intersection@1 | 10 | 577 | 248 | 0.8407 ± 0.0229 | 0.1329 ± 0.0301 | 0.1167 | 0.2004 | 0.3500 | 0.0793 | 0.1237 | 0.0025 | 1.1 | 0.09 | 0.05 | 0.472 | 0.915 | 0.353 | 0.011 | 299.4 |
| categorical_rare | categorical_rare | combination_composite | composite-equal@1 | 10 | 577 | 248 | 0.8435 ± 0.0191 | 0.2009 ± 0.0625 | 0.1250 | 0.2104 | 0.1250 | 0.2104 | 0.1529 | 0.0436 | 12.0 | 0.97 | 0.88 | 0.718 | 0.942 | 0.353 | 0.011 | 304.3 |
| categorical_rare | categorical_rare | combination_union | union@1 | 10 | 577 | 248 | 0.9541 ± 0.0098 | 0.3192 ± 0.0817 | 0.2833 | 0.4621 | 0.2569 | 0.9300 | 0.3962 | 0.0797 | 25.8 | 2.08 | 1.60 | 0.758 | 0.919 | 0.357 | 0.011 | 299.7 |
| categorical_rare | categorical_rare | combination_composite_agreement | composite-agreement@1 | 10 | 577 | 248 | 0.8494 ± 0.0170 | 0.2064 ± 0.0632 | 0.1250 | 0.2104 | 0.1250 | 0.2104 | 0.1529 | 0.0436 | 12.0 | 0.97 | 0.88 | 0.724 | 0.943 | 0.352 | 0.011 | 307.2 |
| categorical_rare | categorical_rare | near_constant_3 | intersection@1 | 10 | 577 | 248 | 0.5036 ± 0.0449 | 0.0446 ± 0.0118 | 0.0333 | 0.0525 | 0.0250 | 0.0200 | 0.0222 | 0.0166 | 4.1 | 0.33 | 0.33 | 0.675 | 0.860 | 0.355 | 0.010 | 312.0 |
| categorical_rare | categorical_rare | categorical_unscaled | intersection@1 | 10 | 577 | 248 | 0.8789 ± 0.0154 | 0.2001 ± 0.0650 | 0.1583 | 0.2646 | 0.4167 | 0.1368 | 0.1947 | 0.0037 | 1.9 | 0.15 | 0.07 | 0.572 | 0.945 | 0.359 | 0.011 | 301.9 |
| text_timestamp | text_timestamp | default | intersection@1 | 10 | 581 | 249 | 0.9406 ± 0.0273 | 0.4824 ± 0.0930 | 0.3750 | 0.6735 | 0.5750 | 0.2333 | 0.3050 | 0.0046 | 2.5 | 0.20 | 0.09 | 0.630 | 0.941 | 0.364 | 0.011 | 307.0 |
| text_timestamp | text_timestamp | combination_composite | composite-equal@1 | 10 | 581 | 249 | 0.9487 ± 0.0185 | 0.4749 ± 0.0827 | 0.3167 | 0.5798 | 0.3167 | 0.5798 | 0.3928 | 0.0339 | 12.0 | 0.96 | 0.68 | 0.686 | 0.966 | 0.363 | 0.011 | 309.5 |
| text_timestamp | text_timestamp | combination_union | union@1 | 10 | 581 | 249 | 0.9438 ± 0.0172 | 0.2848 ± 0.0396 | 0.2333 | 0.4580 | 0.2577 | 0.9232 | 0.3941 | 0.0761 | 24.8 | 1.99 | 1.53 | 0.730 | 0.943 | 0.363 | 0.011 | 311.1 |
| text_timestamp | text_timestamp | combination_composite_agreement | composite-agreement@1 | 10 | 581 | 249 | 0.9477 ± 0.0178 | 0.4784 ± 0.0848 | 0.3167 | 0.5927 | 0.3167 | 0.5927 | 0.3969 | 0.0339 | 12.0 | 0.96 | 0.68 | 0.675 | 0.964 | 0.361 | 0.011 | 307.5 |
| text_timestamp | text_timestamp | near_constant_3 | intersection@1 | 10 | 581 | 249 | 0.9439 ± 0.0141 | 0.3600 ± 0.0722 | 0.3083 | 0.5477 | 0.5817 | 0.2712 | 0.3406 | 0.0062 | 3.3 | 0.27 | 0.12 | 0.604 | 0.954 | 0.363 | 0.011 | 306.2 |
| text_timestamp | text_timestamp | categorical_unscaled | intersection@1 | 10 | 581 | 249 | 0.9289 ± 0.0262 | 0.4025 ± 0.1145 | 0.2917 | 0.5121 | 0.5233 | 0.2619 | 0.3230 | 0.0062 | 3.1 | 0.25 | 0.12 | 0.598 | 0.943 | 0.365 | 0.011 | 307.8 |
| irrelevant_columns | irrelevant_columns | default | intersection@1 | 10 | 588 | 252 | 0.9693 ± 0.0137 | 0.6977 ± 0.0462 | 0.5923 | 0.7313 | 0.6979 | 0.3356 | 0.4385 | 0.0066 | 5.1 | 0.40 | 0.12 | 0.614 | 0.980 | 0.367 | 0.012 | 302.1 |
| irrelevant_columns | irrelevant_columns | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.9605 ± 0.0174 | 0.6780 ± 0.0779 | 0.4923 | 0.6215 | 0.4923 | 0.6215 | 0.5427 | 0.0274 | 13.0 | 1.03 | 0.51 | 0.644 | 0.985 | 0.371 | 0.012 | 303.2 |
| irrelevant_columns | irrelevant_columns | combination_union | union@1 | 10 | 588 | 252 | 0.9553 ± 0.0112 | 0.4381 ± 0.0717 | 0.4308 | 0.5280 | 0.4331 | 0.9492 | 0.5900 | 0.0552 | 23.4 | 1.86 | 1.02 | 0.689 | 0.966 | 0.369 | 0.012 | 303.5 |
| irrelevant_columns | irrelevant_columns | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.9632 ± 0.0161 | 0.6943 ± 0.0740 | 0.5077 | 0.6381 | 0.5077 | 0.6381 | 0.5585 | 0.0265 | 13.0 | 1.03 | 0.49 | 0.680 | 0.986 | 0.367 | 0.012 | 308.7 |
| irrelevant_columns | irrelevant_columns | near_constant_3 | intersection@1 | 10 | 588 | 252 | 0.9710 ± 0.0107 | 0.7088 ± 0.0457 | 0.6231 | 0.7613 | 0.7262 | 0.3753 | 0.4744 | 0.0066 | 5.5 | 0.44 | 0.12 | 0.577 | 0.980 | 0.370 | 0.012 | 299.6 |
| irrelevant_columns | irrelevant_columns | categorical_unscaled | intersection@1 | 10 | 588 | 252 | 0.9711 ± 0.0123 | 0.6956 ± 0.0560 | 0.6231 | 0.7649 | 0.6958 | 0.3534 | 0.4468 | 0.0074 | 5.5 | 0.44 | 0.14 | 0.691 | 0.979 | 0.366 | 0.012 | 307.6 |
| all_categorical | all_categorical | default | intersection@1 | 10 | 588 | 252 | 0.8626 ± 0.0149 | 0.1651 ± 0.0260 | 0.1538 | 0.1845 | 0.0000 | 0.0000 | 0.0000 | 0.0215 | 5.2 | 0.41 | 0.40 | 0.726 | 0.957 | 0.331 | 0.011 | 294.3 |
| all_categorical | all_categorical | combination_composite | composite-equal@1 | 10 | 588 | 252 | 0.8507 ± 0.0144 | 0.1487 ± 0.0181 | 0.1154 | 0.1467 | 0.1154 | 0.1467 | 0.1281 | 0.0477 | 13.0 | 1.03 | 0.88 | 0.720 | 0.961 | 0.333 | 0.011 | 295.3 |
| all_categorical | all_categorical | combination_union | union@1 | 10 | 588 | 252 | 0.8403 ± 0.0155 | 0.1336 ± 0.0159 | 0.0308 | 0.0422 | 0.1692 | 0.3530 | 0.2272 | 0.0772 | 22.4 | 1.78 | 1.43 | 0.820 | 0.959 | 0.332 | 0.011 | 301.9 |
| all_categorical | all_categorical | combination_composite_agreement | composite-agreement@1 | 10 | 588 | 252 | 0.8517 ± 0.0148 | 0.1502 ± 0.0183 | 0.1154 | 0.1467 | 0.1154 | 0.1467 | 0.1281 | 0.0477 | 13.0 | 1.03 | 0.88 | 0.713 | 0.964 | 0.331 | 0.011 | 301.6 |
| all_categorical | all_categorical | near_constant_3 | intersection@1 | 10 | 588 | 252 | 0.8626 ± 0.0149 | 0.1651 ± 0.0260 | 0.1538 | 0.1845 | 0.0000 | 0.0000 | 0.0000 | 0.0215 | 5.2 | 0.41 | 0.40 | 0.726 | 0.957 | 0.331 | 0.011 | 296.8 |
| all_categorical | all_categorical | categorical_unscaled | intersection@1 | 10 | 588 | 252 | 0.8626 ± 0.0150 | 0.1656 ± 0.0265 | 0.1538 | 0.1845 | 0.0000 | 0.0000 | 0.0000 | 0.0220 | 5.3 | 0.42 | 0.41 | 0.742 | 0.955 | 0.332 | 0.011 | 297.4 |
| swamping_clean | swamping | default | intersection@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0073 | 1.1 | 0.15 | 0.14 | 0.394 | 0.902 | 0.416 | 0.010 | 343.5 |
| swamping_clean | swamping | combination_composite | composite-equal@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.661 | 0.949 | 0.419 | 0.010 | 336.7 |
| swamping_clean | swamping | combination_union | union@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1167 | 17.5 | 2.33 | 2.19 | 0.616 | 0.961 | 0.417 | 0.010 | 338.5 |
| swamping_clean | swamping | combination_composite_agreement | composite-agreement@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.615 | 0.944 | 0.419 | 0.010 | 335.3 |
| swamping_clean | swamping | near_constant_3 | intersection@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0087 | 1.3 | 0.17 | 0.16 | 0.472 | 0.911 | 0.411 | 0.009 | 339.4 |
| swamping_clean | swamping | categorical_unscaled | intersection@1 | 10 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0107 | 1.6 | 0.21 | 0.20 | 0.611 | 0.911 | 0.419 | 0.009 | 335.9 |
| swamping_contaminated | swamping | default | intersection@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0067 | 1.0 | 0.13 | 0.12 | 0.694 | 0.920 | 0.409 | 0.010 | 356.2 |
| swamping_contaminated | swamping | combination_composite | composite-equal@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.615 | 0.964 | 0.406 | 0.010 | 353.7 |
| swamping_contaminated | swamping | combination_union | union@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1153 | 17.3 | 2.31 | 2.16 | 0.665 | 0.986 | 0.409 | 0.010 | 350.9 |
| swamping_contaminated | swamping | combination_composite_agreement | composite-agreement@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0533 | 8.0 | 1.07 | 1.00 | 0.681 | 0.970 | 0.411 | 0.010 | 359.1 |
| swamping_contaminated | swamping | near_constant_3 | intersection@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0093 | 1.4 | 0.19 | 0.17 | 0.594 | 0.931 | 0.406 | 0.009 | 363.0 |
| swamping_contaminated | swamping | categorical_unscaled | intersection@1 | 10 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0100 | 1.5 | 0.20 | 0.19 | 0.472 | 0.966 | 0.408 | 0.010 | 362.5 |
<!-- policy-matrix-results-end -->

### What the matrix shows (2026-10-19, 10 seeds)

- **`intersection@1` (the default policy, here at the review budget)** is the most
  conservative policy. Its false-positive rate stays at or below 2.2%, and it flags
  9-41% of the review budget. Even where its ranking is excellent (`point` and
  `clustered`, AP 1.00), its flags catch only 10-15% of the anomalies. At a fixed
  budget, unanimity gives up most of the recall.
- **`composite-equal@1` / `composite-agreement@1`** flag about the full review
  budget (`over_flag_ratio` ≈ 1.0), with recall of 0.58-0.81 on point,
  clustered, text-timestamp and irrelevant-column anomalies, and 0.40 on `local`.
  Neither reaches `varying_density` (recall 0.01). The two weightings differ by
  at most 0.04 ROC-AUC (on `masking` and `varying_density`).
- **`union@1`** is the only policy that flags a useful share of `varying_density`
  anomalies (recall 0.16; 0.83 on `local`), but it flags 1.8-2.3x the review
  budget, breaching the false-positive budget.
- **`contextual`** is near chance under every policy (ROC-AUC 0.60-0.64): no
  ensemble rule fixes it.
- **`all_categorical`** is ranked reasonably by every policy (ROC-AUC 0.84-0.86),
  but `intersection@1` flags none of its anomalies; the composites catch 15% at
  the review budget and `union@1` 35% at 1.8x it.

Whether any of these differences justifies a different policy is decided by a
pre-registered backtest, not by this table: see
[the backtest results](ensemble-policy.md#backtests-when-a-candidate-may-be-said-to-outperform).
Those results, and the decision not to change the default for now, are summarised in
[Where things stand](ensemble-policy.md#where-things-stand-2026-10-15).

### Categorical columns: why 3-value columns are kept, at a fixed distance

Two defaults decide how a low-cardinality column counts: whether a 3-value column
is kept at all (`profiling.near_constant_distinct`, default 2: only 1- and 2-value
columns are dropped as near-constant), and how far apart its one-hot categories are
(`features.categorical_distance`, default 0.5). Each has a row in the matrix that
undoes it, everything else at the default:

| scenario | default | `near_constant_3` (drop 3-value columns) | `categorical_unscaled` (dummies scaled like numerics) |
| --- | --- | --- | --- |
| `local` | 0.93 | 0.95 | 0.31 |
| `varying_density` | 0.63 | 0.81 | 0.37 |
| `categorical_rare` | 0.84 | 0.50 | 0.88 |
| `contextual` | 0.61 | 0.50 | 0.60 |

(ROC-AUC, mean of 10 seeds.) Every scenario carries a 3-level noise column, and
before the fixed distance existed, robust-scaling its one-hot dummies like numeric
columns made them dominate the distance-based detectors: the `categorical_unscaled`
column shows `local` and `varying_density` falling *below chance*. Dropping 3-value
columns hid that, and it is still the better choice on those two scenarios, whose
3-value column is pure noise. It is the wrong choice whenever a 3-value column
carries signal: `categorical_rare` falls to chance, `contextual` to chance, and on
KDDCup99 (20,000 rows, 5 seeds) dropping `protocol_type` costs ROC-AUC 0.918 to
0.916. A default that silently discards a real column is worse than one that keeps a
noise column at a bounded weight.

At 0.5, two rows that differ only in one categorical column are as far apart as
half the interquartile range of a numeric column (the unit robust scaling uses). That is a prior, not a measurement:
[approximations.md](approximations.md) says what it trades. Set
`features.categorical_distance` lower where your categorical columns are mostly
noise, higher where they carry your signal, or `null` for the old scaling, and
`profiling.near_constant_distinct: 3` to drop 3-value columns again.

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
