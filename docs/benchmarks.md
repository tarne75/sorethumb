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
*default* ablation only (plus its `sklearn:*` baselines) — 10 pipeline
ablations exist in total (`AblationSpec`s: PCA on, standard scaler, each
detector alone, all six detectors, composite/union combination), but only
`default` is committed here to keep regeneration time bounded; run
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
fails that suite. It also documents where the *shipped default* combination
mode (`intersection`) genuinely underperforms (`local`, `varying_density` —
see [`approximations.md`](approximations.md)) rather than asserting something the data shows
is false. `scripts/run_benchmark.py` itself refuses to publish a
matrix with a missing or errored cell (`assert_complete_and_error_free`).

<!-- pipeline-benchmark-results-start -->
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md` to regenerate. -->

| scenario | kind | ablation | n_seeds | n_train | n_holdout | roc_auc | average_precision | precision_at_k | recall_at_k | flag_precision | flag_recall | flag_f1 | flag_false_positive_rate | flag_count | fit_seconds | score_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| point | point | default | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | 0.8889 | 0.1178 | 0.1973 | 0.0014 | 1.7 | 0.823 | 0.008 | 244.1 |
| point | point | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | 0.158 | 0.006 | 230.0 |
| point | point | sklearn:lof | 3 | 588 | 252 | 0.3239 ± 0.0298 | 0.0377 ± 0.0100 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.054 | 0.002 | 220.5 |
| point | point | sklearn:one_class_svm | 3 | 588 | 252 | 0.9251 ± 0.0304 | 0.3431 ± 0.0486 | 0.2308 | 0.2504 | n/a | n/a | n/a | n/a | n/a | 0.042 | 0.000 | 217.6 |
| local | local | default | 3 | 588 | 252 | 0.7085 ± 0.0335 | 0.1075 ± 0.0460 | 0.1026 | 0.0958 | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 3.0 | 0.576 | 0.008 | 245.0 |
| local | local | sklearn:isolation_forest | 3 | 588 | 252 | 0.9024 ± 0.0583 | 0.4356 ± 0.1883 | 0.3846 | 0.3973 | n/a | n/a | n/a | n/a | n/a | 0.160 | 0.006 | 230.4 |
| local | local | sklearn:lof | 3 | 588 | 252 | 0.8213 ± 0.0719 | 0.6270 ± 0.1236 | 0.5641 | 0.5777 | n/a | n/a | n/a | n/a | n/a | 0.059 | 0.001 | 220.2 |
| local | local | sklearn:one_class_svm | 3 | 588 | 252 | 0.0073 ± 0.0109 | 0.0269 ± 0.0074 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.046 | 0.000 | 217.5 |
| contextual | contextual | default | 3 | 840 | 360 | 0.4965 ± 0.0842 | 0.0579 ± 0.0091 | 0.0370 | 0.0417 | 0.0333 | 0.0208 | 0.0256 | 0.0204 | 7.3 | 0.674 | 0.010 | 249.0 |
| contextual | contextual | sklearn:isolation_forest | 3 | 840 | 360 | 0.7495 ± 0.0652 | 0.1524 ± 0.0577 | 0.1667 | 0.1838 | n/a | n/a | n/a | n/a | n/a | 0.164 | 0.007 | 230.8 |
| contextual | contextual | sklearn:lof | 3 | 840 | 360 | 0.5295 ± 0.0950 | 0.1012 ± 0.0607 | 0.1111 | 0.1225 | n/a | n/a | n/a | n/a | n/a | 0.064 | 0.003 | 220.7 |
| contextual | contextual | sklearn:one_class_svm | 3 | 840 | 360 | 0.6000 ± 0.0267 | 0.0901 ± 0.0262 | 0.0926 | 0.1017 | n/a | n/a | n/a | n/a | n/a | 0.044 | 0.001 | 218.2 |
| clustered | clustered | default | 3 | 588 | 252 | 1.0000 ± 0.0001 | 0.9988 ± 0.0024 | 0.8718 | 0.9375 | 0.4222 | 0.1606 | 0.2286 | 0.0041 | 2.7 | 0.626 | 0.008 | 244.3 |
| clustered | clustered | sklearn:isolation_forest | 3 | 588 | 252 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8718 | 0.9375 | n/a | n/a | n/a | n/a | n/a | 0.160 | 0.006 | 229.9 |
| clustered | clustered | sklearn:lof | 3 | 588 | 252 | 0.3503 ± 0.1922 | 0.0418 ± 0.0157 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.060 | 0.002 | 220.3 |
| clustered | clustered | sklearn:one_class_svm | 3 | 588 | 252 | 0.9174 ± 0.0337 | 0.2915 ± 0.0327 | 0.2564 | 0.2867 | n/a | n/a | n/a | n/a | n/a | 0.040 | 0.000 | 217.5 |
| masking | masking | default | 3 | 595 | 255 | 0.8813 ± 0.0328 | 0.4686 ± 0.1070 | 0.4825 | 0.4236 | 0.3185 | 0.0394 | 0.0684 | 0.0187 | 5.7 | 0.641 | 0.008 | 244.5 |
| masking | masking | sklearn:isolation_forest | 3 | 595 | 255 | 0.9749 ± 0.0208 | 0.8911 ± 0.0878 | 0.8684 | 0.7674 | n/a | n/a | n/a | n/a | n/a | 0.162 | 0.006 | 230.2 |
| masking | masking | sklearn:lof | 3 | 595 | 255 | 0.6479 ± 0.0817 | 0.3159 ± 0.0725 | 0.4123 | 0.3723 | n/a | n/a | n/a | n/a | n/a | 0.058 | 0.002 | 220.4 |
| masking | masking | sklearn:one_class_svm | 3 | 595 | 255 | 0.8448 ± 0.0466 | 0.4534 ± 0.1458 | 0.4123 | 0.3603 | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.001 | 217.8 |
| varying_density | varying_density | default | 3 | 728 | 312 | 0.6824 ± 0.0341 | 0.0844 ± 0.0552 | 0.0208 | 0.0256 | 0.0000 | 0.0000 | 0.0000 | 0.0155 | 4.7 | 0.552 | 0.009 | 246.5 |
| varying_density | varying_density | sklearn:isolation_forest | 3 | 728 | 312 | 0.9386 ± 0.0155 | 0.3351 ± 0.1728 | 0.2292 | 0.3158 | n/a | n/a | n/a | n/a | n/a | 0.161 | 0.007 | 230.4 |
| varying_density | varying_density | sklearn:lof | 3 | 728 | 312 | 0.8671 ± 0.0970 | 0.4416 ± 0.4803 | 0.4375 | 0.5966 | n/a | n/a | n/a | n/a | n/a | 0.057 | 0.001 | 220.2 |
| varying_density | varying_density | sklearn:one_class_svm | 3 | 728 | 312 | 0.1297 ± 0.0348 | 0.0233 ± 0.0037 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.000 | 217.5 |
| swamping_clean | swamping | default | 3 | 800 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0111 | 1.7 | 0.696 | 0.007 | 249.7 |
| swamping_contaminated | swamping | default | 3 | 880 | 150 | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.7 | 0.655 | 0.007 | 251.0 |
<!-- pipeline-benchmark-results-end -->

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
