# sorethumb documentation

New here? Start with the [README](../README.md): install, the 60-second quickstart,
what it does not do, and whether it is the right tool. Then work down this page.

## Getting started

- [Getting started](getting-started.md): the command line from a single file with no
  configuration to a configured workspace, reading results, and the file formats it
  reads.
- [Adapting to your data](adapting-to-your-data.md): what to change when the defaults
  do not fit, and how to tell.
- [Example runs](example-runs.md): a validation sweep over nine real datasets.

## Reading and trusting the results

- [Limitations](limitations.md): the full list, with where each one is measured.
- [Ensemble policy: the default and the evidence](ensemble-policy.md): what decides
  which rows are flagged, how many, and the evidence behind the default.
- [Explanations: exact, model-specific and heuristic](explanations.md): what each
  reason means and how it was produced.
- [Approximations and error characteristics](approximations.md): every shortcut the
  pipeline takes, and what it costs.
- [Trends: what `sorethumb history` can and cannot tell you](trends.md)
- [Benchmark results](benchmarks.md): headline figures, full matrices, and the
  end-to-end scale benchmark.

## Running it

- [Scale guide](scale.md): time and memory measured at 10,000 to 1,000,000 rows, and
  what drives them.
- [Supported platforms](platforms.md): operating systems, Python versions, and what
  each CI lane runs.
- [Detector models](models.md): the six detectors, choosing an ensemble, and writing
  a detector plugin.

## Reference

- [CLI reference](cli_reference.md)
- [Configuration reference](configuration.md)
- [Configuration examples](configuration-examples.md)
- [API and stability policy](stability.md)

## Project

- [Security policy](../SECURITY.md)
- [Contributing](../CONTRIBUTING.md)
- [Releasing, rehearsing, and rolling back](releasing.md) (maintainer-only)

## Why sorethumb?

There are already good anomaly-detection libraries (PyOD ships far more detectors).
`sorethumb` is not competing on detector count. Its contribution is the surrounding
machinery those libraries leave to the user:

1. **Zero-configuration column handling** — profile, classify, encode, impute, derive.
2. **Ensemble scoring with percentile-calibrated scores.** Each run maps its raw
   detector scores onto [0, 1] against its own training distribution. Scores are
   comparable *across* runs only when a fitted calibrator is reused —
   `sorethumb score --from-run RUN_ID` — since a plain run self-calibrates.
3. **Per-record explanations** in original feature terms, labelled `exact`,
   `model_specific` or `heuristic`, each with the record of how it was produced.
4. **Run history you can reason about.** Models and calibrators are persisted,
   and `sorethumb history` reports four separate signals: population drift,
   score drift, threshold drift and anomaly counts, each with a verdict. A plain
   `backfill` refits each period, so score and threshold drift need a fixed
   anchor run (`history.anchor_run`); see the
   [trends guide](trends.md).
5. **Idempotent, resumable execution** with a completion ledger.

It runs on a single machine, uses Polars throughout, and has no dependency on Spark,
Databricks, or any cloud vendor.
