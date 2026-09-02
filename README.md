# sorethumb

**Unsupervised anomaly detection for tabular data.**

`sorethumb` takes a dataset it knows nothing about, works out how to treat every
column, fits an ensemble of detectors, ranks the records that stand out, and explains
*why* each one stands out in terms of the original columns.

> Documentation and full benchmark results coming in v1.0.

## Installation

```bash
pip install sorethumb
```

## Quick start

```python
from sorethumb import run_detection, Config

# Coming in M1+
```

## Why sorethumb?

There are already good anomaly-detection libraries (PyOD ships far more detectors).
`sorethumb` is not competing on detector count. Its contribution is the surrounding
machinery those libraries leave to the user:

1. **Zero-configuration column handling** — profile, classify, encode, impute, derive.
2. **Ensemble scoring with calibrated, comparable scores** across runs.
3. **Per-record explanations** in original feature terms, labelled exact or heuristic.
4. **Run history that is actually valid** — persisted models and score calibration.
5. **Idempotent, resumable execution** with a completion ledger.

## Honest limitations

- `max_memory_mb` is advisory, not enforced — Python cannot impose a hard ceiling.
- The composite score is an interpretable ranking score, not a probability of anomaly.
- Only Isolation Forest yields exact (TreeSHAP) attributions; all others are heuristic.
- Self-calibrated runs are not comparable to each other — use `score --from-run` for trends.
- Unsupervised anomaly ≠ the thing you care about. The library ranks statistical oddity;
  whether an odd record is *interesting* is a domain judgement it cannot make.
