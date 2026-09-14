"""``BenchmarkRow`` builder for tests that check formatting/serialisation
logic without running the real benchmark harness (see
``tests/benchmark/test_benchmark_harness.py`` for tests that do)."""

from __future__ import annotations

from typing import Any

from sorethumb.evaluate.benchmark import BenchmarkRow


def make_benchmark_row(**overrides: Any) -> BenchmarkRow:
    defaults: dict = {
        "dataset": "d",
        "detector": "det",
        "n_rows": 100,
        "n_features": 4,
        "contamination": 0.05,
        "n_seeds": 3,
        "roc_auc": 0.9,
        "roc_auc_std": 0.01,
        "average_precision": 0.8,
        "average_precision_std": 0.02,
        "precision_at_k": 0.5,
        "precision_at_k_std": 0.0,
        "recall_at_k": 0.5,
        "recall_at_k_std": 0.0,
        "f1_at_contamination": 0.5,
        "f1_at_contamination_std": 0.0,
        "fit_seconds": 0.1,
        "score_seconds": 0.01,
    }
    defaults.update(overrides)
    return BenchmarkRow(**defaults)
