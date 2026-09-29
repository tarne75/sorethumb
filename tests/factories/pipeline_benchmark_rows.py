"""``PipelineBenchmarkRow`` builder for tests that check matrix/formatting
logic without running the real (slow, subprocess-fitting) benchmark harness
-- see ``tests/benchmark/test_pipeline_benchmark_harness.py`` for tests that
do."""

from __future__ import annotations

from typing import Any

from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkRow


def make_pipeline_row(**overrides: Any) -> PipelineBenchmarkRow:
    defaults: dict = {
        "scenario": "point",
        "kind": "point",
        "ablation": "default",
        "n_seeds": 3,
        "n_train": 100,
        "n_holdout": 40,
        "review_budget": 0.05,
        "roc_auc": 0.9,
        "roc_auc_ci95": 0.01,
        "average_precision": 0.8,
        "average_precision_ci95": 0.02,
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "f1_at_contamination": 0.5,
        "fit_seconds": 0.1,
        "score_seconds": 0.01,
        "peak_memory_mb": 200.0,
    }
    defaults.update(overrides)
    return PipelineBenchmarkRow(**defaults)
