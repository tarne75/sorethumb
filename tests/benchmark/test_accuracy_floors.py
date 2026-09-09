"""Accuracy-floor benchmark: every guarded detector must clear a committed
ROC-AUC floor on the network-free synthetic datasets.

Marked ``benchmark``, so it is deselected by default and run explicitly::

    pytest -m benchmark

This is the regression guard the Opus review asked for ("assert every default
detector exceeds an AUC of 0.5"). In particular it locks in the
``kmeans_distance`` CBLOF fix, which previously scored the planted anomaly
cluster as *normal* (ROC-AUC ~0.0002 on ``synthetic_highd``).

Floors were set from an observed run (2026-09-09) with margin below the measured
value. Ratchet a floor UP when a detector reliably improves; do not lower one
without a documented reason in this file.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sorethumb.evaluate.benchmark import BenchmarkRow

pytestmark = pytest.mark.benchmark

# Network-free synthetic datasets from the benchmark registry.
_SYNTHETIC_DATASETS = ("synthetic_gaussian", "synthetic_highd")

# Per-detector ROC-AUC floor, applied on every synthetic dataset. Observed
# values on the 2026-09-09 reference run are in the trailing comments.
_ROC_AUC_FLOORS: dict[str, float] = {
    "isolation_forest": 0.95,  # observed 1.00 / 1.00
    "kmeans_distance": 0.95,  # observed 1.00 / 1.00  — CBLOF-fix regression guard
    "one_class_svm": 0.80,  # observed 0.91 / 0.92
    "ecod": 0.95,  # observed 1.00 / 1.00
    "hbos": 0.95,  # observed 1.00 / 1.00
    # lof is deliberately excluded. The planted anomalies form their own dense
    # blob, so local-density scoring rates them "normal within their own
    # neighbourhood" (observed ROC-AUC ~0.22). That is a documented LOF
    # limitation (docs/models.md), not a regression; guarding it meaningfully
    # needs a varying-density dataset the synthetic registry does not provide.
}

_DEFAULT_ENSEMBLE = ("isolation_forest", "kmeans_distance", "one_class_svm")


@pytest.fixture(scope="session")
def benchmark_rows(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str], BenchmarkRow]:
    """Run the synthetic benchmark once for every guarded detector."""
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=list(_SYNTHETIC_DATASETS),
        detector_names=list(_ROC_AUC_FLOORS),
        cache_dir=tmp_path_factory.mktemp("bench_cache"),
    )
    with warnings.catch_warnings():
        # A warning is not the signal this test guards; ROC-AUC is.
        warnings.simplefilter("ignore")
        rows = run_benchmark(cfg)
    return {(r.dataset, r.detector): r for r in rows}


@pytest.mark.parametrize(
    ("dataset", "detector"),
    [(ds, det) for ds in _SYNTHETIC_DATASETS for det in _ROC_AUC_FLOORS],
)
def test_detector_clears_roc_auc_floor(
    benchmark_rows: dict[tuple[str, str], BenchmarkRow],
    dataset: str,
    detector: str,
) -> None:
    row = benchmark_rows[dataset, detector]
    assert row.error is None, f"{detector} on {dataset} errored: {row.error}"
    floor = _ROC_AUC_FLOORS[detector]
    assert row.roc_auc >= floor, (
        f"{detector} on {dataset}: ROC-AUC {row.roc_auc:.4f} < floor {floor:.2f}. "
        "If this is a real change, update _ROC_AUC_FLOORS in this file with a note."
    )


def test_default_ensemble_beats_random(
    benchmark_rows: dict[tuple[str, str], BenchmarkRow],
) -> None:
    """The Opus-review requirement, stated plainly and independent of the
    (tighter, tunable) per-detector floors: the shipped default ensemble must
    beat random on planted anomalies."""
    for dataset in _SYNTHETIC_DATASETS:
        for detector in _DEFAULT_ENSEMBLE:
            auc = benchmark_rows[dataset, detector].roc_auc
            assert auc > 0.5, f"{detector} on {dataset}: ROC-AUC {auc:.4f} is not better than random"
