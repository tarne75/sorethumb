"""Real-dataset benchmark smoke: the guarded detectors still clear a loose
ROC-AUC floor on network-fetched real data (KDDCup99, Covtype), not just the
synthetic generators in ``test_accuracy_floors.py``.

Marked ``benchmark``, ``slow`` and ``network`` — deselected everywhere by
default and run explicitly on the weekly schedule::

    pytest -m benchmark

This is deliberately loose (floor 0.5, one seed, capped rows): it exists to
catch a broken pipeline or an inverted score direction on real, messy,
non-Gaussian data, not to set a tight accuracy bar. Tight, dataset-specific
floors belong in the full harness rebuild (see prompts/pre-release-plan.md
P3-2); this is the minimal genuine member of the ``slow``/``network`` lanes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sorethumb.evaluate.benchmark import BenchmarkRow

pytestmark = [pytest.mark.benchmark, pytest.mark.slow, pytest.mark.network]

_REAL_DATASETS = ("kddcup99_sa", "covtype")
_DETECTOR = "isolation_forest"
_MAX_ROWS = 20_000  # cap so a nightly run stays bounded even on covtype


@pytest.mark.parametrize("dataset", _REAL_DATASETS)
def test_detector_beats_random_on_real_dataset(
    dataset: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=[dataset],
        detector_names=[_DETECTOR],
        max_rows=_MAX_ROWS,
        cache_dir=tmp_path_factory.mktemp("real_bench_cache"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows: list[BenchmarkRow] = run_benchmark(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row.error is None, f"{_DETECTOR} on {dataset} errored: {row.error}"
    assert row.roc_auc > 0.5, (
        f"{_DETECTOR} on real dataset {dataset}: ROC-AUC {row.roc_auc:.4f} is not "
        "better than random — check for an inverted score direction or a broken fetch/label path."
    )
