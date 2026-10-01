"""Real-dataset benchmark smoke: the guarded detectors still clear a loose
ROC-AUC floor on network-fetched real data (KDDCup99, Covtype), not just the
synthetic generators in ``test_accuracy_floors.py``.

Marked ``benchmark``, ``slow`` and ``network`` — deselected everywhere by
default and run explicitly on the weekly schedule::

    pytest -m benchmark

Offline behaviour: the fetch is attempted first. If it fails with a network
error the test is *skipped* with the underlying reason, so running the lane in
a sandbox without internet access does not report a failure that has nothing to
do with the code. Set ``SORETHUMB_REQUIRE_NETWORK=1`` (the scheduled
``full-nightly`` job does) to turn that skip into a hard failure, so a broken
fetch can never silently disable the real-data check where network is expected.

This is deliberately loose (floor 0.5, one seed, capped rows): it exists to
catch a broken pipeline or an inverted score direction on real, messy,
non-Gaussian data, not to set a tight accuracy bar. Tight, dataset-specific
floors belong in the full harness rebuild (see prompts/pre-release-plan.md
P3-2); this is the minimal genuine member of the ``slow``/``network`` lanes.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sorethumb_ml.evaluate.benchmark import BenchmarkRow

pytestmark = [pytest.mark.benchmark, pytest.mark.slow, pytest.mark.network]

_REAL_DATASETS = ("kddcup99_sa", "covtype")
_DETECTOR = "isolation_forest"
_MAX_ROWS = 20_000  # cap so a nightly run stays bounded even on covtype


_REQUIRE_NETWORK_ENV = "SORETHUMB_REQUIRE_NETWORK"


def _fetch_or_skip(dataset: str, cache_dir: Path) -> None:
    """Warm *cache_dir* with *dataset*, or skip/fail on a network error.

    ``run_benchmark`` swallows a load error and returns no rows, which a test
    can only report as an opaque ``0 == 1``. Fetching here first separates "the
    network is unavailable" from "the benchmark is broken". Only ``OSError`` (which
    covers ``URLError``, connection and timeout errors) counts as a network
    problem; anything else is a real defect and propagates.
    """
    from sorethumb_ml.evaluate.benchmark import DATASETS

    entry = next(d for d in DATASETS if d.name == dataset)
    try:
        entry.load(cache_dir, seed=0)
    except OSError as exc:
        reason = f"cannot fetch real dataset {dataset!r}: {exc}"
        if os.environ.get(_REQUIRE_NETWORK_ENV) == "1":
            pytest.fail(f"{reason} ({_REQUIRE_NETWORK_ENV}=1, so the network is required)")
        pytest.skip(reason)


@pytest.mark.parametrize("dataset", _REAL_DATASETS)
def test_detector_beats_random_on_real_dataset(
    dataset: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from sorethumb_ml.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cache_dir = tmp_path_factory.mktemp("real_bench_cache")
    _fetch_or_skip(dataset, cache_dir)
    cfg = BenchmarkConfig(
        dataset_names=[dataset],
        detector_names=[_DETECTOR],
        max_rows=_MAX_ROWS,
        cache_dir=cache_dir,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows: list[BenchmarkRow] = run_benchmark(cfg)
    assert len(rows) == 1, f"run_benchmark produced no row for {dataset} even though the fetch succeeded"
    row = rows[0]
    assert row.error is None, f"{_DETECTOR} on {dataset} errored: {row.error}"
    assert row.roc_auc > 0.5, (
        f"{_DETECTOR} on real dataset {dataset}: ROC-AUC {row.roc_auc:.4f} is not "
        "better than random — check for an inverted score direction or a broken fetch/label path."
    )
