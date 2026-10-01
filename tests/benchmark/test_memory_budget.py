"""Subprocess peak-memory tests for the preflight budget check and the
matrix-lifetime fix it backs (dead ``full_space`` reference removed,
per-group float64 upcast made conditional on ``explain.enabled`` -- see
``sorethumb_ml._pipeline._group_feature_matrix`` and
``sorethumb_ml.features.build._peak_matrix_multiplier``).

Marked ``benchmark`` (real detector fitting + subprocess spawning, not fast/
deterministic), deselected by default::

    pytest -m benchmark

Each cell runs ``run_detection`` in an isolated, spawned subprocess so
``resource.getrusage(RUSAGE_SELF).ru_maxrss`` measures that attempt's own
real, OS-tracked peak RSS -- not a before/after snapshot delta, and
uncontaminated by pytest's/the test session's own accumulated state (same
rationale as ``evaluate/pipeline_benchmark.py``'s isolated-worker pattern,
reused here rather than imported from a benchmark-only module).

Data is written as Parquet, not CSV: a binary, typed, columnar format reads
back without CSV's text-parsing overhead (which otherwise dominates peak RSS
at these row counts and would make this a memory test of the CSV reader,
not of the feature-matrix lifetime this suite actually guards).

``explain.enabled=False`` throughout: this isolates the fit/score matrix
path the memory fix actually changed. With explain enabled (the default), TreeSHAP
attributes every row of the *full* matrix, uncapped -- correct for accuracy,
but it makes a large run far too slow for a memory test (the existing
``evaluate/pipeline_benchmark.py`` harness disables explain for the same
reason).
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.benchmark


def _peak_memory_mb() -> float | None:
    """Return this process's peak RSS so far, in MB, or None where unavailable (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # POSIX ru_maxrss units are platform-defined: Linux reports KB, macOS bytes.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return max_rss / divisor


def _run_isolated(target: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child_conn))
    proc.start()
    if parent_conn.poll(180):
        result = parent_conn.recv()
    else:
        result = {"peak_memory_mb": None, "outcome": "timeout", "error": "worker timed out after 180s"}
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
    return result


# ---------------------------------------------------------------------------
# full_space lifetime: peak memory must track one group's size, not the
# whole dataset's, across a multi-group run
# ---------------------------------------------------------------------------

_MG_N_GROUPS = 5
_MG_ROWS_PER_GROUP = 100_000
_MG_N_COLS = 100
_MG_WHOLE_DATASET_MB = _MG_N_GROUPS * _MG_ROWS_PER_GROUP * _MG_N_COLS * 4 / (1024 * 1024)
_MG_ONE_GROUP_MB = _MG_ROWS_PER_GROUP * _MG_N_COLS * 4 / (1024 * 1024)


def _make_multi_group_parquet(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_total = _MG_N_GROUPS * _MG_ROWS_PER_GROUP
    data: dict[str, list[float] | np.ndarray] = {
        "id": np.arange(n_total, dtype=np.float64),
        "group": np.repeat(np.arange(_MG_N_GROUPS), _MG_ROWS_PER_GROUP).astype(str),
    }
    for i in range(_MG_N_COLS):
        data[f"num_{i}"] = rng.normal(0.0, 1.0, n_total)
    pl.DataFrame(data).write_parquet(str(path))


def _multi_group_worker(data_path: str, ws_path: str, conn: Any) -> None:
    try:
        from sorethumb_ml import run_detection
        from sorethumb_ml.config import (
            ColumnsConfig,
            Config,
            DetectorConfig,
            ExplainConfig,
            RunConfig,
            SourceConfig,
        )

        cfg = Config(
            source=SourceConfig(uri=data_path, format="parquet"),
            run=RunConfig(workdir=ws_path, seed=0, max_memory_mb=8192),
            columns=ColumnsConfig(group_by=["group"]),
            detectors=[DetectorConfig(name="isolation_forest")],
            explain=ExplainConfig(enabled=False),
        )
        result = run_detection(cfg, no_report=True)
        conn.send(
            {
                "peak_memory_mb": _peak_memory_mb(),
                "n_succeeded": result.n_succeeded,
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- reported back to the parent, not raised in the child
        conn.send({"peak_memory_mb": _peak_memory_mb(), "n_succeeded": 0, "error": str(exc)[:500]})
    finally:
        conn.close()


def test_multi_group_run_completes_with_sane_peak_memory(tmp_path_factory) -> None:
    """Sanity-only real-subprocess check at a representative multi-group
    size: a run over _MG_N_GROUPS groups of ~_MG_ONE_GROUP_MB each (whole
    dataset ~_MG_WHOLE_DATASET_MB) must complete and must not blow past a
    generous ceiling -- catching a catastrophic (multi-GB-scale) regression,
    not asserting a tight per-byte budget. Real per-process memory here is
    dominated by things this test does not try to model precisely (df_raw
    itself staying alive at float64 for the whole run -- necessary, not a
    bug -- plus parquet decode and interpreter/import overhead), so this is
    deliberately loose.

    The *precise*, deterministic regression guard for the specific bug
    this dataset shape is designed to exercise (fit_features' full-dataset
    FeatureSpace staying reachable for the whole group loop) is
    tests/integration/test_feature_matrix_lifetime.py, which asserts via a
    weakref rather than an absolute memory number -- confirmed (by
    temporarily reintroducing the old code) to actually fail without the
    fix, which this RSS-based sanity check alone was too noisy to do
    reliably.
    """
    path = tmp_path_factory.mktemp("mg") / "groups.parquet"
    _make_multi_group_parquet(path)

    result = _run_isolated(_multi_group_worker, (str(path), str(tmp_path_factory.mktemp("mg_ws") / "ws")))
    assert result["error"] is None, result["error"]
    assert result["n_succeeded"] == _MG_N_GROUPS

    peak = result["peak_memory_mb"]
    if peak is not None:  # None on platforms without `resource` (Windows)
        # Observed ~1570MB for this dataset shape (500,000 rows x 100 cols,
        # 5 groups) on 2026-09-27; ceiling leaves wide headroom above that
        # for CI variance while still catching a genuine multi-GB blowup.
        assert peak < 3000, f"peak RSS {peak:.1f}MB far exceeds the expected order of magnitude"


# ---------------------------------------------------------------------------
# Preflight budget check: must reject before completing a run, not after
# ---------------------------------------------------------------------------

_REJECT_N_ROWS = 250_000
_REJECT_N_COLS = 150
_REJECT_BASE_MB = _REJECT_N_ROWS * _REJECT_N_COLS * 4 / (1024 * 1024)  # float32, no multiplier


def _make_wide_parquet(path: Path, n_rows: int, n_cols: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data: dict[str, list[float] | np.ndarray] = {"id": np.arange(n_rows, dtype=np.float64)}
    for i in range(n_cols):
        data[f"num_{i}"] = rng.normal(0.0, 1.0, n_rows)
    pl.DataFrame(data).write_parquet(str(path))


def _budget_worker(data_path: str, ws_path: str, max_memory_mb: int, conn: Any) -> None:
    try:
        from sorethumb_ml import run_detection
        from sorethumb_ml.config import Config, DetectorConfig, ExplainConfig, RunConfig, SourceConfig
        from sorethumb_ml.errors import MemoryBudgetError

        cfg = Config(
            source=SourceConfig(uri=data_path, format="parquet"),
            run=RunConfig(workdir=ws_path, seed=0, max_memory_mb=max_memory_mb),
            detectors=[DetectorConfig(name="isolation_forest")],
            explain=ExplainConfig(enabled=False),
        )
        try:
            run_detection(cfg, no_report=True)
            outcome = "completed"
        except MemoryBudgetError:
            outcome = "rejected"
        conn.send({"peak_memory_mb": _peak_memory_mb(), "outcome": outcome, "error": None})
    except Exception as exc:  # noqa: BLE001 -- reported back to the parent, not raised in the child
        conn.send({"peak_memory_mb": _peak_memory_mb(), "outcome": "crashed", "error": str(exc)[:500]})
    finally:
        conn.close()


@pytest.fixture(scope="module")
def reject_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("memory_budget_reject") / "wide.parquet"
    _make_wide_parquet(path, _REJECT_N_ROWS, _REJECT_N_COLS)
    return path


def test_budget_rejects_before_completing_a_run(reject_parquet: Path, tmp_path: Path) -> None:
    """A budget the *expanded* estimate correctly flags -- but the old,
    single-matrix-only estimate would have missed -- must be rejected via
    MemoryBudgetError, not silently allowed through to a full fit.

    Chosen budget (256, run.max_memory_mb's pydantic-enforced floor -- the
    smallest a real config can ever set) sits strictly between the old and
    new estimates: below _peak_matrix_multiplier's x2 projection (explain
    disabled here), above the bare single-matrix size the previous check
    alone computed -- so this specifically exercises the expanded estimate
    (a build without the fix would have seen "completed", not
    "rejected", at this budget).
    """
    budget = 256
    assert _REJECT_BASE_MB * 1 < budget < _REJECT_BASE_MB * 2, (
        f"test assumption violated: base={_REJECT_BASE_MB:.1f}MB, x2={_REJECT_BASE_MB * 2:.1f}MB "
        f"no longer straddle the {budget}MB floor -- adjust _REJECT_N_ROWS/_REJECT_N_COLS"
    )
    result = _run_isolated(_budget_worker, (str(reject_parquet), str(tmp_path / "ws"), budget))
    assert result["error"] is None, result["error"]
    assert result["outcome"] == "rejected", (
        f"expected MemoryBudgetError, got outcome={result['outcome']!r} (peak={result['peak_memory_mb']})"
    )
