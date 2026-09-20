"""Pipeline-benchmark harness tests: real fitting through sorethumb's full feature
pipeline, in isolated subprocesses. Deselected by default; run explicitly::

    pytest -m benchmark
"""

from __future__ import annotations

import csv
import io

import pytest

pytestmark = pytest.mark.benchmark


def test_run_pipeline_benchmark_single_scenario():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row.scenario == "point"
    assert row.error is None
    assert 0.0 <= row.roc_auc <= 1.0
    assert row.n_train > 0
    assert row.n_holdout > 0


def test_run_pipeline_benchmark_multiple_seeds_reports_ci():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=3, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    row = rows[0]
    assert row.n_seeds == 3
    # A real spread across seeds should produce a non-trivial CI half-width
    # most of the time; assert it is at least well-defined (non-negative).
    assert row.roc_auc_ci95 >= 0.0
    assert row.average_precision_ci95 >= 0.0


def test_run_pipeline_benchmark_multiple_ablations():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"],
        ablation_names=["default", "pca_on"],
        n_seeds=1,
        include_baselines=False,
        include_swamping=False,
    )
    rows = run_pipeline_benchmark(cfg)
    ablation_names = {r.ablation for r in rows}
    assert ablation_names == {"default", "pca_on"}


def test_run_pipeline_benchmark_includes_sklearn_baselines():
    from sorethumb.evaluate.pipeline_benchmark import (
        _SKLEARN_BASELINE_DETECTORS,
        PipelineBenchmarkConfig,
        run_pipeline_benchmark,
    )

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"],
        n_seeds=1,
        include_baselines=True,
        include_swamping=False,
    )
    rows = run_pipeline_benchmark(cfg)
    baseline_ablations = {r.ablation for r in rows if r.ablation.startswith("sklearn:")}
    assert baseline_ablations == {f"sklearn:{d}" for d in _SKLEARN_BASELINE_DETECTORS}
    for row in rows:
        if row.ablation.startswith("sklearn:"):
            assert row.error is None, row.error
            assert 0.0 <= row.roc_auc <= 1.0


def test_run_pipeline_benchmark_includes_swamping():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"],
        n_seeds=1,
        include_baselines=False,
        include_swamping=True,
    )
    rows = run_pipeline_benchmark(cfg)
    swamping_rows = [r for r in rows if r.scenario == "swamping"]
    assert len(swamping_rows) == 1
    assert swamping_rows[0].error is None


def test_run_pipeline_benchmark_reports_peak_memory():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    # None only on platforms without the `resource` module (Windows); this
    # suite runs on POSIX CI, so a real subprocess-measured value is expected.
    assert rows[0].peak_memory_mb is not None
    assert rows[0].peak_memory_mb > 0


def test_to_markdown_returns_table():
    from sorethumb.evaluate.pipeline_benchmark import (
        PipelineBenchmarkConfig,
        run_pipeline_benchmark,
        to_markdown,
    )

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    md = to_markdown(rows)
    assert "| " in md
    assert "point" in md
    assert "roc_auc" in md


def test_to_csv_parseable():
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark, to_csv

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    csv_text = to_csv(rows)
    reader = csv.DictReader(io.StringIO(csv_text))
    parsed = list(reader)
    assert len(parsed) == 1
    assert "roc_auc" in parsed[0]
    assert "peak_memory_mb" in parsed[0]


def test_write_outputs_creates_files(tmp_path):
    from sorethumb.evaluate.pipeline_benchmark import (
        PipelineBenchmarkConfig,
        run_pipeline_benchmark,
        write_outputs,
    )

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    md_path, csv_path = write_outputs(rows, tmp_path / "out")
    assert md_path.exists()
    assert csv_path.exists()
    assert md_path.stat().st_size > 0


def test_inject_into_readme_idempotent(tmp_path):
    from sorethumb.evaluate.pipeline_benchmark import (
        _RESULTS_MARKER_END,
        _RESULTS_MARKER_START,
        PipelineBenchmarkConfig,
        inject_into_readme,
        run_pipeline_benchmark,
    )

    readme = tmp_path / "README.md"
    readme.write_text(f"# Proj\n\n{_RESULTS_MARKER_START}\n{_RESULTS_MARKER_END}\n", encoding="utf-8")

    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"], n_seeds=1, include_baselines=False, include_swamping=False
    )
    rows = run_pipeline_benchmark(cfg)
    assert inject_into_readme(rows, readme) is True
    content_after_first = readme.read_text(encoding="utf-8")
    assert inject_into_readme(rows, readme) is False
    assert readme.read_text(encoding="utf-8") == content_after_first


def test_scenario_error_is_captured_not_raised():
    """An unknown scenario name inside the worker must come back as an error
    row, not crash the whole batch."""
    from sorethumb.evaluate.pipeline_benchmark import DEFAULT_ABLATION, _pipeline_batch_worker, _run_isolated

    batch = _run_isolated(_pipeline_batch_worker, ("does_not_exist", False, DEFAULT_ABLATION, [0]))
    assert batch["error"] is not None
    assert batch["per_seed"] == []
