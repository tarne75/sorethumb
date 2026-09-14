"""Benchmark-harness tests that fit real detectors via ``run_benchmark``.

Moved out of ``tests/unit/test_evaluate.py`` (P1-1): unlike the rest of that
module, these exercise actual detector fitting on synthetic datasets, so they
do not belong in the fast/deterministic unit lane. Deselected by default; run
explicitly::

    pytest -m benchmark
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.benchmark


def test_run_benchmark_synthetic_only(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    assert len(rows) == 1
    assert rows[0].dataset == "synthetic_gaussian"
    assert rows[0].detector == "isolation_forest"
    assert rows[0].n_rows == 200  # capped by max_rows
    assert rows[0].error is None


def test_run_benchmark_all_detectors(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        max_rows=300,
    )
    rows = run_benchmark(cfg)
    detector_names = {r.detector for r in rows}
    assert "isolation_forest" in detector_names
    assert "kmeans_distance" in detector_names


def test_run_benchmark_metrics_in_range(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=500,
    )
    rows = run_benchmark(cfg)
    row = rows[0]
    assert 0.0 <= row.roc_auc <= 1.0
    assert 0.0 <= row.average_precision <= 1.0
    assert row.fit_seconds >= 0
    assert row.score_seconds >= 0


def test_to_markdown_returns_table(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark, to_markdown

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    md = to_markdown(rows)
    assert "| " in md
    assert "isolation_forest" in md
    assert "roc_auc" in md


def test_to_csv_parseable(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark, to_csv

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    csv_text = to_csv(rows)
    reader = csv.DictReader(io.StringIO(csv_text))
    parsed = list(reader)
    assert len(parsed) == 1
    assert "roc_auc" in parsed[0]
    assert "average_precision" in parsed[0]


def test_write_outputs_creates_files(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark, write_outputs

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    md_path, csv_path = write_outputs(rows, tmp_path / "out")
    assert md_path.exists()
    assert csv_path.exists()
    assert md_path.stat().st_size > 0
    assert csv_path.stat().st_size > 0


def test_inject_into_readme_no_markers(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, inject_into_readme, run_benchmark

    readme = tmp_path / "README.md"
    readme.write_text("# My project\n", encoding="utf-8")

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    result = inject_into_readme(rows, readme)
    assert result is False  # no markers → no modification


def test_inject_into_readme_with_markers(tmp_path: Path):
    from sorethumb.evaluate.benchmark import (
        _RESULTS_MARKER_END,
        _RESULTS_MARKER_START,
        BenchmarkConfig,
        inject_into_readme,
        run_benchmark,
    )

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# My project\n\n{_RESULTS_MARKER_START}\nold content\n{_RESULTS_MARKER_END}\n\nEnd.\n",
        encoding="utf-8",
    )

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    result = inject_into_readme(rows, readme)
    assert result is True
    updated = readme.read_text(encoding="utf-8")
    assert "isolation_forest" in updated
    assert "old content" not in updated
    assert "End." in updated


def test_inject_into_readme_idempotent(tmp_path: Path):
    from sorethumb.evaluate.benchmark import (
        _RESULTS_MARKER_END,
        _RESULTS_MARKER_START,
        BenchmarkConfig,
        inject_into_readme,
        run_benchmark,
    )

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# My project\n\n{_RESULTS_MARKER_START}\n{_RESULTS_MARKER_END}\n",
        encoding="utf-8",
    )

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    inject_into_readme(rows, readme)
    content_after_first = readme.read_text(encoding="utf-8")

    result = inject_into_readme(rows, readme)
    assert result is False  # second injection is a no-op
    assert readme.read_text(encoding="utf-8") == content_after_first


def test_benchmark_row_as_dict_has_all_metric_keys(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    d = rows[0].as_dict()
    for key in ("roc_auc", "average_precision", "precision_at_k", "recall_at_k", "fit_seconds"):
        assert key in d, f"Missing key: {key}"


def test_run_benchmark_row_has_no_peak_rss_mb(tmp_path: Path):
    from sorethumb.evaluate.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig(
        dataset_names=["synthetic_gaussian"],
        detector_names=["isolation_forest"],
        cache_dir=tmp_path / "cache",
        max_rows=200,
    )
    rows = run_benchmark(cfg)
    assert "peak_rss_mb" not in rows[0].as_dict()
