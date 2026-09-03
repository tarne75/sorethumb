"""Unit tests for M9: evaluate/metrics.py and evaluate/benchmark.py."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from sorethumb.evaluate.metrics import Metrics, evaluate_scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_scores(n: int = 200, contamination: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) where the detector is perfect."""
    rng = np.random.default_rng(0)
    n_anom = max(1, round(n * contamination))
    labels = np.zeros(n, dtype=int)
    labels[:n_anom] = 1
    perm = rng.permutation(n)
    labels = labels[perm]
    # Anomalies get score 1.0, normals get 0.0 — perfect ranking
    scores = labels.astype(float)
    return scores, labels


def _random_scores(n: int = 200, contamination: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_anom = max(1, round(n * contamination))
    labels = np.concatenate([np.ones(n_anom), np.zeros(n - n_anom)]).astype(int)
    scores = rng.random(n)
    return scores, labels


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------


def test_metrics_fields_present():
    m = Metrics(
        roc_auc=0.9,
        average_precision=0.8,
        precision_at_k=0.7,
        recall_at_k=0.6,
        f1_at_contamination=0.65,
        contamination_used=0.1,
        n_positives=20,
        n_total=200,
        k_used=20,
    )
    assert m.roc_auc == pytest.approx(0.9)
    assert m.average_precision == pytest.approx(0.8)
    assert m.n_positives == 20
    assert m.n_total == 200


def test_metrics_str_contains_key_fields():
    m = Metrics(
        roc_auc=0.9,
        average_precision=0.8,
        precision_at_k=0.7,
        recall_at_k=0.6,
        f1_at_contamination=0.65,
        contamination_used=0.1,
        n_positives=20,
        n_total=200,
        k_used=20,
    )
    s = str(m)
    assert "ROC-AUC" in s
    assert "AP" in s
    assert "F1" in s


# ---------------------------------------------------------------------------
# evaluate_scores: basic correctness
# ---------------------------------------------------------------------------


def test_evaluate_scores_returns_metrics():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels)
    assert isinstance(m, Metrics)


def test_evaluate_scores_roc_auc_range():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels)
    assert 0.0 <= m.roc_auc <= 1.0


def test_evaluate_scores_ap_range():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels)
    assert 0.0 <= m.average_precision <= 1.0


def test_evaluate_scores_precision_at_k_range():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels, contamination=0.1)
    assert 0.0 <= m.precision_at_k <= 1.0


def test_evaluate_scores_recall_at_k_range():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels, contamination=0.1)
    assert 0.0 <= m.recall_at_k <= 1.0


def test_evaluate_scores_f1_range():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels)
    assert 0.0 <= m.f1_at_contamination <= 1.0


def test_evaluate_scores_k_used_matches_contamination():
    scores, labels = _random_scores(n=200, contamination=0.1)
    m = evaluate_scores(scores, labels, contamination=0.1)
    expected_k = max(1, round(200 * 0.1))
    assert m.k_used == expected_k


def test_evaluate_scores_contamination_stored():
    scores, labels = _random_scores()
    m = evaluate_scores(scores, labels, contamination=0.15)
    assert m.contamination_used == pytest.approx(0.15)


def test_evaluate_scores_n_total():
    scores, labels = _random_scores(n=300)
    m = evaluate_scores(scores, labels)
    assert m.n_total == 300


def test_evaluate_scores_n_positives():
    scores, labels = _random_scores(n=200, contamination=0.1)
    m = evaluate_scores(scores, labels)
    assert m.n_positives == int(labels.sum())


# ---------------------------------------------------------------------------
# Perfect detector achieves maximum scores
# ---------------------------------------------------------------------------


def test_perfect_detector_roc_auc_is_1():
    scores, labels = _perfect_scores()
    m = evaluate_scores(scores, labels)
    assert m.roc_auc == pytest.approx(1.0, abs=1e-6)


def test_perfect_detector_ap_is_1():
    scores, labels = _perfect_scores()
    m = evaluate_scores(scores, labels)
    assert m.average_precision == pytest.approx(1.0, abs=1e-6)


def test_perfect_detector_precision_at_k_is_1():
    scores, labels = _perfect_scores(contamination=0.1)
    m = evaluate_scores(scores, labels, contamination=0.1)
    assert m.precision_at_k == pytest.approx(1.0, abs=1e-6)


def test_perfect_detector_recall_at_k_is_1():
    scores, labels = _perfect_scores(contamination=0.1)
    m = evaluate_scores(scores, labels, contamination=0.1)
    assert m.recall_at_k == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge case: all labels the same
# ---------------------------------------------------------------------------


def test_all_normal_returns_zero_metrics():
    scores = np.random.default_rng(0).random(100)
    labels = np.zeros(100, dtype=int)
    m = evaluate_scores(scores, labels)
    assert m.roc_auc == 0.0
    assert m.average_precision == 0.0


def test_all_anomaly_returns_zero_metrics():
    scores = np.random.default_rng(0).random(100)
    labels = np.ones(100, dtype=int)
    m = evaluate_scores(scores, labels)
    assert m.roc_auc == 0.0
    assert m.average_precision == 0.0


def test_all_same_returns_n_positives_and_n_total():
    scores = np.ones(50, dtype=float)
    labels = np.zeros(50, dtype=int)
    m = evaluate_scores(scores, labels)
    assert m.n_positives == 0
    assert m.n_total == 50


# ---------------------------------------------------------------------------
# List / array-like inputs
# ---------------------------------------------------------------------------


def test_list_inputs_accepted():
    scores = [0.1, 0.9, 0.5, 0.8, 0.2]
    labels = [0, 1, 0, 1, 0]
    m = evaluate_scores(scores, labels)
    assert isinstance(m, Metrics)


def test_float32_inputs_accepted():
    scores, labels = _random_scores()
    m = evaluate_scores(scores.astype(np.float32), labels.astype(np.int32))
    assert isinstance(m, Metrics)


# ---------------------------------------------------------------------------
# benchmark module: synthetic datasets and formatters
# ---------------------------------------------------------------------------


def test_synthetic_dataset_load():
    from sorethumb.evaluate.benchmark import DATASETS, SyntheticEntry

    synth = next(d for d in DATASETS if isinstance(d, SyntheticEntry))
    X, y = synth.load(Path("/tmp"))
    assert X.ndim == 2
    assert y.ndim == 1
    assert len(X) == len(y)
    assert set(y.tolist()) == {0, 1}


def test_synthetic_dataset_contamination_matches():
    from sorethumb.evaluate.benchmark import SyntheticEntry

    ds = SyntheticEntry(
        name="t",
        description="",
        licence="",
        provenance="",
        contamination=50 / 1050,
        n_normal=1000,
        n_anomaly=50,
        n_features=4,
    )
    X, y = ds.load(Path("/tmp"))
    assert y.sum() == 50
    assert len(X) == 1050


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


def test_to_markdown_empty():
    from sorethumb.evaluate.benchmark import to_markdown

    assert "No benchmark" in to_markdown([])


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


def test_benchmark_config_defaults():
    from sorethumb.evaluate.benchmark import BenchmarkConfig

    cfg = BenchmarkConfig()
    assert cfg.dataset_names == []
    assert cfg.detector_names == []
    assert cfg.seed == 0
    assert cfg.max_rows == 0


def test_dataset_registry_has_synthetic():
    from sorethumb.evaluate.benchmark import DATASETS

    names = [d.name for d in DATASETS]
    assert "synthetic_gaussian" in names
    assert "synthetic_highd" in names


def test_dataset_entry_has_licence():
    from sorethumb.evaluate.benchmark import DATASETS

    for ds in DATASETS:
        assert ds.licence, f"{ds.name} has no licence"
        assert ds.provenance, f"{ds.name} has no provenance"


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
