"""Unit tests for M9: evaluate/metrics.py and evaluate/benchmark.py.

Detector-fitting/benchmark-harness tests that call ``run_benchmark`` live in
``tests/benchmark/test_benchmark_harness.py`` (moved out in P1-1) since they
are not fast/deterministic in the way the rest of this module is.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sorethumb.evaluate.metrics import Metrics, evaluate_scores
from tests.factories.benchmark_rows import make_benchmark_row

pytestmark = pytest.mark.unit

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


def test_evaluate_scores_on_random_input():
    """One evaluate_scores() call, one coherent set of assertions over every
    field, instead of nine near-identical calls each checking one field."""
    n, contamination = 300, 0.15
    scores, labels = _random_scores(n=n, contamination=contamination)
    m = evaluate_scores(scores, labels, contamination=contamination)

    assert isinstance(m, Metrics)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.average_precision <= 1.0
    assert 0.0 <= m.precision_at_k <= 1.0
    assert 0.0 <= m.recall_at_k <= 1.0
    assert 0.0 <= m.f1_at_contamination <= 1.0
    assert m.contamination_used == pytest.approx(contamination)
    assert m.n_total == n
    assert m.n_positives == int(labels.sum())
    assert m.k_used == max(1, round(n * contamination))


# ---------------------------------------------------------------------------
# Perfect detector achieves maximum scores
# ---------------------------------------------------------------------------


def test_perfect_detector_maximises_every_metric():
    scores, labels = _perfect_scores(contamination=0.1)
    m = evaluate_scores(scores, labels, contamination=0.1)
    assert m.roc_auc == pytest.approx(1.0, abs=1e-6)
    assert m.average_precision == pytest.approx(1.0, abs=1e-6)
    assert m.precision_at_k == pytest.approx(1.0, abs=1e-6)
    assert m.recall_at_k == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge case: all labels the same
# ---------------------------------------------------------------------------


def test_all_normal_returns_nan_metrics():
    # ROC-AUC/AP are mathematically undefined with a single class present.
    # NaN, not 0.0 -- 0.0 reads as "worse than random" (a real, terrible
    # score) and would silently drag down any mean/std computed over it.
    scores = np.random.default_rng(0).random(100)
    labels = np.zeros(100, dtype=int)
    m = evaluate_scores(scores, labels)
    assert math.isnan(m.roc_auc)
    assert math.isnan(m.average_precision)


def test_all_anomaly_returns_nan_metrics():
    scores = np.random.default_rng(0).random(100)
    labels = np.ones(100, dtype=int)
    m = evaluate_scores(scores, labels)
    assert math.isnan(m.roc_auc)
    assert math.isnan(m.average_precision)


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
# Input validation (P2-9): fail closed rather than silently corrupt a metric
# or crash deep inside sklearn with a confusing error.
# ---------------------------------------------------------------------------


def test_evaluate_scores_rejects_2d_scores():
    scores = np.zeros((10, 2))
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="1-D"):
        evaluate_scores(scores, labels)


def test_evaluate_scores_rejects_2d_labels():
    scores = np.zeros(10)
    labels = np.zeros((10, 1), dtype=int)
    with pytest.raises(ValueError, match="1-D"):
        evaluate_scores(scores, labels)


def test_evaluate_scores_rejects_mismatched_length():
    scores = np.zeros(10)
    labels = np.zeros(9, dtype=int)
    with pytest.raises(ValueError, match="equal length"):
        evaluate_scores(scores, labels)


def test_evaluate_scores_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_scores(np.array([]), np.array([]))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_evaluate_scores_rejects_non_finite_scores(bad_value):
    scores = np.array([0.1, 0.2, bad_value, 0.4])
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="finite"):
        evaluate_scores(scores, labels)


def test_evaluate_scores_rejects_non_binary_labels():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 2, 1])
    with pytest.raises(ValueError, match="binary"):
        evaluate_scores(scores, labels)


@pytest.mark.parametrize("bad_contamination", [0.0, 1.0, -0.1, 1.5])
def test_evaluate_scores_rejects_out_of_range_contamination(bad_contamination):
    scores, labels = _random_scores()
    with pytest.raises(ValueError, match="contamination"):
        evaluate_scores(scores, labels, contamination=bad_contamination)


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


def test_to_markdown_empty():
    from sorethumb.evaluate.benchmark import to_markdown

    assert "No benchmark" in to_markdown([])


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


def test_covtype_loader_restricts_to_class_2_vs_class_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """P2-9: the documented ODDS formulation for Covtype is class 2 (normal,
    majority within this pair) versus class 4 (anomaly, rare) -- every other
    cover type (1, 3, 5, 6, 7) must be dropped from the dataset entirely, not
    silently folded into "normal" by a bare `target == 4` comparison."""
    from sorethumb.evaluate.benchmark import DATASETS

    covtype_entry = next(d for d in DATASETS if d.name == "covtype")

    fake_data = np.arange(7 * 3, dtype=np.float64).reshape(7, 3)
    fake_target = np.array([1, 2, 2, 3, 4, 4, 5])

    class _FakeBunch:
        data = fake_data
        target = fake_target

    monkeypatch.setattr("sklearn.datasets.fetch_covtype", lambda **_kwargs: _FakeBunch())

    X, y = covtype_entry.load(tmp_path)

    assert X.shape[0] == 4  # only the 4 rows with target in {2, 4} survive
    assert set(y.tolist()) == {0, 1}
    assert y.sum() == 2  # the two target==4 rows
    np.testing.assert_array_equal(X, fake_data[[1, 2, 4, 5]])
    np.testing.assert_array_equal(y, [0, 0, 1, 1])


# ---------------------------------------------------------------------------
# P0-7: honest benchmark evidence — NaN rendering, no peak_rss_mb, metadata
# ---------------------------------------------------------------------------


def test_benchmark_row_nan_roc_auc_renders_na_in_display_dict():
    row = make_benchmark_row(roc_auc=float("nan"), roc_auc_std=float("nan"))
    d = row.as_display_dict()
    assert d["roc_auc"] == "n/a"


def test_benchmark_row_nan_average_precision_renders_na_in_csv_dict():
    row = make_benchmark_row(average_precision=float("nan"), average_precision_std=float("nan"))
    d = row.as_dict()
    assert d["average_precision"] == "n/a"
    assert d["average_precision_std"] == "n/a"


def test_benchmark_row_nan_does_not_render_as_literal_nan_string():
    from sorethumb.evaluate.benchmark import to_markdown

    row = make_benchmark_row(roc_auc=float("nan"), roc_auc_std=float("nan"))
    md = to_markdown([row])
    assert "nan" not in md.lower().replace("n/a", "")


def test_benchmark_row_has_no_peak_rss_mb_field():
    from sorethumb.evaluate.benchmark import BenchmarkRow

    assert "peak_rss_mb" not in BenchmarkRow.__dataclass_fields__


def test_benchmark_table_cols_has_no_peak_rss_mb():
    from sorethumb.evaluate.benchmark import _TABLE_COLS

    assert "peak_rss_mb" not in _TABLE_COLS


def test_generate_metadata_fields_populated():
    from sorethumb.evaluate.benchmark import generate_metadata

    meta = generate_metadata()
    assert meta.generated_at
    assert meta.platform
    assert meta.python_version
    assert meta.sorethumb_version
    assert meta.numpy_version
    assert meta.scipy_version
    assert meta.scikit_learn_version


def test_generate_metadata_matches_installed_versions():
    import numpy as np
    import scipy
    import sklearn

    from sorethumb import __version__ as sorethumb_version
    from sorethumb.evaluate.benchmark import generate_metadata

    meta = generate_metadata()
    assert meta.sorethumb_version == sorethumb_version
    assert meta.numpy_version == np.__version__
    assert meta.scipy_version == scipy.__version__
    assert meta.scikit_learn_version == sklearn.__version__


def test_to_markdown_with_metadata_includes_summary_line():
    from sorethumb.evaluate.benchmark import generate_metadata, to_markdown

    row = make_benchmark_row()
    meta = generate_metadata()
    md = to_markdown([row], meta)
    assert meta.sorethumb_version in md
    assert meta.platform in md


def test_to_markdown_without_metadata_omits_summary_line():
    from sorethumb.evaluate.benchmark import to_markdown

    row = make_benchmark_row()
    md = to_markdown([row])
    assert "Generated" not in md


def test_inject_into_readme_with_metadata(tmp_path: Path):
    from sorethumb.evaluate.benchmark import (
        _RESULTS_MARKER_END,
        _RESULTS_MARKER_START,
        generate_metadata,
        inject_into_readme,
    )

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# My project\n\n{_RESULTS_MARKER_START}\n{_RESULTS_MARKER_END}\n",
        encoding="utf-8",
    )
    row = make_benchmark_row()
    meta = generate_metadata()
    result = inject_into_readme([row], readme, meta)
    assert result is True
    updated = readme.read_text(encoding="utf-8")
    assert meta.sorethumb_version in updated


def test_write_outputs_with_metadata_creates_json(tmp_path: Path):
    import json

    from sorethumb.evaluate.benchmark import generate_metadata, write_outputs

    row = make_benchmark_row()
    meta = generate_metadata()
    write_outputs([row], tmp_path, meta)
    meta_path = tmp_path / "benchmark_metadata.json"
    assert meta_path.exists()
    parsed = json.loads(meta_path.read_text(encoding="utf-8"))
    assert parsed["sorethumb_version"] == meta.sorethumb_version


def test_write_outputs_without_metadata_skips_json(tmp_path: Path):
    from sorethumb.evaluate.benchmark import write_outputs

    row = make_benchmark_row()
    write_outputs([row], tmp_path)
    assert not (tmp_path / "benchmark_metadata.json").exists()
