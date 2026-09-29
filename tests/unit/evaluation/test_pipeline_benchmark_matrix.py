"""Unit tests for the pipeline-benchmark publication guardrail (P2-1):
``expected_cells``/``assert_complete_and_error_free`` must catch a missing or
errored (scenario, ablation) cell before it ever ships as benchmark evidence.

Pure, fast, no model fitting -- see ``tests/benchmark/test_pipeline_benchmark_harness.py``
for tests that exercise the real harness end to end.
"""

from __future__ import annotations

import math

import pytest

from sorethumb.evaluate.pipeline_benchmark import (
    PipelineBenchmarkConfig,
    assert_complete_and_error_free,
    expected_cells,
)
from tests.factories.pipeline_benchmark_rows import make_pipeline_row

pytestmark = pytest.mark.unit


def test_expected_cells_ordinary_scenarios_only():
    cfg = PipelineBenchmarkConfig(
        scenario_names=["point", "local"],
        ablation_names=["default", "pca_on"],
        include_baselines=False,
        include_swamping=False,
    )
    assert expected_cells(cfg) == {
        ("point", "default"),
        ("point", "pca_on"),
        ("local", "default"),
        ("local", "pca_on"),
    }


def test_expected_cells_includes_baselines():
    cfg = PipelineBenchmarkConfig(
        scenario_names=["point"],
        ablation_names=["default"],
        include_baselines=True,
        include_swamping=False,
    )
    cells = expected_cells(cfg)
    assert ("point", "default") in cells
    assert ("point", "sklearn:isolation_forest") in cells
    assert ("point", "sklearn:lof") in cells
    assert ("point", "sklearn:one_class_svm") in cells
    assert len(cells) == 4


def test_expected_cells_includes_swamping_clean_and_contaminated():
    cfg = PipelineBenchmarkConfig(
        scenario_names=["__no_ordinary_scenarios__"],
        ablation_names=["default", "combination_union"],
        include_baselines=False,
        include_swamping=True,
    )
    assert expected_cells(cfg) == {
        ("swamping_clean", "default"),
        ("swamping_clean", "combination_union"),
        ("swamping_contaminated", "default"),
        ("swamping_contaminated", "combination_union"),
    }


def test_assert_complete_and_error_free_passes_for_a_complete_matrix():
    expected = {("point", "default")}
    rows = [make_pipeline_row(scenario="point", ablation="default")]
    assert_complete_and_error_free(rows, expected)  # must not raise


def test_assert_complete_and_error_free_raises_on_missing_cell():
    expected = {("point", "default"), ("local", "default")}
    rows = [make_pipeline_row(scenario="point", ablation="default")]
    with pytest.raises(RuntimeError, match="missing cell"):
        assert_complete_and_error_free(rows, expected)


def test_assert_complete_and_error_free_raises_on_unexpected_cell():
    expected = {("point", "default")}
    rows = [
        make_pipeline_row(scenario="point", ablation="default"),
        make_pipeline_row(scenario="local", ablation="default"),
    ]
    with pytest.raises(RuntimeError, match="unexpected cell"):
        assert_complete_and_error_free(rows, expected)


def test_assert_complete_and_error_free_raises_on_errored_cell():
    expected = {("point", "default")}
    rows = [make_pipeline_row(scenario="point", ablation="default", error="boom")]
    with pytest.raises(RuntimeError, match="errored cell"):
        assert_complete_and_error_free(rows, expected)


def test_fmt_row_for_table_renders_nan_roc_auc_as_na():
    """A single-class holdout (e.g. swamping's at-risk-normals-only
    population) makes roc_auc/average_precision genuinely undefined (NaN) --
    the table must show 'n/a', never a literal 'nan' string."""
    from sorethumb.evaluate.pipeline_benchmark import _fmt_row_for_table

    row = make_pipeline_row(
        scenario="swamping_contaminated",
        roc_auc=math.nan,
        roc_auc_ci95=0.0,
        average_precision=math.nan,
        average_precision_ci95=0.0,
        flag_precision=0.0,
        flag_recall=0.0,
        flag_f1=0.0,
        flag_false_positive_rate=0.12,
        flag_count=15.0,
    )
    formatted = _fmt_row_for_table(row)
    assert formatted["roc_auc"] == "n/a"
    assert formatted["average_precision"] == "n/a"
    assert "nan" not in formatted["roc_auc"].lower()
