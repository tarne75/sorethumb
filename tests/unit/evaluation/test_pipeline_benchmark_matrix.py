"""Unit tests for the pipeline-benchmark publication guardrail:
``expected_cells``/``assert_complete_and_error_free`` must catch a missing or
errored (scenario, ablation) cell before it ever ships as benchmark evidence.

Pure, fast, no model fitting -- see ``tests/benchmark/test_pipeline_benchmark_harness.py``
for tests that exercise the real harness end to end.
"""

from __future__ import annotations

import math

import pytest

from sorethumb_ml.evaluate.pipeline_benchmark import (
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
    from sorethumb_ml.evaluate.pipeline_benchmark import _fmt_row_for_table

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


# ---------------------------------------------------------------------------
# Policy matrix plumbing (no detectors run)
# ---------------------------------------------------------------------------


def test_every_ablation_resolves_to_a_registered_policy() -> None:
    from sorethumb_ml.evaluate.pipeline_benchmark import ABLATIONS
    from sorethumb_ml.scoring.policies import policy_by_id

    for ablation in ABLATIONS:
        assert policy_by_id(ablation.policy_id) is not None, ablation


def test_the_policy_matrix_covers_every_policy_except_manual_weighting() -> None:
    """composite-manual's behaviour depends on user-supplied weights, so it has no
    data-independent benchmark; every other registered policy must be in the matrix."""
    from sorethumb_ml.evaluate.pipeline_benchmark import ABLATIONS, POLICY_ABLATIONS
    from sorethumb_ml.scoring.policies import POLICIES

    by_name = {a.name: a for a in ABLATIONS}
    covered = {by_name[name].policy_id for name in POLICY_ABLATIONS}
    assert covered == {p.id for p in POLICIES} - {"composite-manual@1"}


def test_flag_jaccard_and_rank_correlation_edge_cases() -> None:
    import numpy as np

    from sorethumb_ml.evaluate.pipeline_benchmark import _flag_jaccard, _rank_correlation

    a = np.array([True, True, False, False])
    assert _flag_jaccard(a, a) == 1.0
    assert _flag_jaccard(a, np.array([True, False, True, False])) == 1 / 3
    assert _flag_jaccard(np.zeros(4, bool), np.zeros(4, bool)) == 1.0  # both flag nothing: they agree
    assert _rank_correlation(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0
    assert np.isnan(_rank_correlation(np.ones(3), np.array([1.0, 2.0, 3.0])))


def test_suite_version_moves_with_the_suite_parameters() -> None:
    from sorethumb_ml.evaluate.pipeline_benchmark import policy_matrix_config, suite_version

    base = suite_version(policy_matrix_config(n_seeds=10, n_bootstrap=3))
    assert base == suite_version(policy_matrix_config(n_seeds=10, n_bootstrap=3))
    assert base != suite_version(policy_matrix_config(n_seeds=5, n_bootstrap=3))
    assert base != suite_version(policy_matrix_config(n_seeds=10, n_bootstrap=0))


def test_to_json_carries_metadata_and_turns_nan_into_null() -> None:
    import json
    import math

    from sorethumb_ml.evaluate.pipeline_benchmark import to_json
    from tests.factories.pipeline_benchmark_rows import make_pipeline_row

    payload = json.loads(to_json([make_pipeline_row(rank_stability=math.nan)], {"suite_version": "abc"}))
    assert payload["metadata"] == {"suite_version": "abc"}
    assert payload["rows"][0]["rank_stability"] is None
    assert payload["rows"][0]["policy"] == "intersection@1"
