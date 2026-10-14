"""One pipeline-benchmark cell fitted in-process (no subprocess), plus the new scenarios.

The benchmark suites run each cell in a spawned subprocess, which coverage
does not see; these exercise the same code directly on single small seeds.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb_ml.evaluate.pipeline_benchmark import (
    ABLATIONS,
    _fit_score_one_seed_pipeline,
    inject_policy_matrix,
)
from sorethumb_ml.evaluate.scenarios import scenario_by_name
from tests.factories.pipeline_benchmark_rows import make_pipeline_row

pytestmark = pytest.mark.unit

_ABLATION = {a.name: a for a in ABLATIONS}


@pytest.mark.parametrize(
    ("name", "column", "dtype"),
    [
        ("categorical_rare", "channel", pl.String),
        ("text_timestamp", "event_time", pl.String),
        ("irrelevant_columns", "noise_19", pl.Float64),
        ("all_categorical", "device", pl.String),
    ],
)
def test_new_scenarios_generate_labelled_frames(name: str, column: str, dtype: type) -> None:
    scenario = scenario_by_name(name)
    assert scenario is not None
    df, y = scenario.generate(0)
    assert len(df) == len(y)
    assert df.schema[column] == dtype
    assert 0 < y.sum() < len(y) * 0.1


def test_categorical_rare_signal_is_only_the_rare_level() -> None:
    df, y = scenario_by_name("categorical_rare").generate(1)  # type: ignore[union-attr]
    channel = df["channel"].to_numpy()
    assert set(channel[y == 1]) == {"fax"}
    assert "fax" not in set(channel[y == 0])


def test_a_time_sorted_cell_scores_against_the_right_labels() -> None:
    """text_timestamp is time-sorted by the pipeline; misaligned scores would sit near chance."""
    out = _fit_score_one_seed_pipeline("text_timestamp", None, _ABLATION["default"], seed=0, n_bootstrap=1)
    assert out["roc_auc"] > 0.75
    assert 0.0 <= out["flag_jaccard_stability"] <= 1.0
    assert -1.0 <= out["rank_stability"] <= 1.0
    assert out["over_flag_ratio"] >= 0.0


def test_a_cell_without_refits_reports_stability_as_nan() -> None:
    out = _fit_score_one_seed_pipeline(
        "categorical_rare", None, _ABLATION["near_constant_2"], seed=0, n_bootstrap=0
    )
    assert math.isnan(out["flag_jaccard_stability"])
    assert math.isnan(out["rank_stability"])
    assert np.isfinite(out["roc_auc"])


def test_all_categorical_has_no_numeric_column_and_anomalies_reuse_common_levels() -> None:
    df, y = scenario_by_name("all_categorical").generate(2)  # type: ignore[union-attr]
    assert all(dtype == pl.String for dtype in df.schema.values())
    normal_levels = {c: set(df[c].to_numpy()[y == 0]) for c in df.columns}
    for c in df.columns:
        assert set(df[c].to_numpy()[y == 1]) <= normal_levels[c]  # no rare level: only the combination is new


def test_a_candidate_rule_records_the_branch_it_took() -> None:
    rule = _ABLATION["rule_discrete_composite"]
    assert _fit_score_one_seed_pipeline("all_categorical", None, rule, seed=0)["policy_branch"] == "discrete"
    assert _fit_score_one_seed_pipeline("point", None, rule, seed=0)["policy_branch"] == "mixed"
    assert _fit_score_one_seed_pipeline("point", None, _ABLATION["default"], seed=0)["policy_branch"] is None


def test_discrete_budget_flags_exactly_the_review_budget_on_all_categorical_data() -> None:
    rule = _ABLATION["rule_discrete_budget"]
    out = _fit_score_one_seed_pipeline("all_categorical", None, rule, seed=0)
    assert out["policy_branch"] == "discrete"
    assert out["flag_count"] == round(0.05 * out["n_holdout"])
    default = _fit_score_one_seed_pipeline("all_categorical", None, _ABLATION["default"], seed=0)
    assert out["average_precision"] == default["average_precision"]  # the default's ranking, unchanged


def test_false_positive_budget_ratio_counts_wrong_flags_against_the_whole_row_budget() -> None:
    out = _fit_score_one_seed_pipeline("point", None, _ABLATION["combination_composite"], seed=0)
    budget_rows = round(0.05 * out["n_holdout"])
    wrong = out["flag_count"] * (1 - out["flag_precision"])
    assert out["false_positive_budget_ratio"] == pytest.approx(wrong / budget_rows)
    assert out["false_positive_budget_ratio"] <= 1.0  # an exact-k policy never exceeds its budget


def test_an_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        _fit_score_one_seed_pipeline("nope", None, _ABLATION["default"], seed=0)


_META = {
    "suite_version": "abc123",
    "budgets_version": 1,
    "n_seeds": 2,
    "n_bootstrap": 1,
    "library_version": "0.1.0",
    "git_sha": "deadbeef",
}


def test_inject_policy_matrix_replaces_only_the_marked_block(tmp_path: Path) -> None:
    doc = tmp_path / "benchmarks.md"
    doc.write_text(
        "intro\n<!-- policy-matrix-results-start -->\nSTALE-TABLE\n<!-- policy-matrix-results-end -->\noutro\n",
        encoding="utf-8",
    )
    assert inject_policy_matrix([make_pipeline_row()], doc, _META) is True
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert "STALE-TABLE" not in text
    assert "Suite version `abc123`" in text
    assert "`deadbeef`" in text
    assert inject_policy_matrix([make_pipeline_row()], doc, _META) is False  # idempotent


def test_inject_policy_matrix_skips_a_doc_without_markers(tmp_path: Path) -> None:
    doc = tmp_path / "benchmarks.md"
    doc.write_text("no markers here\n", encoding="utf-8")
    assert inject_policy_matrix([make_pipeline_row()], doc, _META) is False
    assert doc.read_text(encoding="utf-8") == "no markers here\n"
