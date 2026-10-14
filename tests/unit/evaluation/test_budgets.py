"""Budget checking for the ensemble-policy benchmark matrix (no detectors run here)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sorethumb_ml.evaluate.budgets import check_rows, default_policy_report, load_budgets
from tests.factories.pipeline_benchmark_rows import make_pipeline_row

pytestmark = pytest.mark.unit

_REPO_BUDGETS = Path(__file__).resolve().parents[2] / "benchmark" / "budgets.toml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "budgets.toml"
    path.write_text(body, encoding="utf-8")
    return path


_BASIC = """
version = 3
[defaults]
min_flag_recall = 0.25
max_over_flag_ratio = 1.5
[kinds.point]
min_flag_recall = 0.5
[kinds.swamping]
only = ["max_over_flag_ratio"]
"""


def test_kind_tables_override_defaults_and_only_restricts(tmp_path: Path) -> None:
    budgets = load_budgets(_write(tmp_path, _BASIC))
    assert budgets.version == 3
    assert budgets.bounds_for("point") == {"min_flag_recall": 0.5, "max_over_flag_ratio": 1.5}
    assert budgets.bounds_for("swamping") == {"max_over_flag_ratio": 1.5}


def test_an_unbudgeted_kind_is_an_error_not_silently_unchecked(tmp_path: Path) -> None:
    budgets = load_budgets(_write(tmp_path, _BASIC))
    with pytest.raises(KeyError):
        check_rows([make_pipeline_row(kind="local", scenario="local", flag_recall=0.1)], budgets)


def test_a_bound_name_without_min_or_max_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_ or max_"):
        load_budgets(_write(tmp_path, "version = 1\n[defaults]\nflag_recall = 0.2\n[kinds.point]\n"))


def test_min_and_max_bounds_and_nan_skipping(tmp_path: Path) -> None:
    budgets = load_budgets(_write(tmp_path, _BASIC))
    rows = [
        make_pipeline_row(flag_recall=0.6, over_flag_ratio=2.0),
        make_pipeline_row(scenario="swamping_clean", kind="swamping", over_flag_ratio=math.nan),
        make_pipeline_row(scenario="x", ablation="sklearn:lof", policy=None),  # baselines have no policy
    ]
    checks = {(c.scenario, c.bound): c.passed for c in check_rows(rows, budgets)}
    assert checks == {("point", "min_flag_recall"): True, ("point", "max_over_flag_ratio"): False}


def test_default_policy_report_separates_accepted_unaccepted_and_stale(tmp_path: Path) -> None:
    body = (
        _BASIC
        + """
[kinds.local]
[kinds.varying_density]
[[accepted_limitations]]
policy = "intersection@1"
scenario = "local"
bound = "min_flag_recall"
reason = "unanimity is conservative"
[[accepted_limitations]]
policy = "intersection@1"
scenario = "varying_density"
bound = "min_flag_recall"
reason = "no longer true"
"""
    )
    budgets = load_budgets(_write(tmp_path, body))
    rows = [
        make_pipeline_row(scenario="local", kind="local", flag_recall=0.0, over_flag_ratio=0.2),
        make_pipeline_row(
            scenario="varying_density", kind="varying_density", flag_recall=0.4, over_flag_ratio=0.3
        ),
        make_pipeline_row(scenario="point", kind="point", flag_recall=0.1, over_flag_ratio=0.2),
        make_pipeline_row(
            scenario="point", kind="point", policy="union@1", flag_recall=0.0, over_flag_ratio=3.0
        ),
        # Same policy, non-default setting: evidence, not the shipped default -- never counted.
        make_pipeline_row(scenario="point", kind="point", ablation="near_constant_2", flag_recall=0.0),
    ]
    report = default_policy_report(check_rows(rows, budgets), budgets, "intersection@1")
    assert [(c.scenario, c.bound) for c in report.accepted_misses] == [("local", "min_flag_recall")]
    assert [(c.scenario, c.bound) for c in report.unaccepted_misses] == [("point", "min_flag_recall")]
    assert [(a.scenario, a.bound) for a in report.stale_acceptances] == [
        ("varying_density", "min_flag_recall")
    ]


def test_the_repository_budgets_file_budgets_every_scenario_kind() -> None:
    from sorethumb_ml.evaluate.scenarios import SCENARIOS

    budgets = load_budgets(_REPO_BUDGETS)
    for kind in {s.kind for s in SCENARIOS} | {"swamping"}:
        budgets.bounds_for(kind)  # KeyError if a kind was added without a budget table


def test_every_accepted_limitation_names_a_real_scenario_policy_and_bound() -> None:
    from sorethumb_ml.evaluate.scenarios import SCENARIOS
    from sorethumb_ml.scoring.policies import policy_by_id

    budgets = load_budgets(_REPO_BUDGETS)
    scenario_kinds = {s.name: s.kind for s in SCENARIOS} | {
        "swamping_clean": "swamping",
        "swamping_contaminated": "swamping",
    }
    for entry in budgets.accepted:
        assert policy_by_id(entry.policy) is not None, entry
        assert entry.scenario in scenario_kinds, entry
        assert entry.bound in budgets.bounds_for(scenario_kinds[entry.scenario]), entry
        assert entry.reason.strip(), entry
