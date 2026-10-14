"""The default ensemble policy against the per-anomaly-type budgets, over the full policy matrix.

Runs the published policy-matrix configuration (every scenario x every policy,
10 seeds, 3 stability refits) -- about half an hour -- so it has its own
scheduled CI job (``policy_matrix`` marker), not a per-PR one.

Only the *default* policy is asserted. Other policies are measured and
published (scripts/run_benchmark.py --policy-matrix), not gated: they are
options a user chooses on this evidence, not a promise.

- A budget the default misses must be listed under ``accepted_limitations`` in
  ``budgets.toml`` with a reason. A new, unlisted miss fails: either the code
  regressed, or the trade-off must be accepted explicitly (and documented in
  docs/ensemble-policy.md) -- never by quietly moving a budget.
- An accepted limitation the default now *meets* also fails: the list must stay
  true, so the stale entry is removed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.policy_matrix]

_BUDGETS = Path(__file__).resolve().parent / "budgets.toml"


@pytest.fixture(scope="module")
def report() -> object:
    from sorethumb_ml.evaluate.budgets import check_rows, default_policy_report, load_budgets
    from sorethumb_ml.evaluate.pipeline_benchmark import (
        assert_complete_and_error_free,
        expected_cells,
        policy_matrix_config,
        run_pipeline_benchmark,
    )
    from sorethumb_ml.scoring.policies import DEFAULT_POLICY_ID

    cfg = policy_matrix_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ZeroAnomalyWarning etc. are expected outcomes here
        rows = run_pipeline_benchmark(cfg)
    assert_complete_and_error_free(rows, expected_cells(cfg))
    budgets = load_budgets(_BUDGETS)
    return default_policy_report(check_rows(rows, budgets), budgets, DEFAULT_POLICY_ID)


def test_the_default_policy_misses_no_budget_that_is_not_an_accepted_limitation(report) -> None:
    misses = [f"{c.scenario}: {c.bound} = {c.value:.3f} (bound {c.limit})" for c in report.unaccepted_misses]
    assert not misses, (
        "The default policy misses budget(s) not listed under accepted_limitations in "
        f"tests/benchmark/budgets.toml: {misses}. Fix the regression, or accept the trade-off "
        "explicitly with a reason (and document it in docs/ensemble-policy.md)."
    )


def test_no_accepted_limitation_is_stale(report) -> None:
    stale = [f"{a.scenario}: {a.bound}" for a in report.stale_acceptances]
    assert not stale, f"The default policy now meets {stale}: remove these accepted_limitations entries."
