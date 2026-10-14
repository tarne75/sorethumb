"""The backtest protocol: pre-registration, paired statistics and the decision rule (no fitting)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sorethumb_ml.evaluate.backtest import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    ablation_for,
    backtest_scenarios,
    decide,
    holm_adjust,
    inject_results,
    load_preregistration,
    paired_bootstrap,
    render_results,
)
from sorethumb_ml.evaluate.budgets import load_budgets

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PREREG_DIR = _REPO / "tests" / "benchmark" / "preregistrations"
_RESULTS_DIR = _REPO / "docs" / "benchmark-results" / "backtests"

_PREREG = """
candidate = "composite-agreement@1"
hypothesis = "it flags better"
primary_metric = "flag_f1"
scenarios = ["point", "local"]
require = "{require}"
"""


def _prereg(tmp_path: Path, require: str = "all", **extra: Any) -> Any:
    body = _PREREG.format(require=require) + "".join(f"{k} = {json.dumps(v)}\n" for k, v in extra.items())
    path = tmp_path / "cand-v1.toml"
    path.write_text(body, encoding="utf-8")
    return load_preregistration(path)


_BUDGETS = """
version = 1
[defaults]
max_flag_false_positive_rate = 0.05
[kinds.point]
[kinds.local]
"""


def _seed(f1: float, fpr: float = 0.01, jac: float = 0.9) -> dict[str, float]:
    return {"flag_f1": f1, "flag_false_positive_rate": fpr, "flag_jaccard_stability": jac}


def _per_seed(
    base: list[dict[str, float]], cand: list[dict[str, float]]
) -> dict[str, list[dict[str, float]]]:
    return {"baseline": base, "candidate": cand}


@pytest.fixture
def budgets(tmp_path: Path) -> Any:
    path = tmp_path / "budgets.toml"
    path.write_text(_BUDGETS, encoding="utf-8")
    return load_budgets(path)


_KINDS = {"point": "point", "local": "local"}


def test_development_and_confirmation_seeds_are_disjoint_and_the_matrix_uses_development() -> None:
    from sorethumb_ml.evaluate.pipeline_benchmark import policy_matrix_config

    assert not set(DEVELOPMENT_SEEDS) & set(CONFIRMATION_SEEDS)
    cfg = policy_matrix_config()
    assert {cfg.seed + i for i in range(cfg.n_seeds)} <= set(DEVELOPMENT_SEEDS)


def test_a_preregistration_records_its_hash_and_defaults(tmp_path: Path) -> None:
    prereg = _prereg(tmp_path)
    assert prereg.id == "cand-v1"
    assert prereg.baseline == "intersection@1"
    assert prereg.alpha == 0.05
    assert len(prereg.sha256) == 64


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"candidate": "nope@1"}, "unknown candidate"),
        ({"candidate": "intersection@1"}, "same policy"),
        ({"primary_metric": "flag_false_positive_rate"}, "primary_metric"),
        ({"scenarios": ["not-a-scenario"]}, "scenarios"),
        ({"alpha": 1.5}, "alpha"),
        ({"n_resamples": 10}, "n_resamples"),
        ({"stability_metric": "flag_f1"}, "stability_metric"),
    ],
)
def test_an_invalid_preregistration_is_rejected(tmp_path: Path, extra: dict[str, Any], message: str) -> None:
    body = _PREREG.format(require="all")
    for key, value in extra.items():
        body = "\n".join(line for line in body.splitlines() if not line.startswith(f"{key} ="))
        body += f"\n{key} = {json.dumps(value)}\n"
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_preregistration(path)


def test_paired_bootstrap_separates_a_real_shift_from_none() -> None:
    rng = np.random.default_rng(0)
    shifted = paired_bootstrap(0.2 + rng.normal(0, 0.05, 20), n_resamples=2000, alpha=0.05)
    assert shifted.ci_low > 0
    assert shifted.p_value < 0.01
    identical = paired_bootstrap(np.zeros(20), n_resamples=2000, alpha=0.05)
    assert identical.mean == 0.0
    assert identical.p_value == 1.0
    empty = paired_bootstrap(np.array([math.nan]), n_resamples=2000, alpha=0.05)
    assert empty.n == 0
    assert empty.p_value == 1.0


def test_holm_adjustment_is_step_down_and_monotone() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


def test_a_clear_win_without_regressions_outperforms(tmp_path: Path, budgets: Any) -> None:
    prereg = _prereg(tmp_path)
    per_seed = {
        s: _per_seed([_seed(0.1 + 0.01 * i) for i in range(10)], [_seed(0.5 + 0.01 * i) for i in range(10)])
        for s in _KINDS
    }
    decision = decide(prereg, per_seed, _KINDS, budgets)
    assert decision.outperforms
    assert all(c.improved for c in decision.comparisons)
    assert decision.reasons == ()


def test_require_all_versus_any(tmp_path: Path, budgets: Any) -> None:
    win = _per_seed([_seed(0.1 + 0.01 * i) for i in range(10)], [_seed(0.5 + 0.01 * i) for i in range(10)])
    tie = _per_seed([_seed(0.3)] * 10, [_seed(0.3)] * 10)
    per_seed = {"point": win, "local": tie}
    assert not decide(_prereg(tmp_path, "all"), per_seed, _KINDS, budgets).outperforms
    assert decide(_prereg(tmp_path, "any"), per_seed, _KINDS, budgets).outperforms


def test_a_budget_the_baseline_meets_and_the_candidate_breaches_blocks_a_win(
    tmp_path: Path, budgets: Any
) -> None:
    prereg = _prereg(tmp_path, "any")
    per_seed = {
        "point": _per_seed(
            [_seed(0.1 + 0.01 * i) for i in range(10)], [_seed(0.5 + 0.01 * i) for i in range(10)]
        ),
        "local": _per_seed([_seed(0.0, fpr=0.02)] * 10, [_seed(0.0, fpr=0.06)] * 10),
    }
    decision = decide(prereg, per_seed, _KINDS, budgets)
    assert not decision.outperforms
    assert [(r.scenario, r.bound) for r in decision.budget_regressions] == [
        ("local", "max_flag_false_positive_rate")
    ]


def test_a_budget_the_baseline_already_misses_is_not_a_regression(tmp_path: Path, budgets: Any) -> None:
    per_seed = {
        "point": _per_seed(
            [_seed(0.1 + 0.01 * i, fpr=0.08) for i in range(10)],
            [_seed(0.5 + 0.01 * i, fpr=0.09) for i in range(10)],
        ),
        "local": _per_seed(
            [_seed(0.1 + 0.01 * i) for i in range(10)], [_seed(0.5 + 0.01 * i) for i in range(10)]
        ),
    }
    assert decide(_prereg(tmp_path), per_seed, _KINDS, budgets).budget_regressions == ()


def test_a_stability_drop_beyond_tolerance_blocks_a_win(tmp_path: Path, budgets: Any) -> None:
    prereg = _prereg(tmp_path, "any")
    per_seed = {
        "point": _per_seed(
            [_seed(0.1 + 0.01 * i) for i in range(10)], [_seed(0.5 + 0.01 * i, jac=0.86) for i in range(10)]
        ),
        "local": _per_seed([_seed(0.0)] * 10, [_seed(0.0, jac=0.7)] * 10),
    }
    decision = decide(prereg, per_seed, _KINDS, budgets)
    assert [r.scenario for r in decision.stability_regressions] == ["local"]  # 0.86 is within 0.05 of 0.9
    assert not decision.outperforms


def test_null_metrics_from_a_published_result_are_treated_as_undefined(tmp_path: Path, budgets: Any) -> None:
    nulls: dict[str, Any] = {
        "flag_f1": None,
        "flag_false_positive_rate": None,
        "flag_jaccard_stability": None,
    }
    per_seed = {s: _per_seed([nulls] * 3, [nulls] * 3) for s in _KINDS}
    decision = decide(_prereg(tmp_path), per_seed, _KINDS, budgets)
    assert not decision.outperforms
    assert decision.budget_regressions == ()


def test_every_policy_a_backtest_names_has_a_default_settings_ablation() -> None:
    assert ablation_for("intersection@1").name == "default"
    assert ablation_for("composite-agreement@1").name == "combination_composite_agreement"
    assert ablation_for("rule:discrete-composite@1").name == "rule_discrete_composite"
    with pytest.raises(ValueError, match="no benchmark ablation"):
        ablation_for("composite-manual@1")


def test_backtests_cover_every_scenario_and_the_swamping_pair() -> None:
    from sorethumb_ml.evaluate.scenarios import SCENARIOS

    kinds = backtest_scenarios()
    assert set(kinds) == {s.name for s in SCENARIOS} | {"swamping_clean", "swamping_contaminated"}
    assert kinds["swamping_clean"] == "swamping"


def test_rendering_and_injection(tmp_path: Path) -> None:
    assert "No confirmation backtest" in render_results([], "x")
    doc = tmp_path / "doc.md"
    doc.write_text("a\n<!-- backtest-results-start -->\n<!-- backtest-results-end -->\nb\n", encoding="utf-8")
    result = {
        "preregistration": {
            "id": "c-v1",
            "candidate": "x@1",
            "baseline": "y@1",
            "primary_metric": "flag_f1",
            "hypothesis": "h",
        },
        "seeds": [1, 2],
        "decision": {
            "outperforms": False,
            "comparisons": [
                {
                    "scenario": "point",
                    "estimate": {"mean": 0.1, "ci_low": None, "ci_high": 0.2},
                    "p_holm": 0.5,
                    "improved": False,
                }
            ],
            "reasons": ["because"],
        },
    }
    assert inject_results(doc, [result], "results") is True
    text = doc.read_text(encoding="utf-8")
    assert "**does not outperform**" in text
    assert "[n/a, +0.2000]" in text
    assert "- because" in text
    assert inject_results(doc, [result], "results") is False
    doc.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        inject_results(doc, [result], "results")


# ---------------------------------------------------------------------------
# The repository's own pre-registrations and published results
# ---------------------------------------------------------------------------


def test_every_repository_preregistration_is_valid() -> None:
    paths = sorted(_PREREG_DIR.glob("*.toml"))
    assert paths
    for path in paths:
        prereg = load_preregistration(path)
        assert prereg.expected_outcome, f"{path.name}: state the expected outcome before confirming"


def test_every_published_result_matches_its_preregistration_and_reproduces_its_verdict() -> None:
    """A result is evidence only for the pre-registration it was run under, unedited since,
    and its verdict must follow from its own per-seed values under the current decision rule."""
    from sorethumb_ml.evaluate.backtest import decision_as_dict

    budgets = load_budgets(_REPO / "tests" / "benchmark" / "budgets.toml")
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        prereg = load_preregistration(_REPO / result["preregistration"]["path"])
        assert prereg.sha256 == result["preregistration"]["sha256"], f"{prereg.id} edited after confirmation"
        assert result["seeds"] == list(CONFIRMATION_SEEDS)
        assert result["metadata"]["budgets_version"] == budgets.version, "re-confirm under the new budgets"
        redone = decision_as_dict(decide(prereg, result["per_seed"], result["kinds"], budgets))
        assert redone == result["decision"], path.name


def test_run_policy_seeds_runs_every_scenario_and_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb_ml.evaluate import backtest, pipeline_benchmark

    calls: list[tuple[str, str | None, str, list[int], int]] = []
    row = dict.fromkeys(backtest._PER_SEED_KEYS, 0.5) | {"policy_branch": None, "extra": 1}

    def fake(_target: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        name, variant, ablation, seeds, refits = args
        calls.append((name, variant, ablation.name, seeds, refits))
        return {"per_seed": [row] * len(seeds), "error": None}

    monkeypatch.setattr(pipeline_benchmark, "_run_isolated", fake)
    out = backtest.run_policy_seeds("intersection@1", (1, 2))
    assert set(out) == set(backtest_scenarios())
    assert out["point"] == [{k: row[k] for k in backtest._PER_SEED_KEYS}] * 2  # only the protocol's keys
    assert ("point", "clean", "default", [1, 2], backtest.STABILITY_REFITS) in calls  # swamping pair

    monkeypatch.setattr(pipeline_benchmark, "_run_isolated", lambda _t, _a: {"per_seed": [], "error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        backtest.run_policy_seeds("intersection@1", (1, 2))


def test_a_decision_on_numpy_typed_metrics_serialises_to_json(tmp_path: Path, budgets: Any) -> None:
    """The harness hands back numpy scalars; the published decision must still be plain JSON."""
    from sorethumb_ml.evaluate.backtest import decision_as_dict

    def np_seed(f1: float) -> dict[str, Any]:
        return {k: np.float64(v) for k, v in _seed(f1).items()}

    per_seed = {
        s: _per_seed(
            [np_seed(0.1 + 0.01 * i) for i in range(10)], [np_seed(0.5 + 0.01 * i) for i in range(10)]
        )
        for s in _KINDS
    }
    decision = decide(_prereg(tmp_path), per_seed, _KINDS, budgets)
    assert all(type(c.improved) is bool for c in decision.comparisons)
    assert type(decision.comparisons[0].estimate.p_value) is float
    payload = json.loads(json.dumps(decision_as_dict(decision)))
    assert payload["outperforms"] is True
