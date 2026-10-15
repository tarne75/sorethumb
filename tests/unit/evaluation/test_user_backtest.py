"""The user-data backtest's pure parts: candidate selection, the decision rule, the evidence scope."""

from __future__ import annotations

import math

import pytest

from sorethumb_ml.config import Config
from sorethumb_ml.errors import ConfigError
from sorethumb_ml.evaluate.user_backtest import candidate_policies, decide_candidates
from sorethumb_ml.scoring.policies import policy_scope_hash

pytestmark = pytest.mark.unit


def _cfg(**over: object) -> Config:
    return Config.model_validate({"source": {"uri": "file:///dev/null"}, "run": {"workdir": "ws"}, **over})


def test_every_policy_but_the_default_is_a_candidate_and_manual_needs_weights() -> None:
    ids = [p.id for p in candidate_policies(_cfg())]
    assert ids == ["union@1", "composite-equal@1", "composite-agreement@1"]
    weighted = _cfg(scoring={"weights": {"isolation_forest": 2.0}})
    assert "composite-manual@1" in [p.id for p in candidate_policies(weighted)]


@pytest.mark.parametrize(
    ("requested", "message"), [(["nope@1"], "Unknown"), (["intersection@1"], "baseline")]
)
def test_an_unknown_or_default_candidate_is_rejected(requested: list[str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        candidate_policies(_cfg(), requested)


def _reps(f1: list[float], fp: float = 0.5, jac: float = 0.8, ap: float = 0.4) -> list[dict[str, float]]:
    return [
        {
            "flag_f1": v,
            "false_positive_budget_ratio": fp,
            "flag_jaccard_stability": jac,
            "average_precision": ap,
        }
        for v in f1
    ]


_BASE = [0.30, 0.32, 0.31, 0.29, 0.30, 0.33, 0.28, 0.31]


def _decide(
    per: dict[str, list[dict[str, float]]], *, labelled: bool = True, ranking_tolerance: float | None = 0.10
) -> dict[str, object]:
    out = decide_candidates(
        per,
        [k for k in per if k != "intersection@1"],
        labelled=labelled,
        alpha=0.05,
        stability_tolerance=0.05,
        n_resamples=2000,
        ranking_tolerance=ranking_tolerance,
    )
    return {v.policy_id: v for v in out}


def test_a_clear_win_outperforms() -> None:
    v = _decide({"intersection@1": _reps(_BASE), "union@1": _reps([b + 0.2 for b in _BASE])})["union@1"]
    assert v.verdict == "outperforms"  # type: ignore[attr-defined]
    assert v.mean_difference == pytest.approx(0.2)  # type: ignore[attr-defined]


def test_spending_more_than_the_budget_on_wrong_rows_blocks_a_win() -> None:
    v = _decide({"intersection@1": _reps(_BASE), "union@1": _reps([b + 0.2 for b in _BASE], fp=1.4)})[
        "union@1"
    ]
    assert v.verdict == "does_not_outperform"  # type: ignore[attr-defined]
    assert "review budget" in " ".join(v.reasons)  # type: ignore[attr-defined]


def test_less_stable_flags_block_a_win() -> None:
    v = _decide({"intersection@1": _reps(_BASE), "union@1": _reps([b + 0.2 for b in _BASE], jac=0.6)})[
        "union@1"
    ]
    assert v.verdict == "does_not_outperform"  # type: ignore[attr-defined]
    assert "stable" in " ".join(v.reasons)  # type: ignore[attr-defined]


def test_no_difference_does_not_outperform() -> None:
    v = _decide({"intersection@1": _reps(_BASE), "union@1": _reps(_BASE)})["union@1"]
    assert v.verdict == "does_not_outperform"  # type: ignore[attr-defined]


def test_without_labels_every_candidate_is_not_assessable() -> None:
    nolab = [{"flag_jaccard_stability": 0.8}] * 5
    v = _decide({"intersection@1": nolab, "union@1": nolab}, labelled=False)["union@1"]
    assert v.verdict == "not_assessable"  # type: ignore[attr-defined]
    assert math.isnan(v.mean_difference)  # type: ignore[attr-defined]


def test_the_evidence_scope_ignores_the_policy_and_what_cannot_change_a_flag() -> None:
    base = policy_scope_hash(_cfg(scoring={"contamination": 0.05}))
    assert base == policy_scope_hash(
        _cfg(
            scoring={"contamination": 0.05, "combination": "union"},
            run={"workdir": "ws", "seed": 7, "strict": True},
        )
    )
    assert base == policy_scope_hash(_cfg(scoring={"contamination": 0.05}, explain={"enabled": False}))
    assert base != policy_scope_hash(_cfg(scoring={"contamination": 0.1}))
    assert base != policy_scope_hash(_cfg(scoring={"contamination": 0.05}, detectors=[{"name": "lof"}]))


def test_ranking_the_anomalies_worse_blocks_a_win_under_decision_rule_2() -> None:
    per = {"intersection@1": _reps(_BASE, ap=0.40), "union@1": _reps([b + 0.2 for b in _BASE], ap=0.34)}
    v = _decide(per)["union@1"]
    assert v.verdict == "does_not_outperform"  # type: ignore[attr-defined]
    assert "ranks the anomalies worse" in " ".join(v.reasons)  # type: ignore[attr-defined]
    within = {"intersection@1": _reps(_BASE, ap=0.40), "union@1": _reps([b + 0.2 for b in _BASE], ap=0.37)}
    assert _decide(within)["union@1"].verdict == "outperforms"  # type: ignore[attr-defined]


def test_decision_rule_1_has_no_ranking_check() -> None:
    per = {"intersection@1": _reps(_BASE, ap=0.40), "union@1": _reps([b + 0.2 for b in _BASE], ap=0.10)}
    assert _decide(per, ranking_tolerance=None)["union@1"].verdict == "outperforms"  # type: ignore[attr-defined]
