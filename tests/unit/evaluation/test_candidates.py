"""Candidate ensemble rules: what each version decides from a fitted plan."""

from __future__ import annotations

import polars as pl
import pytest

from sorethumb_ml.config import Config
from sorethumb_ml.evaluate.candidates import CANDIDATE_RULES, candidate_by_id
from sorethumb_ml.profiling.plan import build_feature_plan

pytestmark = pytest.mark.unit


def _plan(df: pl.DataFrame) -> object:
    cfg = Config.model_validate({"source": {"uri": "file:///dev/null"}, "run": {"workdir": "/tmp/unused"}})
    return build_feature_plan(df, cfg)


def test_rule_ids_are_versioned_and_unique() -> None:
    ids = [r.id for r in CANDIDATE_RULES]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("rule:") and "@" in i for i in ids)
    assert candidate_by_id("rule:discrete-composite@1") is CANDIDATE_RULES[0]
    assert candidate_by_id("rule:nope@1") is None


def test_discrete_composite_v1_switches_only_on_all_discrete_plans() -> None:
    rule = candidate_by_id("rule:discrete-composite@1")
    assert rule is not None
    levels = ["a", "b", "c", "d", "e"]
    all_discrete = pl.DataFrame(
        {"colour": [levels[i % 5] for i in range(200)], "shape": [levels[(i * 3) % 5] for i in range(200)]}
    )
    choice = rule.choose(_plan(all_discrete))  # type: ignore[arg-type]
    assert (choice.combination, choice.weighting, choice.branch) == ("composite", "equal", "discrete")

    mixed = all_discrete.with_columns(pl.Series("amount", [float(i % 37) for i in range(200)]))
    choice = rule.choose(_plan(mixed))  # type: ignore[arg-type]
    assert (choice.combination, choice.branch) == ("intersection", "mixed")
