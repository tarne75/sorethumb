"""Candidate ensemble rules: data-dependent policies under backtest, not yet selectable.

A *candidate rule* picks a registered ensemble policy from observable
properties of the fitted feature plan -- never from labels or scores. It is
identified ``rule:name@version`` and, like a policy, never edited in place: a
change to which branch it takes, or what a branch runs, is a new version.

Candidates live here, in the evaluation package, until a pre-registered
backtest (``evaluate.backtest``) shows one outperforming the default. Only
then is it promoted to a user-selectable option with its evidence attached;
a rule that loses never becomes config surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sorethumb_ml.profiling.classify import ColumnClass, Treatment

if TYPE_CHECKING:
    from sorethumb_ml.profiling.plan import FeaturePlan


@dataclass(frozen=True)
class RuleChoice:
    """What a candidate rule decided for one fitted plan."""

    combination: str
    weighting: str
    branch: str  # recorded with every result, so a rule's behaviour is attributable


@dataclass(frozen=True)
class CandidateRule:
    """One versioned data-dependent ensemble rule."""

    name: str
    version: int
    summary: str
    choose: Callable[[FeaturePlan], RuleChoice]

    @property
    def id(self) -> str:
        """``rule:name@version``, the identity recorded with every result."""
        return f"rule:{self.name}@{self.version}"


_DISCRETE_CLASSES = frozenset({ColumnClass.categorical, ColumnClass.boolean})


def _discrete_composite_v1(plan: FeaturePlan) -> RuleChoice:
    """``composite-equal@1`` when every kept input column is categorical or boolean, else the default."""
    kept = [d for d in plan.decisions if d.treatment != Treatment.drop]
    if kept and all(d.col_class in _DISCRETE_CLASSES for d in kept):
        return RuleChoice("composite", "equal", "discrete")
    return RuleChoice("intersection", "equal", "mixed")


CANDIDATE_RULES: tuple[CandidateRule, ...] = (
    CandidateRule(
        name="discrete-composite",
        version=1,
        summary=(
            "composite-equal@1 when every kept input column is categorical or boolean; "
            "intersection@1 otherwise."
        ),
        choose=_discrete_composite_v1,
    ),
)


def candidate_by_id(rule_id: str) -> CandidateRule | None:
    """Look up a candidate rule by its ``rule:name@version`` id, or None."""
    return next((r for r in CANDIDATE_RULES if r.id == rule_id), None)
