"""Check pipeline-benchmark results against per-anomaly-type budgets.

Budgets live in a version-controlled TOML file (``tests/benchmark/budgets.toml``
in this repository): a ``[defaults]`` table of bounds, a ``[kinds.<kind>]``
table per scenario kind overriding any of them, and optionally ``only = [...]``
to restrict a kind to the bounds that are defined for it. A bound is named
``min_<metric>`` or ``max_<metric>``, where ``<metric>`` is a
:class:`~sorethumb_ml.evaluate.pipeline_benchmark.PipelineBenchmarkRow` field.

``[[accepted_limitations]]`` entries record a budget the *default* policy is
known to miss and the project accepts, with the reason -- see
``docs/ensemble-policy.md``. :func:`default_policy_report` separates such
accepted misses from unaccepted ones, and flags an accepted entry that the
default now meets (a stale entry, to be removed so the list stays true).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sorethumb_ml.evaluate.pipeline_benchmark import PipelineBenchmarkRow


@dataclass(frozen=True)
class BudgetCheck:
    """One bound applied to one benchmark cell."""

    scenario: str
    kind: str
    policy: str
    ablation: str
    bound: str  # e.g. "min_flag_recall"
    limit: float
    value: float

    @property
    def passed(self) -> bool:
        """Whether the cell's value is within the bound."""
        if self.bound.startswith("min_"):
            return self.value >= self.limit
        return self.value <= self.limit


@dataclass(frozen=True)
class AcceptedLimitation:
    """A budget the default policy misses, accepted as a design trade-off."""

    policy: str
    scenario: str
    bound: str
    reason: str


@dataclass(frozen=True)
class Budgets:
    """A parsed budgets file."""

    version: int
    by_kind: dict[str, dict[str, float]]
    accepted: tuple[AcceptedLimitation, ...]

    def bounds_for(self, kind: str) -> dict[str, float]:
        """Return the bounds that apply to scenario kind *kind*.

        Raises:
            KeyError: *kind* has no ``[kinds.<kind>]`` table -- every scenario
                kind must be budgeted explicitly, even if only by defaults.

        """
        return self.by_kind[kind]


def load_budgets(path: Path) -> Budgets:
    """Parse a budgets TOML file."""
    with path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    defaults: dict[str, float] = {k: float(v) for k, v in raw.get("defaults", {}).items()}
    by_kind: dict[str, dict[str, float]] = {}
    for kind, table in raw.get("kinds", {}).items():
        merged = {**defaults, **{k: float(v) for k, v in table.items() if k != "only"}}
        only = table.get("only")
        if only is not None:
            merged = {k: v for k, v in merged.items() if k in only}
        _check_bound_names(merged, f"kinds.{kind}")
        by_kind[kind] = merged
    accepted = tuple(
        AcceptedLimitation(e["policy"], e["scenario"], e["bound"], e["reason"])
        for e in raw.get("accepted_limitations", [])
    )
    return Budgets(version=int(raw["version"]), by_kind=by_kind, accepted=accepted)


def _check_bound_names(bounds: dict[str, float], where: str) -> None:
    bad = [name for name in bounds if not name.startswith(("min_", "max_"))]
    if bad:
        msg = f"{where}: bound names must start with min_ or max_, got {bad}"
        raise ValueError(msg)


def check_rows(rows: list[PipelineBenchmarkRow], budgets: Budgets) -> list[BudgetCheck]:
    """Every defined bound applied to every ensemble-policy row (baseline rows have no policy)."""
    checks: list[BudgetCheck] = []
    for row in rows:
        if row.policy is None or row.error is not None:
            continue
        for bound, limit in budgets.bounds_for(row.kind).items():
            value = getattr(row, bound[4:])
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            checks.append(
                BudgetCheck(row.scenario, row.kind, row.policy, row.ablation, bound, limit, float(value))
            )
    return checks


@dataclass(frozen=True)
class DefaultPolicyReport:
    """How the default policy stands against the budgets."""

    unaccepted_misses: tuple[BudgetCheck, ...]
    accepted_misses: tuple[BudgetCheck, ...]
    stale_acceptances: tuple[AcceptedLimitation, ...]


def default_policy_report(
    checks: list[BudgetCheck], budgets: Budgets, default_policy: str, default_ablation: str = "default"
) -> DefaultPolicyReport:
    """Split the shipped default's misses into accepted and unaccepted; find stale acceptances.

    Only *default_ablation* rows count: another ablation may run the same policy
    with a non-default setting (``near_constant_2``, say) -- evidence, not the
    shipped configuration.
    """
    accepted_keys = {(a.scenario, a.bound) for a in budgets.accepted if a.policy == default_policy}
    default_checks = [c for c in checks if c.policy == default_policy and c.ablation == default_ablation]
    misses = [c for c in default_checks if not c.passed]
    missed_keys = {(c.scenario, c.bound) for c in misses}
    checked_keys = {(c.scenario, c.bound) for c in default_checks}
    stale = tuple(
        a
        for a in budgets.accepted
        if a.policy == default_policy
        and (a.scenario, a.bound) in checked_keys
        and (a.scenario, a.bound) not in missed_keys
    )
    return DefaultPolicyReport(
        unaccepted_misses=tuple(c for c in misses if (c.scenario, c.bound) not in accepted_keys),
        accepted_misses=tuple(c for c in misses if (c.scenario, c.bound) in accepted_keys),
        stale_acceptances=stale,
    )
