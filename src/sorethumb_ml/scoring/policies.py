"""Ensemble policies: named, versioned identities for how detector votes become flags.

A *policy* is the decision rule ``ScoreEnsemble`` applies: the combination
mode, plus the weighting where the mode uses one (only ``composite`` does).
Each policy has an id of the form ``name@version`` -- ``intersection@1``,
``composite-agreement@1`` -- recorded on every run, so a result can always be
traced to the exact rule that produced it.

Versioning rule
---------------
A policy's version identifies its *behaviour*, not its name. Any change that
alters which rows a policy flags, or the combined score it produces, for the
same inputs -- including a bug fix -- must bump that policy's ``version``.
``tests/golden/ensemble_policies.json`` pins each current version's output on
fixed inputs, so such a change cannot land under an unchanged id. Results
recorded under an earlier version keep that id; they describe what the code
did at the time, not what the current code does.

The review budget (``scoring.contamination``) and manual weights are inputs to
a policy, not part of its identity: they are recorded with the run's config.

The shipped default is :data:`DEFAULT_POLICY_ID`, deliberately -- see
``docs/ensemble-policy.md`` for the design decision, the limitations it
accepts, and what evidence a different policy should be chosen on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sorethumb_ml.config import Config


@dataclass(frozen=True)
class EnsemblePolicy:
    """One versioned ensemble decision rule."""

    name: str
    version: int
    combination: str
    weighting: str | None  # None: the combination ignores weighting
    summary: str

    @property
    def id(self) -> str:
        """``name@version``, the identity recorded on every run."""
        return f"{self.name}@{self.version}"


POLICIES: tuple[EnsemblePolicy, ...] = (
    EnsemblePolicy(
        name="intersection",
        version=1,
        combination="intersection",
        weighting=None,
        summary="Flag a row only when every configured detector flags it on its own boundary.",
    ),
    EnsemblePolicy(
        name="union",
        version=1,
        combination="union",
        weighting=None,
        summary="Flag a row when any configured detector flags it on its own boundary.",
    ),
    EnsemblePolicy(
        name="composite-equal",
        version=1,
        combination="composite",
        weighting="equal",
        summary="Equal-weight average of calibrated scores, one global threshold.",
    ),
    EnsemblePolicy(
        name="composite-agreement",
        version=1,
        combination="composite",
        weighting="agreement",
        summary="Average weighted by each detector's rank agreement with the consensus, one global threshold.",
    ),
    EnsemblePolicy(
        name="composite-manual",
        version=1,
        combination="composite",
        weighting="manual",
        summary="Average weighted by scoring.weights, one global threshold.",
    ),
)

DEFAULT_POLICY_ID = "intersection@1"


def resolve_policy(combination: str, weighting: str) -> EnsemblePolicy:
    """Return the policy a ``scoring.combination`` / ``scoring.weighting`` pair selects.

    Weighting is ignored by ``intersection`` and ``union`` -- they are votes --
    so it does not change their identity.

    Raises:
        ValueError: the pair names no registered policy.

    """
    for policy in POLICIES:
        if policy.combination != combination:
            continue
        if policy.weighting is None or policy.weighting == weighting:
            return policy
    msg = f"no ensemble policy for combination={combination!r}, weighting={weighting!r}"
    raise ValueError(msg)


def policy_by_id(policy_id: str) -> EnsemblePolicy | None:
    """Look up a current policy by its ``name@version`` id, or None."""
    return next((p for p in POLICIES if p.id == policy_id), None)


def policy_scope_hash(config: Config) -> str:
    """Fingerprint *config* apart from its ensemble policy: what backtest evidence is valid for.

    Evidence that a policy outperforms the default was produced with one
    configuration; it only speaks for a run with the same detectors, features,
    review budget and data handling. The policy itself
    (``scoring.combination``/``scoring.weighting``) is excluded -- the evidence
    compares policies -- and so is everything that cannot change a flag: the
    explain settings, ``run.strict`` and ``run.seed`` (a backtest already varies
    the fit across resamples), on top of what ``Config.config_hash`` excludes.
    """
    scoring_fields = type(config.scoring).model_fields
    scoring = config.scoring.model_copy(
        update={
            "combination": scoring_fields["combination"].default,
            "weighting": scoring_fields["weighting"].default,
        }
    )
    run = config.run.model_copy(
        update={"strict": False, "seed": type(config.run).model_fields["seed"].default}
    )
    neutral = config.model_copy(update={"scoring": scoring, "run": run, "explain": type(config.explain)()})
    return neutral.config_hash()
