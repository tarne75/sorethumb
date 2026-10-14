"""Ensemble policy registry: identities, resolution, and per-version behaviour pins."""

from __future__ import annotations

import hashlib
import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest

from sorethumb_ml.config import ScoringConfig
from sorethumb_ml.scoring.combine import ScoreEnsemble
from sorethumb_ml.scoring.policies import (
    DEFAULT_POLICY_ID,
    POLICIES,
    EnsemblePolicy,
    policy_by_id,
    resolve_policy,
)

pytestmark = pytest.mark.unit

_GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "ensemble_policies.json"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_policy_ids_are_unique() -> None:
    ids = [p.id for p in POLICIES]
    assert len(ids) == len(set(ids))


def test_the_default_policy_is_registered_and_is_what_the_default_config_selects() -> None:
    """The default is a design decision (docs/ensemble-policy.md): pinned here so it
    cannot change as a side effect of editing ScoringConfig's defaults."""
    assert DEFAULT_POLICY_ID == "intersection@1"
    assert policy_by_id(DEFAULT_POLICY_ID) is not None
    cfg = ScoringConfig()
    assert resolve_policy(cfg.combination, cfg.weighting).id == DEFAULT_POLICY_ID


def test_every_configurable_pair_resolves_to_a_policy() -> None:
    combinations = typing.get_args(ScoringConfig.model_fields["combination"].annotation)
    weightings = typing.get_args(ScoringConfig.model_fields["weighting"].annotation)
    for combination in combinations:
        for weighting in weightings:
            policy = resolve_policy(combination, weighting)
            assert policy.combination == combination


@pytest.mark.parametrize("combination", ["intersection", "union"])
def test_weighting_does_not_change_a_vote_policy(combination: str) -> None:
    ids = {resolve_policy(combination, w).id for w in ("equal", "manual", "agreement")}
    assert ids == {f"{combination}@1"}


@pytest.mark.parametrize(
    ("weighting", "expected"),
    [
        ("equal", "composite-equal@1"),
        ("agreement", "composite-agreement@1"),
        ("manual", "composite-manual@1"),
    ],
)
def test_composite_identity_includes_its_weighting(weighting: str, expected: str) -> None:
    assert resolve_policy("composite", weighting).id == expected


def test_an_unknown_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="no ensemble policy"):
        resolve_policy("majority", "equal")


# ---------------------------------------------------------------------------
# Behaviour pins: a policy's version identifies its behaviour
# ---------------------------------------------------------------------------


def _fixed_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Three positively-correlated calibrated detectors over 300 rows, 12 planted anomalies."""
    rng = np.random.default_rng(20261014)
    n = 300
    base = rng.uniform(0.0, 0.8, n)
    base[:12] = rng.uniform(0.9, 1.0, 12)
    scores = {
        name: np.clip(base + rng.normal(0.0, noise, n), 0.0, 1.0)
        for name, noise in (("det_a", 0.03), ("det_b", 0.06), ("det_c", 0.09))
    }
    flags = {name: s >= cutoff for (name, s), cutoff in zip(scores.items(), (0.92, 0.9, 0.88), strict=True)}
    return scores, flags


def _behaviour(policy: EnsemblePolicy) -> dict[str, object]:
    scores, flags = _fixed_inputs()
    out: dict[str, object] = {}
    for contamination in ("auto", 0.05):
        ensemble = ScoreEnsemble(
            weighting=policy.weighting or "equal",
            combination=policy.combination,
            contamination=contamination,
            manual_weights={"det_a": 0.5, "det_b": 0.3, "det_c": 0.2}
            if policy.weighting == "manual"
            else None,
        )
        result = ensemble.combine(scores, flags)
        combined = np.round(np.asarray(result["combined_score"], dtype=np.float64), 9)
        out[str(contamination)] = {
            "flagged_rows": [int(i) for i in np.flatnonzero(result["anomaly_flag"])],
            "combined_score_sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
        }
    return out


def test_every_current_policy_version_matches_its_pinned_behaviour() -> None:
    actual = {p.id: _behaviour(p) for p in POLICIES}
    if os.environ.get("UPDATE_GOLDEN") == "1":
        _GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    assert set(expected) == set(actual), (
        f"tests/golden/ensemble_policies.json pins {sorted(expected)} but the registry has "
        f"{sorted(actual)}. After bumping a policy's version, regenerate with UPDATE_GOLDEN=1."
    )
    changed = sorted(pid for pid in actual if actual[pid] != expected[pid])
    assert not changed, (
        f"Behaviour changed for {changed} under an unchanged id. A policy's version identifies its "
        "behaviour (scoring/policies.py): bump that policy's `version`, then regenerate this golden "
        "file with UPDATE_GOLDEN=1 -- runs recorded under the old id must keep meaning what they did."
    )
