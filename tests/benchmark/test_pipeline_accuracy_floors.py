"""Pipeline-benchmark accuracy floors: the harness's "reject a default
configuration" gate (P3-2). Every guarded (scenario, ablation) cell must clear
a committed ROC-AUC floor on the network-free synthetic scenarios.

Marked ``benchmark``, so it is deselected by default and run explicitly::

    pytest -m benchmark

Floors were set from an observed run (2026-09-20) with margin below the
measured value. Ratchet a floor UP when a change reliably improves it; do not
lower one without a documented reason in this file.

Known, deliberately undocumented-as-floors weak spots
-------------------------------------------------------
The *default* ablation (``combination="intersection"``, the pipeline's real
shipped default) is not floor-tested on ``local`` or ``varying_density``:
OneClassSVM badly misranks both regimes (observed ROC-AUC as low as 0.04 on
``local``), and intersection's ``min()`` aggregation across detectors lets
that one bad member drag the whole ensemble's continuous ranking down with
it -- observed default-ablation ROC-AUC on those two scenarios is *worse
than random*. This is a real, measured property of the shipped default, not
a synthetic-data artifact (see docs/approximations.md), and mirrors the
existing precedent of documenting ``lof``'s weakness on ``clustered``
anomalies (tests/benchmark/test_accuracy_floors.py) rather than asserting
something known to be false. The ``combination_union`` ablation -- which
takes the *best*-performing detector's opinion per row instead of the
worst's -- clears a real floor on both, and *that* is floor-tested here:
if union's robustness on these scenarios regresses, this suite catches it.

``contextual`` is not floor-tested as "beats random" under any ablation: every
combination mode tried (default/intersection, composite, union, all_detectors)
lands at 0.45-0.49 ROC-AUC -- consistently at chance, not occasionally by
seed luck (see docs/approximations.md). None of sorethumb's current
detector/combination choices learn the conditional "normal depends on
context" structure from an ordinary categorical feature column. Rather than
assert something the data shows is false, this is checked only for a
catastrophic regression (a real inversion/bug), not for beating random.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkRow

pytestmark = pytest.mark.benchmark

# (scenario, ablation) -> ROC-AUC floor. Observed values on the 2026-09-20
# reference run are in the trailing comments.
_FLOORS: dict[tuple[str, str], float] = {
    ("point", "default"): 0.85,  # observed 0.97
    ("clustered", "default"): 0.85,  # observed 0.98 (default ensemble excludes lof)
    ("masking", "default"): 0.45,  # observed 0.57 -- masking is deliberately hard
    ("swamping", "default"): 0.70,  # observed 0.97 -- training contamination cost less than expected
    ("local", "combination_union"): 0.65,  # observed 0.84
    ("varying_density", "combination_union"): 0.65,  # observed 0.86
}

# Sanity-only, not "beats random": contextual anomalies genuinely sit at
# chance (0.45-0.49) under every combination mode tried -- see the module
# docstring. Only guards against a catastrophic regression (an inverted
# score, a crash), not against "does not beat random", which is not
# something any current ablation actually achieves here.
_SANITY_ONLY: list[tuple[str, str]] = [("contextual", "default")]
_SANITY_FLOOR = 0.35


@pytest.fixture(scope="session")
def pipeline_rows() -> dict[tuple[str, str], PipelineBenchmarkRow]:
    """Run every guarded (scenario, ablation) cell once, 3 seeds each."""
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    scenarios = sorted({s for s, _ in [*_FLOORS, *_SANITY_ONLY]})
    ablations = sorted({a for _, a in [*_FLOORS, *_SANITY_ONLY]})
    cfg = PipelineBenchmarkConfig(
        scenario_names=scenarios,
        ablation_names=ablations,
        n_seeds=3,
        include_baselines=False,
        # swamping's own generator (point-family) is included separately by
        # run_pipeline_benchmark(include_swamping=True); "swamping" is not a
        # name in SCENARIOS, so it isn't filtered out by scenario_names above.
        include_swamping=("swamping", "default") in _FLOORS,
    )
    with warnings.catch_warnings():
        # A warning (e.g. ZeroAnomalyWarning, AntiCorrelatedMemberWarning) is
        # not the signal this test guards; ROC-AUC is.
        warnings.simplefilter("ignore")
        rows = run_pipeline_benchmark(cfg)
    return {(r.scenario, r.ablation): r for r in rows}


@pytest.mark.parametrize(("scenario", "ablation"), sorted(_FLOORS))
def test_cell_clears_roc_auc_floor(
    pipeline_rows: dict[tuple[str, str], PipelineBenchmarkRow],
    scenario: str,
    ablation: str,
) -> None:
    row = pipeline_rows[scenario, ablation]
    assert row.error is None, f"{scenario}/{ablation} errored: {row.error}"
    floor = _FLOORS[scenario, ablation]
    assert row.roc_auc >= floor, (
        f"{scenario}/{ablation}: ROC-AUC {row.roc_auc:.4f} < floor {floor:.2f}. "
        "If this is a real change, update _FLOORS in this file with a note."
    )


@pytest.mark.parametrize(("scenario", "ablation"), _SANITY_ONLY)
def test_cell_not_catastrophically_broken(
    pipeline_rows: dict[tuple[str, str], PipelineBenchmarkRow],
    scenario: str,
    ablation: str,
) -> None:
    row = pipeline_rows[scenario, ablation]
    assert row.error is None, f"{scenario}/{ablation} errored: {row.error}"
    assert row.roc_auc > _SANITY_FLOOR, (
        f"{scenario}/{ablation}: ROC-AUC {row.roc_auc:.4f} is well below the near-chance "
        f"range this scenario normally sits in ({_SANITY_FLOOR:.2f}) -- likely a real regression, "
        "not the documented 'contextual is hard' limitation."
    )
