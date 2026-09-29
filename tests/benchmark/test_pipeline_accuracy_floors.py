"""Pipeline-benchmark accuracy floors: the harness's "reject a default
configuration" gate (P3-2). Every guarded (scenario, ablation) cell must clear
a committed ROC-AUC floor on the network-free synthetic scenarios.

Marked ``benchmark``, so it is deselected by default and run explicitly::

    pytest -m benchmark

Floors were set from an observed run (2026-09-20, ``local``/``varying_density``
default-ablation floors added 2026-09-27 after P0-8) with margin below the
measured value. Ratchet a floor UP when a change reliably improves it; do not
lower one without a documented reason in this file.

``local``/``varying_density`` under the default ablation (P0-8, 2026-09-27)
-----------------------------------------------------------------------------
Previously undocumented-as-floors here: the *default* ablation
(``combination="intersection"``) scored **worse than random** on these two
scenarios (ROC-AUC as low as 0.04), because the combined *ranking* used
``min()`` across detectors -- letting OneClassSVM's badly-misranked score on
these regimes dominate the whole ensemble's ranking, even with two other
members ranking well. Fixed by ranking on the per-row *median* across
detectors instead (``scoring/combine.py::ScoreEnsemble._combine``,
``combination="intersection"``) -- a statistic that cannot be dominated by a
single anti-correlated member out of three, unlike ``min()``. The *flag*
decision (an AND of all three detectors' independent votes) is unchanged by
this fix and is not floor-tested for accuracy here -- intersection's
naturally conservative, unanimous-agreement requirement still produces very
few flags on these two scenarios, most of them false positives (0.0
flag_precision/flag_recall observed even post-fix; see
``PipelineBenchmarkRow.flag_precision``/``flag_recall`` computed against the
real ``anomaly_flag``, not the ranking) -- a separate, genuine limitation of
requiring unanimous agreement on data these detectors individually misjudge,
not something this phase changed or was asked to fix. See
docs/approximations.md for the full writeup.

``contextual`` is not floor-tested as "beats random" under any ablation: every
combination mode tried (default/intersection, composite, union, all_detectors)
lands at 0.45-0.49 ROC-AUC -- consistently at chance, not occasionally by
seed luck (see docs/approximations.md). None of sorethumb's current
detector/combination choices learn the conditional "normal depends on
context" structure from an ordinary categorical feature column. Rather than
assert something the data shows is false, this is checked only for a
catastrophic regression (a real inversion/bug), not for beating random.

``swamping`` (P2-1, 2026-09-29): redesigned as a matched clean/contaminated
fit pair, both scored against the same designated at-risk-normal holdout --
see ``evaluate.scenarios`` module docstring. This population is single-class
by construction, so ROC-AUC/AP are undefined (NaN); it is guarded separately
below via ``flag_false_positive_rate``, not in ``_FLOORS``. See
``test_swamping_contamination_does_not_inflate_false_positives_on_at_risk_normals``
and docs/approximations.md for what was actually measured -- contrary to the
scenario's original naive premise ("training contamination causes nearby
genuinely-normal points to be falsely flagged more"), unlabelled training
contamination in this pipeline measurably *reduces* (never increases) the
flag rate on these points for the shipped default (intersection) combination
-- a masking effect (the natural-flag threshold widens to absorb the far-out
contamination), not the swamping effect the name originally implied.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkRow

pytestmark = pytest.mark.benchmark

# (scenario, ablation) -> ROC-AUC floor. Observed values on the 2026-09-20
# reference run are in the trailing comments; local/varying_density default
# floors observed 2026-09-27 (post P0-8's median-ranking fix).
_FLOORS: dict[tuple[str, str], float] = {
    ("point", "default"): 0.85,  # observed 0.97
    ("clustered", "default"): 0.85,  # observed 0.98 (default ensemble excludes lof)
    ("masking", "default"): 0.45,  # observed 0.57 -- masking is deliberately hard
    ("local", "combination_union"): 0.65,  # observed 0.84
    ("varying_density", "combination_union"): 0.65,  # observed 0.86
    ("local", "default"): 0.55,  # observed 0.71 -- was 0.04 (worse than random) before P0-8
    ("varying_density", "default"): 0.55,  # observed 0.68 -- was 0.24 before P0-8
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
        include_swamping=False,  # guarded separately -- see swamping_rows below
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


# ---------------------------------------------------------------------------
# Swamping (P2-1): guarded separately from _FLOORS -- see module docstring.
# ROC-AUC is undefined (NaN) on this scenario's single-class at-risk holdout
# by design, so flag_false_positive_rate is what's actually asserted here.
# ---------------------------------------------------------------------------

_SWAMPING_ABLATION = "default"  # the shipped default ensemble/combination
# Observed on a 2026-09-29 reference run (n_seeds=20): clean ~0.0110,
# contaminated ~0.0100 -- contamination measurably *reduces* the flag rate
# on at-risk normals here (masking), never inflates it. Generous margins
# below/above those observed values; update with a note if a real,
# reproducible change moves them.
_SWAMPING_CLEAN_FPR_CEILING = 0.05
_SWAMPING_CONTAMINATED_FPR_REGRESSION_MARGIN = 0.03


@pytest.fixture(scope="session")
def swamping_rows() -> dict[str, PipelineBenchmarkRow]:
    """Run the matched clean/contaminated swamping pair, 10 seeds.

    More seeds than ``pipeline_rows``' 3: the at-risk holdout is only 150
    points and the effect measured here is a small single-digit flag-count
    difference, too noisy at 3 seeds to assert on reliably (confirmed by
    hand: the observed direction flips seed-to-seed at n_seeds=3 for some
    ablations, but is stable by n_seeds=10).
    """
    from sorethumb.evaluate.pipeline_benchmark import PipelineBenchmarkConfig, run_pipeline_benchmark

    cfg = PipelineBenchmarkConfig(
        scenario_names=["__no_ordinary_scenarios__"],  # skip the unrelated per-scenario loop entirely
        ablation_names=[_SWAMPING_ABLATION],
        n_seeds=10,
        include_baselines=False,
        include_swamping=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = run_pipeline_benchmark(cfg)
    return {r.scenario: r for r in rows}


def test_swamping_contamination_does_not_inflate_false_positives_on_at_risk_normals(
    swamping_rows: dict[str, PipelineBenchmarkRow],
) -> None:
    """The regression this scenario actually guards against: unlabelled
    training contamination must never cause the shipped default ensemble to
    flag *more* of the genuinely-normal at-risk holdout than a clean fit
    does, beyond ordinary noise. Evidence so far shows the opposite (a mild
    masking effect, contamination reducing the flag rate) -- documented in
    docs/approximations.md and the module docstring above. This test would
    fail if a future change introduced a real, meaningful swamping
    (false-positive-inflating) effect."""
    clean = swamping_rows["swamping_clean"]
    contaminated = swamping_rows["swamping_contaminated"]
    assert clean.error is None, f"swamping_clean errored: {clean.error}"
    assert contaminated.error is None, f"swamping_contaminated errored: {contaminated.error}"

    assert clean.flag_false_positive_rate is not None
    assert contaminated.flag_false_positive_rate is not None
    assert clean.flag_false_positive_rate <= _SWAMPING_CLEAN_FPR_CEILING, (
        f"swamping_clean: flag_false_positive_rate {clean.flag_false_positive_rate:.4f} "
        f"exceeds the sanity ceiling {_SWAMPING_CLEAN_FPR_CEILING:.2f} -- the clean-fit baseline "
        "itself looks broken, not just the contaminated comparison."
    )
    assert contaminated.flag_false_positive_rate <= (
        clean.flag_false_positive_rate + _SWAMPING_CONTAMINATED_FPR_REGRESSION_MARGIN
    ), (
        f"swamping_contaminated: flag_false_positive_rate {contaminated.flag_false_positive_rate:.4f} "
        f"exceeds clean's {clean.flag_false_positive_rate:.4f} by more than the "
        f"{_SWAMPING_CONTAMINATED_FPR_REGRESSION_MARGIN:.2f} noise margin -- this looks like a real "
        "swamping (false-positive-inflating) regression, not the documented masking effect."
    )
