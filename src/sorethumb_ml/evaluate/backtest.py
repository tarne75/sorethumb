"""Backtest protocol: when a candidate ensemble policy may be said to outperform the default.

A benchmark matrix shows how policies behave; it cannot by itself justify
switching, because a rule tuned while looking at those numbers will look good
on them. This module is the protocol that guards against that:

1. **Pre-registration.** A candidate's comparison is written down before it is
   run on confirmation data: the candidate and baseline, the primary metric,
   the scenarios the hypothesis covers, how many of them must improve, the
   significance level and the stability tolerance. The file's sha256 is
   recorded with the result, so it cannot be edited afterwards unnoticed.
2. **Seed split.** Candidates are developed on :data:`DEVELOPMENT_SEEDS` (the
   policy matrix's seeds) and judged once on the disjoint
   :data:`CONFIRMATION_SEEDS`.
3. **Paired statistics.** Candidate and baseline run on the same seeds, so on
   the same data and split; each hypothesis scenario's per-seed differences in
   the primary metric get a paired bootstrap estimate, and the p-values are
   Holm-corrected across the hypothesis scenarios.
4. **Decision rule** (:func:`decide`). The candidate *outperforms* only if
   (a) the required hypothesis scenarios improve -- positive mean difference
   with a Holm-adjusted p-value at or below ``alpha``; (b) on *every*
   scenario, it meets each budget the baseline meets (means over the
   confirmation seeds); and (c) on every scenario its stability is no more
   than ``stability_tolerance`` below the baseline's.

A candidate that outperforms is not thereby made the default (see
``docs/ensemble-policy.md``); it becomes eligible to ship as a supported,
versioned option with this evidence attached.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sorethumb_ml.evaluate.budgets import BudgetCheck, Budgets
from sorethumb_ml.scoring.policies import DEFAULT_POLICY_ID, policy_by_id

DEVELOPMENT_SEEDS: tuple[int, ...] = tuple(range(10))
CONFIRMATION_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_020))
STABILITY_REFITS = 3

# Primary metrics: per-seed values where higher is better.
PRIMARY_METRICS = frozenset(
    {
        "average_precision",
        "roc_auc",
        "precision_at_k",
        "recall_at_k",
        "flag_precision",
        "flag_recall",
        "flag_f1",
    }
)
STABILITY_METRICS = frozenset({"flag_jaccard_stability", "rank_stability"})
# Matched swamping pair: scenario name -> the variant the harness runs.
SWAMPING_SCENARIOS = {"swamping_clean": "clean", "swamping_contaminated": "contaminated"}


@dataclass(frozen=True)
class Preregistration:
    """A candidate comparison, fixed before it is run on confirmation seeds."""

    id: str
    candidate: str
    baseline: str
    hypothesis: str
    primary_metric: str
    scenarios: tuple[str, ...]
    require: str  # "all" | "any"
    alpha: float
    stability_metric: str
    stability_tolerance: float
    n_resamples: int
    expected_outcome: str
    sha256: str


def _known_policy(policy_id: str) -> bool:
    from sorethumb_ml.evaluate.candidates import candidate_by_id  # noqa: PLC0415

    return policy_by_id(policy_id) is not None or candidate_by_id(policy_id) is not None


def load_preregistration(path: Path) -> Preregistration:
    """Parse and validate a pre-registration TOML file; its id is the file stem.

    Raises:
        ValueError: an unknown policy, scenario or metric, or an invalid setting.

    """
    from sorethumb_ml.evaluate.scenarios import SCENARIOS  # noqa: PLC0415

    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    prereg = Preregistration(
        id=path.stem,
        candidate=str(raw["candidate"]),
        baseline=str(raw.get("baseline", DEFAULT_POLICY_ID)),
        hypothesis=str(raw["hypothesis"]).strip(),
        primary_metric=str(raw["primary_metric"]),
        scenarios=tuple(raw["scenarios"]),
        require=str(raw.get("require", "all")),
        alpha=float(raw.get("alpha", 0.05)),
        stability_metric=str(raw.get("stability_metric", "flag_jaccard_stability")),
        stability_tolerance=float(raw.get("stability_tolerance", 0.05)),
        n_resamples=int(raw.get("n_resamples", 10_000)),
        expected_outcome=str(raw.get("expected_outcome", "")).strip(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    problems: list[str] = []
    for role in ("candidate", "baseline"):
        if not _known_policy(getattr(prereg, role)):
            problems.append(f"unknown {role} {getattr(prereg, role)!r}")
    if prereg.candidate == prereg.baseline:
        problems.append("candidate and baseline are the same policy")
    if prereg.primary_metric not in PRIMARY_METRICS:
        problems.append(f"primary_metric must be one of {sorted(PRIMARY_METRICS)}")
    if prereg.stability_metric not in STABILITY_METRICS:
        problems.append(f"stability_metric must be one of {sorted(STABILITY_METRICS)}")
    known = {s.name for s in SCENARIOS}
    unknown = [s for s in prereg.scenarios if s not in known]
    if not prereg.scenarios or unknown:
        problems.append(f"scenarios must be a non-empty list of benchmark scenarios; unknown: {unknown}")
    if prereg.require not in {"all", "any"}:
        problems.append("require must be 'all' or 'any'")
    if not 0.0 < prereg.alpha < 1.0:
        problems.append("alpha must be in (0, 1)")
    if prereg.stability_tolerance < 0.0:
        problems.append("stability_tolerance must be >= 0")
    if prereg.n_resamples < 1000:
        problems.append("n_resamples must be >= 1000")
    if not prereg.hypothesis:
        problems.append("hypothesis must be stated")
    if problems:
        msg = f"{path}: " + "; ".join(problems)
        raise ValueError(msg)
    return prereg


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedEstimate:
    """Paired-bootstrap estimate of the mean per-seed difference (candidate - baseline)."""

    mean: float
    ci_low: float
    ci_high: float
    p_value: float  # two-sided, for H0: mean difference = 0
    n: int


def paired_bootstrap(
    differences: np.ndarray, *, n_resamples: int, alpha: float, seed: int = 0
) -> PairedEstimate:
    """Resample the per-seed differences with replacement; percentile CI and two-sided p-value.

    The p-value is twice the smaller tail mass of the bootstrap means on either
    side of 0, with the +1 correction so it is never exactly 0. All-zero
    differences (the candidate behaves identically) give p = 1.
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return PairedEstimate(math.nan, math.nan, math.nan, 1.0, 0)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    below = (np.count_nonzero(means <= 0.0) + 1) / (n_resamples + 1)
    above = (np.count_nonzero(means >= 0.0) + 1) / (n_resamples + 1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return PairedEstimate(float(d.mean()), float(lo), float(hi), min(1.0, 2 * min(below, above)), int(d.size))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioComparison:
    """The primary-metric comparison on one hypothesis scenario."""

    scenario: str
    estimate: PairedEstimate
    p_holm: float
    improved: bool


@dataclass(frozen=True)
class Regression:
    """A budget or stability bound the baseline meets and the candidate does not."""

    scenario: str
    bound: str
    limit: float
    baseline: float
    candidate: float


@dataclass(frozen=True)
class BacktestDecision:
    """The verdict and everything it rests on."""

    outperforms: bool
    comparisons: tuple[ScenarioComparison, ...]
    budget_regressions: tuple[Regression, ...]
    stability_regressions: tuple[Regression, ...]
    reasons: tuple[str, ...] = field(default=())


def _num(value: float | None) -> float:
    """Read a stored metric as a float (published results write NaN as null)."""
    return math.nan if value is None else float(value)


def _mean(values: list[float | None]) -> float:
    arr = np.asarray([_num(v) for v in values], dtype=np.float64)
    return math.nan if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def decide(
    prereg: Preregistration,
    per_seed: dict[str, dict[str, list[dict[str, Any]]]],
    kinds: dict[str, str],
    budgets: Budgets,
) -> BacktestDecision:
    """Apply the pre-registered decision rule.

    *per_seed* maps scenario -> {"baseline": [...], "candidate": [...]}, each a
    list of the harness's per-seed metric dicts in the same seed order (so the
    pairs line up). *kinds* maps scenario -> kind, for the budgets.
    """
    comparisons: list[ScenarioComparison] = []
    estimates = []
    for scenario in prereg.scenarios:
        base = per_seed[scenario]["baseline"]
        cand = per_seed[scenario]["candidate"]
        diffs = np.array(
            [
                _num(c[prereg.primary_metric]) - _num(b[prereg.primary_metric])
                for b, c in zip(base, cand, strict=True)
            ],
            dtype=np.float64,
        )
        estimates.append(paired_bootstrap(diffs, n_resamples=prereg.n_resamples, alpha=prereg.alpha))
    adjusted = holm_adjust([e.p_value for e in estimates])
    for scenario, est, p_holm in zip(prereg.scenarios, estimates, adjusted, strict=True):
        improved = est.mean > 0.0 and p_holm <= prereg.alpha
        comparisons.append(ScenarioComparison(scenario, est, p_holm, improved))

    budget_regressions: list[Regression] = []
    stability_regressions: list[Regression] = []
    for scenario in sorted(per_seed):
        base = per_seed[scenario]["baseline"]
        cand = per_seed[scenario]["candidate"]
        for bound, limit in budgets.bounds_for(kinds[scenario]).items():
            metric = bound[4:]
            b_val = _mean([r[metric] for r in base])
            c_val = _mean([r[metric] for r in cand])
            if math.isnan(b_val) or math.isnan(c_val):
                continue

            b_ok = BudgetCheck(scenario, "", "", "", bound, limit, b_val).passed
            c_ok = BudgetCheck(scenario, "", "", "", bound, limit, c_val).passed
            if b_ok and not c_ok:
                budget_regressions.append(Regression(scenario, bound, limit, b_val, c_val))
        b_stab = _mean([r[prereg.stability_metric] for r in base])
        c_stab = _mean([r[prereg.stability_metric] for r in cand])
        if not (math.isnan(b_stab) or math.isnan(c_stab)) and c_stab < b_stab - prereg.stability_tolerance:
            stability_regressions.append(
                Regression(
                    scenario, prereg.stability_metric, b_stab - prereg.stability_tolerance, b_stab, c_stab
                )
            )

    n_improved = sum(c.improved for c in comparisons)
    hypothesis_met = n_improved == len(comparisons) if prereg.require == "all" else n_improved > 0
    reasons: list[str] = []
    if not hypothesis_met:
        reasons.append(
            f"{prereg.primary_metric} improved significantly on {n_improved} of {len(comparisons)} "
            f"hypothesis scenario(s); required: {prereg.require}"
        )
    reasons.extend(
        f"budget regression on {r.scenario}: {r.bound} {r.limit:g} met by the baseline "
        f"({r.baseline:.4f}) but not the candidate ({r.candidate:.4f})"
        for r in budget_regressions
    )
    reasons.extend(
        f"stability regression on {r.scenario}: {r.bound} {r.candidate:.4f} vs baseline {r.baseline:.4f} "
        f"(tolerance {prereg.stability_tolerance:g})"
        for r in stability_regressions
    )
    return BacktestDecision(
        outperforms=not reasons,
        comparisons=tuple(comparisons),
        budget_regressions=tuple(budget_regressions),
        stability_regressions=tuple(stability_regressions),
        reasons=tuple(reasons),
    )


def decision_as_dict(decision: BacktestDecision) -> dict[str, Any]:
    """JSON-ready form of a decision (NaN becomes None)."""

    def _clean(value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [_clean(v) for v in value]
        return value

    return dict(_clean(asdict(decision)))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def ablation_for(policy_id: str) -> Any:
    """Return the pipeline-benchmark ablation that runs *policy_id* with everything else at the default."""
    from sorethumb_ml.evaluate.pipeline_benchmark import ABLATIONS, AblationSpec  # noqa: PLC0415

    defaults = AblationSpec("_")
    for ablation in ABLATIONS:
        same_knobs = (
            ablation.pca == defaults.pca
            and ablation.scaler == defaults.scaler
            and ablation.detector_names == defaults.detector_names
            and ablation.near_constant_distinct == defaults.near_constant_distinct
        )
        if same_knobs and ablation.policy_id == policy_id:
            return ablation
    msg = f"no benchmark ablation runs {policy_id!r} at the default settings"
    raise ValueError(msg)


def backtest_scenarios() -> dict[str, str]:
    """Every scenario a backtest covers, mapped to its kind (the swamping pair included)."""
    from sorethumb_ml.evaluate.scenarios import SCENARIOS  # noqa: PLC0415

    return {s.name: s.kind for s in SCENARIOS} | dict.fromkeys(SWAMPING_SCENARIOS, "swamping")


_PER_SEED_KEYS = (
    "average_precision",
    "roc_auc",
    "precision_at_k",
    "recall_at_k",
    "flag_precision",
    "flag_recall",
    "flag_f1",
    "flag_false_positive_rate",
    "over_flag_ratio",
    "flag_jaccard_stability",
    "rank_stability",
    "policy_branch",
)


def run_policy_seeds(policy_id: str, seeds: tuple[int, ...]) -> dict[str, list[dict[str, Any]]]:
    """Run *policy_id* on every backtest scenario at *seeds*; per-seed metrics per scenario.

    Each (scenario, policy) cell runs in an isolated subprocess, exactly as the
    benchmark matrix does.

    Raises:
        RuntimeError: a cell failed.

    """
    from sorethumb_ml.evaluate.pipeline_benchmark import (  # noqa: PLC0415
        _pipeline_batch_worker,
        _run_isolated,
    )

    ablation = ablation_for(policy_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for scenario in backtest_scenarios():
        variant = SWAMPING_SCENARIOS.get(scenario)
        # A swamping variant takes only "point"'s review budget; its data is its own.
        name = "point" if variant is not None else scenario
        batch = _run_isolated(
            _pipeline_batch_worker, (name, variant, ablation, list(seeds), STABILITY_REFITS)
        )
        if batch["error"] is not None or len(batch["per_seed"]) != len(seeds):
            msg = f"{policy_id} on {scenario}: {batch['error'] or 'missing seeds'}"
            raise RuntimeError(msg)
        out[scenario] = [{k: r[k] for k in _PER_SEED_KEYS} for r in batch["per_seed"]]
    return out


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

BACKTEST_MARKER_START = "<!-- backtest-results-start -->"
BACKTEST_MARKER_END = "<!-- backtest-results-end -->"


def render_results(results: list[dict[str, Any]], results_link: str) -> str:
    """Markdown summary of confirmation results: one verdict table, then each hypothesis's estimates."""
    if not results:
        return "No confirmation backtest has been run yet.\n"
    lines = [
        "| pre-registration | candidate | baseline | primary metric | verdict | seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        pre = res["preregistration"]
        verdict = "outperforms" if res["decision"]["outperforms"] else "does not outperform"
        lines.append(
            f"| [`{pre['id']}`]({results_link}/{pre['id']}.json) | `{pre['candidate']}` | `{pre['baseline']}` "
            f"| {pre['primary_metric']} | **{verdict}** | {len(res['seeds'])} |"
        )
    for res in results:
        pre = res["preregistration"]
        lines += [
            "",
            f"**`{pre['id']}`** -- {pre['hypothesis']}",
            "",
            "| hypothesis scenario | mean difference | 95% CI | p (Holm) | improved |",
            "| --- | --- | --- | --- | --- |",
        ]
        for comp in res["decision"]["comparisons"]:
            est = comp["estimate"]
            lines.append(
                f"| {comp['scenario']} | {_fmt(est['mean'])} | [{_fmt(est['ci_low'])}, {_fmt(est['ci_high'])}] "
                f"| {comp['p_holm']:.4f} | {'yes' if comp['improved'] else 'no'} |"
            )
        reasons = res["decision"]["reasons"]
        if reasons:
            lines += ["", "Why it does not outperform:", ""]
            lines += [f"- {reason}" for reason in reasons]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def inject_results(doc_path: Path, results: list[dict[str, Any]], results_link: str) -> bool:
    """Replace the block between the backtest markers in *doc_path*. Returns True if it changed."""
    original = doc_path.read_text(encoding="utf-8")
    if BACKTEST_MARKER_START not in original or BACKTEST_MARKER_END not in original:
        msg = f"{doc_path} has no backtest-results markers"
        raise ValueError(msg)
    before = original[: original.index(BACKTEST_MARKER_START)]
    after = original[original.index(BACKTEST_MARKER_END) + len(BACKTEST_MARKER_END) :]
    auto_gen = (
        "<!-- AUTO-GENERATED — do not edit manually; run "
        "`python scripts/run_backtest.py --phase confirmation` to add a result. -->"
    )
    block = (
        f"{BACKTEST_MARKER_START}\n{auto_gen}\n\n{render_results(results, results_link)}{BACKTEST_MARKER_END}"
    )
    updated = before + block + after
    if updated == original:
        return False
    doc_path.write_text(updated, encoding="utf-8")
    return True
