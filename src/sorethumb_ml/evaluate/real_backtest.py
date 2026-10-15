"""Pre-registered backtests of an ensemble policy on real labelled datasets.

The synthetic rounds (``evaluate.backtest``) test policies on known anomaly
types. This module applies the same discipline to real data, through the same
code a user's ``sorethumb backtest`` runs (``evaluate.user_backtest``):

* **Pre-registration** (``tests/benchmark/preregistrations/real/<id>.toml``):
  candidate, baseline, the datasets the hypothesis covers, whether all or any
  must improve, ``alpha``, the stability tolerance and the expected outcome --
  committed before the confirmation run, its sha256 recorded with the result.
* **Development vs confirmation data.** Each dataset has two disjoint slices:
  a development slice to form the hypothesis on, and a confirmation slice
  judged once. For a dataset tested at random, they are disjoint row samples;
  for a time-ordered one, disjoint stretches of time (fit each period, score
  the next -- ``mode="forward"``).
* **Decision.** Per dataset, the candidate's per-replicate flag F1 is compared
  with the baseline's by a paired bootstrap; the p-values are Holm-corrected
  across the hypothesis datasets. The candidate outperforms only if the
  required datasets improve and, on every dataset, it neither spends more than
  the review budget on wrong rows where the baseline does not nor loses more
  than the tolerance in flag stability -- the rule ``sorethumb backtest`` uses.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sorethumb_ml.evaluate.backtest import holm_adjust, paired_bootstrap
from sorethumb_ml.evaluate.user_backtest import PRIMARY_METRIC, decide_candidates
from sorethumb_ml.scoring.policies import DEFAULT_POLICY_ID, policy_by_id

# Dataset ids the pre-registrations may name; their loaders live in
# scripts/run_real_backtest.py (they need the network and the `benchmark` extra).
REAL_DATASETS = ("kddcup99_sa", "covtype_2v4", "creditcard")


@dataclass(frozen=True)
class RealPreregistration:
    """A real-data comparison, fixed before its confirmation run."""

    id: str
    candidate: str
    baseline: str
    hypothesis: str
    datasets: tuple[str, ...]
    require: str
    alpha: float
    stability_tolerance: float
    n_resamples: int
    expected_outcome: str
    sha256: str
    # None: decision rule 1, as pre-registrations made before rule 2 were judged.
    ranking_tolerance: float | None = None


def load_real_preregistration(path: Path) -> RealPreregistration:
    """Parse and validate a real-data pre-registration; its id is the file stem.

    Raises:
        ValueError: an unknown policy or dataset, or an invalid setting.

    """
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    pre = RealPreregistration(
        id=path.stem,
        candidate=str(raw["candidate"]),
        baseline=str(raw.get("baseline", DEFAULT_POLICY_ID)),
        hypothesis=str(raw["hypothesis"]).strip(),
        datasets=tuple(raw["datasets"]),
        require=str(raw.get("require", "all")),
        alpha=float(raw.get("alpha", 0.05)),
        stability_tolerance=float(raw.get("stability_tolerance", 0.05)),
        n_resamples=int(raw.get("n_resamples", 10_000)),
        expected_outcome=str(raw.get("expected_outcome", "")).strip(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        ranking_tolerance=float(raw["ranking_tolerance"]) if "ranking_tolerance" in raw else None,
    )
    problems = []
    if policy_by_id(pre.candidate) is None:
        problems.append(f"unknown candidate {pre.candidate!r}")
    if pre.baseline != DEFAULT_POLICY_ID:
        problems.append(f"baseline must be the default, {DEFAULT_POLICY_ID}")
    if pre.candidate == pre.baseline:
        problems.append("candidate and baseline are the same policy")
    unknown = [d for d in pre.datasets if d not in REAL_DATASETS]
    if not pre.datasets or unknown or len(set(pre.datasets)) != len(pre.datasets):
        problems.append(f"datasets must be distinct, from {list(REAL_DATASETS)}; unknown: {unknown}")
    if pre.require not in {"all", "any"}:
        problems.append("require must be 'all' or 'any'")
    if not 0.0 < pre.alpha < 1.0:
        problems.append("alpha must be in (0, 1)")
    if pre.n_resamples < 1000:
        problems.append("n_resamples must be >= 1000")
    if not pre.hypothesis:
        problems.append("hypothesis must be stated")
    if problems:
        msg = f"{path}: " + "; ".join(problems)
        raise ValueError(msg)
    return pre


@dataclass(frozen=True)
class DatasetComparison:
    """The candidate against the baseline on one dataset."""

    dataset: str
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float
    improved: bool
    regressions: tuple[str, ...]


@dataclass(frozen=True)
class RealDecision:
    """The verdict across datasets and everything it rests on."""

    outperforms: bool
    comparisons: tuple[DatasetComparison, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (NaN becomes None)."""

        def _clean(v: Any) -> Any:
            if isinstance(v, float) and math.isnan(v):
                return None
            if isinstance(v, np.generic):
                return _clean(v.item())
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()}
            if isinstance(v, list | tuple):
                return [_clean(x) for x in v]
            return v

        return dict(_clean(asdict(self)))


def _num(value: float | None) -> float:
    return math.nan if value is None else float(value)


def decide_real(
    pre: RealPreregistration, per_dataset: dict[str, dict[str, list[dict[str, float]]]]
) -> RealDecision:
    """Apply the pre-registered rule to per-dataset, per-replicate metrics (pure; no fitting).

    *per_dataset* maps dataset -> {policy id -> [metrics per replicate]}, with the
    candidate's and baseline's lists in the same (paired) order.
    """
    estimates = []
    for d in pre.datasets:
        base, cand = per_dataset[d][pre.baseline], per_dataset[d][pre.candidate]
        diffs = np.array(
            [_num(c[PRIMARY_METRIC]) - _num(b[PRIMARY_METRIC]) for b, c in zip(base, cand, strict=True)],
            dtype=np.float64,
        )
        estimates.append(paired_bootstrap(diffs, n_resamples=pre.n_resamples, alpha=pre.alpha))
    adjusted = holm_adjust([e.p_value for e in estimates])

    comparisons = []
    reasons: list[str] = []
    for d, est, p_holm in zip(pre.datasets, estimates, adjusted, strict=True):
        # The budget and stability checks are sorethumb backtest's own; its flag F1
        # test is replaced by the cross-dataset Holm correction above.
        (single,) = decide_candidates(
            {DEFAULT_POLICY_ID: per_dataset[d][pre.baseline], pre.candidate: per_dataset[d][pre.candidate]},
            [pre.candidate],
            labelled=True,
            alpha=pre.alpha,
            stability_tolerance=pre.stability_tolerance,
            n_resamples=pre.n_resamples,
            ranking_tolerance=pre.ranking_tolerance,
        )
        regressions = tuple(r for r in single.reasons if not r.startswith("flag F1 not significantly"))
        improved = bool(est.mean > 0.0 and p_holm <= pre.alpha)
        comparisons.append(
            DatasetComparison(
                d, est.mean, est.ci_low, est.ci_high, est.p_value, float(p_holm), improved, regressions
            )
        )
        reasons.extend(f"{d}: {r}" for r in regressions)

    n_improved = sum(c.improved for c in comparisons)
    met = n_improved == len(comparisons) if pre.require == "all" else n_improved > 0
    if not met:
        reasons.insert(
            0,
            f"flag F1 improved significantly on {n_improved} of {len(comparisons)} dataset(s); "
            f"required: {pre.require}",
        )
    return RealDecision(outperforms=not reasons, comparisons=tuple(comparisons), reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

REAL_MARKER_START = "<!-- real-backtest-results-start -->"
REAL_MARKER_END = "<!-- real-backtest-results-end -->"


def render_real_results(results: list[dict[str, Any]], results_link: str) -> str:
    """Markdown summary of real-data confirmation results."""
    if not results:
        return "No real-data confirmation backtest has been run yet.\n"
    lines: list[str] = []
    for res in results:
        pre, dec = res["preregistration"], res["decision"]
        verdict = "outperforms" if dec["outperforms"] else "does not outperform"
        lines += [
            f"**[`{pre['id']}`]({results_link}/{pre['id']}.json)**: `{pre['candidate']}` vs "
            f"`{pre['baseline']}` -- **{verdict}**. {pre['hypothesis']}",
            "",
            "| dataset | slice | flag F1 difference | 95% CI | p (Holm) | improved | regressions |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for comp in dec["comparisons"]:
            info = res["datasets"][comp["dataset"]]
            lo, hi = comp["ci_low"], comp["ci_high"]
            ci = "n/a" if lo is None or hi is None else f"[{lo:+.4f}, {hi:+.4f}]"
            diff = "n/a" if comp["mean_difference"] is None else f"{comp['mean_difference']:+.4f}"
            lines.append(
                f"| {comp['dataset']} | {info['slice']} | {diff} | {ci} | {comp['p_holm']:.4f} | "
                f"{'yes' if comp['improved'] else 'no'} | {'; '.join(comp['regressions']) or '--'} |"
            )
        if dec["reasons"]:
            lines += ["", "Why it does not outperform:", ""]
            lines += [f"- {r}" for r in dec["reasons"]]
        lines.append("")
    return "\n".join(lines)


def inject_real_results(doc_path: Path, results: list[dict[str, Any]], results_link: str) -> bool:
    """Replace the block between the real-backtest markers in *doc_path*. Returns True if it changed."""
    original = doc_path.read_text(encoding="utf-8")
    if REAL_MARKER_START not in original or REAL_MARKER_END not in original:
        msg = f"{doc_path} has no real-backtest-results markers"
        raise ValueError(msg)
    before = original[: original.index(REAL_MARKER_START)]
    after = original[original.index(REAL_MARKER_END) + len(REAL_MARKER_END) :]
    auto_gen = (
        "<!-- AUTO-GENERATED — do not edit manually; run "
        "`python scripts/run_real_backtest.py --phase confirmation` to add a result. -->"
    )
    block = (
        f"{REAL_MARKER_START}\n{auto_gen}\n\n{render_real_results(results, results_link)}{REAL_MARKER_END}"
    )
    updated = before + block + after
    if updated == original:
        return False
    doc_path.write_text(updated, encoding="utf-8")
    return True
