"""The trend as four separate signals, built once and rendered by the CLI, ``--json`` and the report.

1. **Population**: did the input data change? Per-field PSI between windows,
   and volume (``period_stats``). No model is involved.
2. **Scores**: do the anchor run's fixed models score the data differently?
   The rate above the anchor's fixed threshold, score PSI against the anchor's
   own scores, and the share above its 95th percentile. Needs an anchor.
3. **Threshold**: did each period's own recalibrated threshold move against
   the anchor's? The gap between the own and anchor-scale rates, and its share
   of the change in count. Needs an anchor.
4. **Counts**: the own (self-fitted) anomaly rate per window, with intervals,
   and, with an anchor, the exact split of its change into volume, score drift
   and threshold drift.

Every section carries a verdict per window: ``ok``, ``caveats`` (with the
reasons) or ``not_available`` (with why). See docs/trends.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sorethumb_ml.history.windows import (
    DecompositionResult,
    DriftResult,
    TrendBasis,
    WindowResult,
    compute_count_decomposition,
    compute_population_drift,
    compute_rolling_windows,
)

if TYPE_CHECKING:
    from sorethumb_ml.config import Config
    from sorethumb_ml.store.db import Store

SECTIONS = ("population", "scores", "threshold", "counts")
REPORT_VERSION = 1


@dataclass(frozen=True)
class Verdict:
    """How far one section's figure for one window can be trusted."""

    status: str  # "ok" | "caveats" | "not_available"
    reasons: tuple[str, ...] = ()


@dataclass
class TrendReport:
    """Everything ``sorethumb history`` shows, for one dataset, configuration and reference period."""

    reference_label: str
    in_progress: bool
    config_hash: str
    granularity: str
    basis: TrendBasis
    anchor_configured: str | None
    windows: list[WindowResult] = field(default_factory=list)
    drift: list[DriftResult] = field(default_factory=list)
    decomposition: list[DecompositionResult] = field(default_factory=list)

    @property
    def window_sizes(self) -> list[int]:
        """Window sizes in ascending order."""
        return [w.window_size for w in self.windows]

    def verdict(self, section: str, window_size: int) -> Verdict:
        """Return the verdict for one section and window."""
        w = next(r for r in self.windows if r.window_size == window_size)
        d = next((r for r in self.drift if r.window_size == window_size), None)
        c = next((r for r in self.decomposition if r.window_size == window_size), None)
        if section == "population":
            return _population_verdict(w, d)
        if section in ("scores", "threshold"):
            return _anchor_verdict(
                c, self.anchor_configured, budget=section == "threshold" and self.basis == "budget"
            )
        if section == "counts":
            return _counts_verdict(w, self.basis)
        msg = f"unknown section {section!r}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of the whole report, verdicts included."""
        by_window: list[dict[str, Any]] = []
        for w in self.windows:
            d = next((r for r in self.drift if r.window_size == w.window_size), None)
            c = next((r for r in self.decomposition if r.window_size == w.window_size), None)
            counts = _as_dict(w)
            counts.pop("current_labels", None)
            counts.pop("prior_labels", None)
            by_window.append(
                {
                    "window_size": w.window_size,
                    "current_periods": w.current_labels,
                    "prior_periods": w.prior_labels,
                    "population": _as_dict(d) if d else None,
                    "anchor": _as_dict(c) if c else None,
                    "counts": counts,
                    "verdicts": {s: dataclasses.asdict(self.verdict(s, w.window_size)) for s in SECTIONS},
                }
            )
        return {
            "report_version": REPORT_VERSION,
            "reference_period": self.reference_label,
            "reference_in_progress": self.in_progress,
            "config_hash": self.config_hash,
            "granularity": self.granularity,
            "basis": self.basis,
            "anchor_run": self.anchor_configured,
            "windows": by_window,
        }


def _as_dict(obj: Any) -> dict[str, Any]:
    out = dataclasses.asdict(obj)
    for key, value in list(out.items()):
        if isinstance(value, tuple):
            out[key] = list(value)
    return out


def _population_verdict(w: WindowResult, d: DriftResult | None) -> Verdict:
    if d is None or not d.current_stats or not d.prior_stats:
        return Verdict("not_available", ("no per-period statistics on one side",))
    reasons = [
        r for r in w.comparability if r in ("group_mix_changed", "incomplete_current", "incomplete_prior")
    ]
    if not d.fields or all(v["psi"] is None for v in d.fields.values()):
        return Verdict("not_available", ("no field could be compared",))
    return Verdict("caveats" if reasons else "ok", tuple(reasons))


def _anchor_verdict(c: DecompositionResult | None, anchor: str | None, *, budget: bool) -> Verdict:
    if c is None or c.anchor_run_id is None:
        why = "no period in these windows was scored against the anchor" if anchor else "no anchor run is set"
        return Verdict("not_available", (why,))
    if c.anchor_run_id == "mixed":
        return Verdict("not_available", ("more than one anchor run in these windows",))
    if c.volume is None:
        return Verdict("not_available", ("no group has anchored periods in both windows",))
    reasons: list[str] = []
    if c.cells_excluded:
        reasons.append(f"{c.cells_excluded} (group, period) cell(s) left out")
    if budget:
        reasons.append("the own rate is fixed by the review budget, so threshold drift mirrors score drift")
    return Verdict("caveats" if reasons else "ok", tuple(reasons))


def _counts_verdict(w: WindowResult, basis: TrendBasis) -> Verdict:
    if w.current_rate is None:
        return Verdict("not_available", ("no totals in the current window",))
    reasons = list(w.comparability) + (["low_volume"] if w.low_volume else [])
    if basis == "budget":
        reasons.insert(0, "rate set by the review budget")
    return Verdict("caveats" if reasons else "ok", tuple(reasons))


def build_trend_report(
    store: Store,
    *,
    dataset_fp: str,
    config_hash: str,
    config: Config,
    reference_label: str,
    windows: list[int],
    in_progress: bool = False,
    group_keys: list[str] | None = None,
    match_groups: bool = True,
) -> TrendReport:
    """Compute every section of the trend for one dataset, configuration and reference period."""
    from sorethumb_ml.history.windows import trend_basis  # noqa: PLC0415

    granularity = config.history.period_granularity
    basis = trend_basis(config.scoring.combination, config.scoring.contamination)
    return TrendReport(
        reference_label=reference_label,
        in_progress=in_progress,
        config_hash=config_hash,
        granularity=granularity,
        basis=basis,
        anchor_configured=config.history.anchor_run,
        windows=compute_rolling_windows(
            store,
            dataset_fp,
            config_hash,
            reference_label,
            windows,
            granularity,
            group_keys=group_keys,
            low_volume_threshold=config.history.low_volume_threshold,
            basis=basis,
            roll_non_business=config.history.roll_non_business,
            match_groups=match_groups,
        ),
        drift=compute_population_drift(
            store,
            dataset_fp,
            config_hash,
            reference_label,
            windows,
            granularity,
            group_keys,
            match_groups=match_groups,
        ),
        decomposition=compute_count_decomposition(
            store, dataset_fp, config_hash, reference_label, windows, granularity, group_keys
        ),
    )
