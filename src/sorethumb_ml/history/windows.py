"""Rolling window statistics over the anomaly history.

What these numbers are
----------------------
Every period run fits its own detectors and calibrates on its own rows, so a
period's anomaly rate is measured against *that period's own* notion of
normal. The trend is therefore a sequence of independently calibrated
snapshots, and what its rate can mean depends on how flags were decided
(:func:`trend_basis`):

``budget``
    ``combination="composite"`` with a numeric ``contamination``: every period
    flags exactly ``round(n * contamination)`` rows. The rate is fixed by the
    review budget, so a change in it is rounding, not a change in the data.
    Counts are reported; the change between windows is not.
``self_referenced``
    Anything else: each period flags against boundaries fitted to its own rows.
    A change mixes real change with how far each period's boundary moved, so
    it is a relative signal, not an absolute anomaly level.

How the windows are computed
----------------------------
- Rates are volume-weighted: ``sum(anomalies) / sum(population)``. Rows of
  unknown population (sentinel -1) are excluded from both sides.
- **Matched groups.** By default only groups present in both the current and
  the prior window are summed, so a group appearing or disappearing cannot move
  the rate on its own (a Simpson's-paradox artefact). The composition is
  reported. When either window has no groups at all there is nothing to match
  and each window keeps all of its own groups.
- **Coverage.** Each window reports how many of its expected periods have
  totals (weekend days are not expected when ``roll_non_business`` is set).
- **Uncertainty.** Each rate carries a Wilson 95% interval and the change a
  Newcombe (hybrid score) interval. They treat rows as independent Bernoulli
  trials, so they capture sampling noise only, not the boundary movement
  described above. The relative change (``pct_change``) is withheld when the
  prior window has fewer than ``min_prior_count`` anomalies.
- **Comparability.** A window lists every reason its comparison may mislead:
  another configuration has totals in the span (``other_config_in_span``),
  runs in the span used different library versions or ensemble policies,
  coverage is incomplete, or the groups differ between the windows.

Aggregation is always scoped to one ``config_hash``: totals for the same
(dataset_fp, group_key, period_label) under a different configuration are
separate rows (migration 002) and are never summed together.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sorethumb_ml.history.periods import PeriodGranularity, filter_non_business, period_range, step_back

if TYPE_CHECKING:
    from sorethumb_ml.store.db import Store

logger = logging.getLogger(__name__)

_UNKNOWN_POPULATION: int = -1
_Z95 = 1.959963984540054

TrendBasis = Literal["budget", "self_referenced"]

# Comparability codes, with what each means (shown in docs and the report).
COMPARABILITY: dict[str, str] = {
    "other_config_in_span": "another configuration has totals in this span; they are excluded from these sums",
    "mixed_library_versions": "runs in this span used different sorethumb versions",
    "mixed_ensemble_policies": "runs in this span combined detector votes under different ensemble policies",
    "incomplete_current": "the current window is missing some of its periods",
    "incomplete_prior": "the prior window is missing some of its periods",
    "group_mix_changed": "some groups appear in only one window; they are left out of both",
}


def trend_basis(combination: str, contamination: str | float) -> TrendBasis:
    """Return what a period's anomaly rate can mean under this scoring configuration (see the module docstring)."""
    if combination == "composite" and not isinstance(contamination, str):
        return "budget"
    return "self_referenced"


def wilson_interval(count: int, n: int, z: float = _Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion; None when *n* is 0."""
    if n <= 0:
        return None
    p = count / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Exact at the edges: no rounding residue such as 7e-18 for a count of 0.
    lower = 0.0 if count == 0 else max(0.0, centre - half)
    upper = 1.0 if count == n else min(1.0, centre + half)
    return lower, upper


def newcombe_interval(
    count_1: int, n_1: int, count_2: int, n_2: int, z: float = _Z95
) -> tuple[float, float] | None:
    """Newcombe's hybrid score interval for ``p1 - p2`` (method 10); None when either n is 0."""
    ci_1 = wilson_interval(count_1, n_1, z)
    ci_2 = wilson_interval(count_2, n_2, z)
    if ci_1 is None or ci_2 is None:
        return None
    p1, p2 = count_1 / n_1, count_2 / n_2
    (l1, u1), (l2, u2) = ci_1, ci_2
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


@dataclass(frozen=True)
class WindowResult:
    """Rolling window comparison for one window size at one reference period."""

    window_size: int
    period_label: str
    current_anomaly_count: int
    current_population: int
    current_rate: float | None
    prior_anomaly_count: int
    prior_population: int
    prior_rate: float | None
    absolute_change: float | None  # None under the "budget" basis or without a prior rate
    pct_change: float | None  # also None when the prior count is below min_prior_count
    low_volume: bool  # either window's population is below the threshold
    calibration_break: bool  # "other_config_in_span" (kept for existing readers)
    basis: TrendBasis = "self_referenced"
    current_rate_ci: tuple[float, float] | None = None
    prior_rate_ci: tuple[float, float] | None = None
    change_ci: tuple[float, float] | None = None
    change_significant: bool | None = None  # change_ci excludes 0
    current_periods_covered: int = 0
    current_periods_expected: int = 0
    prior_periods_covered: int = 0
    prior_periods_expected: int = 0
    groups_matched: int = 0
    groups_current_only: int = 0
    groups_prior_only: int = 0
    comparability: tuple[str, ...] = ()
    current_labels: list[str] = field(default_factory=list, compare=False)
    prior_labels: list[str] = field(default_factory=list, compare=False)

    @property
    def coverage_complete(self) -> bool:
        """Both windows have totals for every period they expect."""
        return (
            self.current_periods_covered == self.current_periods_expected
            and self.prior_periods_covered == self.prior_periods_expected
        )


def compute_rolling_windows(
    store: Store,
    dataset_fp: str,
    config_hash: str,
    reference_label: str,
    windows: list[int],
    granularity: PeriodGranularity,
    group_keys: list[str] | None = None,
    low_volume_threshold: int = 100,
    *,
    basis: TrendBasis = "self_referenced",
    roll_non_business: bool = False,
    match_groups: bool = True,
    min_prior_count: int = 10,
) -> list[WindowResult]:
    """Compute rolling window stats for each window size ending at reference_label.

    Parameters
    ----------
    store:
        Active store for totals queries.
    dataset_fp:
        Dataset fingerprint.
    config_hash:
        Configuration whose totals to aggregate. Mandatory: totals recorded
        under a different configuration for the same dataset/period/group are
        separate rows and must never be summed into this trend.
    reference_label:
        The period being reported on (end of the "current" window).
    windows:
        Window sizes in periods (e.g. [1, 7, 14, 28]).
    granularity:
        Period granularity string; passed to period navigation helpers.
    group_keys:
        Optional allow-list: only these groups contribute.
    low_volume_threshold:
        A window is low volume when its current or prior population is below this.
    basis:
        What the rate means (:func:`trend_basis`). Under ``"budget"`` the
        change between windows is not reported.
    roll_non_business:
        Weekend days are not expected periods (``history.roll_non_business``).
    match_groups:
        Sum only groups present in both windows (see the module docstring).
    min_prior_count:
        Withhold ``pct_change`` when the prior window has fewer anomalies.

    Returns
    -------
    One WindowResult per window size, sorted by window_size ascending.

    """
    if not windows:
        return []

    results: list[WindowResult] = []
    for w in sorted(windows):
        cur_start = step_back(reference_label, granularity, w - 1)
        cur_labels = period_range(cur_start, reference_label, granularity)
        pri_end = step_back(cur_start, granularity, 1)
        pri_start = step_back(cur_start, granularity, w)
        pri_labels = period_range(pri_start, pri_end, granularity)

        cur_rows = _known_rows(store.totals_for_periods(dataset_fp, cur_labels, config_hash, group_keys))
        pri_rows = _known_rows(store.totals_for_periods(dataset_fp, pri_labels, config_hash, group_keys))

        cur_groups = {str(r["group_key"]) for r in cur_rows}
        pri_groups = {str(r["group_key"]) for r in pri_rows}
        matched = cur_groups & pri_groups
        if match_groups and cur_groups and pri_groups:
            cur_rows = [r for r in cur_rows if r["group_key"] in matched]
            pri_rows = [r for r in pri_rows if r["group_key"] in matched]

        cur_count, cur_pop = _sum(cur_rows)
        pri_count, pri_pop = _sum(pri_rows)
        cur_rate = cur_count / cur_pop if cur_pop > 0 else None
        pri_rate = pri_count / pri_pop if pri_pop > 0 else None

        abs_change: float | None = None
        pct_change: float | None = None
        change_ci: tuple[float, float] | None = None
        if basis == "self_referenced" and cur_rate is not None and pri_rate is not None:
            abs_change = cur_rate - pri_rate
            change_ci = newcombe_interval(cur_count, cur_pop, pri_count, pri_pop)
            if pri_rate > 0.0 and pri_count >= min_prior_count:
                pct_change = abs_change / pri_rate

        cur_expected = filter_non_business(cur_labels, granularity, roll_non_business)
        pri_expected = filter_non_business(pri_labels, granularity, roll_non_business)
        cur_covered = len({str(r["period_label"]) for r in cur_rows} & set(cur_expected))
        pri_covered = len({str(r["period_label"]) for r in pri_rows} & set(pri_expected))

        all_labels = pri_labels + cur_labels
        other_configs = bool(store.other_config_hashes_for_periods(dataset_fp, all_labels, config_hash))
        comparability = _comparability(
            store,
            run_ids={str(r["run_id"]) for r in cur_rows + pri_rows},
            other_configs=other_configs,
            current_incomplete=cur_covered < len(cur_expected),
            prior_incomplete=pri_covered < len(pri_expected),
            group_mix_changed=bool(cur_groups and pri_groups and cur_groups != pri_groups),
        )

        results.append(
            WindowResult(
                window_size=w,
                period_label=reference_label,
                current_anomaly_count=cur_count,
                current_population=cur_pop,
                current_rate=cur_rate,
                prior_anomaly_count=pri_count,
                prior_population=pri_pop,
                prior_rate=pri_rate,
                absolute_change=abs_change,
                pct_change=pct_change,
                low_volume=cur_pop < low_volume_threshold
                or (bool(pri_rows) and pri_pop < low_volume_threshold),
                calibration_break=other_configs,
                basis=basis,
                current_rate_ci=wilson_interval(cur_count, cur_pop),
                prior_rate_ci=wilson_interval(pri_count, pri_pop),
                change_ci=change_ci,
                change_significant=None if change_ci is None else not (change_ci[0] <= 0.0 <= change_ci[1]),
                current_periods_covered=cur_covered,
                current_periods_expected=len(cur_expected),
                prior_periods_covered=pri_covered,
                prior_periods_expected=len(pri_expected),
                groups_matched=len(matched),
                groups_current_only=len(cur_groups - pri_groups),
                groups_prior_only=len(pri_groups - cur_groups),
                comparability=comparability,
                current_labels=cur_labels,
                prior_labels=pri_labels,
            )
        )

    return results


def default_reference_label(
    store: Store, dataset_fp: str, config_hash: str, current_label: str
) -> tuple[str | None, bool]:
    """Return the period a trend should end at, as ``(label, in_progress)``.

    The most recent period with totals that has already ended (its label
    before *current_label*, the period now in progress). When the only totals
    are for the period in progress, that period is returned with
    ``in_progress=True`` so the caller can say so; ``(None, False)`` when
    there are no totals at all.
    """
    labels = store.period_labels_with_totals(dataset_fp, config_hash)
    if not labels:
        return None, False
    elapsed = [label for label in labels if label < current_label]
    if elapsed:
        return max(elapsed), False
    return max(labels), True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _known_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows of unknown population (sentinel -1): they count on neither side."""
    return [r for r in rows if int(r["population"]) != _UNKNOWN_POPULATION]


def _sum(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(int(r["anomaly_count"]) for r in rows), sum(int(r["population"]) for r in rows)


def _comparability(
    store: Store,
    *,
    run_ids: set[str],
    other_configs: bool,
    current_incomplete: bool,
    prior_incomplete: bool,
    group_mix_changed: bool,
) -> tuple[str, ...]:
    codes: list[str] = []
    if other_configs:
        codes.append("other_config_in_span")
    provenance = store.run_provenance(sorted(run_ids)) if run_ids else {}
    if len({p.get("library_version") for p in provenance.values()}) > 1:
        codes.append("mixed_library_versions")
    if len({p.get("ensemble_policy") for p in provenance.values()}) > 1:
        codes.append("mixed_ensemble_policies")
    if current_incomplete:
        codes.append("incomplete_current")
    if prior_incomplete:
        codes.append("incomplete_prior")
    if group_mix_changed:
        codes.append("group_mix_changed")
    return tuple(codes)
