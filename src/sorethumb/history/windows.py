"""Rolling window statistics over the anomaly history.

Rates are volume-weighted: sum(anomalies) / sum(population). An unweighted
mean of per-group rates is mathematically wrong when groups differ in size.
Unknown-population rows (sentinel -1) are excluded from both sides of the ratio.
A window that spans a calibration-mode change is flagged rather than averaged,
because comparing self-calibrated and reference-calibrated scores is invalid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sorethumb.history.periods import PeriodGranularity, period_range, step_back

if TYPE_CHECKING:
    from sorethumb.store.db import Store

logger = logging.getLogger(__name__)

_UNKNOWN_POPULATION: int = -1


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
    absolute_change: float | None
    pct_change: float | None
    low_volume: bool
    calibration_break: bool
    current_labels: list[str] = field(default_factory=list, compare=False)
    prior_labels: list[str] = field(default_factory=list, compare=False)


def compute_rolling_windows(
    store: Store,
    dataset_fp: str,
    reference_label: str,
    windows: list[int],
    granularity: PeriodGranularity,
    group_keys: list[str] | None = None,
    low_volume_threshold: int = 100,
) -> list[WindowResult]:
    """Compute rolling window stats for each window size ending at reference_label.

    Parameters
    ----------
    store:
        Active store for totals queries.
    dataset_fp:
        Dataset fingerprint.
    reference_label:
        The period being reported on (end of the "current" window).
    windows:
        Window sizes in periods (e.g. [1, 7, 14, 28]).
    granularity:
        Period granularity string; passed to period navigation helpers.
    group_keys:
        Optional allow-list. Only these groups contribute, preventing a phantom
        rate collapse when the run scope narrows temporarily.
    low_volume_threshold:
        Flag the shortest window as low volume when current_population < threshold.

    Returns
    -------
    One WindowResult per window size, sorted by window_size ascending.

    """
    if not windows:
        return []

    shortest = min(windows)
    results: list[WindowResult] = []

    for w in sorted(windows):
        # Current window: w periods ending at reference_label (inclusive)
        cur_start = step_back(reference_label, granularity, w - 1)
        cur_labels = period_range(cur_start, reference_label, granularity)

        # Prior window: w periods immediately before the current window
        pri_end = step_back(cur_start, granularity, 1)
        pri_start = step_back(cur_start, granularity, w)
        pri_labels = period_range(pri_start, pri_end, granularity)

        cur = _aggregate(store, dataset_fp, cur_labels, group_keys)
        pri = _aggregate(store, dataset_fp, pri_labels, group_keys)

        # Calibration break: multiple distinct modes across the combined span
        all_labels = pri_labels + cur_labels
        cal_break = len(store.calibration_modes_for_periods(dataset_fp, all_labels)) > 1

        cur_rate = _safe_rate(cur["anomaly_count"], cur["population"])
        pri_rate = _safe_rate(pri["anomaly_count"], pri["population"])

        abs_change: float | None = None
        pct_change: float | None = None
        if cur_rate is not None and pri_rate is not None:
            abs_change = cur_rate - pri_rate
            if pri_rate > 0.0:
                pct_change = abs_change / pri_rate

        is_low_volume = (w == shortest) and (cur["population"] < low_volume_threshold)

        results.append(
            WindowResult(
                window_size=w,
                period_label=reference_label,
                current_anomaly_count=cur["anomaly_count"],
                current_population=cur["population"],
                current_rate=cur_rate,
                prior_anomaly_count=pri["anomaly_count"],
                prior_population=pri["population"],
                prior_rate=pri_rate,
                absolute_change=abs_change,
                pct_change=pct_change,
                low_volume=is_low_volume,
                calibration_break=cal_break,
                current_labels=cur_labels,
                prior_labels=pri_labels,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate(
    store: Store,
    dataset_fp: str,
    period_labels: list[str],
    group_keys: list[str] | None,
) -> dict[str, int]:
    """Sum anomaly_count and population over period_labels, excluding unknown population."""
    rows = store.totals_for_periods(dataset_fp, period_labels, group_keys)
    anomaly_count = 0
    population = 0
    for row in rows:
        if int(row["population"]) == _UNKNOWN_POPULATION:
            continue
        anomaly_count += int(row["anomaly_count"])
        population += int(row["population"])
    return {"anomaly_count": anomaly_count, "population": population}


def _safe_rate(anomaly_count: int, population: int) -> float | None:
    if population <= 0:
        return None
    return anomaly_count / population
