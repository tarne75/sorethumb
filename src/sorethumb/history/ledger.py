"""Completion ledger built on the run / run_group / totals tables.

The totals table is the source of truth for period-level completion: a period is
considered done once compute_totals() has upserted rows for it. This lets
zero-anomaly periods be correctly distinguished from unprocessed periods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sorethumb.history.periods import PeriodGranularity, period_range, step_back, step_forward

if TYPE_CHECKING:
    from sorethumb.store.db import Store

logger = logging.getLogger(__name__)


def mark_run(
    store: Store,
    run_id: str,
    dataset_fp: str,
    config_json: str,
    seed: int,
) -> None:
    """Upsert a run row (idempotent)."""
    store.insert_run(run_id, dataset_fp, config_json, seed)


def mark_group(
    store: Store,
    run_id: str,
    group_key: str,
    group_values_json: str,
    group_label: str,
    *,
    status: str = "complete",
    record_count: int | None = None,
    anomaly_count: int | None = None,
) -> None:
    """Upsert a run_group row (idempotent)."""
    store.upsert_run_group(
        run_id,
        group_key,
        group_values_json,
        group_label,
        status=status,
        record_count=record_count,
        anomaly_count=anomaly_count,
    )


def last_complete_period(store: Store, dataset_fp: str) -> str | None:
    """Return the most recent period_label with at least one totals row, or None."""
    return store.last_complete_period_label(dataset_fp)


def completed_groups(store: Store, dataset_fp: str, period_label: str) -> list[str]:
    """Return group_keys that have a totals row for this (dataset_fp, period_label)."""
    return store.completed_group_keys(dataset_fp, period_label)


def periods_missing_groups(
    store: Store,
    dataset_fp: str,
    requested_groups: list[str],
    granularity: PeriodGranularity,
    lookback_periods: int,
    reference_label: str,
) -> list[str]:
    """Return periods that have some totals but are missing one or more requested groups.

    Bounds the requested list to groups the dataset has actually produced, so a
    group that never occurs never re-queues the same periods forever.
    """
    if not requested_groups:
        return []

    existing = store.groups_seen_for_dataset(dataset_fp)
    bounded = [g for g in requested_groups if g in existing]
    if not bounded:
        return []

    start_label = step_back(reference_label, granularity, lookback_periods)
    all_labels = period_range(start_label, reference_label, granularity)

    missing: list[str] = []
    for label in all_labels:
        done = set(completed_groups(store, dataset_fp, label))
        if done and any(g not in done for g in bounded):
            missing.append(label)
    return missing


def clear_period(store: Store, dataset_fp: str, period_label: str) -> None:
    """Remove all totals rows for this period, forcing it to reprocess."""
    store.delete_totals_for_period(dataset_fp, period_label)
    logger.info("Cleared period %s for dataset %s.", period_label, dataset_fp)


def resolve_backfill_range(
    store: Store,
    dataset_fp: str,
    reference_label: str,
    granularity: PeriodGranularity,
    bootstrap_periods: int,
    lookback_periods: int,
    max_backfill_periods: int,
) -> list[str]:
    """Return an inclusive list of period labels to backfill.

    The caller owns the reference period. This function returns labels ending at
    reference − 1 so the backfill and the live run never race over the same period.

    Three branches — all span exactly bootstrap_periods or lookback_periods:
    1. Cold start  (no complete periods): bootstrap_periods back from reference.
    2. Warm        (last_complete < reference): from last_complete + 1,
                   capped at lookback_periods back.
    3. Already complete (last_complete >= reference): full lookback_periods scan
                   to pick up any periods that were skipped.

    The result is clamped to max_backfill_periods (most recent). An empty list
    is a normal outcome when there is nothing to do.
    """
    end_label = step_back(reference_label, granularity, 1)
    last_complete = last_complete_period(store, dataset_fp)

    if last_complete is None:
        # Branch 1: cold start
        start = step_back(reference_label, granularity, bootstrap_periods)
        labels = period_range(start, end_label, granularity)
        logger.info(
            "Cold-start backfill for %s: %d periods (%s → %s).",
            dataset_fp,
            len(labels),
            start,
            end_label,
        )

    elif last_complete >= reference_label:
        # Branch 3: reference already complete — full lookback scan for gaps
        start = step_back(reference_label, granularity, lookback_periods)
        labels = period_range(start, end_label, granularity)
        logger.info(
            "Reference-complete backfill scan for %s: %d periods (%s → %s).",
            dataset_fp,
            len(labels),
            start,
            end_label,
        )

    else:
        # Branch 2: warm continuation
        start = step_forward(last_complete, granularity, 1)
        furthest_allowed = step_back(reference_label, granularity, lookback_periods)
        start = max(start, furthest_allowed)
        labels = period_range(start, end_label, granularity)
        logger.info(
            "Warm backfill for %s: %d periods (%s → %s).",
            dataset_fp,
            len(labels),
            start,
            end_label,
        )

    if len(labels) > max_backfill_periods:
        labels = labels[-max_backfill_periods:]

    if not labels:
        logger.info("Nothing to backfill for dataset %s.", dataset_fp)

    return labels


def iter_pending_periods(
    store: Store,
    dataset_fp: str,
    backfill_labels: list[str],
    forced_periods: list[str] | None = None,
) -> list[str]:
    """Return sorted pending periods from backfill_labels, minus already-complete ones.

    Forced periods bypass the completion check. A zero-anomaly period that was
    previously processed is still complete and must be skipped — that is the
    entire point of keeping a ledger.
    """
    forced: set[str] = set(forced_periods or [])
    pending: list[str] = []
    for label in sorted(backfill_labels):
        if label in forced:
            pending.append(label)
            continue
        if not completed_groups(store, dataset_fp, label):
            pending.append(label)
    return pending
