"""Completion ledger built on the run / run_group / totals / period_execution tables.

``period_execution`` (migration 006) is the source of truth for period-level
completion, scoped to one (dataset_fp, period_label, config_hash): a period is
done only when a run against that exact configuration recorded it complete —
i.e. every group it discovered reached a non-failed terminal status. This lets
zero-anomaly periods be correctly distinguished from unprocessed ones, and
keeps two different configurations touching the same period_label from making
each other look done (or from double-counting when their totals are summed).

Every function here therefore takes an explicit ``config_hash``; there is no
default, because silently aggregating across configurations is exactly the
bug this module exists to prevent.
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


def last_complete_period(store: Store, dataset_fp: str, config_hash: str) -> str | None:
    """Return the most recent period_label completed under *config_hash*, or None."""
    return store.last_complete_period_label(dataset_fp, config_hash)


def completed_groups(store: Store, dataset_fp: str, period_label: str, config_hash: str) -> list[str]:
    """Return group_keys with a totals row for this (dataset_fp, period_label, config_hash)."""
    return store.completed_group_keys(dataset_fp, period_label, config_hash)


def periods_missing_groups(
    store: Store,
    dataset_fp: str,
    config_hash: str,
    requested_groups: list[str],
    granularity: PeriodGranularity,
    lookback_periods: int,
    reference_label: str,
) -> list[str]:
    """Return periods (under *config_hash*) that have some totals but are
    missing one or more requested groups.

    Bounds the requested list to groups this configuration has actually
    produced before, so a group that never occurs never re-queues the same
    periods forever. Typical caller: pass the group set discovered from the
    live dataset to catch periods that were completed before the group_by
    dimension widened.
    """
    if not requested_groups:
        return []

    existing = store.groups_seen_for_dataset(dataset_fp, config_hash)
    bounded = [g for g in requested_groups if g in existing]
    if not bounded:
        return []

    start_label = step_back(reference_label, granularity, lookback_periods)
    all_labels = period_range(start_label, reference_label, granularity)

    missing: list[str] = []
    for label in all_labels:
        done = set(completed_groups(store, dataset_fp, label, config_hash))
        if done and any(g not in done for g in bounded):
            missing.append(label)
    return missing


def clear_period(store: Store, dataset_fp: str, period_label: str, config_hash: str) -> None:
    """Remove this (dataset_fp, period_label, config_hash)'s totals + completion record."""
    store.delete_totals_for_period(dataset_fp, period_label, config_hash)
    logger.info("Cleared period %s (config %s) for dataset %s.", period_label, config_hash[:8], dataset_fp)


def resolve_backfill_range(
    store: Store,
    dataset_fp: str,
    config_hash: str,
    reference_label: str,
    granularity: PeriodGranularity,
    bootstrap_periods: int,
    lookback_periods: int,
    max_backfill_periods: int,
) -> list[str]:
    """Return an inclusive list of period labels to backfill under *config_hash*.

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
    last_complete = last_complete_period(store, dataset_fp, config_hash)

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
    config_hash: str,
    backfill_labels: list[str],
    forced_periods: list[str] | None = None,
) -> list[str]:
    """Return sorted pending periods from backfill_labels, minus ones complete under *config_hash*.

    Forced periods bypass the completion check. A zero-anomaly period that was
    previously processed is still complete and must be skipped — that is the
    entire point of keeping a ledger. A period where some but not all groups
    succeeded (one failed, or the process was interrupted before completion
    was recorded) is not complete and is returned for retry.
    """
    forced: set[str] = set(forced_periods or [])
    pending: list[str] = []
    for label in sorted(backfill_labels):
        if label in forced:
            pending.append(label)
            continue
        if not store.period_is_complete(dataset_fp, label, config_hash):
            pending.append(label)
    return pending
