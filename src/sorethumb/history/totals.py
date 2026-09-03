"""Anomaly count aggregation and population join at (group, period) grain.

Never copy a whole-period population onto every group row — that understates
the rate by the group count and breaks any later summation. Each group's
population comes from the population frame aggregated at the same grain.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import polars as pl

from sorethumb.errors import PopulationMismatchWarning
from sorethumb.store.workspace import make_group_key

if TYPE_CHECKING:
    from sorethumb.store.db import Store

logger = logging.getLogger(__name__)

_UNKNOWN_POPULATION: int = -1


def compute_totals(
    store: Store,
    results: pl.DataFrame,
    population: pl.DataFrame | None,
    group_by: list[str],
    period_label: str,
    dataset_fp: str,
    run_id: str,
) -> pl.DataFrame:
    """Aggregate anomaly counts and join to population at (group, period) grain.

    Parameters
    ----------
    store:
        Active store; rows are upserted on the natural key so re-running
        a period yields exactly one row per (dataset_fp, group_key, period_label).
    results:
        Per-row result frame. Must contain a boolean ``anomaly_flag`` column plus
        all columns named in group_by.
    population:
        Optional frame with the group_by columns and a numeric ``population``
        column. Aggregated at the same group grain before joining — never copied
        wholesale onto every result group.
    group_by:
        Columns defining the group dimension. Empty list means a single global group.
    period_label, dataset_fp, run_id:
        Stored on every upserted row for cross-period queries.

    Returns
    -------
    pl.DataFrame with columns: group_key, anomaly_count, population, rate.

    """
    agg = _aggregate_results(results, group_by, make_group_key)
    pop_agg = _aggregate_population(population, group_by, make_group_key, period_label)

    totals = _join_population(agg, pop_agg, period_label)

    # Upsert each row into the store
    for row in totals.iter_rows(named=True):
        gk = str(row["group_key"])
        anomaly_count = int(row["anomaly_count"])
        pop = int(row["population"])
        rate = row.get("rate")
        store.upsert_total(dataset_fp, gk, period_label, anomaly_count, pop, rate, run_id)

    return totals.select(["group_key", "anomaly_count", "population", "rate"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_results(
    results: pl.DataFrame,
    group_by: list[str],
    make_group_key: object,
) -> pl.DataFrame:
    if group_by:
        agg = (
            results.group_by(group_by)
            .agg(
                pl.col("anomaly_flag").cast(pl.Int64).sum().alias("anomaly_count"),
            )
            .with_columns(
                pl.struct(group_by)
                .map_elements(
                    lambda s: make_group_key({k: str(s[k]) for k in group_by}),  # type: ignore[operator]
                    return_dtype=pl.Utf8,
                )
                .alias("group_key")
            )
        )
    else:
        total = int(results["anomaly_flag"].cast(pl.Int64).sum())
        agg = pl.DataFrame({"group_key": ["__all__"], "anomaly_count": [total]})

    return agg


def _aggregate_population(
    population: pl.DataFrame | None,
    group_by: list[str],
    make_group_key: object,
    period_label: str,
) -> pl.DataFrame | None:
    if population is None or not group_by:
        return None

    missing_cols = [c for c in group_by if c not in population.columns]
    if missing_cols:
        warnings.warn(
            f"Population frame is missing group columns {missing_cols!r} "
            f"for period {period_label!r} (group misalignment). "
            "Population set to unknown (-1) and rate to null.",
            PopulationMismatchWarning,
            stacklevel=4,
        )
        return None

    if "population" not in population.columns:
        warnings.warn(
            f"Population frame has no 'population' column for period {period_label!r}. "
            "Population set to unknown (-1) and rate to null.",
            PopulationMismatchWarning,
            stacklevel=4,
        )
        return None

    return (
        population.group_by(group_by)
        .agg(pl.col("population").sum().alias("population"))
        .with_columns(
            pl.struct(group_by)
            .map_elements(
                lambda s: make_group_key({k: str(s[k]) for k in group_by}),  # type: ignore[operator]
                return_dtype=pl.Utf8,
            )
            .alias("group_key")
        )
        .select(["group_key", "population"])
    )


def _join_population(
    agg: pl.DataFrame,
    pop_agg: pl.DataFrame | None,
    period_label: str,
) -> pl.DataFrame:
    if pop_agg is not None:
        totals = agg.join(pop_agg, on="group_key", how="left").with_columns(
            pl.col("population").fill_null(_UNKNOWN_POPULATION)
        )
        # Warn if the join produced no matches at all (period misalignment)
        if totals["population"].eq(_UNKNOWN_POPULATION).all():
            warnings.warn(
                f"Population frame matched no groups for period {period_label!r}. "
                "This suggests period misalignment rather than column misalignment.",
                PopulationMismatchWarning,
                stacklevel=4,
            )
    else:
        totals = agg.with_columns(pl.lit(_UNKNOWN_POPULATION).alias("population"))

    return totals.with_columns(
        pl.when(pl.col("population") > 0)
        .then(pl.col("anomaly_count").cast(pl.Float64) / pl.col("population").cast(pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("rate")
    )
