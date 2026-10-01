"""Cohort-level contrast: how the flagged cohort differs from the rest as a group.

Per-record explanations answer "why is this row odd". Contrast answers "how does the
flagged set differ from the unflagged set". It is diagnostic output computed after
flagging; a failure here must never fail a run — all exceptions are caught and logged.

``contrast_score`` is on one bounded scale for every column kind, so numeric and
categorical columns can be ranked in the same table: the largest difference in
probability mass the two cohorts assign to any one event, in ``[0, 1]``. For a
numeric column that is the two-sample Kolmogorov-Smirnov statistic (events are
half-lines); for a categorical column it is the total variation distance (events
are sets of categories). 0 means identical distributions, 1 means disjoint
support, and neither depends on how many rows either cohort has.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 2  # need at least 2 samples per group for meaningful stats


def compute_contrast(
    flagged: pl.DataFrame,
    unflagged: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    top_n: int = 10,
) -> pl.DataFrame:
    """Compare flagged vs unflagged cohorts column by column.

    Returns a Polars frame sorted by descending contrast score with columns:
    feature, kind (numeric|categorical), stat_name, stat_value, contrast_score
    (``stat_value`` equals ``contrast_score``; ``stat_name`` says which statistic it is).

    Failures on individual columns are caught and skipped so a single bad column
    never aborts the contrast computation.
    """
    rows: list[dict[str, Any]] = []

    for col in numeric_cols:
        if col not in flagged.columns or col not in unflagged.columns:
            continue
        try:
            row = _numeric_contrast(col, flagged[col], unflagged[col])
            if row:
                rows.append(row)
        except Exception:  # noqa: BLE001
            logger.debug("Contrast skipped for column %r.", col)

    for col in categorical_cols:
        if col not in flagged.columns or col not in unflagged.columns:
            continue
        try:
            row = _categorical_contrast(col, flagged[col], unflagged[col])
            if row:
                rows.append(row)
        except Exception:  # noqa: BLE001
            logger.debug("Contrast skipped for column %r.", col)

    if not rows:
        return pl.DataFrame(
            schema={
                "feature": pl.Utf8,
                "kind": pl.Utf8,
                "stat_name": pl.Utf8,
                "stat_value": pl.Float64,
                "contrast_score": pl.Float64,
            }
        )

    df = pl.DataFrame(rows).sort("contrast_score", descending=True)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# Per-column helpers
# ---------------------------------------------------------------------------


def _numeric_contrast(
    col: str,
    flagged_series: pl.Series,
    unflagged_series: pl.Series,
) -> dict[str, Any] | None:
    from scipy.stats import ks_2samp  # noqa: PLC0415

    a = flagged_series.drop_nulls().to_numpy().astype(float)
    b = unflagged_series.drop_nulls().to_numpy().astype(float)

    if len(a) < _MIN_SAMPLES or len(b) < _MIN_SAMPLES:
        return None

    ks_stat, _ = ks_2samp(a, b)

    return {
        "feature": col,
        "kind": "numeric",
        "stat_name": "ks_statistic",
        "stat_value": float(ks_stat),
        "contrast_score": float(ks_stat),
    }


def _categorical_contrast(
    col: str,
    flagged_series: pl.Series,
    unflagged_series: pl.Series,
) -> dict[str, Any] | None:
    a = flagged_series.drop_nulls().cast(pl.Utf8)
    b = unflagged_series.drop_nulls().cast(pl.Utf8)

    if len(a) < _MIN_SAMPLES or len(b) < _MIN_SAMPLES:
        return None

    total_a = len(a)
    total_b = len(b)

    freq_a = {v: c / total_a for v, c in a.value_counts().iter_rows()}
    freq_b = {v: c / total_b for v, c in b.value_counts().iter_rows()}

    # Total variation distance: half the L1 distance between the two category
    # distributions. A category present in only one cohort contributes its full
    # frequency; nothing is divided by a possibly-zero count.
    tvd = 0.5 * sum(abs(freq_a.get(cat, 0.0) - freq_b.get(cat, 0.0)) for cat in freq_a.keys() | freq_b.keys())

    return {
        "feature": col,
        "kind": "categorical",
        "stat_name": "total_variation",
        "stat_value": float(tvd),
        "contrast_score": float(tvd),
    }
