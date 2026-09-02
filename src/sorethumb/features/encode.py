"""Categorical encoding: one-hot, frequency, and width-control demotion.

One-hot encodes categoricals with <= one_hot_max_cardinality distinct values; above
that threshold, frequency encoding is used instead. When the projected feature matrix
would exceed features.max_feature_width, columns are demoted from one-hot to frequency
encoding in descending cardinality order (most columns first) until it fits — deterministic
ordering is essential for reproducibility.

The __other bucket in one-hot encoding captures unseen values at score time so they are
never silently collapsed into the zero vector. Null values are never routed to __other —
they produce an all-zero one-hot row, with their missingness captured by __is_missing when
the missing-indicator rule applies.

Frequency-encoded unseen values map to 0.0, which correctly reads as "never observed in
the fitting data" — this is a feature, not a default.
"""

from __future__ import annotations

import logging
import warnings

import polars as pl

from sorethumb.errors import FeatureWidthWarning
from sorethumb.profiling.classify import Treatment
from sorethumb.profiling.plan import ColumnDecision, FeaturePlan

logger = logging.getLogger(__name__)

# Polars dt accessor mapping for supported time derivatives
_DERIVATIVE_EXPRS: dict[str, object] = {
    "hour": lambda col: pl.col(col).dt.hour().cast(pl.Int32),
    "dayofweek": lambda col: pl.col(col).dt.weekday().cast(pl.Int32),
    "day": lambda col: pl.col(col).dt.day().cast(pl.Int32),
    "month": lambda col: pl.col(col).dt.month().cast(pl.Int32),
    "year": lambda col: pl.col(col).dt.year().cast(pl.Int32),
    "quarter": lambda col: pl.col(col).dt.quarter().cast(pl.Int32),
}

# Aliases matching plan.py's _TIME_DERIVATIVE_ALIASES
_DERIVATIVE_SUFFIXES: dict[str, str] = {
    "hour": "__hour",
    "dayofweek": "__dayofweek",
    "day": "__day",
    "month": "__month",
    "year": "__year",
    "quarter": "__quarter",
}

_NUMERIC_INNER_TYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)


# ---------------------------------------------------------------------------
# Width-control demotion
# ---------------------------------------------------------------------------


def compute_demotions(plan: FeaturePlan, max_feature_width: int) -> set[str]:
    """Return the set of one-hot column names to demote to frequency encoding.

    Demotions are computed in descending cardinality order (widest columns first) so that
    each demotion recovers the most width per step. The result is deterministic: columns
    with equal cardinality are broken by their appearance order in plan.decisions.
    """
    current_width = len(plan.output_features)
    if current_width <= max_feature_width:
        return set()

    # Gather one-hot columns sorted by descending category count
    one_hot: list[tuple[ColumnDecision, int]] = [
        (dec, len(plan.one_hot_categories.get(dec.column, [])))
        for dec in plan.decisions
        if dec.treatment == Treatment.one_hot
    ]
    one_hot.sort(key=lambda x: -x[1])

    demoted: set[str] = set()
    for dec, n_cats in one_hot:
        if current_width <= max_feature_width:
            break
        # Demoting removes (n_cats + 1) one-hot columns and adds 1 frequency column
        current_width -= n_cats  # net change = -(n_cats + 1) + 1 = -n_cats
        demoted.add(dec.column)
        warnings.warn(
            f"Demoting '{dec.column}' from one-hot to frequency encoding "
            f"(n_unique={n_cats}) to stay within max_feature_width={max_feature_width}.",
            FeatureWidthWarning,
            stacklevel=4,
        )

    return demoted


# ---------------------------------------------------------------------------
# Expression builders
# ---------------------------------------------------------------------------


def _missing_indicator_expr(col: str) -> pl.Expr:
    return pl.col(col).is_null().cast(pl.Int8).alias(f"{col}__is_missing")


def _one_hot_exprs(col: str, cats: list[str]) -> list[pl.Expr]:
    """One dummy per category, plus __other for unseen non-null values."""
    exprs: list[pl.Expr] = []
    for cat in cats:
        exprs.append((pl.col(col) == pl.lit(cat)).cast(pl.Int8).fill_null(0).alias(f"{col}__{cat}"))
    # __other: 1 when value is present but not in any known category
    exprs.append(
        (pl.col(col).is_not_null() & ~pl.col(col).is_in(cats)).cast(pl.Int8).alias(f"{col}____other")
    )
    return exprs


def _frequency_expr(col: str, freq_map: dict[str, float]) -> pl.Expr:
    """Replace each value with its relative frequency. Unseen values → 0.0."""
    if not freq_map:
        # Reference col to preserve row count; is_null()*0 is 0.0 for every row
        return pl.col(col).is_null().cast(pl.Float64).mul(0.0).alias(col)
    keys = list(freq_map.keys())
    vals = [freq_map[k] for k in keys]
    return pl.col(col).replace_strict(keys, vals, default=0.0).cast(pl.Float64).fill_null(0.0).alias(col)


def _time_derivative_exprs(col: str, derivatives: list[str], dtype_str: str) -> list[pl.Expr]:
    """Extract configured time derivatives from a temporal column."""
    is_date_only = dtype_str.startswith("Date") and not dtype_str.startswith("Datetime")
    exprs: list[pl.Expr] = []
    for deriv in derivatives:
        if deriv == "hour" and is_date_only:
            continue  # Date has no hour component
        expr_fn = _DERIVATIVE_EXPRS.get(deriv)
        if expr_fn is None:
            logger.warning("Unknown time derivative '%s'; skipping.", deriv)
            continue
        suffix = _DERIVATIVE_SUFFIXES.get(deriv, f"__{deriv}")
        exprs.append(expr_fn(col).alias(f"{col}{suffix}"))  # type: ignore[operator]
    return exprs


def _array_derive_exprs(col: str, schema: pl.Schema) -> list[pl.Expr]:
    """Derive scalar features from a List column."""
    exprs: list[pl.Expr] = [
        pl.col(col).list.len().fill_null(0).alias(f"{col}__len"),
        pl.col(col).is_null().cast(pl.Int8).alias(f"{col}__is_null"),
        (pl.col(col).is_not_null() & (pl.col(col).list.len().fill_null(0) == 0))
        .cast(pl.Int8)
        .alias(f"{col}__is_empty"),
    ]
    dtype = schema.get(col)
    if dtype is not None and isinstance(dtype, pl.List):
        inner = dtype.inner
        if isinstance(inner, _NUMERIC_INNER_TYPES):
            exprs.extend(
                [
                    pl.col(col).list.mean().alias(f"{col}__mean"),
                    pl.col(col).list.min().alias(f"{col}__min"),
                    pl.col(col).list.max().alias(f"{col}__max"),
                ]
            )
    return exprs


# ---------------------------------------------------------------------------
# Frame-level encoder
# ---------------------------------------------------------------------------


def build_encoding_exprs(
    schema: pl.Schema,
    plan: FeaturePlan,
    demoted: set[str],
    extra_freq_maps: dict[str, dict[str, float]] | None = None,
) -> list[pl.Expr]:
    """Build the ordered list of polars expressions that produce the encoded frame.

    Parameters
    ----------
    schema:
        Schema of the *input* frame (pre-encoding).
    plan:
        Fitted FeaturePlan. Used for categories, freq maps, medians, and decisions.
    demoted:
        Columns demoted from one-hot to frequency due to width control.
    extra_freq_maps:
        Additional frequency maps for demoted columns (computed at fit time).

    """
    freq_maps = dict(plan.frequency_maps)
    if extra_freq_maps:
        freq_maps.update(extra_freq_maps)

    exprs: list[pl.Expr] = []

    for dec in plan.decisions:
        col = dec.column
        treatment = dec.treatment

        # Override treatment for width-demoted columns
        if col in demoted:
            treatment = Treatment.frequency

        if treatment == Treatment.drop:
            if dec.emit_missing_indicator:
                exprs.append(_missing_indicator_expr(col))
            continue

        if treatment == Treatment.indicator_only:
            if dec.emit_missing_indicator:
                exprs.append(_missing_indicator_expr(col))
            continue

        if treatment == Treatment.cast_int:
            exprs.append(pl.col(col).cast(pl.Int8).fill_null(0).alias(col))

        elif treatment == Treatment.impute_median:
            median = plan.imputation_medians.get(col, 0.0)
            exprs.append(pl.col(col).fill_null(median).cast(pl.Float64).alias(col))

        elif treatment == Treatment.derive_time:
            dtype_str = str(schema.get(col, pl.Null))
            exprs.extend(_time_derivative_exprs(col, plan.time_derivatives, dtype_str))

        elif treatment == Treatment.derive_array:
            exprs.extend(_array_derive_exprs(col, schema))

        elif treatment == Treatment.one_hot:
            cats = plan.one_hot_categories.get(col, [])
            exprs.extend(_one_hot_exprs(col, cats))

        elif treatment == Treatment.frequency:
            exprs.append(_frequency_expr(col, freq_maps.get(col, {})))

        elif treatment == Treatment.passthrough:
            exprs.append(pl.col(col).alias(col))

        if dec.emit_missing_indicator and treatment != Treatment.indicator_only:
            exprs.append(_missing_indicator_expr(col))

    return exprs
