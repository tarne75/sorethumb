"""Single-pass column profiling.

One ``df.select(aggs)`` call collects every statistic needed by the classifier.
A second, narrower pass gathers mean string length and example values for string
columns; this is run only on the string subset.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import polars as pl

from sorethumb.config import ProfilingConfig

logger = logging.getLogger(__name__)

_NUMERIC_DTYPES = frozenset(
    {
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
    }
)


@dataclass
class ColumnProfile:
    """Statistics for a single column collected during profiling."""

    name: str
    dtype_str: str
    null_count: int
    non_null_count: int
    null_ratio: float
    n_unique: int
    cardinality_ratio: float  # n_unique / non_null_count (0 if non_null_count == 0)
    is_empty: bool  # all values are null
    is_constant: bool  # exactly one distinct non-null value
    min_val: float | None  # numeric only
    max_val: float | None  # numeric only
    mean_val: float | None  # numeric only
    std_val: float | None  # numeric only
    mean_length: float | None  # string only
    examples: list[str] = field(default_factory=list)


def profile_columns(df: pl.DataFrame, config: ProfilingConfig) -> list[ColumnProfile]:
    """Profile every column of *df* with a minimal number of passes.

    Pass 1: aggregate counts, distinct counts, and numeric stats.
    Pass 2 (string columns only): mean length and example values.
    """
    n_rows = len(df)
    profiles: dict[str, ColumnProfile] = {}

    # --- Pass 1: universal aggregations ---
    aggs: list[pl.Expr] = []
    for col in df.columns:
        dtype = df.schema[col]
        aggs.append(pl.col(col).null_count().alias(f"{col}__null_count"))
        aggs.append(pl.col(col).n_unique().alias(f"{col}__n_unique"))

        if dtype in _NUMERIC_DTYPES:
            aggs.append(pl.col(col).min().cast(pl.Float64).alias(f"{col}__min"))
            aggs.append(pl.col(col).max().cast(pl.Float64).alias(f"{col}__max"))
            aggs.append(pl.col(col).mean().cast(pl.Float64).alias(f"{col}__mean"))
            aggs.append(pl.col(col).std().cast(pl.Float64).alias(f"{col}__std"))

    stats_row = df.select(aggs).row(0, named=True)

    for col in df.columns:
        dtype = df.schema[col]
        null_count = int(stats_row[f"{col}__null_count"])
        n_unique = int(stats_row[f"{col}__n_unique"])
        non_null = n_rows - null_count
        null_ratio = null_count / n_rows if n_rows > 0 else 0.0
        # n_unique from polars counts null as a distinct value; subtract it
        # only when all rows are null (n_unique == 1 but all values are null)
        # In practice: n_unique includes null as one of the distinct values.
        # We want n_unique among non-null values only.
        has_null = null_count > 0
        n_unique_non_null = max(0, n_unique - (1 if has_null else 0))
        cardinality_ratio = n_unique_non_null / non_null if non_null > 0 else 0.0
        is_empty = non_null == 0
        is_constant = n_unique_non_null == 1 and not is_empty

        min_val = max_val = mean_val = std_val = None
        if dtype in _NUMERIC_DTYPES:
            raw_min = stats_row.get(f"{col}__min")
            raw_max = stats_row.get(f"{col}__max")
            raw_mean = stats_row.get(f"{col}__mean")
            raw_std = stats_row.get(f"{col}__std")
            min_val = float(raw_min) if raw_min is not None and not _isnan(raw_min) else None
            max_val = float(raw_max) if raw_max is not None and not _isnan(raw_max) else None
            mean_val = float(raw_mean) if raw_mean is not None and not _isnan(raw_mean) else None
            std_val = float(raw_std) if raw_std is not None and not _isnan(raw_std) else None

        profiles[col] = ColumnProfile(
            name=col,
            dtype_str=str(dtype),
            null_count=null_count,
            non_null_count=non_null,
            null_ratio=null_ratio,
            n_unique=n_unique_non_null,
            cardinality_ratio=cardinality_ratio,
            is_empty=is_empty,
            is_constant=is_constant,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            mean_length=None,
            examples=[],
        )

    # --- Pass 2: string columns ---
    string_cols = [c for c in df.columns if df.schema[c] == pl.String]
    if string_cols:
        sample_n = config.sample_rows_for_examples
        aggs2: list[pl.Expr] = []
        for col in string_cols:
            aggs2.append(pl.col(col).drop_nulls().str.len_chars().mean().alias(f"{col}__mean_len"))
        len_row = df.select(aggs2).row(0, named=True)

        for col in string_cols:
            raw_len = len_row.get(f"{col}__mean_len")
            profiles[col].mean_length = float(raw_len) if raw_len is not None else None

            sample = df.lazy().select(pl.col(col).drop_nulls()).limit(sample_n).collect()[col].to_list()
            profiles[col].examples = [str(v) for v in sample]

    return [profiles[c] for c in df.columns]


def _isnan(v: object) -> bool:
    try:
        return math.isnan(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
