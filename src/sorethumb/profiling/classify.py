"""Column classification: maps a ColumnProfile to a ColumnClass and Treatment.

Classification priority (first rule that matches wins):
  1. ignored       — name matches config.columns.ignore patterns
  2. empty         — all values null
  3. constant      — exactly one distinct non-null value
  4. near_constant — n_unique <= profiling.near_constant_distinct
  5. high_null     — null_ratio > profiling.null_ratio_drop
  6. boolean       — pl.Boolean dtype
  7. temporal      — Date / Datetime / Time / Duration dtype
  8. numeric       — numeric dtype (integers and floats)
  9. array_derived — List dtype (handled later by derive_array_features)
 10. identifier_like — high-cardinality string matching UUID / hex / int sequence
 11. free_text     — mean value length > profiling.free_text_mean_length
 12. categorical   — cardinality_ratio <= profiling.categorical_cardinality_ratio
 13. categorical   — fall-through for remaining strings (frequency-encode)
 14. unsupported   — anything else (Struct, Binary, Object, …)
"""

from __future__ import annotations

import fnmatch
import re
from enum import Enum

import polars as pl

from sorethumb.config import ColumnsConfig, FeaturesConfig, ProfilingConfig
from sorethumb.profiling.profile import ColumnProfile

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_LONG_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)

_NUMERIC_DTYPE_STRS = frozenset(
    {
        str(pl.Int8),
        str(pl.Int16),
        str(pl.Int32),
        str(pl.Int64),
        str(pl.UInt8),
        str(pl.UInt16),
        str(pl.UInt32),
        str(pl.UInt64),
        str(pl.Float32),
        str(pl.Float64),
    }
)
_TEMPORAL_DTYPE_PREFIXES = ("Date", "Time", "Datetime", "Duration")


class ColumnClass(str, Enum):
    """The logical kind of a column, as determined by profiling."""

    empty = "empty"
    constant = "constant"
    near_constant = "near_constant"
    high_null = "high_null"
    identifier_like = "identifier_like"
    free_text = "free_text"
    categorical = "categorical"
    numeric = "numeric"
    boolean = "boolean"
    temporal = "temporal"
    array_derived = "array_derived"
    ignored = "ignored"
    unsupported = "unsupported"


class Treatment(str, Enum):
    """How a column is handled during feature construction."""

    drop = "drop"  # excluded from features entirely
    passthrough = "passthrough"  # kept as-is (typically after imputation)
    one_hot = "one_hot"  # one-hot encoding for low-cardinality strings
    frequency = "frequency"  # frequency encoding for high-cardinality strings
    impute_median = "impute_median"  # numeric: impute nulls with median, then passthrough
    cast_int = "cast_int"  # boolean → 0/1 integer
    derive_time = "derive_time"  # temporal: extract cyclical/ordinal derivatives
    derive_array = "derive_array"  # List column: replace with __len/__mean/etc.
    indicator_only = "indicator_only"  # drop original, keep __is_missing indicator


def classify_column(
    profile: ColumnProfile,
    config: ProfilingConfig,
    columns_config: ColumnsConfig,
    protected_columns: set[str],
) -> tuple[ColumnClass, str]:
    """Return ``(ColumnClass, human-readable reason)`` for *profile*.

    *protected_columns* must include ``time_column``, all ``group_by`` columns,
    ``id_column``, and ``reference_column`` — these are never classified as
    identifier_like or dropped by pattern matching.
    """
    col = profile.name

    # 1. Ignored by pattern
    if _is_ignored(col, profile.dtype_str, columns_config.ignore, protected_columns):
        return ColumnClass.ignored, "matches ignore pattern"

    # 2. Empty
    if profile.is_empty:
        return ColumnClass.empty, "all values are null"

    # 3. Constant
    if profile.is_constant:
        return ColumnClass.constant, "exactly one distinct non-null value"

    # 4–5: dtype-based typed columns always win over near_constant / high_null
    dtype = profile.dtype_str

    if dtype == str(pl.Boolean):
        return ColumnClass.boolean, "Boolean dtype"

    if _is_temporal(dtype):
        return ColumnClass.temporal, f"temporal dtype ({dtype})"

    # 6. Near-constant (applies to numeric, string, list, and other types)
    if profile.n_unique <= config.near_constant_distinct and profile.n_unique > 0:
        return (
            ColumnClass.near_constant,
            f"only {profile.n_unique} distinct value(s) (threshold={config.near_constant_distinct})",
        )

    # 7. High null (drop threshold — above the flag threshold)
    if profile.null_ratio > config.null_ratio_drop:
        return (
            ColumnClass.high_null,
            f"null_ratio={profile.null_ratio:.2%} > drop threshold {config.null_ratio_drop:.2%}",
        )

    # 8–9: remaining dtype-based checks
    if dtype in _NUMERIC_DTYPE_STRS:
        return ColumnClass.numeric, f"numeric dtype ({dtype})"

    if _is_list(dtype):
        return ColumnClass.array_derived, f"List dtype ({dtype})"

    if dtype != str(pl.String):
        return ColumnClass.unsupported, f"unsupported dtype ({dtype})"

    # Remaining rules apply to string columns only

    # 10. Identifier-like (protected columns are exempt)
    if col not in protected_columns and config.identifier_detection != "off":
        id_class, reason = _check_identifier(profile, config)
        if id_class is not None:
            return id_class, reason

    # 11. Free text
    if profile.mean_length is not None and profile.mean_length > config.free_text_mean_length:
        return (
            ColumnClass.free_text,
            f"mean length={profile.mean_length:.1f} > {config.free_text_mean_length}",
        )

    # 12 & 13. Categorical
    return ColumnClass.categorical, f"string with cardinality_ratio={profile.cardinality_ratio:.4f}"


def treatment_for(
    col_class: ColumnClass,
    profile: ColumnProfile,
    features_config: FeaturesConfig,
) -> Treatment:
    """Map a ``ColumnClass`` to the appropriate ``Treatment``."""
    if col_class in (
        ColumnClass.empty,
        ColumnClass.constant,
        ColumnClass.near_constant,
        ColumnClass.identifier_like,
        ColumnClass.free_text,
        ColumnClass.ignored,
        ColumnClass.unsupported,
    ):
        return Treatment.drop

    if col_class == ColumnClass.high_null:
        return Treatment.indicator_only

    if col_class == ColumnClass.boolean:
        return Treatment.cast_int

    if col_class == ColumnClass.temporal:
        return Treatment.derive_time

    if col_class == ColumnClass.array_derived:
        return Treatment.derive_array

    if col_class == ColumnClass.numeric:
        return Treatment.impute_median

    if col_class == ColumnClass.categorical:
        if profile.n_unique <= features_config.one_hot_max_cardinality:
            return Treatment.one_hot
        return Treatment.frequency

    return Treatment.drop  # unreachable given complete ColumnClass enum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_ignored(
    col: str,
    dtype_str: str,
    patterns: list[str],
    protected: set[str],
) -> bool:
    if col in protected:
        return False
    for pat in patterns:
        if pat.startswith("type:"):
            rest = pat[5:].strip()
            parts = rest.split(None, 1)
            if len(parts) != 2:
                continue
            type_prefix, glob = parts
            if dtype_str.startswith(type_prefix) and fnmatch.fnmatch(col, glob):
                return True
        elif fnmatch.fnmatch(col, pat):
            return True
    return False


def _is_temporal(dtype_str: str) -> bool:
    return any(dtype_str.startswith(p) for p in _TEMPORAL_DTYPE_PREFIXES)


def _is_list(dtype_str: str) -> bool:
    return dtype_str.startswith("List(")


def _check_identifier(
    profile: ColumnProfile,
    config: ProfilingConfig,
) -> tuple[ColumnClass | None, str]:
    """Return (ColumnClass.identifier_like, reason) if the column looks like an ID."""
    if config.identifier_detection == "aggressive":
        if profile.cardinality_ratio > config.identifier_cardinality_ratio:
            return (
                ColumnClass.identifier_like,
                f"aggressive mode: cardinality_ratio={profile.cardinality_ratio:.4f} > "
                f"{config.identifier_cardinality_ratio}",
            )

    # Conservative: only flag UUID/hex patterns at high cardinality
    if profile.cardinality_ratio > config.identifier_cardinality_ratio and profile.examples:
        sample = profile.examples[:50]
        uuid_hits = sum(1 for v in sample if _UUID_RE.match(v))
        hex_hits = sum(1 for v in sample if _LONG_HEX_RE.match(v))
        if uuid_hits == len(sample):
            return (
                ColumnClass.identifier_like,
                f"UUID pattern detected in sample (n={len(sample)})",
            )
        if hex_hits == len(sample):
            return (
                ColumnClass.identifier_like,
                f"long-hex pattern detected in sample (n={len(sample)})",
            )

    return None, ""
