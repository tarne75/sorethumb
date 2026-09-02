"""Nested-type handling: struct flattening and array feature derivation.

Struct columns are recursively expanded into ``parent_child`` named columns
until no structs remain or ``max_depth`` is reached.

Array (List) columns are not flattened; instead, a set of scalar derived
features replaces each one:
    - ``<col>__len``          — element count (0 for null/empty)
    - ``<col>__is_null``      — 1 if the element is null
    - ``<col>__is_empty``     — 1 if the list is empty (length == 0)
    - ``<col>__mean``         — mean of elements (numeric arrays only)
    - ``<col>__min``          — minimum of elements (numeric arrays only)
    - ``<col>__max``          — maximum of elements (numeric arrays only)

Arrays whose maximum observed length across the whole frame is 0 are dropped
with a ``ColumnDroppedWarning``.
"""

from __future__ import annotations

import logging
import warnings

import polars as pl

from sorethumb.errors import ColumnDroppedWarning, SchemaError

logger = logging.getLogger(__name__)

_NUMERIC_BASE_TYPES = (
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


def unnest_all(df: pl.DataFrame, max_depth: int = 5) -> pl.DataFrame:
    """Recursively expand struct columns using ``parent_child`` naming.

    Each field of a struct column named ``parent`` becomes ``parent_field``.
    Raises ``SchemaError`` if expanding would create a duplicate column name.
    Stops after *max_depth* iterations even if structs remain.
    """
    for depth in range(max_depth):
        struct_cols = [c for c in df.columns if isinstance(df.schema[c], pl.Struct)]
        if not struct_cols:
            break

        non_struct = [c for c in df.columns if c not in struct_cols]
        seen: set[str] = set(non_struct)
        new_names: list[tuple[str, str, str]] = []  # (parent_col, field_name, output_name)

        for col_name in struct_cols:
            dtype = df.schema[col_name]
            assert isinstance(dtype, pl.Struct)
            for field in dtype.fields:
                out = f"{col_name}_{field.name}"
                if out in seen:
                    raise SchemaError(
                        f"Struct unnesting at depth {depth} would create a duplicate "
                        f"column '{out}' (from struct '{col_name}', field '{field.name}')."
                    )
                seen.add(out)
                new_names.append((col_name, field.name, out))

        exprs: list[pl.Expr] = [pl.col(c) for c in non_struct]
        for parent_col, field_name, out_name in new_names:
            exprs.append(pl.col(parent_col).struct.field(field_name).alias(out_name))

        df = df.select(exprs)

    remaining = [c for c in df.columns if isinstance(df.schema[c], pl.Struct)]
    if remaining:
        logger.warning(
            "max_depth=%d reached; %d struct column(s) not unnested: %s", max_depth, len(remaining), remaining
        )
    return df


def derive_array_features(df: pl.DataFrame) -> pl.DataFrame:
    """Replace each List column with derived scalar features.

    Arrays with max observed length 0 across the frame are dropped with a
    ``ColumnDroppedWarning``.  Non-numeric arrays only get the length/null/empty
    features; numeric arrays also get mean, min, and max.
    """
    list_cols = [c for c in df.columns if isinstance(df.schema[c], pl.List)]
    if not list_cols:
        return df

    keep_exprs: list[pl.Expr] = [pl.col(c) for c in df.columns if c not in list_cols]
    derived_exprs: list[pl.Expr] = []

    for col_name in list_cols:
        col_dtype = df.schema[col_name]
        assert isinstance(col_dtype, pl.List)

        max_len = df.select(pl.col(col_name).list.len().max().alias("mx"))["mx"][0]

        if max_len is None or max_len == 0:
            warnings.warn(
                f"Array column '{col_name}' has max length 0; dropping it.",
                ColumnDroppedWarning,
                stacklevel=2,
            )
            continue

        derived_exprs.extend(
            [
                pl.col(col_name).list.len().alias(f"{col_name}__len"),
                pl.col(col_name).is_null().cast(pl.Int8).alias(f"{col_name}__is_null"),
                (pl.col(col_name).list.len() == 0).cast(pl.Int8).alias(f"{col_name}__is_empty"),
            ]
        )

        inner = col_dtype.inner
        if isinstance(inner, _NUMERIC_BASE_TYPES):
            derived_exprs.extend(
                [
                    pl.col(col_name).list.mean().alias(f"{col_name}__mean"),
                    pl.col(col_name).list.min().alias(f"{col_name}__min"),
                    pl.col(col_name).list.max().alias(f"{col_name}__max"),
                ]
            )

    return df.select([*keep_exprs, *derived_exprs])
