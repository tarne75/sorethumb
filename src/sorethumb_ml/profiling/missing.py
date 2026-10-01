"""Missing-value normalisation: float NaN is a missing value, same as null (P0-5).

Polars distinguishes ``null`` (missing) from ``NaN`` (a float value that sorts
above everything and propagates through arithmetic). Profiling statistics,
median imputation, missing indicators and scaler fitting all key off
*null*; a NaN left in a float column therefore slips past every one of them
and only surfaces later as a non-finite value in the scaled matrix. Treating
NaN as "missing" at the boundary -- before profiling, and again inside the
encoder for values *derived* from the source (e.g. the mean of a list that
contains a NaN) -- gives both kinds of missing the same, already-tested path:
profiled as missing, imputed with the fitted median, and (where the
missing-indicator rule applies) flagged.

Infinities are deliberately *not* normalised here: no transformation can
preserve what +/-inf means, so they are rejected with a clear error where
they would enter the feature matrix (``features.build``).
"""

from __future__ import annotations

import polars as pl


def float_nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Return *df* with NaN replaced by null in every float column.

    Non-float columns and the frame's schema are unchanged; returns *df*
    itself when it has no float columns.
    """
    exprs = [pl.col(name).fill_nan(None) for name, dtype in df.schema.items() if dtype.is_float()]
    return df.with_columns(exprs) if exprs else df
