"""Pure data-frame transforms: train/holdout splitting and label mapping.

No file I/O — every function here takes and returns in-memory ``pl.DataFrame``/
``pl.Series`` objects, so they are directly unit-testable with tiny synthetic
frames.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

# Row-identifier column stamped onto each split so held-out predictions can be
# joined back to their true label by plain positional lookup (row_id i ==
# holdout_df row i), without a real join key existing in the source data.
SPLIT_ROW_ID_COLUMN = "_validation_row_id"


def split_train_holdout(
    df: pl.DataFrame, holdout_frac: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split *df* into (train, holdout) by a seeded random row permutation.

    Deterministic given (len(df), holdout_frac, seed): the same inputs always
    produce the same split, independent of the row *values* (a positional
    permutation, not a content-aware split). ``holdout_frac`` is clamped so
    both splits always get at least one row when ``len(df) >= 2``.
    """
    n = len(df)
    if n < 2:
        return df, df.clear()

    n_holdout = round(n * holdout_frac)
    n_holdout = max(1, min(n - 1, n_holdout))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]

    train_df = df[train_idx.tolist()]
    holdout_df = df[holdout_idx.tolist()]
    return train_df, holdout_df


def stamp_row_id(df: pl.DataFrame, column: str = SPLIT_ROW_ID_COLUMN) -> pl.DataFrame:
    """Add a 0..len(df)-1 positional id column, for joining results back to labels."""
    return df.with_columns(pl.arange(0, len(df)).alias(column))


def label_to_anomaly_array(series: pl.Series, is_anomaly: Callable[[pl.Series], pl.Series]) -> np.ndarray:
    """Map a raw label column to a 0/1 int array via *is_anomaly* (Series -> bool Series)."""
    mask = is_anomaly(series)
    return mask.cast(pl.Int8).to_numpy().astype(int)
