"""Correlation-based feature reduction.

Computes the Pearson correlation matrix on the scaled feature matrix, groups columns
into connected components where |r| > threshold using a union-find, and keeps one
member of each component (by plan feature order). Dropped columns are recorded in the
plan so that score-forward runs apply the same reduction without re-computing.

Correlation is computed on a sample of at most 200 000 rows. Pearson coefficients are
stable at that sample size; see docs/approximations.md for the justification.

No graph library is used: the union-find is a 20-line implementation that is exactly
as capable as the task requires.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_CORR_SAMPLE = 200_000


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self._rank[ri] < self._rank[rj]:
            ri, rj = rj, ri
        self._parent[rj] = ri
        if self._rank[ri] == self._rank[rj]:
            self._rank[ri] += 1


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def drop_correlated(
    df: pl.DataFrame,
    threshold: float = 0.95,
) -> tuple[pl.DataFrame, list[str]]:
    """Drop one column from each highly-correlated pair.

    Returns the trimmed DataFrame and the list of dropped column names.
    The kept member of each connected component is always the *first* by column order in
    *df*, ensuring deterministic results when several columns are mutually correlated.
    """
    cols = df.columns
    n = len(cols)
    if n < 2:
        return df, []

    matrix = _sample_matrix(df)
    if matrix is None:
        return df, []

    corr = np.corrcoef(matrix, rowvar=False)

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = corr[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                uf.union(i, j)

    # For each component, keep only the first column by index
    seen_roots: set[int] = set()
    keep_mask = []
    for i in range(n):
        root = uf.find(i)
        keep_mask.append(root not in seen_roots)
        seen_roots.add(root)

    kept = [c for c, keep in zip(cols, keep_mask, strict=True) if keep]
    dropped = [c for c, keep in zip(cols, keep_mask, strict=True) if not keep]

    if dropped:
        logger.info(
            "Correlation reduction dropped %d feature(s) at threshold=%.2f: %s",
            len(dropped),
            threshold,
            dropped,
        )

    return df.select(kept), dropped


def correlated_pairs(df: pl.DataFrame, threshold: float = 0.95) -> pl.DataFrame:
    """Return de-duplicated pairs with |r| >= threshold, sorted by |r| descending.

    Useful for human inspection and for the report.
    """
    cols = df.columns
    n = len(cols)
    if n < 2:
        return pl.DataFrame(schema={"feature_a": pl.String, "feature_b": pl.String, "pearson_r": pl.Float64})

    matrix = _sample_matrix(df)
    if matrix is None:
        return pl.DataFrame(schema={"feature_a": pl.String, "feature_b": pl.String, "pearson_r": pl.Float64})

    corr = np.corrcoef(matrix, rowvar=False)

    rows: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = corr[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                rows.append((cols[i], cols[j], float(v)))

    rows.sort(key=lambda x: -abs(x[2]))
    return pl.DataFrame(
        {
            "feature_a": [r[0] for r in rows],
            "feature_b": [r[1] for r in rows],
            "pearson_r": [r[2] for r in rows],
        },
        schema={"feature_a": pl.String, "feature_b": pl.String, "pearson_r": pl.Float64},
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _sample_matrix(df: pl.DataFrame) -> np.ndarray | None:
    """Convert df to float64 numpy array, sampling if large."""
    if df.is_empty():
        return None
    sample = df if len(df) <= _CORR_SAMPLE else df.sample(_CORR_SAMPLE, seed=0)
    try:
        return sample.cast(pl.Float64).to_numpy()
    except (TypeError, ValueError, pl.exceptions.InvalidOperationError):
        logger.warning("Could not convert feature matrix to numpy for correlation; skipping reduction.")
        return None
