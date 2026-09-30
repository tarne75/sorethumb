"""Cohort-level contrast: how the flagged cohort differs from the rest as a group.

Per-record explanations answer "why is this row odd". Contrast answers "how does the
flagged set differ from the unflagged set". It is diagnostic output computed after
flagging; a failure here must never fail a run — all exceptions are caught and logged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
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
    feature, kind (numeric|categorical), stat_name, stat_value, contrast_score.

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

    cohens_d = _cohens_d(a, b)
    ks_stat, _ = ks_2samp(a, b)

    # Combined score: abs Cohen's d boosted by KS (both range ~0→∞ and 0→1)
    contrast_score = abs(cohens_d) + ks_stat

    return {
        "feature": col,
        "kind": "numeric",
        "stat_name": "cohens_d",
        "stat_value": float(cohens_d),
        "contrast_score": float(contrast_score),
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

    max_lift = 0.0
    for cat, pa in freq_a.items():
        pb = freq_b.get(cat, 0.0)
        lift = pa / pb if pb > 0 else pa * total_b  # treat as infinite lift, bound by count
        max_lift = max(max_lift, lift)

    return {
        "feature": col,
        "kind": "categorical",
        "stat_name": "max_lift",
        "stat_value": float(max_lift),
        "contrast_score": float(max_lift),
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    pooled_std = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    if pooled_std == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)
