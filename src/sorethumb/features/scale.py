"""Scaler fit and apply: standard (mean/std) and robust (median/IQR).

Why percentile-rank robust scaling: the IQR is resistant to extremes — a single 10x
outlier compresses the rest of a min-max range but barely moves the 25th/75th percentiles.

Fitting on data that still contains the anomalies:
  - robust mode (the default) uses median + IQR, which the anomalous rows barely move.
  - standard mode computes mean + std over each column's central 98% (values winsorised
    to the 1st–99th percentile) so a handful of extreme rows cannot inflate the centre
    or the spread. This keeps z-scores meaningful for the normal bulk of the data.

Degenerate columns: when the fitted spread (IQR or std) is essentially zero — a constant
column, or a binary/sparse indicator whose quartiles coincide — the scale is set to 1.0
so ``(x - centre) / scale`` stays on the column's natural 0/1 (or all-zero) footing
rather than exploding. A genuinely small but non-zero spread is used as-is: it is no
longer clamped up to 1.0, which previously flattened fine-grained columns to near-zero
variance.

All per-column quantiles are computed in batched polars aggregations so a wide feature
matrix triggers a small constant number of full passes rather than one pass per feature.
"""

from __future__ import annotations

import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

ScalerParams = dict[str, dict[str, float]]  # col → {center, scale}

# Spread at or below this is treated as "no spread" — use 1.0 so a constant or
# binary column is not divided by a near-zero number.
_ZERO_SPREAD = 1e-12
# Standard-mode mean/std are computed over this central quantile range so the
# anomalous tails do not distort the fit.
_TRIM_LOW = 0.01
_TRIM_HIGH = 0.99


def _spread_or_unit(value: float) -> float:
    """Return *value* as the scale, or 1.0 when it is effectively zero."""
    return value if value > _ZERO_SPREAD else 1.0


def fit_scaler(
    df: pl.DataFrame,
    cols: list[str],
    scaler_type: Literal["standard", "robust"],
) -> ScalerParams:
    """Compute center and scale parameters for each column.

    For robust scaling: center = median, scale = IQR.
    For standard scaling: center = mean, scale = std, both computed over the
    column's 1st–99th percentile range so extreme rows do not distort the fit.

    A spread of effectively zero (constant / degenerate column) becomes a scale
    of 1.0; any genuine non-zero spread is used unchanged.
    """
    if not cols:
        return {}

    if scaler_type == "robust":
        aggs: list[pl.Expr] = []
        for col in cols:
            aggs.append(pl.col(col).median().alias(f"{col}__med"))
            aggs.append(pl.col(col).quantile(0.25).alias(f"{col}__q25"))
            aggs.append(pl.col(col).quantile(0.75).alias(f"{col}__q75"))
        row = df.select(aggs).row(0, named=True)
        params: ScalerParams = {}
        for col in cols:
            med = row.get(f"{col}__med") or 0.0
            q25 = row.get(f"{col}__q25") or 0.0
            q75 = row.get(f"{col}__q75") or 0.0
            iqr = q75 - q25
            params[col] = {"center": float(med), "scale": _spread_or_unit(float(iqr))}
        return params

    # standard — trimmed mean/std over the central [_TRIM_LOW, _TRIM_HIGH] range.
    bound_aggs: list[pl.Expr] = []
    for col in cols:
        bound_aggs.append(pl.col(col).quantile(_TRIM_LOW).alias(f"{col}__lo"))
        bound_aggs.append(pl.col(col).quantile(_TRIM_HIGH).alias(f"{col}__hi"))
    bounds = df.select(bound_aggs).row(0, named=True)

    stat_aggs: list[pl.Expr] = []
    for col in cols:
        lo = bounds.get(f"{col}__lo")
        hi = bounds.get(f"{col}__hi")
        clipped = pl.col(col).clip(lo, hi)
        stat_aggs.append(clipped.mean().alias(f"{col}__mean"))
        stat_aggs.append(clipped.std().alias(f"{col}__std"))
    stats = df.select(stat_aggs).row(0, named=True)

    params2: ScalerParams = {}
    for col in cols:
        mean = stats.get(f"{col}__mean") or 0.0
        std = stats.get(f"{col}__std")
        std_f = 0.0 if std is None else float(std)
        params2[col] = {"center": float(mean), "scale": _spread_or_unit(std_f)}
    return params2


def apply_scaler(
    df: pl.DataFrame,
    scaler_params: ScalerParams,
    cols: list[str] | None = None,
) -> pl.DataFrame:
    """Apply stored center/scale parameters to *df* in a single select pass.

    Columns not present in scaler_params (or not in *cols* when supplied) are
    passed through unchanged. Column order matches the input frame.
    """
    if not scaler_params:
        return df

    scale_set = set(scaler_params) if cols is None else (set(cols) & set(scaler_params))

    exprs: list[pl.Expr] = []
    for col in df.columns:
        if col in scale_set:
            p = scaler_params[col]
            exprs.append(((pl.col(col).cast(pl.Float64) - p["center"]) / p["scale"]).alias(col))
        else:
            exprs.append(pl.col(col))

    return df.select(exprs)
