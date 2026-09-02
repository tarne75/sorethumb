"""Scaler fit and apply: standard (mean/std) and robust (median/IQR).

Why percentile-rank robust scaling: the IQR is resistant to extremes — a single 10x
outlier compresses the rest of a min-max range but barely moves the 25th/75th percentiles.
A zero IQR (constant column) is clamped to 1.0 so scaling never produces NaN or Inf.
The same clamp is applied for std=0 in standard mode, so constant columns survive as zeros
after centering rather than producing NaN.

All per-column quantiles are computed in a single polars aggregation so that a wide feature
matrix triggers one full pass rather than n_features passes.
"""

from __future__ import annotations

import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

ScalerParams = dict[str, dict[str, float]]  # col → {center, scale}


def fit_scaler(
    df: pl.DataFrame,
    cols: list[str],
    scaler_type: Literal["standard", "robust"],
) -> ScalerParams:
    """Compute center and scale parameters for each column in one polars pass.

    For standard scaling: center = mean, scale = std (clamped to 1.0 if zero).
    For robust scaling: center = median, scale = IQR (clamped to 1.0 if zero).
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
            params[col] = {"center": float(med), "scale": float(max(iqr, 1.0))}
        return params

    # standard
    aggs2: list[pl.Expr] = []
    for col in cols:
        aggs2.append(pl.col(col).mean().alias(f"{col}__mean"))
        aggs2.append(pl.col(col).std().alias(f"{col}__std"))
    row2 = df.select(aggs2).row(0, named=True)
    params2: ScalerParams = {}
    for col in cols:
        mean = row2.get(f"{col}__mean") or 0.0
        std = row2.get(f"{col}__std") or 1.0
        if std is None or float(std) == 0.0:
            std = 1.0
        params2[col] = {"center": float(mean), "scale": float(std)}
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
