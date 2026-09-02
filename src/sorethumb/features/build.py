"""Feature construction: apply a FeaturePlan to produce a FeatureSpace.

Two public entry points:

``fit_features(df, plan, config)``
    Encodes *df* using the plan's artefacts, fits the scaler, does correlation
    reduction, and optionally fits PCA. Mutates *plan* to store scaler_params,
    correlation_drop_list, pca_components, pca_mean, pca_explained_variance_ratio,
    scaler_type, and output_dtype. Returns a FeatureSpace whose matrix is ready for
    detector training.

``apply_feature_plan(df, plan)``
    Pure apply path for score-forward runs: encodes *df* using the plan's artefacts
    and applies all stored parameters (scaler, correlation list, PCA) without re-fitting
    any of them. The resulting FeatureSpace has the same feature_schema_hash as the
    training FeatureSpace — the hash is what makes model reuse safe to assert on.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import polars as pl

from sorethumb.config import Config
from sorethumb.errors import MemoryBudgetError, NonFiniteWarning
from sorethumb.features.correlate import drop_correlated
from sorethumb.features.encode import build_encoding_exprs, compute_demotions
from sorethumb.features.reduce import apply_pca, fit_pca
from sorethumb.features.scale import apply_scaler, fit_scaler
from sorethumb.features.space import FeatureSpace
from sorethumb.profiling.plan import FeaturePlan

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, type[np.floating]] = {
    "float32": np.float32,
    "float64": np.float64,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_features(df: pl.DataFrame, plan: FeaturePlan, config: Config) -> FeatureSpace:
    """Encode *df*, fit scaler/correlation/PCA, store params in *plan*, return FeatureSpace.

    Mutates *plan* with:
    - scaler_params, scaler_type, output_dtype
    - correlation_drop_list
    - pca_components, pca_mean, pca_explained_variance_ratio
    - frequency_maps extended for any width-demoted columns
    """
    plan.scaler_type = config.features.scaler
    plan.output_dtype = config.features.dtype

    # Sort by time column if present
    if plan.chosen_time_column and plan.chosen_time_column in df.columns:
        df = df.sort(plan.chosen_time_column)

    row_ids = np.arange(len(df), dtype=np.int64)

    # Width demotion
    demoted = compute_demotions(plan, config.features.max_feature_width)

    # Compute frequency maps for demoted columns (not stored in plan at build_feature_plan time)
    extra_freq: dict[str, dict[str, float]] = {}
    for col in demoted:
        vc = df[col].value_counts(normalize=True, sort=True)
        extra_freq[col] = {str(row[col]): float(row["proportion"]) for row in vc.iter_rows(named=True)}
        plan.frequency_maps[col] = extra_freq[col]

    # Build encoded polars frame
    enc_df = _encode(df, plan, demoted, extra_freq)

    # Pre-flight memory check
    n_rows, n_cols = len(enc_df), len(enc_df.columns)
    dtype_bytes = 4 if config.features.dtype == "float32" else 8
    estimated_mb = n_rows * n_cols * dtype_bytes / (1024 * 1024)
    if estimated_mb > config.run.max_memory_mb:
        raise MemoryBudgetError(
            f"Projected feature matrix ({n_rows} x {n_cols} x {dtype_bytes}B = "
            f"{estimated_mb:.0f} MB) exceeds run.max_memory_mb={config.run.max_memory_mb}. "
            "Reduce one_hot_max_cardinality, enable correlation_reduction, or increase the budget."
        )

    feature_cols = enc_df.columns

    # Fit scaler
    scaler_params = fit_scaler(enc_df, feature_cols, config.features.scaler)
    plan.scaler_params = scaler_params

    # Apply scaler
    scaled_df = apply_scaler(enc_df, scaler_params, feature_cols)

    # Correlation reduction
    drop_list: list[str] = []
    if config.features.correlation_reduction:
        scaled_df, drop_list = drop_correlated(scaled_df, config.features.correlation_threshold)
        plan.correlation_drop_list = drop_list
        if drop_list:
            # Re-fit scaler on the trimmed set
            feature_cols = scaled_df.columns
            scaler_params = fit_scaler(enc_df.select(feature_cols), feature_cols, config.features.scaler)
            plan.scaler_params = scaler_params
            scaled_df = apply_scaler(enc_df.select(feature_cols), scaler_params, feature_cols)

    feature_names = scaled_df.columns
    matrix = _to_matrix(scaled_df, config.features.dtype)

    # PCA
    if config.features.pca and matrix.shape[1] > 1:
        components, mean, evr = fit_pca(matrix, config.features, seed=config.run.seed)
        plan.pca_components = components.tolist()
        plan.pca_mean = mean.tolist()
        plan.pca_explained_variance_ratio = evr.tolist()
        n_features = matrix.shape[1]
        n_components = components.shape[0]
        matrix = apply_pca(matrix, components, mean, n_features, n_components)
        feature_names = [f"pc_{i}" for i in range(matrix.shape[1])]

    matrix = _sanitize(matrix, config.features.dtype)

    return FeatureSpace(
        matrix=matrix,
        feature_names=feature_names,
        row_ids=row_ids,
        plan=plan,
        feature_schema_hash=FeatureSpace.make_hash(feature_names),
    )


def apply_feature_plan(df: pl.DataFrame, plan: FeaturePlan) -> FeatureSpace:
    """Apply a fully-fitted plan to new data without re-fitting anything.

    Use for score-forward runs: load the plan, call this, compare feature_schema_hash
    to the training run's hash — a mismatch means the feature space changed and the
    persisted model cannot be safely reused.
    """
    if plan.chosen_time_column and plan.chosen_time_column in df.columns:
        df = df.sort(plan.chosen_time_column)

    row_ids = np.arange(len(df), dtype=np.int64)

    enc_df = _encode(df, plan, set(), None)
    scaled_df = apply_scaler(enc_df, plan.scaler_params, enc_df.columns)
    if plan.correlation_drop_list:
        keep = [c for c in scaled_df.columns if c not in plan.correlation_drop_list]
        scaled_df = scaled_df.select(keep)

    feature_names = scaled_df.columns
    matrix = _to_matrix(scaled_df, plan.output_dtype)

    if plan.pca_components is not None and plan.pca_mean is not None:
        components = np.array(plan.pca_components, dtype=np.float64)
        mean = np.array(plan.pca_mean, dtype=np.float64)
        n_features = components.shape[1]
        n_components = components.shape[0]
        matrix = apply_pca(matrix, components, mean, n_features, n_components)
        feature_names = [f"pc_{i}" for i in range(matrix.shape[1])]

    matrix = _sanitize(matrix, plan.output_dtype)

    return FeatureSpace(
        matrix=matrix,
        feature_names=feature_names,
        row_ids=row_ids,
        plan=plan,
        feature_schema_hash=FeatureSpace.make_hash(feature_names),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode(
    df: pl.DataFrame,
    plan: FeaturePlan,
    demoted: set[str],
    extra_freq: dict[str, dict[str, float]] | None,
) -> pl.DataFrame:
    """Build the encoded polars DataFrame from plan artefacts."""
    exprs = build_encoding_exprs(df.schema, plan, demoted, extra_freq)
    if not exprs:
        return pl.DataFrame()
    encoded = df.select(exprs)
    # Cast everything to Float64 so the scaler operates uniformly
    cast_exprs = [pl.col(c).cast(pl.Float64) for c in encoded.columns]
    return encoded.select(cast_exprs)


def _to_matrix(df: pl.DataFrame, dtype_str: str) -> np.ndarray:
    """Convert polars DataFrame to numpy array with the requested dtype."""
    np_dtype = np.float32 if dtype_str == "float32" else np.float64
    return df.to_numpy().astype(np_dtype)


def _sanitize(matrix: np.ndarray, dtype_str: str) -> np.ndarray:
    """Replace NaN/±Inf with 0.0, warning if any are found."""
    bad = ~np.isfinite(matrix)
    if bad.any():
        n_bad = int(bad.sum())
        warnings.warn(
            f"{n_bad} non-finite value(s) (NaN or ±Inf) found in the feature matrix "
            "and replaced with 0.0. An earlier pipeline stage may have misbehaved — "
            "check profiling logs for columns with high null ratios or extreme values.",
            NonFiniteWarning,
            stacklevel=3,
        )
        matrix = matrix.copy()
        matrix[bad] = 0.0
    np_dtype = np.float32 if dtype_str == "float32" else np.float64
    return matrix.astype(np_dtype)
