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

from sorethumb_ml.config import Config
from sorethumb_ml.errors import MemoryBudgetError, NonFiniteWarning, PlanError
from sorethumb_ml.features.correlate import drop_correlated
from sorethumb_ml.features.encode import build_encoding_exprs, compute_demotions
from sorethumb_ml.features.reduce import apply_pca, fit_pca
from sorethumb_ml.features.scale import apply_scaler, fit_scaler
from sorethumb_ml.features.space import FeatureSpace
from sorethumb_ml.io.fingerprint import INTERNAL_ROW_ID_COLUMN, schema_fingerprint
from sorethumb_ml.profiling.plan import FeaturePlan, recompute_output_features

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, type[np.floating]] = {
    "float32": np.float32,
    "float64": np.float64,
}


def _extract_row_ids(df: pl.DataFrame) -> np.ndarray:
    """Row identity for *df*'s current row order.

    Prefers the pipeline's stable global stamp (``INTERNAL_ROW_ID_COLUMN``,
    set once on the raw source frame before any period filter, group filter,
    or time sort) so values stay unique and joinable across groups. Falls
    back to a plain positional range for direct callers of ``fit_features``/
    ``apply_feature_plan`` that never stamped one.
    """
    if INTERNAL_ROW_ID_COLUMN in df.columns:
        return df[INTERNAL_ROW_ID_COLUMN].to_numpy().astype(np.int64)
    return np.arange(len(df), dtype=np.int64)


def _peak_matrix_multiplier(config: Config) -> float:
    """Estimate how many "base matrix" units can be alive simultaneously (P1-1).

    The base unit is one matrix at the configured dtype (``n_rows x n_cols x
    dtype_bytes`` -- what ``estimated_mb`` below already computes). Real
    per-group peak memory is a multiple of that, not that figure itself:

    - 1x for the group's own matrix (``apply_feature_plan``'s
      ``FeatureSpace.matrix``, always at the configured dtype).
    - Up to 2x more when ``explain.enabled`` (the default) and the configured
      dtype is float32: TreeSHAP, the ECOD/HBOS exact decomposition, and the
      OneClassSVM/LOF gradient path all need a float64 copy (see
      ``_pipeline._group_feature_matrix``) -- float64 is 8 bytes/element vs
      float32's 4, i.e. 2 "base units" on top of the 1x original. Zero extra
      when the configured dtype is already float64 (that copy is skipped
      entirely, ``astype(..., copy=False)``) or when explain is disabled
      (nothing needs the upcast).
    - +1x flat headroom for polars encode/scale intermediates
      (``_encode``/``apply_scaler``'s output frames), transiently alive
      alongside the matrix during fitting/apply, win or lose depending on
      polars' own column-store overhead relative to a dense numpy array.

    Deliberately does *not* attempt to model detector-internal memory (e.g.
    OneClassSVM's kernel-based fit can need O(train_rows^2), dwarfing the
    feature matrix for a large, uncapped ``train_row_cap``) -- that is
    detector- and hyperparameter-specific in a way a single matrix-shaped
    formula cannot honestly capture; ``train_row_cap`` remains the lever for
    it. This estimates matrix memory only, same as before P1-1, just less
    wrong about how many copies of it actually coexist.
    """
    multiplier = 1.0 + 1.0  # the group's own matrix + encode/scale headroom
    if config.explain.enabled and config.features.dtype == "float32":
        multiplier += 2.0
    return multiplier


def _check_memory_budget(n_rows: int, n_cols: int, config: Config) -> None:
    """Raise MemoryBudgetError if the projected matrix (see ``_peak_matrix_multiplier``) is too big."""
    dtype_bytes = 4 if config.features.dtype == "float32" else 8
    base_mb = n_rows * n_cols * dtype_bytes / (1024 * 1024)
    multiplier = _peak_matrix_multiplier(config)
    estimated_mb = base_mb * multiplier
    if estimated_mb > config.run.max_memory_mb:
        raise MemoryBudgetError(
            f"Projected feature matrix ({n_rows} x {n_cols} x {dtype_bytes}B = "
            f"{base_mb:.0f} MB), x{multiplier:.0f} for matrices/copies live at once "
            f"(explain={config.explain.enabled}, dtype={config.features.dtype!r}) "
            f"= {estimated_mb:.0f} MB, exceeds run.max_memory_mb={config.run.max_memory_mb}. "
            "Reduce one_hot_max_cardinality, enable correlation_reduction, or increase the "
            "budget; setting explain.enabled=False also drops the x4 (vs x2) multiplier above "
            "by skipping the float64 explain copy. Detector-internal memory (e.g. a "
            "kernel-based fit's O(train_rows^2) usage) is not included here; see train_row_cap."
        )


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

    *plan* is cleared of every fitted artefact first: correlation_reduction
    or pca being off for *this* call must not leave a previous call's
    correlation_drop_list/pca_* sitting on the plan, silently applied by a
    later apply_feature_plan() as if they still described the current fit.
    """
    _clear_fitted_artefacts(plan)

    plan.scaler_type = config.features.scaler
    plan.output_dtype = config.features.dtype

    # Sort by time column if present
    if plan.chosen_time_column and plan.chosen_time_column in df.columns:
        df = df.sort(plan.chosen_time_column)

    row_ids = _extract_row_ids(df)

    # Width demotion — store result in plan so apply_feature_plan uses the same treatment
    demoted = compute_demotions(plan, config.features.max_feature_width)
    plan.demoted_columns = demoted

    # Compute frequency maps for demoted columns (not stored in plan at build_feature_plan time)
    extra_freq: dict[str, dict[str, float]] = {}
    for col in demoted:
        vc = df[col].value_counts(normalize=True, sort=True)
        extra_freq[col] = {str(row[col]): float(row["proportion"]) for row in vc.iter_rows(named=True)}
        plan.frequency_maps[col] = extra_freq[col]

    # A demoted column's output is now a single frequency feature, not the
    # one-hot dummies build_feature_plan originally listed for it — recompute
    # so the plan keeps describing the matrix it actually produces.
    plan.output_features, plan.derived_to_original = recompute_output_features(
        plan, demoted, df.schema, config.features
    )

    _assert_width_within_budget(plan.output_features, config.features.max_feature_width)

    # Build encoded polars frame
    enc_df = _encode(df, plan, demoted, extra_freq)
    _assert_encoded_frame_is_usable(enc_df)

    # Pre-flight memory check (P1-1: sized for how many copies of this
    # matrix can realistically be alive at once, not just one).
    _check_memory_budget(len(enc_df), len(enc_df.columns), config)

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

    # Sanitize before PCA, not just after: a non-finite value here (e.g. an
    # empty-list row's __mean/__min/__max going through polars null -> numpy
    # NaN on the to_numpy() conversion above) would otherwise reach sklearn's
    # PCA fit directly, which does not handle NaN/Inf gracefully.
    matrix = _sanitize(matrix, config.features.dtype)

    # Snapshot the matrix's exact column order/identity right before the
    # optional PCA step below — this, not plan.output_features (which
    # predates demotion and correlation-drop), is what pca_components'
    # loadings operate on and what back-projection must invert.
    plan.pre_pca_feature_names = list(feature_names)

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
        # Defensive second pass: legitimate finite input can't make PCA
        # itself produce non-finite output, but this is cheap insurance
        # against a numerically degenerate (near-singular) fit.
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

    Raises PlanError immediately if *df*'s raw column names/dtypes do not match
    the schema the plan was fitted on (``plan.schema_fingerprint``). Every
    encoding/scaling artefact below is keyed by column name, not validated by
    dtype at apply time — a drifted column does not necessarily error; it can
    silently mis-encode, or (see ``apply_scaler``) land unscaled in the feature
    matrix. Same schema (names + dtypes) is required; different *values* for
    those columns — the normal score-forward case — is exactly what this
    fingerprint does not object to.
    """
    fp = schema_fingerprint(df)
    if fp != plan.schema_fingerprint:
        msg = (
            f"apply_feature_plan: input schema fingerprint {fp!r} does not match "
            f"the fitted plan's {plan.schema_fingerprint!r}. The source schema has "
            "drifted (a column was added, removed, renamed, or changed dtype) "
            "since the plan was fitted -- re-fit rather than reuse this plan."
        )
        raise PlanError(msg)

    if plan.chosen_time_column and plan.chosen_time_column in df.columns:
        df = df.sort(plan.chosen_time_column)

    row_ids = _extract_row_ids(df)

    enc_df = _encode(df, plan, plan.demoted_columns, None)
    _assert_encoded_frame_is_usable(enc_df)
    if plan.correlation_drop_list:
        # Drop before scaling, mirroring fit_features: when correlation reduction
        # ran, plan.scaler_params was refit on exactly the surviving columns, so
        # a dropped column never has (and doesn't need) a fitted scale param.
        enc_df = enc_df.select([c for c in enc_df.columns if c not in plan.correlation_drop_list])
    scaled_df = apply_scaler(enc_df, plan.scaler_params, enc_df.columns)

    feature_names = scaled_df.columns
    matrix = _to_matrix(scaled_df, plan.output_dtype)
    # Sanitize before PCA -- see the matching comment in fit_features.
    matrix = _sanitize(matrix, plan.output_dtype)

    if plan.pca_components is not None and plan.pca_mean is not None:
        components = np.array(plan.pca_components, dtype=np.float64)
        mean = np.array(plan.pca_mean, dtype=np.float64)
        n_features = components.shape[1]
        n_components = components.shape[0]
        matrix = apply_pca(matrix, components, mean, n_features, n_components)
        feature_names = [f"pc_{i}" for i in range(matrix.shape[1])]
        matrix = _sanitize(matrix, plan.output_dtype)  # defensive second pass

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


def _clear_fitted_artefacts(plan: FeaturePlan) -> None:
    """Reset every features/build.py-populated field before a (re-)fit.

    correlation_reduction or pca being off for *this* call must not leave a
    previous call's correlation_drop_list/pca_* sitting on the plan, applied
    by a later apply_feature_plan() as if it still described the current fit.
    """
    plan.correlation_drop_list = []
    plan.pca_components = None
    plan.pca_mean = None
    plan.pca_explained_variance_ratio = None
    plan.pre_pca_feature_names = None


def _assert_width_within_budget(output_features: list[str], max_feature_width: int) -> None:
    """Fail loudly if the plan's width still exceeds the budget after demotion.

    compute_demotions() only ever demotes one-hot columns; if that alone
    isn't enough (e.g. many non-one-hot-derived features -- array
    derivatives, time derivatives, indicators -- push width over the budget
    on their own), it silently returns having done its best. Without this,
    the pipeline would quietly ship a matrix wider than configured.
    """
    width = len(output_features)
    if width > max_feature_width:
        raise PlanError(
            f"Feature matrix has {width} columns even after demoting every "
            f"eligible one-hot column to frequency encoding, exceeding "
            f"features.max_feature_width={max_feature_width}. Lower "
            "one_hot_max_cardinality, drop more columns, disable unneeded "
            "derived features (array/time derivatives, missing indicators), "
            "or raise max_feature_width."
        )


def _assert_encoded_frame_is_usable(enc_df: pl.DataFrame) -> None:
    """Fail closed on an empty encoded frame no detector could train on.

    Every input column ignored/dropped/reduced to nothing (e.g. an
    all-identifier or all-high-null schema) would otherwise reach sklearn as
    a (n_rows, 0) array, failing deep inside with a confusing error instead
    of a clear one naming the actual cause.
    """
    if len(enc_df.columns) == 0:
        raise PlanError(
            "The encoded feature matrix has zero columns -- every input column was "
            "dropped, ignored, or reduced to nothing by the current configuration. "
            "Check columns.ignore, profiling thresholds, and features.* settings."
        )


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
    _assert_no_duplicate_output_names(exprs, plan)
    encoded = df.select(exprs)
    # Cast everything to Float64 so the scaler operates uniformly
    cast_exprs = [pl.col(c).cast(pl.Float64) for c in encoded.columns]
    return encoded.select(cast_exprs)


def _assert_no_duplicate_output_names(exprs: list[pl.Expr], plan: FeaturePlan) -> None:
    """Fail closed, with the colliding source columns named.

    This runs before ``select`` would otherwise raise polars' own
    less-actionable ``DuplicateError``. The one-hot "__other"/"__is_missing"
    sentinel collisions are resolved
    rather than merely detected (see
    ``profiling.plan.resolve_one_hot_reserved_names``); this is the residual
    defence-in-depth net for any other way two *different* source columns'
    independently-derived feature names could still collide (e.g. a column
    literally named to mimic another column's derived-feature convention).
    """
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for expr in exprs:
        name = expr.meta.output_name()
        original = plan.derived_to_original.get(name, "?")
        if name in seen and seen[name] != original:
            dupes.append(f"{name!r} (from both {seen[name]!r} and {original!r})")
        seen[name] = original
    if dupes:
        raise PlanError(
            f"Encoded feature matrix would have {len(dupes)} duplicate generated "
            f"column name(s), produced by different source columns: {dupes}. "
            "Rename one of the colliding source columns to resolve this."
        )


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
