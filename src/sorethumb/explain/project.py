"""Back-projection from PCA/feature space to original column space.

Two steps, always applied in order:

1. PCA back-projection (only when PCA is on):
   contribution_features = |loadings|ᵀ @ contribution_components
   where |loadings| is the element-wise absolute value of the PCA component
   matrix (shape n_components × n_features). This distributes each component's
   contribution to the features it loads on, weighted by loading magnitude.

2. Derived → original aggregation:
   Sum the absolute contributions of all derived features that map to the same
   original column (via plan.derived_to_original). This prevents a single
   original column (e.g. a categorical with many one-hot dummies) from
   appearing multiple times in the top-N list.

   Aggregate FIRST, then take top_n.

The ExplainError raised on a loadings shape mismatch is deliberate: never
infer orientation from a bare shape comparison — a square matrix is ambiguous,
and a wrongly transposed matrix corrupts every explanation silently.
"""

from __future__ import annotations

import logging

import numpy as np

from sorethumb.errors import ExplainError

logger = logging.getLogger(__name__)


def back_project_pca(
    contribution_components: np.ndarray,
    loadings: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Map PCA-space attributions back to feature space.

    Parameters
    ----------
    contribution_components:
        Shape (n_rows, n_components). Attribution per PCA component per row.
    loadings:
        PCA component matrix, shape (n_components, n_features), as stored in
        plan.pca_components.
    n_features:
        Expected n_features. Used to assert the loadings shape.

    Returns
    -------
    np.ndarray, shape (n_rows, n_features)

    """
    n_components_contrib = contribution_components.shape[1]
    if loadings.shape != (n_components_contrib, n_features):
        msg = (
            f"PCA loadings shape mismatch: expected ({n_components_contrib}, {n_features}), "
            f"got {loadings.shape}. "
            "The plan's pca_components may be corrupted or transposed."
        )
        raise ExplainError(msg)

    abs_loadings = np.abs(loadings)  # shape (n_components, n_features)
    # contribution_components @ abs_loadings → (n_rows, n_features)
    # Each component's contribution is spread to features proportionally to |loading|
    return contribution_components @ abs_loadings


def aggregate_to_original(
    feature_attributions: np.ndarray,
    feature_names: list[str],
    derived_to_original: dict[str, str],
) -> dict[str, np.ndarray]:
    """Sum absolute feature attributions per original column.

    Parameters
    ----------
    feature_attributions:
        Shape (n_rows, n_features). Can be positive or negative; absolute
        value is used so directions don't cancel.
    feature_names:
        Ordered feature names matching axis-1 of feature_attributions.
    derived_to_original:
        Mapping from derived feature name → original column name.

    Returns
    -------
    dict mapping original column name → np.ndarray of shape (n_rows,).

    """
    original_cols: dict[str, np.ndarray] = {}

    for feat_idx, feat_name in enumerate(feature_names):
        orig = derived_to_original.get(feat_name, feat_name)
        col_contrib = np.abs(feature_attributions[:, feat_idx])
        if orig in original_cols:
            original_cols[orig] += col_contrib
        else:
            original_cols[orig] = col_contrib.copy()

    return original_cols


def top_n_reasons(
    row_idx: int,
    original_attributions: dict[str, np.ndarray],
    raw_row: dict[str, object],
    top_n: int,
) -> list[dict[str, object]]:
    """Return the top-N contributing original columns for one row.

    Parameters
    ----------
    row_idx:
        Index into the scored population.
    original_attributions:
        Mapping from original column name → per-row attribution array.
    raw_row:
        Dict of {col: raw_value} from the pre-encoding frame.
    top_n:
        Maximum number of reasons to return.

    Returns
    -------
    List of dicts with keys "column", "raw_value", "attribution".
    Padded with null placeholders when fewer than top_n columns are available.

    """
    scored: list[tuple[str, float]] = [
        (col, float(arr[row_idx])) for col, arr in original_attributions.items()
    ]
    scored.sort(key=lambda x: -x[1])

    reasons: list[dict[str, object]] = []
    for col, attr in scored[:top_n]:
        reasons.append(
            {
                "column": col,
                "raw_value": raw_row.get(col),
                "attribution": attr,
            }
        )

    # Pad with null placeholders so callers always get top_n entries
    while len(reasons) < top_n:
        reasons.append({"column": None, "raw_value": None, "attribution": None})

    return reasons


def permutation_importance(
    detector: object,
    X: np.ndarray,
    feature_names: list[str],
    derived_to_original: dict[str, str],
    n_repeats: int = 5,
    max_rows: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute permutation importance per original column, min-max scaled to [0, 1].

    Parameters
    ----------
    detector:
        Any fitted Detector with score_samples(X) method.
    X:
        Float64 feature matrix.
    feature_names:
        Ordered feature names matching X columns.
    derived_to_original:
        Mapping from derived feature name → original column name.
    n_repeats:
        Number of permutations per feature.
    max_rows:
        Row cap for computational feasibility.
    seed:
        Base random seed; each repeat uses seed + repeat_idx.

    Returns
    -------
    dict mapping original column → importance score in [0, 1].

    """
    if X.shape[0] > max_rows:
        logger.info("permutation_importance: capping %d rows to max_rows=%d.", X.shape[0], max_rows)
        X = X[:max_rows]

    n_features = X.shape[1]
    base_scores = detector.score_samples(X)  # type: ignore[attr-defined]
    base_mean = float(np.mean(base_scores))

    raw_importance: dict[str, float] = {}

    for feat_idx in range(n_features):
        drop_acum = 0.0
        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed + repeat)
            X_perm = X.copy()
            X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
            perm_scores = detector.score_samples(X_perm)  # type: ignore[attr-defined]
            # More anomalous = lower score_samples → permutation that drops score hurts
            # We want importance to reflect how much permuting hurts anomaly detection
            # Use negated score so "more anomalous" = higher metric; importance = metric drop
            drop = base_mean - float(np.mean(perm_scores))
            drop_acum += drop

        feat_name = feature_names[feat_idx]
        orig = derived_to_original.get(feat_name, feat_name)
        raw_importance[orig] = raw_importance.get(orig, 0.0) + drop_acum / n_repeats

    # Min-max scale to [0, 1]
    vals = list(raw_importance.values())
    min_v, max_v = min(vals), max(vals)
    rng_v = max_v - min_v
    if rng_v == 0.0:
        return dict.fromkeys(raw_importance, 0.5)
    return {k: (v - min_v) / rng_v for k, v in raw_importance.items()}
