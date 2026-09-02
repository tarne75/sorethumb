"""PCA compression: off by default, fully supported.

Why PCA is opt-in: with PCA off, attributions from SHAP or gradient methods are natively
in feature space, which is interpretable. With PCA on, attribution values are in component
space and must be back-projected — adding a whole class of possible back-projection bugs and
making explanations harder to reason about. See M4/explain/ for the back-projection logic.

k is capped at n_features - 1: after centering, the last principal component is degenerate
(the covariance matrix has rank n_features - 1 at most). Including it can tip a borderline
matrix into a LinAlgError with an unhelpful message; the cap prevents that.

The components_ shape from sklearn is (n_components, n_features). We assert this against the
plan's recorded n_features and n_components rather than guessing orientation from a bare
shape comparison — a square matrix is ambiguous and a wrongly-transposed projection corrupts
every explanation without raising.
"""

from __future__ import annotations

import logging

import numpy as np

from sorethumb.config import FeaturesConfig
from sorethumb.errors import PlanError

logger = logging.getLogger(__name__)


def fit_pca(
    matrix: np.ndarray,
    config: FeaturesConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on *matrix* and return (components, mean, explained_variance_ratio).

    components shape: (n_components, n_features) — sklearn's convention.
    A LowVarianceWarning is emitted when cumulative explained variance is below
    pca_min_explained_variance.

    Raises PlanError if sklearn raises LinAlgError, naming the likely cause.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415

    n_features = matrix.shape[1]
    k = max(1, min(config.pca_max_components, n_features - 1))

    try:
        pca = PCA(n_components=k, random_state=seed)
        pca.fit(matrix)
    except Exception as exc:
        # scipy/numpy may raise LinAlgError or similar for degenerate matrices
        raise PlanError(
            f"PCA failed ({type(exc).__name__}): {exc}. "
            "Likely cause: the feature matrix is degenerate (fewer independent rows "
            "than features, or near-zero variance columns that survived scaling). "
            "Try disabling PCA or increasing the minimum row count."
        ) from exc

    components: np.ndarray = pca.components_  # type: ignore[assignment]
    mean: np.ndarray = pca.mean_  # type: ignore[assignment]
    evr: np.ndarray = pca.explained_variance_ratio_  # type: ignore[assignment]

    assert components.shape == (k, n_features), (
        f"BUG: PCA components_ shape {components.shape} != ({k}, {n_features})"
    )

    cum_var = float(evr.sum())
    logger.info(
        "PCA: %d components explain %.1f%% of variance (target %.0f%%).",
        k,
        cum_var * 100,
        config.pca_min_explained_variance * 100,
    )

    if cum_var < config.pca_min_explained_variance:
        import warnings  # noqa: PLC0415

        from sorethumb.errors import LowVarianceWarning  # noqa: PLC0415

        warnings.warn(
            f"PCA: {k} components explain only {cum_var:.1%} of variance "
            f"(< pca_min_explained_variance={config.pca_min_explained_variance:.0%}). "
            "Increase pca_max_components or disable PCA.",
            LowVarianceWarning,
            stacklevel=3,
        )

    return components, mean, evr


def apply_pca(
    matrix: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    n_features: int,
    n_components: int,
) -> np.ndarray:
    """Project *matrix* into PCA space using stored components and mean.

    Asserts the components shape against the recorded (n_components, n_features) and raises
    PlanError on mismatch rather than transposing hopefully — a wrongly-oriented matrix
    produces nonsense projections with no error.
    """
    if components.shape != (n_components, n_features):
        raise PlanError(
            f"PCA components shape mismatch: plan recorded ({n_components}, {n_features}) "
            f"but got {components.shape}. The plan may have been saved with a different "
            "feature set and cannot be applied here."
        )
    if matrix.shape[1] != n_features:
        raise PlanError(
            f"Feature matrix has {matrix.shape[1]} column(s) but PCA expects {n_features}. "
            "The feature space changed after the plan was fitted — check for schema drift."
        )
    centred = matrix - mean
    return centred @ components.T
