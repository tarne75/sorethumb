"""Gradient-based (finite-difference) attributions.

Central finite-difference of the detector's score_samples per feature
dimension. Step size is derived from each dimension's scaled standard
deviation so it adapts to the data's scale.

Cost: 2 * n_features model evaluations per row.
Total cost for a batch: 2 * n_rows * n_features score_samples calls.

The batch is split into individual rows — each row needs its own perturbation
— so cost scales linearly with both rows and features. Log the projected cost
before starting and enforce explain.max_rows to keep it bounded.

Note on sign convention: score_samples returns higher = more normal.
We negate the finite-difference gradient so positive attribution means
"this feature pushes the row toward being anomalous".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_STEP_FACTOR = 0.01  # step = factor * per-dimension std


def gradient_attributions(
    detector: Any,
    X: np.ndarray,
    max_rows: int = 5000,
    step_factor: float = _DEFAULT_STEP_FACTOR,
) -> tuple[np.ndarray, str]:
    """Compute central finite-difference attributions for each row in X.

    Parameters
    ----------
    detector:
        Any fitted Detector with a score_samples(X) method.
    X:
        Float64 feature matrix, shape (n_rows, n_features).
    max_rows:
        Hard cap — rows beyond this index are silently skipped (callers
        are expected to pre-filter to flagged rows only).
    step_factor:
        h = step_factor * std(X[:, d]) per dimension. Clamped to 1e-6 when
        the dimension has zero variance.

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly.
    tag:
        Always "heuristic".

    """
    n_rows, n_features = X.shape
    if n_rows > max_rows:
        logger.warning("gradient_attributions: capping %d rows to max_rows=%d.", n_rows, max_rows)
        X = X[:max_rows]
        n_rows = max_rows

    stds = X.std(axis=0)
    # When std is zero (single row or constant column), fall back to step relative to
    # the feature's own magnitude so the perturbation is never negligibly small.
    abs_mean = np.abs(X).mean(axis=0)
    fallback = np.where(abs_mean > 0, step_factor * abs_mean, 1e-3)
    steps = np.where(stds > 0, step_factor * stds, fallback)

    projected_calls = 2 * n_rows * n_features
    logger.info(
        "gradient_attributions: %d rows x %d features → %d score_samples calls.",
        n_rows,
        n_features,
        projected_calls,
    )

    attributions = np.zeros((n_rows, n_features), dtype=np.float64)

    for i in range(n_rows):
        row = X[i]
        for d in range(n_features):
            h = steps[d]
            row_plus = row.copy()
            row_plus[d] += h
            row_minus = row.copy()
            row_minus[d] -= h
            batch = np.stack([row_plus, row_minus])
            scores = detector.score_samples(batch)
            # (score_plus - score_minus) / (2h) → positive means dim pushes score up (more normal)
            # Negate so positive attribution = more anomalous
            attributions[i, d] = -(scores[0] - scores[1]) / (2.0 * h)

    return attributions, "heuristic"


def kernel_shap_attributions(
    detector: Any,
    X: np.ndarray,
    background_k: int = 50,
    max_rows: int = 5000,
) -> tuple[np.ndarray, str]:
    """KernelSHAP attributions using a k-means summary background.

    Much slower than gradient or TreeSHAP; the tag stays "heuristic".
    Only called when explain.kernel_shap = True.

    Parameters
    ----------
    detector:
        Any fitted Detector with a score_samples(X) method.
    X:
        Float64 feature matrix.
    background_k:
        Number of k-means clusters to use as the background summary.
    max_rows:
        Rows beyond this cap are silently skipped.

    """
    import shap  # noqa: PLC0415

    if X.shape[0] > max_rows:
        logger.warning("kernel_shap_attributions: capping %d rows to max_rows=%d.", X.shape[0], max_rows)
        X = X[:max_rows]

    background = shap.kmeans(X, background_k)
    explainer = shap.KernelExplainer(detector.score_samples, background)
    # nworkers=-1 = use all CPUs
    shap_values = explainer.shap_values(X, nsamples="auto")
    # Negate: SHAP positive = more normal → negate for more anomalous
    attributions = -np.asarray(shap_values, dtype=np.float64)
    return attributions, "heuristic"
