"""Gradient-based (finite-difference) attributions.

Central finite-difference of the detector's score_samples per feature
dimension. Step size is derived from each dimension's standard deviation in a
*reference* matrix -- the population the detector was fitted on (or its normal
rows) -- never from the target rows being explained. The targets are the few
rows the detector flagged: their own spread is tiny, contaminated by the very
anomalies being explained, and changes with whichever other rows happen to be
flagged, so deriving the scale from them made a row's attribution depend on its
batch (and gave a lone flagged row an arbitrary fallback step).

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
import warnings
from typing import Any

import numpy as np

from sorethumb_ml.errors import ExplainError, FallbackAttributionWarning

logger = logging.getLogger(__name__)

_DEFAULT_STEP_FACTOR = 0.01  # step = factor * per-dimension std (of the reference)

# KernelSHAP summarises the reference with k-means; clustering millions of rows
# buys nothing for a ~50-centre summary, so a larger reference is subsampled
# (deterministically: a fixed seed, so a run's explanations are reproducible).
_REFERENCE_SAMPLE_CAP = 10_000


def _check_reference(X: np.ndarray, reference: np.ndarray, caller: str) -> None:
    """Fail clearly if *reference* cannot stand in for the population *X* was drawn from."""
    if reference.ndim != 2 or X.ndim != 2:
        msg = f"{caller}: X and reference must both be 2-D (n_rows, n_features)."
        raise ExplainError(msg)
    if reference.shape[1] != X.shape[1]:
        msg = (
            f"{caller}: reference has {reference.shape[1]} feature(s) but the target rows have "
            f"{X.shape[1]}; both must be in the detector's feature space."
        )
        raise ExplainError(msg)
    if reference.shape[0] == 0:
        msg = (
            f"{caller}: reference has no rows; pass the full or normal population the detector was fitted on."
        )
        raise ExplainError(msg)


def gradient_attributions(
    detector: Any,
    X: np.ndarray,
    *,
    reference: np.ndarray,
    max_rows: int = 5000,
    step_factor: float = _DEFAULT_STEP_FACTOR,
) -> tuple[np.ndarray, str]:
    """Compute central finite-difference attributions for each target row in X.

    Parameters
    ----------
    detector:
        Any fitted Detector with a score_samples(X) method.
    X:
        Float64 *target* rows to explain (typically only the flagged rows),
        shape (n_rows, n_features). Gradients are computed for these rows only.
    reference:
        Float64 matrix in the same feature space: the full population, or the
        normal (unflagged) population, the detector was fitted on. Per-dimension
        perturbation scales come from this, never from *X*. Required (no
        default) so a caller cannot silently fall back to scaling by the targets.
    max_rows:
        Hard cap — rows beyond this index are silently skipped (callers
        are expected to pre-filter to flagged rows only).
    step_factor:
        h = step_factor * std(reference[:, d]) per dimension. When the
        reference has zero variance in a dimension, a step relative to that
        feature's own reference magnitude (or 1e-3 for an all-zero feature).

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly.
    tag:
        Always "heuristic".

    """
    _check_reference(X, reference, "gradient_attributions")
    n_rows, n_features = X.shape
    if n_rows > max_rows:
        logger.warning("gradient_attributions: capping %d rows to max_rows=%d.", n_rows, max_rows)
        X = X[:max_rows]
        n_rows = max_rows

    stds = reference.std(axis=0)
    # When the reference's std is zero (constant column), fall back to a step relative to
    # the feature's own magnitude so the perturbation is never negligibly small.
    abs_mean = np.abs(reference).mean(axis=0)
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
    *,
    reference: np.ndarray,
    background_k: int = 50,
    max_rows: int = 5000,
) -> tuple[np.ndarray, str]:
    """KernelSHAP attributions for the target rows *X*, against a k-means summary of *reference*.

    Much slower than gradient or TreeSHAP; the tag stays "heuristic".
    Only called when explain.kernel_shap = True. Falls back to the plain
    finite-difference gradient method (same "heuristic" tag either way, so
    the caller sees no difference) when shap is not installed -- it lives in
    the optional ``explain`` extra, not a core dependency.

    Parameters
    ----------
    detector:
        Any fitted Detector with a score_samples(X) method.
    X:
        Float64 *target* rows to explain, shape (n_rows, n_features).
    reference:
        Float64 matrix in the same feature space: the full or normal population
        the detector was fitted on. The KernelSHAP background -- what "feature
        absent" is replaced by -- is summarised from this, never from the
        targets (explaining a flagged row against a background made of flagged
        rows measures its difference from other anomalies, not from normal data).
        Required (no default).
    background_k:
        Number of k-means clusters to use as the background summary (clamped to
        the number of reference rows; a reference above 10,000 rows is first
        subsampled with a fixed seed).
    max_rows:
        Rows beyond this cap are silently skipped.

    """
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:
        warnings.warn(
            "shap is not installed; KernelSHAP attributions are unavailable. "
            "Falling back to gradient attributions. Install with: pip install 'sorethumb-ml[explain]'.",
            FallbackAttributionWarning,
            stacklevel=2,
        )
        logger.warning("KernelSHAP unavailable (shap not installed): %s", exc)
        return gradient_attributions(detector, X, reference=reference, max_rows=max_rows)

    _check_reference(X, reference, "kernel_shap_attributions")
    if X.shape[0] > max_rows:
        logger.warning("kernel_shap_attributions: capping %d rows to max_rows=%d.", X.shape[0], max_rows)
        X = X[:max_rows]

    if reference.shape[0] > _REFERENCE_SAMPLE_CAP:
        picked = np.random.default_rng(0).choice(
            reference.shape[0], size=_REFERENCE_SAMPLE_CAP, replace=False
        )
        reference = reference[np.sort(picked)]
    background = shap.kmeans(reference, min(background_k, reference.shape[0]))
    explainer = shap.KernelExplainer(detector.score_samples, background)
    # nworkers=-1 = use all CPUs
    shap_values = explainer.shap_values(X, nsamples="auto")
    # Negate: SHAP positive = more normal → negate for more anomalous
    attributions = -np.asarray(shap_values, dtype=np.float64)
    return attributions, "heuristic"
