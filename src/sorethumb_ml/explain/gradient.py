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

Saturating scores (``saturating=True``). IsolationForest's score is piecewise
constant in each input, and an RBF One-Class SVM's decision function flattens
out once a row is far from every support vector. For a row outside the
reference population's range, a small perturbation then changes nothing in the
very dimension that put the row out there, so the gradient there is zero (or
numerically tiny) and whatever residual slope the other dimensions carry wins
the ranking: an ``amount`` of 900 against a reference of about 50 gets
explained as ``lat`` or ``region``. With ``saturating=True``, such rows -- and
any row whose gradient comes back exactly zero -- are attributed with the
reference-scaled marginal deviation instead (see
``marginal_deviation_attributions``). The tag stays ``"heuristic"``.
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


def _robust_scale(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension (centre, scale) of *reference*: median, and IQR/1.349 -> MAD*1.4826 -> std -> 1."""
    centre = np.median(reference, axis=0)
    q75, q25 = np.percentile(reference, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    mad = np.median(np.abs(reference - centre), axis=0) * 1.4826
    scale = np.where(scale > 0, scale, mad)
    scale = np.where(scale > 0, scale, reference.std(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return centre, scale


def marginal_deviation_attributions(X: np.ndarray, *, reference: np.ndarray) -> tuple[np.ndarray, str]:
    """Reference-scaled marginal deviation ``|x - median| / scale`` per dimension.

    The fallback for a saturating score (see the module docstring): it ignores
    the detector and only asks how far each value sits from the bulk of the
    reference population, in units of that dimension's robust spread (IQR/1.349,
    falling back to the scaled MAD, then the standard deviation, then 1 for a
    constant column). Unlike a gradient it keeps growing with the distance, so
    the dimension a row is furthest out in ranks first. It cannot see
    interactions (a row that is unusual only as a combination of in-range
    values gets no signal from it), which is why it is used only where the
    gradient has already lost the signal.

    ECOD's per-feature tail probabilities were considered instead: they need a
    separately fitted ECOD, and they saturate too -- every value beyond the
    reference's extreme gets the same capped tail probability, so 150 and 900
    would tie against a reference that tops out at 90.

    Returns
    -------
    attributions:
        Shape (n_rows, n_features), non-negative.
    tag:
        Always "heuristic".

    """
    _check_reference(X, reference, "marginal_deviation_attributions")
    centre, scale = _robust_scale(reference)
    return np.abs(X - centre) / scale, "heuristic"


def _outside_reference_range(X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Boolean mask: rows of *X* below the reference minimum or above its maximum in any dimension."""
    below = np.less(X, reference.min(axis=0))
    above = np.greater(X, reference.max(axis=0))
    return np.asarray((below | above).any(axis=1))


def gradient_attributions(
    detector: Any,
    X: np.ndarray,
    *,
    reference: np.ndarray,
    max_rows: int = 5000,
    step_factor: float = _DEFAULT_STEP_FACTOR,
    saturating: bool = False,
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
    saturating:
        Set for detectors whose score goes flat outside the data (IsolationForest,
        RBF One-Class SVM). Rows outside the reference range in any dimension are
        then attributed with ``marginal_deviation_attributions`` without
        computing a gradient, as are rows whose gradient is exactly zero.

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

    use_marginal = _outside_reference_range(X, reference) if saturating else np.zeros(n_rows, dtype=bool)
    n_gradient_rows = int((~use_marginal).sum())

    projected_calls = 2 * n_gradient_rows * n_features
    logger.info(
        "gradient_attributions: %d rows x %d features → %d score_samples calls.",
        n_gradient_rows,
        n_features,
        projected_calls,
    )

    attributions = np.zeros((n_rows, n_features), dtype=np.float64)

    for i in range(n_rows):
        if use_marginal[i]:
            continue
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

    if saturating:
        # A row inside the reference range can still sit on a flat stretch of a
        # piecewise-constant score; an all-zero gradient says nothing about it.
        use_marginal |= ~attributions.any(axis=1)
        if use_marginal.any():
            marginal, _ = marginal_deviation_attributions(X[use_marginal], reference=reference)
            attributions[use_marginal] = marginal
            logger.info(
                "gradient_attributions: %d of %d row(s) outside the reference range or with a "
                "zero gradient; used the reference-scaled marginal deviation for them.",
                int(use_marginal.sum()),
                n_rows,
            )

    return attributions, "heuristic"


def kernel_shap_attributions(
    detector: Any,
    X: np.ndarray,
    *,
    reference: np.ndarray,
    background_k: int = 50,
    max_rows: int = 5000,
    saturating: bool = False,
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
    saturating:
        Passed to the gradient fallback used when shap is not installed (see
        ``gradient_attributions``); KernelSHAP itself does not use it.

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
        return gradient_attributions(
            detector, X, reference=reference, max_rows=max_rows, saturating=saturating
        )

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
