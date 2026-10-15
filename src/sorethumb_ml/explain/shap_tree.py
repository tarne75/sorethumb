"""TreeSHAP attributions for IsolationForest.

Interventional TreeSHAP: shap.TreeExplainer on the fitted sklearn
IsolationForest, with the reference population as its background
(``feature_perturbation="interventional"``). A feature's attribution is then
how much the row's value moves the forest's output relative to *normal* rows
having it, which is the question an anomaly explanation asks. The
path-dependent alternative (no background) weighs each branch by the share of
*training* rows that took it, contaminated rows included.

The background is the reference summary's k-means centres, each repeated in
proportion to the share of rows it stands for (``_TREE_BACKGROUND_ROWS`` rows
in all), since TreeExplainer averages over its background uniformly.

Additivity is checked: the values plus the base value must sum to the forest's
output. If shap's check fails, the values are recomputed without it and
reported with the ``treeshap_additivity_unchecked`` flag. Either way the tag is
"model_specific", not "exact": what TreeSHAP decomposes is the forest's
expected path length, of which IsolationForest's score is a monotone, nonlinear
transform, so an exact decomposition of path length is not an exact
decomposition of the score the user sees (flag ``treeshap_path_length``).

The known failure where a tree collapses to a single root node causes shap to
index past the end of the node array; we catch that, fall back to the gradient
method, and emit FallbackAttributionWarning -- the result tag becomes
"heuristic". The same fallback applies when shap itself is not installed (it
lives in the optional ``explain`` extra, not a core dependency). That fallback
runs with ``saturating=True`` (see ``explain/gradient.py``): rows outside the
reference range, where the forest's score is flat, are attributed by their
reference-scaled marginal deviation instead of a zero gradient.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from sorethumb_ml.errors import FallbackAttributionWarning
from sorethumb_ml.explain.reference import ReferenceSummary

if TYPE_CHECKING:
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

logger = logging.getLogger(__name__)

# Rows in the interventional background (centres repeated by weight).
_TREE_BACKGROUND_ROWS = 100


def _tree_background(reference: np.ndarray | ReferenceSummary) -> np.ndarray:
    """Return the interventional background: summary centres repeated in proportion to their weight."""
    if isinstance(reference, ReferenceSummary):
        summary = reference
    else:
        matrix = np.asarray(reference, dtype=np.float64)
        summary = ReferenceSummary.from_rows(
            matrix, reference_id="supplied", feature_names=[f"f{i}" for i in range(matrix.shape[1])]
        )
    if len(summary.background) == 0:
        msg = "TreeSHAP needs a reference with a background; the summary has none."
        raise ValueError(msg)
    counts = np.maximum(1, np.round(summary.background_weights * _TREE_BACKGROUND_ROWS)).astype(int)
    return np.repeat(summary.background, counts, axis=0)


def tree_shap_attributions(
    detector: IsolationForestDetector,
    X: np.ndarray,
    group_name: str = "",
    *,
    reference: np.ndarray | ReferenceSummary,
) -> tuple[np.ndarray, str]:
    """Compute per-row TreeSHAP attributions for an IsolationForest.

    See ``tree_shap_attributions_detailed``; this returns only
    ``(attributions, tag)``.
    """
    attributions, tag, _, _, _ = tree_shap_attributions_detailed(detector, X, group_name, reference=reference)
    return attributions, tag


def tree_shap_attributions_detailed(
    detector: IsolationForestDetector,
    X: np.ndarray,
    group_name: str = "",
    *,
    reference: np.ndarray | ReferenceSummary,
) -> tuple[np.ndarray, str, str, np.ndarray, list[str]]:
    """Per-row TreeSHAP attributions, with the method actually used, any marginal-fallback rows and caveats.

    Parameters
    ----------
    detector:
        A fitted IsolationForestDetector.
    X:
        Float64 *target* rows to explain, shape (n_rows, n_features).
    group_name:
        Used in warning messages when the fallback is triggered.
    reference:
        The reference population, as a matrix in the same feature space or a
        :class:`ReferenceSummary`. Its background is TreeSHAP's interventional
        background; the gradient fallback scales its perturbations from it.

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly.
        The sign is flipped from SHAP's natural direction (which attributes
        toward "more normal") so higher attribution = more anomalous.
    tag:
        "model_specific" on success, "heuristic" after fallback.
    method:
        ``"tree_shap"`` on success, ``"finite_difference_gradient"`` after fallback.
    marginal_rows:
        Rows the fallback attributed by marginal deviation (all-False on success).
    flags:
        Approximation flags for the explanation record: ``treeshap_path_length``
        on success, plus ``treeshap_additivity_unchecked`` when shap's
        additivity check failed; empty after fallback.

    """
    try:
        import shap  # noqa: PLC0415

        explainer = shap.TreeExplainer(
            detector._model,  # noqa: SLF001
            data=_tree_background(reference),
            feature_perturbation="interventional",
        )
        flags = ["treeshap_path_length"]
        try:
            shap_values = explainer.shap_values(X, check_additivity=True)
        except Exception as exc:  # shap's additivity error has no stable public type
            if "additivity" not in str(exc).lower():
                raise
            logger.warning("TreeSHAP additivity check failed for group %r: %s", group_name, exc)
            shap_values = explainer.shap_values(X, check_additivity=False)
            flags.append("treeshap_additivity_unchecked")
        # SHAP convention: positive = pushes the output (path length) up = more normal.
        # Negate so positive attribution means more anomalous.
        attributions = -np.asarray(shap_values, dtype=np.float64)
        return attributions, "model_specific", "tree_shap", np.zeros(len(attributions), dtype=bool), flags

    except ImportError as exc:
        warnings.warn(
            FallbackAttributionWarning(
                f"shap is not installed; TreeSHAP attributions for {group_name!r} are "
                "unavailable. Falling back to gradient/marginal-deviation attributions (heuristic). "
                'Install with: pip install "sorethumb-ml[explain]".',
                detector=group_name,
                rows_affected=len(X),
            ),
            stacklevel=2,
        )
        logger.warning("TreeSHAP unavailable for group %r (shap not installed): %s", group_name, exc)

    except (IndexError, ValueError, RuntimeError) as exc:
        warnings.warn(
            FallbackAttributionWarning(
                f"TreeSHAP failed for {group_name!r} ({type(exc).__name__}: {exc}); "
                "falling back to gradient/marginal-deviation attributions (heuristic).",
                detector=group_name,
                rows_affected=len(X),
            ),
            stacklevel=2,
        )
        logger.warning("TreeSHAP fallback triggered for group %r: %s", group_name, exc)

    # Fallback: import here to avoid circular dependency at module load time
    from sorethumb_ml.explain.gradient import gradient_attributions_detailed  # noqa: PLC0415

    # saturating=True: IsolationForest's score is piecewise constant, so far-out
    # rows get a zero gradient in exactly the dimension that isolated them.
    attrs, marginal_rows = gradient_attributions_detailed(detector, X, reference=reference, saturating=True)
    return attrs, "heuristic", "finite_difference_gradient", marginal_rows, []
