"""TreeSHAP attributions for IsolationForest.

Uses shap.TreeExplainer on the fitted sklearn IsolationForest. The known failure
where a tree collapses to a single root node causes shap to index past the end of
the node array; we catch that, fall back to the gradient method, and emit
FallbackAttributionWarning — the result tag becomes "heuristic".

The explainer is constructed with check_additivity=False because IsolationForest's
score_samples does not equal the SHAP sum of its base value and contributions —
additivity checking would otherwise raise on every call. Two things follow from
that, and both mean the successful result is NOT tagged "exact": (1) additivity
is unverified, so there is no guarantee the values actually sum to the model
output; and (2) even where TreeSHAP's decomposition is exact, what it decomposes
is path length, which IsolationForest's score is a nonlinear transform of — so
"exact for path length" is not "exact for the score the user sees" either way.
The tag is "model_specific": more principled than a generic gradient/centroid
heuristic (it uses the fitted trees' actual structure via the standard TreeSHAP
algorithm), but not an additivity-guaranteed decomposition of the final score.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from sorethumb.errors import FallbackAttributionWarning

if TYPE_CHECKING:
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

logger = logging.getLogger(__name__)


def tree_shap_attributions(
    detector: IsolationForestDetector,
    X: np.ndarray,
    group_name: str = "",
) -> tuple[np.ndarray, str]:
    """Compute per-row TreeSHAP attributions for an IsolationForest.

    Parameters
    ----------
    detector:
        A fitted IsolationForestDetector.
    X:
        Float64 feature matrix, shape (n_rows, n_features).
    group_name:
        Used in warning messages when the fallback is triggered.

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly.
        The sign is flipped from SHAP's natural direction (which attributes
        toward "more normal") so higher attribution = more anomalous.
    tag:
        "model_specific" on success (TreeSHAP over the fitted trees, additivity
        unverified — not "exact"), "heuristic" after fallback.

    """
    import shap  # noqa: PLC0415

    try:
        explainer = shap.TreeExplainer(detector._model)  # noqa: SLF001
        # check_additivity=False because IF path-length scores are not strictly additive
        shap_values = explainer.shap_values(X, check_additivity=False)
        # shap_values: (n_rows, n_features) — SHAP convention: positive = pushes score higher = more normal
        # Negate so positive attribution means more anomalous (consistent with calibrated score direction)
        attributions = -np.asarray(shap_values, dtype=np.float64)
        return attributions, "model_specific"

    except (IndexError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"TreeSHAP failed for group {group_name!r} ({type(exc).__name__}: {exc}); "
            "falling back to gradient attributions (heuristic).",
            FallbackAttributionWarning,
            stacklevel=2,
        )
        logger.warning("TreeSHAP fallback triggered for group %r: %s", group_name, exc)

    # Fallback: import here to avoid circular dependency at module load time
    from sorethumb.explain.gradient import gradient_attributions  # noqa: PLC0415

    attrs, _ = gradient_attributions(detector, X)
    return attrs, "heuristic"
