"""Native per-feature attributions for detectors whose score is already additive.

ECOD and HBOS both define their score as an unweighted average of independent
per-feature terms (an empirical-CDF tail probability for ECOD, a histogram
bin log-density for HBOS — see each detector's ``feature_contributions``).
Recovering the per-feature contribution is not an approximation of
anything: it is exactly what the score already sums, so the tag is "exact" —
summing the returned per-row vector recovers the score with zero error, by
construction, not merely "high fidelity". This is a stronger guarantee than
TreeSHAP's "model_specific" tag (see ``shap_tree.py``), which is explicitly
*not* additivity-verified.

This is also why finite-difference gradients are a poor fit for these two
specifically (see the dispatch in ``_pipeline.py::_compute_attributions``):
the underlying quantity is a discrete rank/bin lookup with no sub-resolution
structure, so a small perturbation of a genuinely anomalous (far-tail) row
routinely lands in the exact same rank position or histogram bin, producing
an exact zero finite difference for precisely the row that matters most.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sorethumb.detectors.ecod import ECODDetector
    from sorethumb.detectors.hbos import HBOSDetector


def ecod_attributions(detector: ECODDetector, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Exact per-feature decomposition of the ECOD outlier score.

    Parameters
    ----------
    detector:
        A fitted ECODDetector.
    X:
        Float64 feature matrix, shape (n_rows, n_features).

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly. Summing
        across features recovers the row's ECOD outlier score exactly.
    tag:
        Always "exact".

    """
    return detector.feature_contributions(X), "exact"


def hbos_attributions(detector: HBOSDetector, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Exact per-feature decomposition of the HBOS outlier score.

    Parameters
    ----------
    detector:
        A fitted HBOSDetector.
    X:
        Float64 feature matrix, shape (n_rows, n_features).

    Returns
    -------
    attributions:
        Shape (n_rows, n_features). Positive = pushes toward anomaly. Summing
        across features recovers the row's HBOS outlier score exactly.
    tag:
        Always "exact".

    """
    return detector.feature_contributions(X), "exact"
