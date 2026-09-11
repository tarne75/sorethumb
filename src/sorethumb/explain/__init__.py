"""Explanation layer: per-row attributions over original input columns.

Each attribution is tagged ``exact``, ``model_specific``, or ``heuristic``;
the tag travels with the result through blending and back-projection so
callers always know which method produced it, and blending a mix of tiers
keeps only the weakest one (see ``blend.py``). ``exact`` means summing the
attribution recovers the score with zero error by construction (ECOD, HBOS —
see ``native.py``); ``model_specific`` means it uses the detector's actual
internal structure but isn't an additivity-verified decomposition of the
score (TreeSHAP — see ``shap_tree.py``); ``heuristic`` covers everything else
(centroid distance, input gradient, KernelSHAP).
"""

from sorethumb.explain.blend import blend
from sorethumb.explain.centroid import centroid_attributions
from sorethumb.explain.gradient import gradient_attributions
from sorethumb.explain.native import ecod_attributions, hbos_attributions
from sorethumb.explain.project import aggregate_to_original, back_project_pca
from sorethumb.explain.shap_tree import tree_shap_attributions

__all__ = [
    "aggregate_to_original",
    "back_project_pca",
    "blend",
    "centroid_attributions",
    "ecod_attributions",
    "gradient_attributions",
    "hbos_attributions",
    "tree_shap_attributions",
]
