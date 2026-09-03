"""Explanation layer: per-row attributions over original input columns.

Each attribution is tagged ``exact`` or ``heuristic``; the tag travels with
the result through blending and back-projection so callers always know which
method produced it.
"""

from sorethumb.explain.blend import blend
from sorethumb.explain.centroid import centroid_attributions
from sorethumb.explain.gradient import gradient_attributions
from sorethumb.explain.project import aggregate_to_original, back_project_pca
from sorethumb.explain.shap_tree import tree_shap_attributions

__all__ = [
    "aggregate_to_original",
    "back_project_pca",
    "blend",
    "centroid_attributions",
    "gradient_attributions",
    "tree_shap_attributions",
]
