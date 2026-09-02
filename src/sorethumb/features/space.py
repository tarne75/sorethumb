"""FeatureSpace: the fitted, scaled feature matrix ready for detector training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from sorethumb.profiling.plan import FeaturePlan


@dataclass
class FeatureSpace:
    """Fitted, scaled feature matrix ready for detector training or scoring.

    The matrix is in plan.output_dtype (float32 by default). Row IDs are monotonically
    increasing integers that map matrix rows back to the original data — join on them,
    never rely on positional alignment.

    feature_schema_hash is a digest of the ordered feature names used for model-reuse
    validation: if it differs between the training run and a score-forward run, the
    persisted estimator cannot be safely reused (§9.4).
    """

    matrix: np.ndarray
    feature_names: list[str]
    row_ids: np.ndarray
    plan: FeaturePlan
    feature_schema_hash: str

    @staticmethod
    def make_hash(names: list[str]) -> str:
        """Digest of ordered feature names. Changes when features are added, dropped, or reordered."""
        return hashlib.sha256("|".join(names).encode()).hexdigest()[:32]
