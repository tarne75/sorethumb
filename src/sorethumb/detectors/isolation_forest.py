"""IsolationForest detector.

Wraps sklearn's IsolationForest. score_samples() returns values where higher means
more normal, matching sklearn's own convention (no sign flip needed).

contamination is never passed to sklearn: thresholding is the scoring layer's job.
Passing it here would give two competing thresholds and make contamination=auto
ambiguous. The natural_flag boundary uses sklearn's fitted offset_ (≈ −0.5 for the
default contamination='auto' mode, which corresponds to the original paper's threshold).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np


class IsolationForestDetector:
    """sklearn IsolationForest wrapped to satisfy the Detector protocol."""

    name: ClassVar[str] = "isolation_forest"
    supports_tree_shap: ClassVar[bool] = True
    default_train_row_cap: ClassVar[int] = 250_000

    def __init__(self, n_estimators: int = 200, max_samples: str | int = "auto") -> None:
        """Initialise with sklearn IsolationForest hyper-parameters."""
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._model: Any = None

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        """Fit the isolation forest on X."""
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415

        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            random_state=seed,
        )
        self._model.fit(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more normal (sklearn convention, no flip)."""
        return self._model.score_samples(X)  # type: ignore[no-any-return]

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows below the model's fitted offset (≈ −0.5 at default contamination).

        Uses sklearn's offset_ so the threshold tracks the training distribution
        rather than a fixed constant.
        """
        return scores < float(self._model.offset_)

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        return {"n_estimators": self._n_estimators, "max_samples": self._max_samples}
