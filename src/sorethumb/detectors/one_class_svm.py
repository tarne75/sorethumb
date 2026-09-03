"""OneClassSVM detector.

score_samples() returns the *negated* sklearn decision_function value so that
higher = more normal, matching the Detector protocol convention. sklearn's
decision_function is positive for the normal class and negative for outliers,
so negating it gives negative-for-normal, which is wrong — we negate once
more so the final sign is: higher (closer to zero from below) = more normal.

Wait — re-reading sklearn docs: decision_function returns positive for inliers
and negative for outliers. So we need to *negate* to get "higher = more normal"
from "higher = more anomalous". Actually, sklearn docs say:

  Signed distance to the separating hyperplane.
  Positive for an inlier, negative for an outlier.

So decision_function already gives higher = more normal (inliers are positive).
We do NOT negate. natural_flag is `scores < 0` (below the hyperplane).

nu="auto" trains with nu=0.1 as a reasonable default contamination estimate.
OneClassSVM is O(n²) for the kernel matrix; the train-row cap (25 000 default)
keeps it tolerable. A SlowStageWarning is emitted if the fit exceeds
slow_stage_seconds (configurable; default 60 s).
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, ClassVar

import numpy as np

from sorethumb.errors import SlowStageWarning

logger = logging.getLogger(__name__)

_DEFAULT_NU = 0.1
_SLOW_THRESHOLD_SECONDS = 60.0


class OneClassSVMDetector:
    """sklearn OneClassSVM wrapped to satisfy the Detector protocol."""

    name: ClassVar[str] = "one_class_svm"
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 25_000

    def __init__(
        self,
        nu: float | str = "auto",
        kernel: str = "rbf",
        gamma: str | float = "scale",
        slow_stage_seconds: float = _SLOW_THRESHOLD_SECONDS,
    ) -> None:
        """Initialise with OneClassSVM hyper-parameters. nu='auto' uses 0.1."""
        self._nu = nu
        self._kernel = kernel
        self._gamma = gamma
        self._slow_stage_seconds = slow_stage_seconds
        self._resolved_nu: float | None = None
        self._model: Any = None

    def fit(self, X: np.ndarray, *, seed: int) -> None:  # noqa: ARG002
        """Fit the one-class SVM on X. seed is accepted for API uniformity but unused."""
        from sklearn.svm import OneClassSVM  # noqa: PLC0415

        nu = _DEFAULT_NU if self._nu == "auto" else float(self._nu)
        self._resolved_nu = nu
        logger.info(
            "OneClassSVM: fitting nu=%.3f kernel=%s on %d rows x %d features.",
            nu,
            self._kernel,
            X.shape[0],
            X.shape[1],
        )
        self._model = OneClassSVM(nu=nu, kernel=self._kernel, gamma=self._gamma)
        t0 = time.monotonic()
        self._model.fit(X)
        elapsed = time.monotonic() - t0
        logger.info("OneClassSVM: fit completed in %.1f s.", elapsed)
        if elapsed > self._slow_stage_seconds:
            warnings.warn(
                f"OneClassSVM fit took {elapsed:.1f} s "
                f"(threshold={self._slow_stage_seconds:.0f} s). "
                "Consider reducing default_train_row_cap or switching to IsolationForest.",
                SlowStageWarning,
                stacklevel=2,
            )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Higher = more normal (sklearn decision_function sign)."""
        return self._model.decision_function(X)  # type: ignore[no-any-return]

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Flag rows below the hyperplane (decision_function < 0 = outlier per sklearn)."""
        return scores < 0.0

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters."""
        return {
            "nu": self._nu,
            "resolved_nu": self._resolved_nu,
            "kernel": self._kernel,
            "gamma": self._gamma,
        }
