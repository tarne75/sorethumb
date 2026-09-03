"""Detector protocol: the structural interface every detector must satisfy.

Every detector returns ``score_samples`` values where **higher = more normal**,
matching sklearn's IsolationForest convention. The sign flip to "higher = more
anomalous" happens once, in ``scoring/combine.py``, so no shared code branches
on model type. Applying the flip inside a detector would break composite scoring.

``natural_flag`` uses the model's own learned boundary rather than a fixed
contamination quota. This is the value used by ``contamination: auto`` to estimate
the expected anomaly rate from the data, not from a user guess.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

_REQUIRED_CLASS_ATTRS: frozenset[str] = frozenset({"name", "supports_tree_shap", "default_train_row_cap"})
_REQUIRED_METHODS: frozenset[str] = frozenset({"fit", "score_samples", "natural_flag", "get_params"})


@runtime_checkable
class Detector(Protocol):
    """Structural interface for anomaly detectors."""

    name: ClassVar[str]
    supports_tree_shap: ClassVar[bool]
    default_train_row_cap: ClassVar[int]

    def fit(self, X: np.ndarray, *, seed: int) -> None:
        """Fit the detector on *X*. Called once per (group, run)."""
        ...

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-row anomaly scores. Higher = more normal."""
        ...

    def natural_flag(self, scores: np.ndarray) -> np.ndarray:
        """Return a boolean array marking rows the model considers anomalous.

        Uses the model's own boundary (not a fixed quota) so that
        ``contamination: auto`` has a principled rate to work from.
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serialisable hyper-parameters for manifest storage."""
        ...


def check_protocol(cls: type) -> None:
    """Raise DetectorError if *cls* does not satisfy the Detector protocol.

    Checks both the required ClassVar attributes and the required methods.
    Called at registration time so a bad detector fails loudly on import,
    not silently during a long run.
    """
    from sorethumb.errors import DetectorError  # noqa: PLC0415

    missing: list[str] = []
    for attr in _REQUIRED_CLASS_ATTRS:
        if not hasattr(cls, attr):
            missing.append(attr)
    for method in _REQUIRED_METHODS:
        if not callable(getattr(cls, method, None)):
            missing.append(method)
    if missing:
        raise DetectorError(
            f"Detector class {cls.__qualname__!r} does not satisfy the Detector protocol. "
            f"Missing or non-callable: {sorted(missing)}"
        )
