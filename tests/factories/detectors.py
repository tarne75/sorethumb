"""Detector-fitting test helpers.

Detectors follow the protocol convention "higher score_samples() = more
normal" (see src/sorethumb/detectors/_protocol.py); anomaly labels use the
opposite convention (1 = anomaly). ``detector_auc`` writes the sign flip
once so callers never have to remember it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    import pytest


class _ScoringDetector(Protocol):
    def fit(self, X: np.ndarray, *, seed: int) -> object: ...
    def score_samples(self, X: np.ndarray) -> np.ndarray: ...


def detector_auc(detector: _ScoringDetector, X: np.ndarray, y_true: np.ndarray, *, seed: int = 42) -> float:
    """Fit ``detector`` on ``X`` and return its ROC-AUC against ``y_true``.

    ``y_true`` uses the usual convention (1 = anomaly). Handles the
    higher-is-more-normal sign flip internally.
    """
    detector.fit(X, seed=seed)
    scores = detector.score_samples(X)
    return float(roc_auc_score(y_true, -scores))


class _FitAttempted(BaseException):
    """Raised by ``ban_all_fitting``'s patches. A ``BaseException`` so a
    per-group ``except Exception`` handler in the pipeline cannot swallow
    it — it must reach the test."""


def ban_all_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every detector-fitting and feature-fitting entry point raise.

    For score-forward tests that must prove *nothing* gets refit: any call
    to a pipeline-level fit function, a detector wrapper's ``fit``, or the
    underlying sklearn estimator's ``fit`` raises ``_FitAttempted`` instead
    of running.
    """
    import sorethumb._pipeline as pipe
    from sorethumb.detectors.isolation_forest import IsolationForestDetector
    from sorethumb.detectors.kmeans_distance import KMeansDetector
    from sorethumb.detectors.one_class_svm import OneClassSVMDetector

    def _boom(*_a: object, **_k: object) -> None:
        raise _FitAttempted

    # Pipeline-level fitting entry points.
    monkeypatch.setattr(pipe, "fit_features", _boom)
    monkeypatch.setattr(pipe, "build_feature_plan", _boom)
    monkeypatch.setattr(pipe, "save_model", _boom)
    # Every detector wrapper's fit …
    for cls in (IsolationForestDetector, KMeansDetector, OneClassSVMDetector):
        monkeypatch.setattr(cls, "fit", _boom)
    # … and the underlying sklearn estimators, in case a wrapper is bypassed.
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    for cls in (IsolationForest, KMeans, OneClassSVM):
        monkeypatch.setattr(cls, "fit", _boom)
