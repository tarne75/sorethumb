"""Detector registry.

Built-in detectors are registered here. Third-party detectors can be registered
via the ``sorethumb.detectors`` entry-point group or by calling ``register()``
directly at import time.

Registry contract
-----------------
Every entry must satisfy the Detector protocol (``check_protocol`` is called at
registration time). The dict key is ``cls.name`` — the canonical string used in
FeaturePlan and manifest JSON.

Usage
-----
    from sorethumb.detectors import registry, register

    register(MyCustomDetector)              # raises DetectorError if invalid
    det = registry["isolation_forest"]()   # instantiate the built-in IF detector
"""

from __future__ import annotations

import importlib.metadata
import logging

from sorethumb.detectors._protocol import Detector, check_protocol
from sorethumb.detectors.isolation_forest import IsolationForestDetector
from sorethumb.detectors.kmeans_distance import KMeansDetector
from sorethumb.detectors.one_class_svm import OneClassSVMDetector

logger = logging.getLogger(__name__)

registry: dict[str, type[Detector]] = {}


def register(cls: type) -> None:
    """Register a detector class, raising DetectorError if it fails the protocol check."""
    check_protocol(cls)
    registry[cls.name] = cls  # type: ignore[attr-defined]
    logger.debug("Registered detector: %s", cls.name)  # type: ignore[attr-defined]


# Register built-ins
register(IsolationForestDetector)
register(KMeansDetector)
register(OneClassSVMDetector)

# Load entry-point extensions
try:
    eps = importlib.metadata.entry_points(group="sorethumb.detectors")
    for ep in eps:
        try:
            cls = ep.load()
            register(cls)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load detector entry-point %r; skipping.", ep.name)
except Exception:  # noqa: BLE001
    logger.debug("Entry-point discovery failed; only built-in detectors available.")

__all__ = [
    "Detector",
    "IsolationForestDetector",
    "KMeansDetector",
    "OneClassSVMDetector",
    "register",
    "registry",
]
