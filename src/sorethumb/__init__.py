"""sorethumb — unsupervised anomaly detection for tabular data.

Public API
----------
run_detection(config)          — full pipeline: profile → features → detect → explain → report
score_with_existing(...)       — score new data with persisted models (no refit)
load_dataset(source_config)    — resolve a source, read and unnest
build_feature_plan(df, config) — profile + classify + plan (no fitting)
apply_feature_plan(df, plan)   — apply a pre-fitted plan to new data
list_detectors()               — names of registered detectors
evaluate_scores(scores, ...)   — ROC-AUC, AP, P@k, R@k, F1

Types
-----
Config, SourceConfig, RunResult, GroupSummary, FeaturePlan, FeatureSpace,
Metrics, Workspace, SorethumbError
"""

__version__ = "0.1.0"

# Pipeline entry points
from sorethumb._pipeline import (
    GroupSummary,
    RunResult,
    list_detectors,
    load_dataset,
    run_detection,
)

# Config
from sorethumb.config import Config, SourceConfig

# Detector protocol (for third-party implementations)
from sorethumb.detectors._protocol import Detector

# Errors
from sorethumb.errors import SorethumbError

# Evaluation
from sorethumb.evaluate.metrics import Metrics, evaluate_scores

# Feature plan
from sorethumb.features.build import apply_feature_plan
from sorethumb.features.space import FeatureSpace

# Profiling
from sorethumb.profiling.plan import FeaturePlan, build_feature_plan

# Workspace
from sorethumb.store.workspace import Workspace

__all__ = [
    "Config",
    "Detector",
    "FeaturePlan",
    "FeatureSpace",
    "GroupSummary",
    "Metrics",
    "RunResult",
    "SorethumbError",
    "SourceConfig",
    "Workspace",
    "__version__",
    "apply_feature_plan",
    "build_feature_plan",
    "evaluate_scores",
    "list_detectors",
    "load_dataset",
    "run_detection",
]
