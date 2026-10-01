"""sorethumb — unsupervised anomaly detection for tabular data.

Public API
----------
run_detection(config)          — full pipeline: profile → features → detect → explain → report
score_forward(config, run_id)  — score new data with a prior run's plan + models (no refit)
render_report_for_run(ws, run_id) — (re-)render a run's HTML report from persisted state
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

import importlib.metadata

try:
    # Single source of truth: the installed distribution's metadata (which
    # comes from pyproject.toml's `version` at build time), not a second
    # hard-coded copy that can drift from it. Works for an editable dev
    # install too -- uv/pip both write real dist-info metadata for those.
    # Looked up by the PyPI *distribution* name ("sorethumb-ml"), which
    # differs from this import package's own name.
    __version__ = importlib.metadata.version("sorethumb-ml")
except importlib.metadata.PackageNotFoundError:
    # sorethumb was imported from source without being installed at all
    # (no pip/uv install step) -- there is no metadata to read.
    __version__ = "0+unknown"

# Pipeline entry points
from sorethumb_ml._pipeline import (
    GroupSummary,
    RunResult,
    list_detectors,
    load_dataset,
    render_report_for_run,
    run_detection,
    score_forward,
)

# Config
from sorethumb_ml.config import Config, SourceConfig

# Detector protocol (for third-party implementations)
from sorethumb_ml.detectors._protocol import Detector

# Errors
from sorethumb_ml.errors import SorethumbError

# Evaluation
from sorethumb_ml.evaluate.metrics import Metrics, evaluate_scores

# Feature plan
from sorethumb_ml.features.build import apply_feature_plan
from sorethumb_ml.features.space import FeatureSpace

# Profiling
from sorethumb_ml.profiling.plan import FeaturePlan, build_feature_plan

# Workspace
from sorethumb_ml.store.workspace import Workspace

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
    "render_report_for_run",
    "run_detection",
    "score_forward",
]
