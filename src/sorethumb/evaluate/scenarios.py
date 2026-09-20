"""Synthetic anomaly-detection scenarios: mixed data with a named taxonomy of anomaly types.

Each with known ground truth and no network access. Every scenario mixes
continuous numeric columns with a handful of categorical
columns (so the harness that consumes these must go through real feature
encoding, not just hand a bare numeric matrix to a detector) and returns
``(pl.DataFrame, y)`` where ``y`` is a 0/1 numpy array, 1 = anomaly.

Taxonomy
--------
point
    Individually far-from-bulk points (the "classic" case every detector
    should clear easily).
local
    Points in the low-density gap between two normal clusters -- not extreme
    in any single feature, only locally sparse relative to their neighbours.
contextual
    A categorical "context" column changes what "normal" means for a numeric
    feature; anomalies have a numeric value that is normal for a *different*
    context than the one they are labelled with. Deliberately not modelled
    via group_by (which would let a group-aware detector trivially solve it
    by fitting one model per context) -- the context is an ordinary feature,
    testing whether encoding alone lets a detector learn the conditional
    structure.
clustered
    Anomalies form their own small, tight cluster elsewhere in feature space.
    Documented LOF weakness (docs/approximations.md): a local-density method
    rates a point normal *within its own neighbourhood*.
masking
    A larger contiguous block of near-identical anomalies -- enough that
    per-point methods can under-detect some of them because they mutually
    support each other's local density/isolation depth.
swamping
    Contamination is injected into the *training* split itself (unlabelled),
    shifting the fitted "normal" boundary; the held-out split's genuinely
    normal points near where that contamination sat can get swamped
    (falsely flagged). Compare this scenario's held-out accuracy against
    "point" (same held-out anomaly pattern, clean training) to see the
    degradation swamping causes.
varying_density
    Two normal clusters of different spread (one tight, one loose); a
    distance that is clearly anomalous relative to the tight cluster is
    unremarkable relative to the loose one. The classic case local-density
    methods (LOF) are designed for and fixed-threshold/global-distance
    methods (OneClassSVM, KMeans-distance) are not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

_CONTEXT_LEVELS = ("north", "south", "east", "west")
_TIER_LEVELS = ("gold", "silver", "bronze")


def _categorical_columns(rng: np.random.Generator, n: int) -> dict[str, list[str]]:
    """Two categorical columns carrying no anomaly signal of their own (noise features)."""
    return {
        "region": rng.choice(_CONTEXT_LEVELS, size=n, p=[0.4, 0.3, 0.2, 0.1]).tolist(),
        "tier": rng.choice(_TIER_LEVELS, size=n, p=[0.2, 0.3, 0.5]).tolist(),
    }


def _assemble(numeric: dict[str, np.ndarray], categorical: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame({**numeric, **categorical})


def _shuffle(df: pl.DataFrame, y: np.ndarray, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(df))
    return df[perm.tolist()], y[perm]


def point_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 40, n_numeric: int = 5
) -> tuple[pl.DataFrame, np.ndarray]:
    """Individually far-from-bulk points in an otherwise Gaussian cluster."""
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal)
    anomaly = rng.uniform(-8, -6, size=(n_anomaly, n_numeric))
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    return _shuffle(_assemble(numeric, categorical), y, seed)


def local_anomalies(
    seed: int, n_per_cluster: int = 400, n_anomaly: int = 40, n_numeric: int = 4
) -> tuple[pl.DataFrame, np.ndarray]:
    """Points in the sparse gap between two normal clusters of the same density."""
    rng = np.random.default_rng(seed)
    sep = 10.0
    center_a = np.zeros(n_numeric)
    center_b = np.full(n_numeric, sep)
    cluster_a = rng.multivariate_normal(center_a, np.eye(n_numeric), size=n_per_cluster)
    cluster_b = rng.multivariate_normal(center_b, np.eye(n_numeric), size=n_per_cluster)
    # Midpoint of the two clusters, jittered -- locally sparse (far from both
    # clusters' mass) without being a global extreme (it sits inside the
    # overall data's bounding range).
    midpoint = np.full(n_numeric, sep / 2)
    anomaly = midpoint + rng.normal(0.0, 0.6, size=(n_anomaly, n_numeric))
    X = np.vstack([cluster_a, cluster_b, anomaly])
    y = np.concatenate([np.zeros(2 * n_per_cluster), np.ones(n_anomaly)]).astype(int)
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    return _shuffle(_assemble(numeric, categorical), y, seed)


def contextual_anomalies(
    seed: int, n_per_context: int = 400, anomaly_rate: float = 0.05, n_noise_numeric: int = 3
) -> tuple[pl.DataFrame, np.ndarray]:
    """Inject a numeric value normal for one context under another context's label.

    ``context`` is an ordinary categorical feature, not a group_by column --
    see the module docstring for why.
    """
    rng = np.random.default_rng(seed)
    contexts = ("A", "B", "C")
    context_means = {"A": 0.0, "B": 10.0, "C": -10.0}

    rows_context: list[str] = []
    rows_signal: list[float] = []
    labels: list[int] = []
    for ctx in contexts:
        n_anom = max(1, round(n_per_context * anomaly_rate))
        n_norm = n_per_context - n_anom
        rows_context.extend([ctx] * n_norm)
        rows_signal.extend(rng.normal(context_means[ctx], 1.0, size=n_norm).tolist())
        labels.extend([0] * n_norm)

        other_ctx = rng.choice([c for c in contexts if c != ctx], size=n_anom)
        rows_context.extend([ctx] * n_anom)
        rows_signal.extend([rng.normal(context_means[str(oc)], 1.0) for oc in other_ctx])
        labels.extend([1] * n_anom)

    n = len(rows_context)
    numeric = {"num_signal": np.asarray(rows_signal)}
    for i in range(n_noise_numeric):
        numeric[f"num_noise_{i}"] = rng.normal(0.0, 1.0, size=n)
    categorical = {"context": rows_context, **_categorical_columns(rng, n)}
    y = np.asarray(labels, dtype=int)
    return _shuffle(_assemble(numeric, categorical), y, seed)


def clustered_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 40, n_numeric: int = 5
) -> tuple[pl.DataFrame, np.ndarray]:
    """Anomalies forming their own small, tight cluster (documented LOF weakness)."""
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal)
    anomaly_center = np.full(n_numeric, -7.0)
    anomaly = rng.multivariate_normal(anomaly_center, 0.3 * np.eye(n_numeric), size=n_anomaly)
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    return _shuffle(_assemble(numeric, categorical), y, seed)


def masking_anomalies(
    seed: int, n_normal: int = 700, n_anomaly: int = 150, n_numeric: int = 5
) -> tuple[pl.DataFrame, np.ndarray]:
    """Generate a large contiguous anomaly block (~17% of rows) that can mask itself.

    Enough near-identical anomalies that per-point isolation/density methods
    can rate some of them "normal" relative to each other.
    """
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal)
    anomaly_center = np.full(n_numeric, -6.0)
    anomaly = rng.multivariate_normal(anomaly_center, 0.5 * np.eye(n_numeric), size=n_anomaly)
    X = np.vstack([normal, anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    return _shuffle(_assemble(numeric, categorical), y, seed)


def swamping_train_reference(
    seed: int,
    n_normal: int = 800,
    n_train_contamination: int = 80,
    n_numeric: int = 5,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Generate a *training/reference* split with unlabelled contamination baked in.

    Returns a frame with no held-out anomalies of its own (``y`` is all
    zeros) -- pair this with ``point_anomalies`` (same generator family) as
    the held-out split: fit on this, score that, and compare against fitting
    on a clean reference to see the accuracy swamping costs.
    """
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal)
    contamination = rng.uniform(-8, -6, size=(n_train_contamination, n_numeric))
    X = np.vstack([normal, contamination])
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    # Unlabelled by design: the fitted model never sees ground truth, only
    # the (contaminated) reference distribution.
    y = np.zeros(len(X), dtype=int)
    return _shuffle(_assemble(numeric, categorical), y, seed)


def varying_density_anomalies(
    seed: int,
    n_tight: int = 500,
    n_loose: int = 500,
    n_anomaly: int = 40,
    n_numeric: int = 4,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Two normal clusters of different spread; anomalies sit near the tight one.

    A distance that is clearly anomalous relative to the tight cluster is
    unremarkable relative to the loose one -- classic density-normalisation
    case (LOF's design purpose; a trap for fixed-threshold/global-distance
    methods).
    """
    rng = np.random.default_rng(seed)
    tight_center = np.zeros(n_numeric)
    loose_center = np.full(n_numeric, 12.0)
    tight = rng.multivariate_normal(tight_center, 0.3 * np.eye(n_numeric), size=n_tight)
    loose = rng.multivariate_normal(loose_center, 3.0 * np.eye(n_numeric), size=n_loose)
    # Offset from the tight cluster that is far outside its own spread
    # (sigma=0.3) but well inside the loose cluster's (sigma=3.0).
    offset = np.full(n_numeric, 2.5)
    anomaly = tight_center + offset + rng.normal(0.0, 0.2, size=(n_anomaly, n_numeric))
    X = np.vstack([tight, loose, anomaly])
    y = np.concatenate([np.zeros(n_tight + n_loose), np.ones(n_anomaly)]).astype(int)
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    return _shuffle(_assemble(numeric, categorical), y, seed)


@dataclass(frozen=True)
class Scenario:
    """One registered synthetic scenario."""

    name: str
    kind: str
    description: str
    generate: Callable[[int], tuple[pl.DataFrame, np.ndarray]]
    # Fixed review budget for precision@k/recall@k/F1@k on this scenario
    # (see evaluate_scores) -- never derived from y.mean().
    review_budget: float = 0.05


SCENARIOS: list[Scenario] = [
    Scenario(
        "point",
        "point",
        "Individually far-from-bulk points in an otherwise Gaussian cluster.",
        point_anomalies,
    ),
    Scenario(
        "local",
        "local",
        "Points in the low-density gap between two normal clusters of equal density.",
        local_anomalies,
    ),
    Scenario(
        "contextual",
        "contextual",
        "A numeric value normal for one categorical context, mislabelled under another.",
        contextual_anomalies,
    ),
    Scenario(
        "clustered",
        "clustered",
        "Anomalies forming their own small, tight cluster (documented LOF weakness).",
        clustered_anomalies,
    ),
    Scenario(
        "masking",
        "masking",
        "A large contiguous anomaly block (~17%) that can mask itself from per-point methods.",
        masking_anomalies,
        review_budget=0.15,
    ),
    Scenario(
        "varying_density",
        "varying_density",
        "Two normal clusters of different spread; a fixed-distance trap for global methods.",
        varying_density_anomalies,
    ),
]


def scenario_by_name(name: str) -> Scenario | None:
    """Look up a Scenario by name, or None if unregistered."""
    return next((s for s in SCENARIOS if s.name == name), None)
