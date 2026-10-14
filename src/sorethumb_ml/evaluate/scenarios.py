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
    shifting the fitted "normal" boundary. Unlike every other scenario here,
    this is a *matched pair*: ``swamping_clean_reference`` and
    ``swamping_train_reference`` share the identical normal population (same
    seed -> same draw) and differ by exactly the injected contamination, so
    any difference in outcome is attributable to the contamination alone,
    not to sampling noise between two unrelated fits. Both are scored
    against ``swamping_at_risk_holdout``: genuinely normal points *selected*
    (not fabricated) from the tail of the same normal law nearest the
    contamination region -- the specific population actually at risk of
    being swamped, rather than a random holdout draw that mostly never comes
    near the contamination in the first place and so cannot demonstrate the
    mechanism regardless of how much accuracy the contaminated fit actually
    loses.
varying_density
    Two normal clusters of different spread (one tight, one loose); a
    distance that is clearly anomalous relative to the tight cluster is
    unremarkable relative to the loose one. The classic case local-density
    methods (LOF) are designed for and fixed-threshold/global-distance
    methods (OneClassSVM, KMeans-distance) are not.
categorical_rare
    The anomaly is a rare level of a 3-value category and nothing numeric
    differs -- exactly the column ``profiling.near_constant_distinct``
    decides whether to drop, so it is the evidence for that default.
text_timestamp
    Unusual-hour activity recorded only in an ISO text timestamp column:
    reachable only through date parsing and the derived hour.
irrelevant_columns
    Moderate point anomalies in two columns among twenty pure-noise
    columns: whether the signal survives dilution.
all_categorical
    The one scenario with no numeric column at all (the mixed-data rule
    above does not apply): five categorical columns, and anomalies that are
    unseen combinations of individually common levels. Every column has
    five levels, so ``profiling.near_constant_distinct`` drops none of them.
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


def _swamping_normal_population(seed: int, n_normal: int, n_numeric: int) -> np.ndarray:
    """Generate the normal population shared by both halves of the swamping comparison.

    Called with the same ``(seed, n_normal, n_numeric)`` from both
    ``swamping_clean_reference`` and ``swamping_train_reference`` -- a fresh
    ``default_rng(seed)`` reproduces the identical first draw regardless of
    what either caller does with its own rng afterward, so the two training
    sets differ by exactly the injected contamination and nothing else.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal)


def swamping_clean_reference(
    seed: int, n_normal: int = 800, n_numeric: int = 5
) -> tuple[pl.DataFrame, np.ndarray]:
    """Generate the "clean fit" half of the matched swamping comparison.

    The identical normal population ``swamping_train_reference`` (same seed)
    trains on, with no contamination injected -- fit both, score each
    against ``swamping_at_risk_holdout``, and any difference in outcome is
    attributable to the contamination alone.
    """
    normal = _swamping_normal_population(seed, n_normal, n_numeric)
    categorical = _categorical_columns(np.random.default_rng(seed + 2000), len(normal))
    numeric = {f"num_{i}": normal[:, i] for i in range(n_numeric)}
    y = np.zeros(len(normal), dtype=int)
    return _shuffle(_assemble(numeric, categorical), y, seed)


def swamping_train_reference(
    seed: int,
    n_normal: int = 800,
    n_train_contamination: int = 80,
    n_numeric: int = 5,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Generate the "contaminated fit" half of the matched swamping comparison.

    Same normal population as ``swamping_clean_reference`` (same seed), plus
    unlabelled contamination baked in. Returns a frame with no held-out
    anomalies of its own (``y`` is all zeros) -- the model never sees ground
    truth, only the (contaminated) reference distribution it fits on.
    """
    normal = _swamping_normal_population(seed, n_normal, n_numeric)
    rng = np.random.default_rng(seed + 1)
    contamination = rng.uniform(-8, -6, size=(n_train_contamination, n_numeric))
    X = np.vstack([normal, contamination])
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = _categorical_columns(rng, len(X))
    y = np.zeros(len(X), dtype=int)
    return _shuffle(_assemble(numeric, categorical), y, seed)


def swamping_at_risk_holdout(
    seed: int, n_at_risk: int = 150, n_numeric: int = 5, n_candidates: int = 20_000
) -> tuple[pl.DataFrame, np.ndarray]:
    """Generate held-out genuinely-normal points nearest the contamination region.

    The population actually "at risk" of being swamped by a contaminated
    fit. Selection, not fabrication: every point here is an ordinary draw from
    the same ``N(0, I)`` law every scenario's normal class comes from (all
    labelled 0 -- there is no positive class in this holdout, since the
    point is to measure the false-positive rate a contaminated fit inflicts
    on points that were never actually anomalous). Of ``n_candidates`` such
    draws, the ``n_at_risk`` with the most negative sum across dimensions
    (closest to the uniform(-8, -6)-per-axis contamination region every
    training dimension shares) are kept -- a random holdout draw would
    almost never land this close by chance and so could never demonstrate
    the swamping mechanism regardless of how much accuracy a contaminated
    fit actually loses.
    """
    rng = np.random.default_rng(seed + 4242)
    candidates = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_candidates)
    nearest = np.argsort(candidates.sum(axis=1))[:n_at_risk]
    at_risk = candidates[nearest]
    categorical = _categorical_columns(rng, len(at_risk))
    numeric = {f"num_{i}": at_risk[:, i] for i in range(n_numeric)}
    y = np.zeros(len(at_risk), dtype=int)
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


def categorical_rare_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 25, n_numeric: int = 3
) -> tuple[pl.DataFrame, np.ndarray]:
    """Make the anomaly a rare level of a 3-value category; numeric columns are normal for everyone.

    Normal rows use "web" or "store" for ``channel``; anomalous rows, and only
    those, use "fax". The column therefore has exactly three distinct values --
    the case ``profiling.near_constant_distinct`` decides whether to drop -- and
    the only signal in the dataset is which level a row has.
    """
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal + n_anomaly)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    channel = [*rng.choice(["web", "store"], size=n_normal, p=[0.6, 0.4]).tolist(), *(["fax"] * n_anomaly)]
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = {**_categorical_columns(rng, len(X)), "channel": channel}
    return _shuffle(_assemble(numeric, categorical), y, seed)


def text_timestamp_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 30, n_numeric: int = 3
) -> tuple[pl.DataFrame, np.ndarray]:
    """Activity at an unusual hour, recorded only in a text timestamp column.

    Normal rows fall in business hours (09:00-17:59) on weekdays; anomalous rows
    at 01:00-03:59. Numeric columns are normal for everyone, so the signal is
    reachable only by parsing ``event_time`` (ISO text) into a datetime and
    deriving its hour.
    """
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(n_numeric), np.eye(n_numeric), size=n_normal + n_anomaly)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    days = rng.integers(0, 20, size=n_normal + n_anomaly)  # 2026-01-05 is a Monday; 4 weeks of weekdays
    day_offsets = days + 2 * (days // 5)
    hours = np.concatenate([rng.integers(9, 18, size=n_normal), rng.integers(1, 4, size=n_anomaly)])
    minutes = rng.integers(0, 60, size=n_normal + n_anomaly)
    event_time = [
        f"2026-01-{5 + int(d):02d}T{int(h):02d}:{int(m):02d}:00"
        for d, h, m in zip(day_offsets, hours, minutes, strict=True)
    ]
    numeric = {f"num_{i}": X[:, i] for i in range(n_numeric)}
    categorical = {**_categorical_columns(rng, len(X)), "event_time": event_time}
    return _shuffle(_assemble(numeric, categorical), y, seed)


def irrelevant_columns_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 40, n_signal: int = 2, n_noise: int = 20
) -> tuple[pl.DataFrame, np.ndarray]:
    """Moderate point anomalies in two columns, diluted by many irrelevant ones.

    Anomalies sit 4-5 sigma out in the two signal columns only; twenty more
    independent Gaussian columns carry nothing. Tests whether the pipeline (and
    each policy) keeps the signal when most of the feature space is noise.
    """
    rng = np.random.default_rng(seed)
    n = n_normal + n_anomaly
    signal = rng.normal(0.0, 1.0, size=(n, n_signal))
    signal[n_normal:] = rng.uniform(4.0, 5.0, size=(n_anomaly, n_signal)) * rng.choice(
        [-1, 1], size=(n_anomaly, 1)
    )
    noise = rng.normal(0.0, 1.0, size=(n, n_noise))
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    numeric = {f"signal_{i}": signal[:, i] for i in range(n_signal)}
    numeric |= {f"noise_{i}": noise[:, i] for i in range(n_noise)}
    categorical = _categorical_columns(rng, n)
    return _shuffle(_assemble(numeric, categorical), y, seed)


_PROFILE_COLUMNS = ("region", "product", "channel", "payment", "device")


def all_categorical_anomalies(
    seed: int, n_normal: int = 800, n_anomaly: int = 40, n_profiles: int = 3, home_p: float = 0.85
) -> tuple[pl.DataFrame, np.ndarray]:
    """Every column categorical; anomalies combine levels that never co-occur in normal rows.

    Normal rows belong to one of *n_profiles* latent customer profiles; each of
    the five columns takes its profile's "home" level with probability
    *home_p* and any other level otherwise. An anomalous row takes each
    column's home level from an independently chosen profile, with at least
    two different profiles among its columns -- so every level it has is
    common, and only the combination is unusual.
    """
    rng = np.random.default_rng(seed)
    n_levels = 5
    levels = {c: [f"{c}_{i}" for i in range(n_levels)] for c in _PROFILE_COLUMNS}

    def draw(home: int) -> int:
        if rng.random() < home_p:
            return home
        return int(rng.choice([i for i in range(n_levels) if i != home]))

    rows: list[list[int]] = []
    for _ in range(n_normal):
        profile = int(rng.integers(n_profiles))
        rows.append([draw(profile) for _ in _PROFILE_COLUMNS])
    for _ in range(n_anomaly):
        profiles = rng.integers(n_profiles, size=len(_PROFILE_COLUMNS))
        while len(set(profiles.tolist())) < 2:
            profiles = rng.integers(n_profiles, size=len(_PROFILE_COLUMNS))
        rows.append([int(p) for p in profiles])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)]).astype(int)
    categorical = {c: [levels[c][r[j]] for r in rows] for j, c in enumerate(_PROFILE_COLUMNS)}
    return _shuffle(_assemble({}, categorical), y, seed)


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
    Scenario(
        "categorical_rare",
        "categorical_rare",
        "The anomaly is a rare level of a 3-value category; nothing numeric differs.",
        categorical_rare_anomalies,
    ),
    Scenario(
        "text_timestamp",
        "text_timestamp",
        "Unusual-hour activity visible only in a text timestamp column.",
        text_timestamp_anomalies,
    ),
    Scenario(
        "irrelevant_columns",
        "irrelevant_columns",
        "Point anomalies in 2 columns diluted by 20 irrelevant numeric columns.",
        irrelevant_columns_anomalies,
    ),
    Scenario(
        "all_categorical",
        "all_categorical",
        "No numeric columns; anomalies are unseen combinations of individually common levels.",
        all_categorical_anomalies,
    ),
]


def scenario_by_name(name: str) -> Scenario | None:
    """Look up a Scenario by name, or None if unregistered."""
    return next((s for s in SCENARIOS if s.name == name), None)
