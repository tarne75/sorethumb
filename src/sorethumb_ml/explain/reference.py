"""The reference population explanations are measured against.

Every attribution method needs a notion of "normal": finite-difference step
sizes and the marginal-deviation fallback need each dimension's typical value
and spread, KernelSHAP and TreeSHAP need a background to stand in for an
"absent" feature, centroid distances are scaled by it, and the explanation
record compares each raw value with it. Explaining an anomaly against a
population that contains anomalies measures its distance from other
anomalies, so the reference is chosen from rows the detectors agree are normal,
and is never the whole (contaminated) group:

1. ``consensus_normal`` -- rows no detector flags, neither the ensemble nor
   any single detector's own boundary (``natural_flag``). Used when at least
   ``MIN_REFERENCE_ROWS`` rows qualify.
2. ``lower_half`` -- otherwise, the unflagged rows whose composite score is at
   or below the group's median: the less anomalous half.
3. Otherwise there is no reference: the group's flagged rows get no
   explanation (``explain.reference_unavailable``).

The chosen rows are reduced to a :class:`ReferenceSummary` -- per-dimension
median, robust scale, standard deviation, mean absolute value, range, a
k-means background of at most ``BACKGROUND_K`` centres, and each original
field's typical raw value. A fitting run saves it as
``models/<run>/<group>/reference.json`` (plain JSON, never unpickled), and
``score --from-run`` explains new rows against the fitting run's summary, so a
forward-scored row is measured against the same "normal" the models were fit
on, not against whatever else happens to be in the new batch.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REFERENCE_FILENAME = "reference.json"
SUMMARY_VERSION = 1

# Fewer rows than this and a "normal" population is too small to trust.
MIN_REFERENCE_ROWS = 10
# k-means centres kept as the SHAP background.
BACKGROUND_K = 50
# Rows clustered for the background; a larger reference is subsampled (fixed seed).
_BACKGROUND_SAMPLE_CAP = 10_000


def select_reference_rows(
    *,
    n_rows: int,
    flagged_idx: np.ndarray,
    natural_flags: dict[str, np.ndarray],
    composite_score: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Return ``(row mask, reference_id)`` for the reference, or ``(None, None)`` when there is none."""
    unflagged = np.ones(n_rows, dtype=bool)
    unflagged[np.asarray(flagged_idx, dtype=int)] = False
    consensus = unflagged.copy()
    for flags in natural_flags.values():
        consensus &= ~np.asarray(flags, dtype=bool)
    if int(consensus.sum()) >= MIN_REFERENCE_ROWS:
        return consensus, "consensus_normal"
    if n_rows:
        lower = unflagged & (np.asarray(composite_score) <= float(np.median(composite_score)))
        if int(lower.sum()) >= MIN_REFERENCE_ROWS:
            return lower, "lower_half"
    return None, None


def robust_scale(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension (centre, scale) of *reference*: median, and IQR/1.349 -> MAD*1.4826 -> std -> 1."""
    centre = np.median(reference, axis=0)
    q75, q25 = np.percentile(reference, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    mad = np.median(np.abs(reference - centre), axis=0) * 1.4826
    scale = np.where(scale > 0, scale, mad)
    scale = np.where(scale > 0, scale, reference.std(axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    return centre, scale


def _background(reference: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """At most *k* k-means centres of *reference* and the share of rows each one stands for (none for k=0)."""
    if k <= 0:
        return np.empty((0, reference.shape[1]), dtype=np.float64), np.empty(0, dtype=np.float64)
    if reference.shape[0] <= k:
        n = reference.shape[0]
        return reference.astype(np.float64, copy=True), np.full(n, 1.0 / n)
    sample = reference
    if reference.shape[0] > _BACKGROUND_SAMPLE_CAP:
        picked = np.random.default_rng(seed).choice(
            reference.shape[0], size=_BACKGROUND_SAMPLE_CAP, replace=False
        )
        sample = reference[np.sort(picked)]
    from sklearn.cluster import KMeans  # noqa: PLC0415

    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(sample)
    counts = np.bincount(km.labels_, minlength=k).astype(np.float64)
    keep = counts > 0
    return np.asarray(km.cluster_centers_[keep], dtype=np.float64), counts[keep] / counts[keep].sum()


@dataclass
class ReferenceSummary:
    """What explanations need to know about the reference population (see the module docstring)."""

    reference_id: str  # consensus_normal | lower_half
    n_rows: int
    feature_names: list[str]
    median: np.ndarray
    scale: np.ndarray
    std: np.ndarray
    abs_mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    background: np.ndarray  # (k, n_features)
    background_weights: np.ndarray  # (k,), sums to 1
    raw_values: dict[str, tuple[Any, str]] = field(default_factory=dict)
    source_run_id: str | None = None  # set when loaded for another run (score-forward)

    @classmethod
    def from_rows(
        cls,
        reference: np.ndarray,
        *,
        reference_id: str,
        feature_names: list[str],
        raw_values: dict[str, tuple[Any, str]] | None = None,
        background_k: int = BACKGROUND_K,
        seed: int = 0,
    ) -> ReferenceSummary:
        """Summarise the reference rows (shape ``(n, n_features)``, n >= 1)."""
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[0] == 0:
            msg = "ReferenceSummary.from_rows needs a non-empty 2-D matrix."
            raise ValueError(msg)
        centre, scale = robust_scale(reference)
        background, weights = _background(reference, background_k, seed)
        return cls(
            reference_id=reference_id,
            n_rows=int(reference.shape[0]),
            feature_names=list(feature_names),
            median=centre,
            scale=scale,
            std=reference.std(axis=0),
            abs_mean=np.abs(reference).mean(axis=0),
            minimum=reference.min(axis=0),
            maximum=reference.max(axis=0),
            background=background,
            background_weights=weights,
            raw_values=dict(raw_values or {}),
        )

    @property
    def n_features(self) -> int:
        """Width of the feature space the summary describes."""
        return int(self.median.shape[0])

    @property
    def label(self) -> str:
        """``reference_id`` as recorded: suffixed with the fitting run when it came from one."""
        return f"{self.reference_id}:{self.source_run_id}" if self.source_run_id else self.reference_id

    def to_json(self) -> str:
        """Serialise to JSON (arrays as lists; raw values with a type tag so they read back comparable)."""
        payload = {
            "summary_version": SUMMARY_VERSION,
            "reference_id": self.reference_id,
            "n_rows": self.n_rows,
            "feature_names": self.feature_names,
            **{
                name: getattr(self, name).tolist()
                for name in ("median", "scale", "std", "abs_mean", "minimum", "maximum", "background")
            },
            "background_weights": self.background_weights.tolist(),
            "raw_values": {k: [*_encode_value(v), kind] for k, (v, kind) in self.raw_values.items()},
        }
        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_json(cls, text: str, *, source_run_id: str | None = None) -> ReferenceSummary:
        """Inverse of :meth:`to_json`. Raises ``ValueError`` on an unknown version or a malformed file."""
        try:
            d = json.loads(text)
            if d.get("summary_version") != SUMMARY_VERSION:
                msg = f"unsupported reference summary version {d.get('summary_version')!r}"
                raise ValueError(msg)
            arrays = {
                name: np.asarray(d[name], dtype=np.float64)
                for name in ("median", "scale", "std", "abs_mean", "minimum", "maximum", "background_weights")
            }
            background = np.asarray(d["background"], dtype=np.float64).reshape(-1, len(d["feature_names"]))
            summary = cls(
                reference_id=str(d["reference_id"]),
                n_rows=int(d["n_rows"]),
                feature_names=[str(n) for n in d["feature_names"]],
                background=background,
                raw_values={
                    k: (_decode_value(tag, v), kind) for k, (tag, v, kind) in d["raw_values"].items()
                },
                source_run_id=source_run_id,
                **arrays,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            msg = f"malformed reference summary: {exc}"
            raise ValueError(msg) from exc
        widths = {
            a.shape[0] for a in (summary.median, summary.scale, summary.std, summary.minimum, summary.maximum)
        }
        if widths != {len(summary.feature_names)} or summary.background.shape[0] != len(
            summary.background_weights
        ):
            msg = "malformed reference summary: inconsistent array shapes"
            raise ValueError(msg)
        return summary


def _encode_value(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float):
        return ("float", value) if math.isfinite(value) else ("str", str(value))
    if isinstance(value, dt.datetime):
        return "datetime", value.isoformat()
    if isinstance(value, dt.date):
        return "date", value.isoformat()
    if isinstance(value, dt.time):
        return "time", value.isoformat()
    if isinstance(value, dt.timedelta):
        return "timedelta", value.total_seconds()
    return "str", str(value)


def _decode_value(tag: str, value: Any) -> Any:
    decoders: dict[str, Any] = {
        "bool": bool,
        "int": int,
        "float": float,
        "datetime": dt.datetime.fromisoformat,
        "date": dt.date.fromisoformat,
        "time": dt.time.fromisoformat,
        "timedelta": lambda s: dt.timedelta(seconds=s),
        "str": str,
    }
    if tag not in decoders:
        msg = f"unknown value tag {tag!r}"
        raise TypeError(msg)
    return decoders[tag](value)


def as_summary(reference: np.ndarray | ReferenceSummary) -> ReferenceSummary:
    """Accept either a reference matrix or a prepared summary (the explainers take both).

    A matrix is summarised without a background (``background`` has no rows):
    the callers that need one (KernelSHAP, TreeSHAP) build it from the matrix
    themselves, and computing k-means centres nobody reads would be waste.
    """
    if isinstance(reference, ReferenceSummary):
        return reference
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] == 0:
        msg = "reference must be a non-empty 2-D matrix (n_rows, n_features)."
        raise ValueError(msg)
    names = [f"f{i}" for i in range(reference.shape[1])]
    return ReferenceSummary.from_rows(reference, reference_id="supplied", feature_names=names, background_k=0)
