"""The explanation record: what produced each reason, in long format.

``reason_<k>`` is a label (``amount=900``). Behind each one this module builds
a record that says how it was produced, so a reader can judge it:

- ``original_field``, ``raw_value`` -- the input column and the row's value.
- ``reference_value`` and ``direction`` -- the reference population's typical
  value for that field (median for numbers and dates, most common value
  otherwise) and where the row sits against it (``above``, ``below``,
  ``same``, ``different``, ``missing``).
- ``signed_contribution`` -- the blended attribution summed over the field's
  transformed features: positive pushes the row toward anomaly, negative
  toward normal. ``contribution_magnitude`` is the sum of absolute values, the
  quantity reasons are ranked by (opposite-signed dummies of one field cancel
  in the signed sum but not in the magnitude).
- ``methods`` -- each attribution source's share: detector, method, kind, its
  weight in this row's blend and its signed contribution, in the units it
  enters the blend. ``signed_contribution`` is the weight-sum of these.
- ``transformed_features`` -- the derived features behind the field (one-hot
  dummies, date parts, scaled numerics) with their signed contribution, the
  ``_TOP_TRANSFORMED`` largest by absolute value.
- ``approximation_flags`` -- every caveat that applies to this row's
  attribution (see :data:`APPROXIMATION_FLAGS`).
- ``explainer_version`` and ``reference_id`` -- which explanation logic and
  which reference population produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from sorethumb_ml.explain.blend import _informative_rows, blend_components
from sorethumb_ml.explain.project import aggregate_signed_to_original, aggregate_to_original

# Bump when a change alters what an explanation means (method, reference,
# aggregation), so stored explanations from different logic are never compared
# as if they were alike.
EXPLAINER_VERSION = "explain@1"

# Derived features listed per reason.
_TOP_TRANSFORMED = 5

# Every flag a record can carry. ``<det>`` is a detector name.
APPROXIMATION_FLAGS: dict[str, str] = {
    "treeshap_additivity_unchecked": (
        "TreeSHAP ran with check_additivity=False and decomposes path length, not the score itself."
    ),
    "treeshap_path_dependent": (
        "TreeSHAP used the trees' own training-path weights as its background, not the reference population."
    ),
    "kernel_shap_sampled": "KernelSHAP estimated the values from sampled coalitions against a k-means background.",
    "unsigned_method:<det>": (
        "This detector's method (centroid distance) reports how far each dimension is from the "
        "nearest centroid, never which way the dimension pushes the score; its share is non-negative."
    ),
    "marginal_deviation_fallback": (
        "The score is flat here, so the row was attributed by its reference-scaled distance "
        "from the median per dimension; unsigned, and blind to the detector."
    ),
    "fallback_used:<det>": "This detector's preferred method failed or was unavailable; a heuristic ran instead.",
    "not_covered_by:<det>": "Beyond this detector's explain.max_rows; it contributes nothing to this row.",
    "no_signal_from:<det>": "This detector's attribution for the row was zero; it contributes nothing to it.",
    "not_explained:<det>": "This detector's share of the score has no explanation (no method, or it failed).",
    "pca_back_projected": (
        "Attributions were mapped from PCA components back to features through |loadings|; "
        "a loading's sign is not carried back."
    ),
    "reference_includes_flagged_rows": (
        "Too few unflagged rows, so the reference population includes the flagged rows."
    ),
}


@dataclass
class SourceAttribution:
    """One detector's attributions for a group's flagged rows."""

    detector: str
    method: str  # tree_shap | centroid_distance | ecod_decomposition | ...
    kind: str  # exact | model_specific | heuristic
    weight: float  # ensemble weight
    values: np.ndarray  # (n_flagged, n_features), signed, zero where not covered
    covered: np.ndarray  # (n_flagged,) bool
    marginal_rows: np.ndarray  # (n_flagged,) bool
    flags: tuple[str, ...] = ()  # apply to every row this source covers
    informative: np.ndarray = field(init=False)  # (n_flagged,) bool: covered and a non-negligible vector

    def __post_init__(self) -> None:
        """Mark the rows this source actually says something about (see ``blend._informative_rows``)."""
        self.informative = self.covered & _informative_rows(np.linalg.norm(self.values, axis=1))


@dataclass
class GroupAttribution:
    """Everything the explanation stage computed for one group, before labelling."""

    kind: str  # blended attribution kind
    sources: list[SourceAttribution]
    feature_names: list[str]  # after any PCA back-projection
    blended: np.ndarray  # (n_flagged, len(feature_names)): blend output, back-projected
    scaled: list[np.ndarray]  # per source, in feature_names space, as blended
    row_weights: np.ndarray  # (n_flagged, n_sources)
    unexplained_detectors: list[str]
    reference_id: str
    group_flags: list[str] = field(default_factory=list)


def decompose(sources: list[SourceAttribution]) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-source blend terms and per-row weights (see ``blend_components``)."""
    return blend_components([s.values for s in sources], [s.weight for s in sources])


def row_gaps(attr: GroupAttribution, row: int) -> list[str]:
    """Detectors whose share of the score this row's explanation leaves out, as flags."""
    gaps = [f"not_explained:{d}" for d in sorted(attr.unexplained_detectors)]
    for s_idx, src in enumerate(attr.sources):
        if src.weight <= 0.0:
            continue
        if not src.covered[row]:
            gaps.append(f"not_covered_by:{src.detector}")
        elif attr.row_weights[row, s_idx] == 0.0 or not src.informative[row]:
            gaps.append(f"no_signal_from:{src.detector}")
    return gaps


def has_signal(attr: GroupAttribution, row: int) -> bool:
    """Return whether at least one source says something about this row and carries weight in its blend."""
    return any(
        src.informative[row] and attr.row_weights[row, s_idx] > 0.0 for s_idx, src in enumerate(attr.sources)
    )


def row_flags(attr: GroupAttribution, row: int) -> list[str]:
    """All approximation flags for one row, sorted."""
    flags: set[str] = set(attr.group_flags)
    for s_idx, src in enumerate(attr.sources):
        if not src.informative[row] or attr.row_weights[row, s_idx] == 0.0:
            continue
        flags.update(src.flags)
        if src.marginal_rows[row]:
            flags.add("marginal_deviation_fallback")
    flags.update(row_gaps(attr, row))
    return sorted(flags)


def _format_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return str(value)


def reference_values(df_reference: pl.DataFrame, fields: list[str]) -> dict[str, tuple[Any, str]]:
    """Typical value of each field in the reference rows: ``{field: (value, "ordered" | "categorical")}``.

    Numbers and dates use the median (``ordered``: the row can sit above or
    below it); anything else the most common non-null value, ties broken by
    sort order (``categorical``). A field absent from the frame, entirely null,
    or of a type that supports neither is omitted.
    """
    out: dict[str, tuple[Any, str]] = {}
    for name in fields:
        if name not in df_reference.columns:
            continue
        series = df_reference[name].drop_nulls()
        if series.dtype.is_float():
            series = series.filter(series.is_not_nan())
        if series.is_empty():
            continue
        try:
            if series.dtype.is_numeric() or series.dtype.is_temporal():
                value = series.median()
                if value is not None:
                    out[name] = (value, "ordered")
                    continue
            mode = series.mode().sort()
            out[name] = (mode[0], "categorical")
        except (pl.exceptions.PolarsError, TypeError):  # e.g. a list or struct column
            continue
    return out


def direction(raw: Any, reference: tuple[Any, str] | None) -> str | None:
    """Where *raw* sits against the reference value: above, below, same, different or missing."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "missing"
    if reference is None:
        return None
    ref_value, ref_kind = reference
    if ref_kind == "ordered":
        try:
            if raw > ref_value:
                return "above"
            if raw < ref_value:
                return "below"
        except TypeError:
            return None
        return "same"
    return "same" if raw == ref_value else "different"


def build(
    attr: GroupAttribution,
    *,
    rows: list[int],
    derived_to_original: dict[str, str],
    raw_rows: list[dict[str, Any]],
    row_ids: list[Any],
    reference: dict[str, tuple[Any, str]],
    top_n: int,
) -> tuple[dict[int, list[tuple[str, Any]]], list[dict[str, Any]]]:
    """Rank the reasons of each row in *rows* and build its records.

    *rows* are positions within the flagged set; *raw_rows* and *row_ids* are
    aligned with the flagged set. Returns ``(reasons, records)``: per row, its
    top-*top_n* ``(field, raw value)`` pairs, ranked exactly as
    ``top_n_reasons`` ranks them; and the flat record list (one dict per row
    and reason).
    """
    blended = attr.blended
    magnitude = aggregate_to_original(blended, attr.feature_names, derived_to_original)
    signed = aggregate_signed_to_original(blended, attr.feature_names, derived_to_original)
    per_source = [
        aggregate_signed_to_original(m, attr.feature_names, derived_to_original) for m in attr.scaled
    ]
    feature_index: dict[str, list[int]] = {}
    for f_idx, name in enumerate(attr.feature_names):
        feature_index.setdefault(derived_to_original.get(name, name), []).append(f_idx)

    reasons: dict[int, list[tuple[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for row in rows:
        raw_row = raw_rows[row]
        ranked = sorted(magnitude, key=lambda col: -float(magnitude[col][row]))[:top_n]
        reasons[row] = [(col, raw_row.get(col)) for col in ranked]
        flags = row_flags(attr, row)
        for rank, col in enumerate(ranked, start=1):
            raw = raw_row.get(col)
            feats = sorted(feature_index[col], key=lambda f: -abs(float(blended[row, f])))
            records.append(
                {
                    "row_id": row_ids[row],
                    "reason_rank": rank,
                    "original_field": col,
                    "raw_value": _format_value(raw),
                    "reference_value": _format_value(reference[col][0]) if col in reference else None,
                    "direction": direction(raw, reference.get(col)),
                    "signed_contribution": float(signed[col][row]),
                    "contribution_magnitude": float(magnitude[col][row]),
                    "attribution_kind": attr.kind,
                    "methods": [
                        {
                            "detector": src.detector,
                            "method": src.method,
                            "attribution_kind": src.kind,
                            "weight": float(attr.row_weights[row, s_idx]),
                            "signed_contribution": float(per_source[s_idx][col][row]),
                        }
                        for s_idx, src in enumerate(attr.sources)
                        if attr.row_weights[row, s_idx] > 0.0 and src.informative[row]
                    ],
                    "transformed_features": [
                        {"feature": attr.feature_names[f], "signed_contribution": float(blended[row, f])}
                        for f in feats[:_TOP_TRANSFORMED]
                    ],
                    "approximation_flags": flags,
                    "explainer_version": EXPLAINER_VERSION,
                    "reference_id": attr.reference_id,
                }
            )
    return reasons, records


RECORD_SCHEMA: dict[str, Any] = {
    "reason_rank": pl.Int32,
    "original_field": pl.String,
    "raw_value": pl.String,
    "reference_value": pl.String,
    "direction": pl.String,
    "signed_contribution": pl.Float64,
    "contribution_magnitude": pl.Float64,
    "attribution_kind": pl.String,
    "methods": pl.List(
        pl.Struct(
            {
                "detector": pl.String,
                "method": pl.String,
                "attribution_kind": pl.String,
                "weight": pl.Float64,
                "signed_contribution": pl.Float64,
            }
        )
    ),
    "transformed_features": pl.List(pl.Struct({"feature": pl.String, "signed_contribution": pl.Float64})),
    "approximation_flags": pl.List(pl.String),
    "explainer_version": pl.String,
    "reference_id": pl.String,
}


def records_frame(records: list[dict[str, Any]], row_id_dtype: pl.DataType | None = None) -> pl.DataFrame:
    """Return the records as a frame with a fixed schema (``row_id`` keeps the detection output's dtype)."""
    schema: dict[str, Any] = {"row_id": row_id_dtype} if row_id_dtype is not None else {}
    schema.update(RECORD_SCHEMA)
    if not records:
        return pl.DataFrame(schema={k: (v if v is not None else pl.Int64) for k, v in schema.items()})
    df = pl.DataFrame(records, schema_overrides=dict(RECORD_SCHEMA), strict=False)
    if row_id_dtype is not None:
        df = df.with_columns(pl.col("row_id").cast(row_id_dtype))
    return df.select(list(schema))
