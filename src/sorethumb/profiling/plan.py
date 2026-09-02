"""FeaturePlan: the fully-fitted encoding specification for a dataset.

``build_feature_plan`` profiles a ``pl.DataFrame``, classifies every column,
collects the encoding artefacts (categories, frequency maps, imputation
medians) needed to apply the same transformation to new data, and returns a
``FeaturePlan`` that can be serialised to JSON and restored exactly.

Fields populated in M2 (``features/build.py``) are left empty here:
  - scaler_params
  - correlation_drop_list
  - pca_components / pca_mean / pca_explained_variance_ratio
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import polars as pl

from sorethumb.config import ColumnsConfig, Config, FeaturesConfig, ProfilingConfig
from sorethumb.errors import PlanError
from sorethumb.io.fingerprint import schema_fingerprint
from sorethumb.profiling.classify import (
    ColumnClass,
    Treatment,
    classify_column,
    treatment_for,
)
from sorethumb.profiling.profile import ColumnProfile, profile_columns

_NUMERIC_BASE_TYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)
_TIME_DERIVATIVE_ALIASES: dict[str, str] = {
    "hour": "__hour",
    "dayofweek": "__dayofweek",
    "day": "__day",
    "month": "__month",
    "year": "__year",
    "quarter": "__quarter",
}


@dataclass
class ColumnDecision:
    """The profiling verdict and encoding action for one input column."""

    column: str
    col_class: ColumnClass
    reason: str
    treatment: Treatment
    emit_missing_indicator: bool


@dataclass
class FeaturePlan:
    """Everything needed to reproduce the feature transformation on new data.

    Serialise with ``to_json()``; restore with ``from_json()``.
    Every field in ``output_features`` MUST have an entry in
    ``derived_to_original``; ``__post_init__`` enforces this invariant.
    """

    schema_fingerprint: str
    n_rows: int
    decisions: list[ColumnDecision]
    output_features: list[str]
    derived_to_original: dict[str, str]
    one_hot_categories: dict[str, list[str]]
    frequency_maps: dict[str, dict[str, float]]
    imputation_medians: dict[str, float]
    chosen_time_column: str | None
    time_derivatives: list[str]

    # Populated by features/build.py (M2)
    scaler_params: dict[str, dict[str, float]] = field(default_factory=dict)
    correlation_drop_list: list[str] = field(default_factory=list)
    pca_components: list[list[float]] | None = None
    pca_mean: list[float] | None = None
    pca_explained_variance_ratio: list[float] | None = None

    def __post_init__(self) -> None:
        """Validate that every output feature has a derived_to_original entry."""
        missing = [f for f in self.output_features if f not in self.derived_to_original]
        if missing:
            raise PlanError(
                f"BUG: {len(missing)} output feature(s) have no derived_to_original entry: "
                f"{missing[:5]!r}{'...' if len(missing) > 5 else ''}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to a JSON string. Round-trips losslessly via ``from_json``."""
        d: dict[str, object] = {
            "schema_fingerprint": self.schema_fingerprint,
            "n_rows": self.n_rows,
            "decisions": [
                {
                    "column": dec.column,
                    "col_class": dec.col_class.value,
                    "reason": dec.reason,
                    "treatment": dec.treatment.value,
                    "emit_missing_indicator": dec.emit_missing_indicator,
                }
                for dec in self.decisions
            ],
            "output_features": self.output_features,
            "derived_to_original": self.derived_to_original,
            "one_hot_categories": self.one_hot_categories,
            "frequency_maps": self.frequency_maps,
            "imputation_medians": self.imputation_medians,
            "chosen_time_column": self.chosen_time_column,
            "time_derivatives": self.time_derivatives,
            "scaler_params": self.scaler_params,
            "correlation_drop_list": self.correlation_drop_list,
            "pca_components": self.pca_components,
            "pca_mean": self.pca_mean,
            "pca_explained_variance_ratio": self.pca_explained_variance_ratio,
        }
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> FeaturePlan:
        """Restore a ``FeaturePlan`` from a JSON string produced by ``to_json``."""
        d = json.loads(s)
        decisions = [
            ColumnDecision(
                column=dec["column"],
                col_class=ColumnClass(dec["col_class"]),
                reason=dec["reason"],
                treatment=Treatment(dec["treatment"]),
                emit_missing_indicator=dec["emit_missing_indicator"],
            )
            for dec in d["decisions"]
        ]
        return cls(
            schema_fingerprint=d["schema_fingerprint"],
            n_rows=d["n_rows"],
            decisions=decisions,
            output_features=d["output_features"],
            derived_to_original=d["derived_to_original"],
            one_hot_categories=d["one_hot_categories"],
            frequency_maps={
                col: {k: float(v) for k, v in fmap.items()} for col, fmap in d["frequency_maps"].items()
            },
            imputation_medians={k: float(v) for k, v in d["imputation_medians"].items()},
            chosen_time_column=d["chosen_time_column"],
            time_derivatives=d["time_derivatives"],
            scaler_params=d.get("scaler_params", {}),
            correlation_drop_list=d.get("correlation_drop_list", []),
            pca_components=d.get("pca_components"),
            pca_mean=d.get("pca_mean"),
            pca_explained_variance_ratio=d.get("pca_explained_variance_ratio"),
        )


# ------------------------------------------------------------------
# Public builder
# ------------------------------------------------------------------


def build_feature_plan(df: pl.DataFrame, config: Config) -> FeaturePlan:
    """Profile *df*, classify columns, collect encoding artefacts, return a ``FeaturePlan``.

    The returned plan is fully fitted: applying it to any frame with the same
    schema will produce an identical feature space (before scaling / PCA).
    """
    protected = _build_protected(config.columns)

    profiles = profile_columns(df, config.profiling)

    # ---- classify ----
    classified: list[tuple[ColumnProfile, ColumnClass, str]] = []
    for p in profiles:
        col_class, reason = classify_column(p, config.profiling, config.columns, protected)
        classified.append((p, col_class, reason))

    # ---- pick chosen temporal column ----
    chosen_time = _pick_time_column(classified, config.columns)

    # ---- collect encoding artefacts ----
    one_hot_cats: dict[str, list[str]] = {}
    freq_maps: dict[str, dict[str, float]] = {}
    medians: dict[str, float] = {}

    _collect_artefacts(df, classified, chosen_time, config, one_hot_cats, freq_maps, medians)

    # ---- build decisions and output feature list ----
    decisions: list[ColumnDecision] = []
    output_features: list[str] = []
    d2o: dict[str, str] = {}

    for p, col_class, reason in classified:
        treatment = treatment_for(col_class, p, config.features)

        # Non-chosen temporal columns are dropped
        if col_class == ColumnClass.temporal and p.name != chosen_time:
            treatment = Treatment.drop

        emit_indicator = _should_emit_indicator(p, config.profiling, config.features)

        decisions.append(
            ColumnDecision(
                column=p.name,
                col_class=col_class,
                reason=reason,
                treatment=treatment,
                emit_missing_indicator=emit_indicator,
            )
        )

        col_feats = _features_for_column(
            p.name,
            treatment,
            emit_indicator,
            one_hot_cats,
            df.schema,
            config.features,
        )
        for feat in col_feats:
            output_features.append(feat)
            d2o[feat] = p.name

    fp = schema_fingerprint(df)

    return FeaturePlan(
        schema_fingerprint=fp,
        n_rows=len(df),
        decisions=decisions,
        output_features=output_features,
        derived_to_original=d2o,
        one_hot_categories=one_hot_cats,
        frequency_maps=freq_maps,
        imputation_medians=medians,
        chosen_time_column=chosen_time,
        time_derivatives=list(config.features.time_derivatives),
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_protected(columns_config: ColumnsConfig) -> set[str]:
    protected: set[str] = set()
    if columns_config.time_column:
        protected.add(columns_config.time_column)
    protected.update(columns_config.group_by)
    if columns_config.id_column:
        protected.add(columns_config.id_column)
    if columns_config.reference_column:
        protected.add(columns_config.reference_column)
    return protected


def _pick_time_column(
    classified: list[tuple[ColumnProfile, ColumnClass, str]],
    columns_config: ColumnsConfig,
) -> str | None:
    if columns_config.time_column:
        return columns_config.time_column

    temporal = [p for p, col_class, _ in classified if col_class == ColumnClass.temporal]
    if not temporal:
        return None
    return max(temporal, key=lambda p: p.n_unique).name


def _collect_artefacts(
    df: pl.DataFrame,
    classified: list[tuple[ColumnProfile, ColumnClass, str]],
    chosen_time: str | None,
    config: Config,
    one_hot_cats: dict[str, list[str]],
    freq_maps: dict[str, dict[str, float]],
    medians: dict[str, float],
) -> None:
    median_cols: list[str] = []
    one_hot_cols: list[str] = []
    freq_cols: list[str] = []

    for p, col_class, _ in classified:
        treatment = treatment_for(col_class, p, config.features)
        if col_class == ColumnClass.temporal and p.name != chosen_time:
            continue

        if treatment == Treatment.impute_median:
            median_cols.append(p.name)
        elif treatment == Treatment.one_hot:
            one_hot_cols.append(p.name)
        elif treatment == Treatment.frequency:
            freq_cols.append(p.name)

    if median_cols:
        med_row = df.select([pl.col(c).median().alias(c) for c in median_cols]).row(0, named=True)
        for col in median_cols:
            v = med_row[col]
            medians[col] = float(v) if v is not None else 0.0

    for col in one_hot_cols:
        cats = df[col].drop_nulls().unique().sort().to_list()
        one_hot_cats[col] = [str(c) for c in cats]

    for col in freq_cols:
        vc = df[col].value_counts(normalize=True, sort=True)
        freq_maps[col] = {str(row[col]): float(row["proportion"]) for row in vc.iter_rows(named=True)}


def _should_emit_indicator(
    profile: ColumnProfile,
    profiling_config: ProfilingConfig,
    features_config: FeaturesConfig,
) -> bool:
    if not features_config.missing_indicators:
        return False
    # All-null columns: skip — the indicator would be a useless constant-1 feature
    if profile.is_empty:
        return False
    return profile.null_ratio > profiling_config.null_ratio_flag


def _features_for_column(
    col: str,
    treatment: Treatment,
    emit_indicator: bool,
    one_hot_cats: dict[str, list[str]],
    schema: pl.Schema,
    features_config: FeaturesConfig,
) -> list[str]:
    """Return the ordered list of output feature names this column produces."""
    feats: list[str] = []

    if treatment == Treatment.indicator_only:
        # The indicator IS the only output for high_null columns
        if emit_indicator:
            feats.append(f"{col}__is_missing")
        return feats

    # For dropped columns: the indicator is the only output (if any)
    if treatment == Treatment.drop:
        if emit_indicator:
            feats.append(f"{col}__is_missing")
        return feats

    if treatment == Treatment.one_hot:
        for cat in one_hot_cats.get(col, []):
            feats.append(f"{col}__{cat}")
        feats.append(f"{col}____other")

    elif treatment == Treatment.derive_time:
        for deriv in features_config.time_derivatives:
            suffix = _TIME_DERIVATIVE_ALIASES.get(deriv, f"__{deriv}")
            feats.append(f"{col}{suffix}")

    elif treatment == Treatment.derive_array:
        feats.extend(
            [
                f"{col}__len",
                f"{col}__is_null",
                f"{col}__is_empty",
            ]
        )
        dtype = schema.get(col)
        if dtype is not None and isinstance(dtype, pl.List):
            inner = dtype.inner
            if isinstance(inner, _NUMERIC_BASE_TYPES):
                feats.extend(
                    [
                        f"{col}__mean",
                        f"{col}__min",
                        f"{col}__max",
                    ]
                )

    else:
        # cast_int, impute_median, passthrough, frequency
        feats.append(col)

    if emit_indicator and treatment != Treatment.indicator_only:
        feats.append(f"{col}__is_missing")

    return feats
