"""The explanation record: signed per-field aggregation, per-method shares, flags, reference values."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector
from sorethumb_ml.detectors.one_class_svm import OneClassSVMDetector
from sorethumb_ml.explain import record
from sorethumb_ml.explain.blend import blend, blend_components
from sorethumb_ml.explain.gradient import gradient_attributions_detailed
from sorethumb_ml.explain.project import aggregate_signed_to_original, top_n_reasons
from sorethumb_ml.explain.shap_tree import tree_shap_attributions_detailed

pytestmark = pytest.mark.unit


def _source(name: str, values: np.ndarray, **kw: object) -> record.SourceAttribution:
    n = values.shape[0]
    return record.SourceAttribution(
        detector=name,
        method=str(kw.pop("method", "ecod_decomposition")),
        kind=str(kw.pop("kind", "exact")),
        weight=float(kw.pop("weight", 1.0)),  # type: ignore[arg-type]
        values=values,
        covered=kw.pop("covered", np.ones(n, dtype=bool)),  # type: ignore[arg-type]
        marginal_rows=kw.pop("marginal_rows", np.zeros(n, dtype=bool)),  # type: ignore[arg-type]
        flags=tuple(kw.pop("flags", ())),  # type: ignore[arg-type]
    )


def _group(
    sources: list[record.SourceAttribution], names: list[str], **kw: object
) -> record.GroupAttribution:
    blended, kind = blend([(s.values, s.kind) for s in sources], [s.weight for s in sources])
    scaled, weights = record.decompose(sources)
    return record.GroupAttribution(
        kind=kind,
        sources=sources,
        feature_names=names,
        blended=blended,
        scaled=scaled,
        row_weights=weights,
        unexplained_detectors=list(kw.pop("unexplained", [])),  # type: ignore[call-overload]
        reference_id=str(kw.pop("reference_id", "unflagged_rows")),
        group_flags=list(kw.pop("group_flags", [])),  # type: ignore[call-overload]
    )


# ---------------------------------------------------------------------------
# blend_components / signed aggregation
# ---------------------------------------------------------------------------


def test_blend_components_reassemble_the_blend() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 4))
    b = rng.normal(size=(6, 4)) * 50
    b[2] = 0.0  # uninformative for b: a alone carries row 2
    blended, _ = blend([(a, "exact"), (b, "heuristic")], [1.0, 3.0])
    scaled, weights = blend_components([a, b], [1.0, 3.0])
    rebuilt = weights[:, 0, None] * scaled[0] + weights[:, 1, None] * scaled[1]
    np.testing.assert_allclose(rebuilt, blended, rtol=1e-12, atol=1e-12)
    assert weights[2].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_blend_components_single_source_is_unscaled() -> None:
    a = np.arange(6, dtype=float).reshape(2, 3)
    scaled, weights = blend_components([a], [0.4])
    np.testing.assert_array_equal(scaled[0], a)
    assert weights.tolist() == [[1.0], [1.0]]


def test_blend_components_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="one weight per source"):
        blend_components([np.zeros((1, 1))], [])


def test_signed_aggregation_keeps_sign_and_lets_dummies_cancel() -> None:
    attrs = np.array([[2.0, -0.5, 0.5, -3.0]])
    names = ["amount", "cat__A", "cat__B", "lat"]
    d2o = {"cat__A": "cat", "cat__B": "cat"}
    signed = aggregate_signed_to_original(attrs, names, d2o)
    assert signed["amount"].tolist() == [2.0]
    assert signed["cat"].tolist() == [0.0]
    assert signed["lat"].tolist() == [-3.0]


# ---------------------------------------------------------------------------
# reference values and direction
# ---------------------------------------------------------------------------


def test_reference_values_use_median_for_ordered_and_mode_for_the_rest() -> None:
    df = pl.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, None, float("nan")],
            "cat": ["b", "a", "a", "b", None],
            "day": [dt.date(2026, 1, d) for d in (1, 2, 3, 4, 5)],
            "empty": [None, None, None, None, None],
            "nested": [[1], [2], [3], [4], [5]],
        }
    )
    ref = record.reference_values(df, ["amount", "cat", "day", "empty", "missing_field"])
    assert ref["amount"] == (2.0, "ordered")
    assert ref["cat"] == ("a", "categorical")  # tie between a and b broken by sort order
    assert ref["day"][1] == "ordered"
    assert "empty" not in ref
    assert "missing_field" not in ref


@pytest.mark.parametrize(
    ("raw", "ref", "expected"),
    [
        (5.0, (2.0, "ordered"), "above"),
        (1.0, (2.0, "ordered"), "below"),
        (2.0, (2.0, "ordered"), "same"),
        ("x", ("a", "categorical"), "different"),
        ("a", ("a", "categorical"), "same"),
        (None, (2.0, "ordered"), "missing"),
        (float("nan"), (2.0, "ordered"), "missing"),
        (5.0, None, None),
        ("text", (2.0, "ordered"), None),  # incomparable types
    ],
)
def test_direction(raw: object, ref: tuple[object, str] | None, expected: str | None) -> None:
    assert record.direction(raw, ref) == expected


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _two_source_group() -> record.GroupAttribution:
    names = ["amount", "cat__A", "cat__B", "lat"]
    a = np.array([[3.0, 0.2, -0.1, -1.0], [0.1, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    b = np.array([[-1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    b_covered = np.array([True, True, False])  # row 2 beyond b's max_rows
    marginal = np.array([True, False, False])
    sources = [
        _source("ecod", a),
        _source(
            "ocsvm",
            np.where(b_covered[:, None], b, 0.0),
            method="finite_difference_gradient",
            kind="heuristic",
            weight=3.0,
            covered=b_covered,
            marginal_rows=marginal,
        ),
    ]
    return _group(sources, names, group_flags=["reference_includes_flagged_rows"], reference_id="all_rows")


def test_build_ranks_like_top_n_reasons_and_records_the_decomposition() -> None:
    attr = _two_source_group()
    d2o = {"cat__A": "cat", "cat__B": "cat"}
    raw_rows = [
        {"amount": 900.0, "cat": "A", "lat": 1.0},
        {"amount": 50.0, "cat": "B", "lat": 80.0},
        {"amount": 51.0, "cat": "A", "lat": 2.0},
    ]
    reference = {"amount": (50.0, "ordered"), "cat": ("A", "categorical"), "lat": (2.0, "ordered")}
    reasons, records = record.build(
        attr,
        rows=[0, 1, 2],
        derived_to_original=d2o,
        raw_rows=raw_rows,
        row_ids=[10, 11, 12],
        reference=reference,
        top_n=2,
    )

    # Same ranking as the legacy ranking helper on the magnitude aggregation.
    from sorethumb_ml.explain.project import aggregate_to_original

    magnitude = aggregate_to_original(attr.blended, attr.feature_names, d2o)
    for row in range(3):
        legacy = [r["column"] for r in top_n_reasons(row, magnitude, raw_rows[row], 2)]
        assert [col for col, _ in reasons[row]] == legacy

    assert len(records) == 6
    for rec in records:
        # signed contribution is the weight-sum of the per-method shares
        total = sum(m["weight"] * m["signed_contribution"] for m in rec["methods"])
        assert rec["signed_contribution"] == pytest.approx(total)
        assert rec["contribution_magnitude"] >= abs(rec["signed_contribution"]) - 1e-12
        assert rec["explainer_version"] == record.EXPLAINER_VERSION
        assert rec["reference_id"] == "all_rows"
        assert "reference_includes_flagged_rows" in rec["approximation_flags"]

    first = records[0]
    assert (first["row_id"], first["reason_rank"], first["original_field"]) == (10, 1, "amount")
    assert first["raw_value"] == "900.0"
    assert first["reference_value"] == "50.0"
    assert first["direction"] == "above"
    assert "marginal_deviation_fallback" in first["approximation_flags"]
    assert {m["detector"] for m in first["methods"]} == {"ecod", "ocsvm"}

    # Row 1: ocsvm covered it but its vector is all zero -> no signal, not in methods.
    row1 = [r for r in records if r["row_id"] == 11]
    assert all("no_signal_from:ocsvm" in r["approximation_flags"] for r in row1)
    assert all({m["detector"] for m in r["methods"]} == {"ecod"} for r in row1)
    # Row 2: beyond ocsvm's max_rows.
    row2 = [r for r in records if r["row_id"] == 12]
    assert all("not_covered_by:ocsvm" in r["approximation_flags"] for r in row2)
    assert record.row_gaps(attr, 0) == []


def test_build_lists_transformed_features_by_absolute_contribution() -> None:
    names = [f"cat__{c}" for c in "ABCDEFG"]
    values = np.array([[0.1, -0.7, 0.3, 0.0, 0.2, -0.05, 0.6]])
    attr = _group([_source("ecod", values)], names)
    _, records = record.build(
        attr,
        rows=[0],
        derived_to_original=dict.fromkeys(names, "cat"),
        raw_rows=[{"cat": "B"}],
        row_ids=["r0"],
        reference={},
        top_n=3,
    )
    assert len(records) == 1  # one original field, however many reasons were asked for
    feats = records[0]["transformed_features"]
    assert [f["feature"] for f in feats] == ["cat__B", "cat__G", "cat__C", "cat__E", "cat__A"]
    assert records[0]["signed_contribution"] == pytest.approx(values.sum())
    assert records[0]["reference_value"] is None
    assert records[0]["direction"] is None


def test_unexplained_detectors_and_source_flags_reach_every_row() -> None:
    values = np.array([[1.0, 2.0], [0.0, 0.0]])
    attr = _group(
        [_source("iforest", values, flags=["treeshap_additivity_unchecked", "fallback_used:iforest"])],
        ["x", "y"],
        unexplained=["lof"],
    )
    assert record.has_signal(attr, 0)
    assert not record.has_signal(attr, 1)
    flags = record.row_flags(attr, 0)
    assert flags == sorted(["fallback_used:iforest", "not_explained:lof", "treeshap_additivity_unchecked"])
    # A row the source says nothing about does not inherit its method flags.
    assert "treeshap_additivity_unchecked" not in record.row_flags(attr, 1)


def test_a_row_only_a_zero_weight_source_informs_has_no_signal() -> None:
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    dropped = np.array([[5.0, 1.0], [1.0, 1.0]])
    attr = _group([_source("a", a), _source("dropped", dropped, weight=0.0)], ["x", "y"])
    assert not record.has_signal(attr, 0)
    assert record.has_signal(attr, 1)


def test_zero_weight_source_is_not_a_gap() -> None:
    values = np.array([[1.0, 2.0]])
    attr = _group([_source("a", values), _source("dropped", np.zeros((1, 2)), weight=0.0)], ["x", "y"])
    assert record.row_gaps(attr, 0) == []


def test_records_frame_schema_is_fixed_even_when_empty() -> None:
    empty = record.records_frame([], row_id_dtype=pl.Int64)
    assert empty.schema["row_id"] == pl.Int64
    assert empty.schema["methods"] == record.RECORD_SCHEMA["methods"]
    attr = _group([_source("ecod", np.array([[1.0, -2.0]]))], ["x", "y"])
    _, recs = record.build(
        attr,
        rows=[0],
        derived_to_original={},
        raw_rows=[{"x": 1, "y": None}],
        row_ids=[7],
        reference={},
        top_n=2,
    )
    df = record.records_frame(recs, row_id_dtype=pl.Int32)
    assert df.schema["row_id"] == pl.Int32
    assert df.columns == ["row_id", *record.RECORD_SCHEMA]
    assert df["raw_value"].to_list() == [None, "1"]  # y ranks first; its raw value is null
    assert df["direction"].to_list() == ["missing", None]


def test_every_flag_the_pipeline_emits_is_documented() -> None:
    documented = {k.split(":", 1)[0] for k in record.APPROXIMATION_FLAGS}
    emitted = {
        "treeshap_additivity_unchecked",
        "treeshap_path_dependent",
        "kernel_shap_sampled",
        "unsigned_method",
        "marginal_deviation_fallback",
        "fallback_used",
        "not_covered_by",
        "no_signal_from",
        "not_explained",
        "pca_back_projected",
        "reference_includes_flagged_rows",
    }
    assert emitted == documented


# ---------------------------------------------------------------------------
# explainers report method and marginal rows
# ---------------------------------------------------------------------------


def _population(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(150, 3))
    targets = np.array([[50.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    return normal, targets


def test_gradient_detailed_marks_out_of_range_rows_as_marginal() -> None:
    normal, targets = _population()
    det = OneClassSVMDetector()
    det.fit(np.vstack([normal, targets]), seed=0)
    attrs, marginal = gradient_attributions_detailed(det, targets, reference=normal, saturating=True)
    assert attrs.shape == targets.shape
    assert marginal[0]
    _, not_saturating = gradient_attributions_detailed(det, targets, reference=normal, saturating=False)
    assert not not_saturating.any()


def test_tree_shap_detailed_reports_the_fallback_method(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    normal, targets = _population()
    det = IsolationForestDetector(n_estimators=20)
    det.fit(np.vstack([normal, targets]), seed=0)
    monkeypatch.setitem(sys.modules, "shap", None)
    with pytest.warns(UserWarning, match="shap is not installed"):
        _, tag, method, marginal = tree_shap_attributions_detailed(det, targets, reference=normal)
    assert (tag, method) == ("heuristic", "finite_difference_gradient")
    assert marginal[0]
