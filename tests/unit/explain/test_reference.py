"""The reference population: which rows qualify, the summary, and its JSON round trip."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from sorethumb_ml.detectors.kmeans_distance import KMeansDetector
from sorethumb_ml.explain import reference as ref_mod
from sorethumb_ml.explain.centroid import centroid_attributions
from sorethumb_ml.explain.gradient import (
    gradient_attributions,
    kernel_shap_attributions,
    marginal_deviation_attributions,
)
from sorethumb_ml.explain.reference import (
    MIN_REFERENCE_ROWS,
    ReferenceSummary,
    as_summary,
    select_reference_rows,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# select_reference_rows
# ---------------------------------------------------------------------------


def test_consensus_normal_excludes_rows_any_detector_flags() -> None:
    n = 40
    flagged = np.array([0, 1])
    a = np.zeros(n, dtype=bool)
    a[2:5] = True  # detector a's own boundary flags rows the ensemble did not
    b = np.zeros(n, dtype=bool)
    b[5] = True
    mask, rid = select_reference_rows(
        n_rows=n, flagged_idx=flagged, natural_flags={"a": a, "b": b}, composite_score=np.zeros(n)
    )
    assert rid == "consensus_normal"
    assert mask is not None
    assert mask.sum() == n - 6
    assert not mask[:6].any()


def test_falls_back_to_the_less_anomalous_half_of_unflagged_rows() -> None:
    n = 30
    score = np.arange(n, dtype=float)  # higher = more anomalous
    flagged = np.array([29])
    everyone = np.ones(n, dtype=bool)  # some detector flags every row
    mask, rid = select_reference_rows(
        n_rows=n, flagged_idx=flagged, natural_flags={"loud": everyone}, composite_score=score
    )
    assert rid == "lower_half"
    assert mask is not None
    assert mask.sum() == 15  # scores 0..14 are at or below the median 14.5
    assert not mask[29]
    assert mask[:15].all()


def test_no_reference_when_neither_population_is_large_enough() -> None:
    n = 2 * MIN_REFERENCE_ROWS - 2
    mask, rid = select_reference_rows(
        n_rows=n,
        flagged_idx=np.array([0]),
        natural_flags={"loud": np.ones(n, dtype=bool)},
        composite_score=np.arange(n, dtype=float),
    )
    assert (mask, rid) == (None, None)
    empty, none_id = select_reference_rows(
        n_rows=0, flagged_idx=np.array([], dtype=int), natural_flags={}, composite_score=np.array([])
    )
    assert (empty, none_id) == (None, None)


def test_the_reference_never_contains_a_flagged_row() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(20, 60))
        flagged = rng.choice(n, size=int(rng.integers(1, n // 2)), replace=False)
        flags = {"d": rng.random(n) < rng.random()}
        mask, _ = select_reference_rows(
            n_rows=n, flagged_idx=flagged, natural_flags=flags, composite_score=rng.random(n)
        )
        if mask is not None:
            assert not mask[flagged].any()


# ---------------------------------------------------------------------------
# ReferenceSummary
# ---------------------------------------------------------------------------


def _summary(n: int = 300, d: int = 3, k: int = 50) -> ReferenceSummary:
    rows = np.random.default_rng(0).normal(size=(n, d))
    return ReferenceSummary.from_rows(
        rows,
        reference_id="consensus_normal",
        feature_names=[f"x{i}" for i in range(d)],
        raw_values={
            "amount": (12.5, "ordered"),
            "n": (3, "ordered"),
            "day": (dt.date(2026, 1, 2), "ordered"),
            "ts": (dt.datetime(2026, 1, 2, 3, 4, 5), "ordered"),
            "t": (dt.time(1, 2), "ordered"),
            "gap": (dt.timedelta(hours=2), "ordered"),
            "flag": (True, "categorical"),
            "cat": ("A", "categorical"),
            "odd": (float("inf"), "ordered"),
        },
        background_k=k,
    )


def test_summary_statistics_and_background() -> None:
    s = _summary()
    assert s.n_rows == 300
    assert s.n_features == 3
    assert s.background.shape == (50, 3)
    assert s.background_weights.sum() == pytest.approx(1.0)
    assert (s.scale > 0).all()
    assert (s.minimum <= s.median).all()
    assert (s.median <= s.maximum).all()
    assert s.label == "consensus_normal"


def test_a_small_reference_is_its_own_background() -> None:
    rows = np.arange(12, dtype=float).reshape(4, 3)
    s = ReferenceSummary.from_rows(rows, reference_id="lower_half", feature_names=["a", "b", "c"])
    np.testing.assert_array_equal(s.background, rows)
    np.testing.assert_allclose(s.background_weights, 0.25)


def test_a_large_reference_is_subsampled_for_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ref_mod, "_BACKGROUND_SAMPLE_CAP", 100)
    s = _summary(n=400, k=5)
    assert s.background.shape[0] <= 5
    assert s.n_rows == 400


def test_summary_round_trips_through_json() -> None:
    s = _summary()
    back = ReferenceSummary.from_json(s.to_json(), source_run_id="run_abc")
    for name in (
        "median",
        "scale",
        "std",
        "abs_mean",
        "minimum",
        "maximum",
        "background",
        "background_weights",
    ):
        np.testing.assert_allclose(getattr(back, name), getattr(s, name))
    assert back.feature_names == s.feature_names
    assert back.label == "consensus_normal:run_abc"
    for key, value in s.raw_values.items():
        if key == "odd":
            assert back.raw_values[key] == ("inf", "ordered")  # non-finite floats are kept as text
        else:
            assert back.raw_values[key] == value
            assert type(back.raw_values[key][0]) is type(value[0])


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"summary_version": 99}',
        '{"summary_version": 1}',
    ],
)
def test_malformed_summaries_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="reference summary"):
        ReferenceSummary.from_json(text)


def test_inconsistent_shapes_and_unknown_tags_are_rejected() -> None:
    import json

    d = json.loads(_summary().to_json())
    d["median"] = d["median"][:-1]
    with pytest.raises(ValueError, match="inconsistent"):
        ReferenceSummary.from_json(json.dumps(d))
    d = json.loads(_summary().to_json())
    d["raw_values"]["cat"][0] = "mystery"
    with pytest.raises(ValueError, match="malformed"):
        ReferenceSummary.from_json(json.dumps(d))


def test_from_rows_and_as_summary_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ReferenceSummary.from_rows(np.empty((0, 2)), reference_id="x", feature_names=["a", "b"])
    with pytest.raises(ValueError, match="non-empty"):
        as_summary(np.empty((0, 2)))


def test_as_summary_passes_a_summary_through_and_skips_the_background_for_a_matrix() -> None:
    s = _summary()
    assert as_summary(s) is s
    m = as_summary(np.ones((5, 2)))
    assert m.background.shape == (0, 2)


# ---------------------------------------------------------------------------
# Explainers give the same answer from a matrix or its summary
# ---------------------------------------------------------------------------


class _Linear:
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -(X @ np.array([1.0, 2.0, 0.0]))


def test_gradient_and_marginal_agree_for_matrix_and_summary() -> None:
    rows = np.random.default_rng(1).normal(size=(200, 3))
    s = ReferenceSummary.from_rows(rows, reference_id="consensus_normal", feature_names=["a", "b", "c"])
    targets = np.array([[0.5, 0.5, 0.5], [9.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        gradient_attributions(_Linear(), targets, reference=rows)[0],
        gradient_attributions(_Linear(), targets, reference=s)[0],
    )
    np.testing.assert_allclose(
        marginal_deviation_attributions(targets, reference=rows)[0],
        marginal_deviation_attributions(targets, reference=s)[0],
    )


def test_centroid_distances_are_scaled_by_the_reference_spread() -> None:
    rng = np.random.default_rng(2)
    rows = np.column_stack([rng.normal(0, 10, 300), rng.normal(0, 0.1, 300)])
    det = KMeansDetector()
    det.fit(rows, seed=0)
    target = np.array([[5.0, 0.5]])
    unscaled, _ = centroid_attributions(det, target)
    scaled, _ = centroid_attributions(det, target, reference=rows)
    # The second dimension is far further out relative to how little normal rows vary in it.
    assert unscaled[0, 0] > unscaled[0, 1]
    assert scaled[0, 1] > scaled[0, 0]


def test_kernel_shap_uses_the_summary_background(monkeypatch: pytest.MonkeyPatch) -> None:
    shap = pytest.importorskip("shap")
    s = _summary(n=60, k=4)
    seen: dict[str, object] = {}
    real = shap.KernelExplainer

    def _spy(f, data, *a, **kw):
        seen["data"] = data
        return real(f, data, *a, **kw)

    monkeypatch.setattr(shap, "KernelExplainer", _spy)
    attrs, _ = kernel_shap_attributions(_Linear(), np.array([[3.0, 0.0, 0.0]]), reference=s)
    assert attrs.shape == (1, 3)
    background = seen["data"]
    np.testing.assert_allclose(background.data, s.background)  # type: ignore[attr-defined]
    np.testing.assert_allclose(background.weights, s.background_weights)  # type: ignore[attr-defined]


def test_kernel_shap_rejects_a_summary_without_background() -> None:
    pytest.importorskip("shap")
    from sorethumb_ml.errors import ExplainError

    with pytest.raises(ExplainError, match="no background"):
        kernel_shap_attributions(_Linear(), np.zeros((1, 3)), reference=as_summary(np.ones((4, 3))))


# ---------------------------------------------------------------------------
# Interventional TreeSHAP
# ---------------------------------------------------------------------------


def _forest() -> tuple[object, np.ndarray, np.ndarray]:
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

    rng = np.random.default_rng(3)
    normal = rng.normal(size=(300, 3))
    target = np.array([[0.0, 12.0, 0.0]])
    det = IsolationForestDetector(n_estimators=30)
    det.fit(np.vstack([normal, target]), seed=0)
    return det, normal, target


def test_tree_shap_is_interventional_against_the_reference_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shap = pytest.importorskip("shap")
    from sorethumb_ml.explain.shap_tree import tree_shap_attributions_detailed

    det, normal, target = _forest()
    summary = ReferenceSummary.from_rows(
        normal, reference_id="consensus_normal", feature_names=["a", "b", "c"]
    )
    seen: dict[str, object] = {}
    real = shap.TreeExplainer

    def _spy(model, **kw):
        seen.update(kw)
        return real(model, **kw)

    monkeypatch.setattr(shap, "TreeExplainer", _spy)
    attrs, tag, method, marginal, flags = tree_shap_attributions_detailed(det, target, reference=summary)  # type: ignore[arg-type]
    assert (tag, method, flags) == ("model_specific", "tree_shap", ["treeshap_path_length"])
    assert seen["feature_perturbation"] == "interventional"
    background = np.asarray(seen["data"])
    assert 50 <= len(background) <= 150  # centres repeated by weight, about 100 rows
    assert {tuple(r) for r in background} <= {tuple(r) for r in summary.background}
    assert int(np.argmax(attrs[0])) == 1  # the planted dimension
    assert not marginal.any()


def test_tree_shap_reports_a_failed_additivity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    shap = pytest.importorskip("shap")
    from sorethumb_ml.explain.shap_tree import tree_shap_attributions_detailed

    det, normal, target = _forest()
    real = shap.TreeExplainer

    class _Strict:
        def __init__(self, model, **kw):
            self._inner = real(model, **kw)

        def shap_values(self, X, check_additivity=True):
            if check_additivity:
                raise RuntimeError("Additivity check failed in TreeExplainer!")
            return self._inner.shap_values(X, check_additivity=False)

    monkeypatch.setattr(shap, "TreeExplainer", _Strict)
    _, tag, method, _, flags = tree_shap_attributions_detailed(det, target, reference=normal)  # type: ignore[arg-type]
    assert (tag, method) == ("model_specific", "tree_shap")
    assert flags == ["treeshap_path_length", "treeshap_additivity_unchecked"]


def test_tree_shap_other_errors_still_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    shap = pytest.importorskip("shap")
    from sorethumb_ml.explain.shap_tree import tree_shap_attributions_detailed

    det, normal, target = _forest()

    class _Broken:
        def __init__(self, model, **kw):
            pass

        def shap_values(self, X, check_additivity=True):
            raise RuntimeError("something else entirely")

    monkeypatch.setattr(shap, "TreeExplainer", _Broken)
    with pytest.warns(UserWarning, match="TreeSHAP failed"):
        _, tag, method, _, flags = tree_shap_attributions_detailed(det, target, reference=normal)  # type: ignore[arg-type]
    assert (tag, method, flags) == ("heuristic", "finite_difference_gradient", [])
