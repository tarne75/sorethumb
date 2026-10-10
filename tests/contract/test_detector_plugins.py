"""Detector plugin contract: the ``Detector`` protocol every detector class
must satisfy, the registry every built-in detector is exported through, the
``available_extra_params()`` enumeration used by docs and `sorethumb init`,
and the per-detector contract table below (fit/score shape, seed-sensitivity,
score orientation, natural-boundary mechanism, training cap, extra-params
support, attribution kind, and a save_model/load_model round-trip). Detector-
*specific* nuance (CBLOF, elbow-k selection, ECOD/HBOS feature contributions,
LOF's small-dataset clamp, ...) lives in tests/unit/detectors/test_detectors.py;
this file is about the shape every plugin -- built-in or third-party -- must
have in common.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import app
from sorethumb_ml.detectors import registry
from sorethumb_ml.detectors._protocol import (
    check_protocol,
    score_and_flag,
    validate_flags,
    validate_scores,
)
from sorethumb_ml.detectors.ecod import ECODDetector
from sorethumb_ml.detectors.hbos import HBOSDetector
from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector
from sorethumb_ml.detectors.kmeans_distance import KMeansDetector
from sorethumb_ml.detectors.lof import LOFDetector
from sorethumb_ml.detectors.one_class_svm import OneClassSVMDetector
from sorethumb_ml.errors import DetectorError
from sorethumb_ml.store.models import load_model, plan_digest, save_model
from sorethumb_ml.store.workspace import make_group_key
from tests.factories.frames import write_grouped_csv
from tests.factories.workspaces import open_workspace

pytestmark = pytest.mark.contract

# ---------------------------------------------------------------------------
# Protocol: check_protocol
# ---------------------------------------------------------------------------


class _GoodDetector:
    name = "good"
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


class _MissingName:
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


class _MissingMethod:
    name = "bad"
    supports_tree_shap = False
    default_train_row_cap = 1000

    def fit(self, X, *, seed):
        pass

    def score_samples(self, X):
        return np.zeros(len(X))

    # natural_flag deliberately missing

    def get_params(self):
        return {}


def test_check_protocol_passes_good_detector():
    check_protocol(_GoodDetector)


def test_check_protocol_raises_missing_name():
    with pytest.raises(DetectorError, match="name"):
        check_protocol(_MissingName)


def test_check_protocol_raises_missing_method():
    with pytest.raises(DetectorError, match="natural_flag"):
        check_protocol(_MissingMethod)


class _NonCallableFit:
    name = "bad"
    supports_tree_shap = False
    default_train_row_cap = 1000
    fit = "not_a_function"  # non-callable

    def score_samples(self, X):
        return np.zeros(len(X))

    def natural_flag(self, scores):
        return scores < 0

    def get_params(self):
        return {}


def test_check_protocol_raises_non_callable_method():
    """A method *present* but overridden with a non-callable is a distinct
    failure from a missing method entirely (test_check_protocol_raises_missing_method)."""
    with pytest.raises(DetectorError, match="fit"):
        check_protocol(_NonCallableFit)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_builtins():
    assert "isolation_forest" in registry
    assert "kmeans_distance" in registry
    assert "one_class_svm" in registry


def test_registry_values_are_detector_classes():
    assert registry["isolation_forest"] is IsolationForestDetector
    assert registry["kmeans_distance"] is KMeansDetector
    assert registry["one_class_svm"] is OneClassSVMDetector


def test_registry_register_valid():
    from sorethumb_ml.detectors import register

    try:
        register(_GoodDetector)
        assert "good" in registry
    finally:
        registry.pop("good", None)


def test_registry_register_invalid_raises():
    from sorethumb_ml.detectors import register

    with pytest.raises(DetectorError):
        register(_MissingName)


def test_detectors_all_list():
    """__all__ is the plugin surface: adding a detector means exporting it here."""
    import sorethumb_ml.detectors as det

    expected = {
        "Detector",
        "ECODDetector",
        "HBOSDetector",
        "IsolationForestDetector",
        "KMeansDetector",
        "LOFDetector",
        "OneClassSVMDetector",
        "register",
        "registry",
    }
    assert set(det.__all__) == expected


# ---------------------------------------------------------------------------
# available_extra_params() — enumeration for docs / `sorethumb init`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "curated"),
    [
        (IsolationForestDetector, {"n_estimators", "max_samples"}),
        (LOFDetector, {"n_neighbors"}),
        (KMeansDetector, {"k", "k_min", "k_max", "n_init", "large_cluster_coverage"}),
        (OneClassSVMDetector, {"nu", "kernel", "gamma"}),
    ],
)
def test_available_extra_params_excludes_managed_and_curated(cls, curated):
    keys = set(cls.available_extra_params())
    assert keys, f"{cls.name} should expose at least one extra_params key"
    # Nothing sorethumb manages, and nothing already a wrapper argument.
    assert not (keys & {"random_state", "seed", "contamination", "novelty", "n_clusters"})
    assert not (keys & curated)


@pytest.mark.parametrize(
    "cls",
    [IsolationForestDetector, LOFDetector, KMeansDetector, OneClassSVMDetector],
)
def test_available_extra_params_all_accept_when_passed_back(cls):
    """Every advertised key must actually be accepted as an extra_param."""
    advertised = cls.available_extra_params()
    # Passing the whole set (at its sklearn default) must not raise.
    cls(extra_params=dict(advertised))


def test_available_extra_params_is_sorted():
    keys = list(IsolationForestDetector.available_extra_params())
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# The detector contract table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorSpec:
    """One row of the detector plugin contract.

    ``min_rows`` is the smallest sample count every detector is guaranteed to
    fit and score on without raising (verified here, not just claimed);
    ``stochastic`` records whether ``seed`` actually changes the fit (checked
    against the *_hyperparams reserved-key list, not sklearn source-diving);
    ``natural_boundary`` names the mechanism ``natural_flag`` uses;
    ``training_cap`` must match the class's ``default_train_row_cap``;
    ``attribution_kind`` is the tag the explain pipeline routes this detector
    to (see src/sorethumb_ml/explain/{native,shap_tree,centroid,gradient}.py).
    """

    id: str
    cls: type
    min_rows: int
    stochastic: bool
    natural_boundary: str
    training_cap: int
    supports_extra_params: bool
    attribution_kind: str


DETECTOR_SPECS = [
    DetectorSpec(
        id="isolation_forest",
        cls=IsolationForestDetector,
        min_rows=10,
        stochastic=True,
        natural_boundary="fitted_offset",
        training_cap=250_000,
        supports_extra_params=True,
        attribution_kind="model_specific",
    ),
    DetectorSpec(
        id="kmeans_distance",
        cls=KMeansDetector,
        min_rows=10,
        stochastic=True,
        natural_boundary="tukey_fence",
        training_cap=200_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="one_class_svm",
        cls=OneClassSVMDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="zero_hyperplane",
        training_cap=25_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="lof",
        cls=LOFDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="fitted_offset",
        training_cap=50_000,
        supports_extra_params=True,
        attribution_kind="heuristic",
    ),
    DetectorSpec(
        id="ecod",
        cls=ECODDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="percentile_95",
        training_cap=500_000,
        supports_extra_params=False,
        attribution_kind="exact",
    ),
    DetectorSpec(
        id="hbos",
        cls=HBOSDetector,
        min_rows=10,
        stochastic=False,
        natural_boundary="percentile_95",
        training_cap=500_000,
        supports_extra_params=False,
        attribution_kind="exact",
    ),
]

_SPEC_IDS = [s.id for s in DETECTOR_SPECS]


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_class_vars_match_contract_table(spec: DetectorSpec) -> None:
    assert spec.cls.name == spec.id
    assert spec.cls.default_train_row_cap == spec.training_cap
    # ECOD/HBOS don't advertise available_extra_params() at all -- they have no
    # underlying estimator to forward keys to, so its absence *is* "no support".
    advertised = getattr(spec.cls, "available_extra_params", None)
    assert bool(advertised and advertised()) == spec.supports_extra_params


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_fits_and_scores_at_min_rows(spec: DetectorSpec) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((spec.min_rows, 4)).astype(np.float64)
    det = spec.cls()
    det.fit(X, seed=0)  # must not raise at the documented minimum

    scores = det.score_samples(X)
    assert scores.shape == (spec.min_rows,)
    assert scores.dtype.kind == "f"

    flags = det.natural_flag(scores)
    assert flags.shape == (spec.min_rows,)
    assert flags.dtype == bool

    params = det.get_params()
    assert isinstance(params, dict)


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_higher_score_means_more_normal(spec: DetectorSpec) -> None:
    """The shared protocol invariant: score_samples() ranks outliers lower."""
    rng = np.random.default_rng(1)
    n_normal, n_outlier = 40, 10
    X_normal = rng.standard_normal((n_normal, 4))
    X_outlier = rng.standard_normal((n_outlier, 4)) + 10.0
    X = np.vstack([X_normal, X_outlier])

    det = spec.cls()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    assert scores[:n_normal].mean() > scores[n_normal:].mean(), (
        f"{spec.id}: outliers must score lower (more anomalous) than normals"
    )


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_seed_sensitivity_matches_contract_table(spec: DetectorSpec) -> None:
    """Same seed -> identical scores always; a different seed changes the fit
    only for detectors the table marks stochastic."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 4)).astype(np.float64)

    det_a = spec.cls()
    det_a.fit(X, seed=0)
    det_b = spec.cls()
    det_b.fit(X, seed=0)
    np.testing.assert_array_equal(det_a.score_samples(X), det_b.score_samples(X))

    det_c = spec.cls()
    det_c.fit(X, seed=123)
    changed = not np.array_equal(det_a.score_samples(X), det_c.score_samples(X))
    assert changed == spec.stochastic, (
        f"{spec.id}: stochastic={spec.stochastic} in the table, but changing the "
        f"seed {'changed nothing' if spec.stochastic else 'changed the fit'}"
    )


@pytest.mark.parametrize("spec", DETECTOR_SPECS, ids=_SPEC_IDS)
def test_detector_save_load_model_roundtrip(spec: DetectorSpec, tmp_path) -> None:
    """Every detector -- not just the shipped default three -- must round-trip
    through save_model/load_model with an identical calibrated score."""
    from sorethumb_ml.scoring.calibrate import Calibrator

    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 4)).astype(np.float64)

    det = spec.cls()
    det.fit(X, seed=0)
    scores = det.score_samples(X)
    cal = Calibrator()
    cal.fit(scores)

    with open_workspace(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", len(X), 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", "hash_abc", len(X), 0)

        det2, cal2, manifest = load_model(ws, "run1", gk, spec.id, expected_plan_digest=plan_digest("{}"))

    assert det2.name == spec.id
    np.testing.assert_array_equal(det2.score_samples(X), scores)
    np.testing.assert_allclose(cal2.transform(scores), cal.transform(scores))
    assert manifest["detector_name"] == spec.id


# ---------------------------------------------------------------------------
# Output contract: validate_scores / validate_flags / score_and_flag
# ---------------------------------------------------------------------------

_N = 6
_OK_SCORES = np.linspace(-1.0, 0.0, _N)
_OK_FLAGS = np.array([True, False, False, False, False, False])


@pytest.mark.parametrize(
    ("bad", "condition"),
    [
        (_OK_SCORES[:-1], "wrong length 5 for 6 input rows"),
        (np.append(_OK_SCORES, 0.0), "wrong length 7 for 6 input rows"),
        (_OK_SCORES.reshape(-1, 1), "2-dimensional, not 1-D"),
        (_OK_SCORES.reshape(2, 3), "2-dimensional, not 1-D"),
        (np.array(0.5), "0-dimensional, not 1-D"),
        (np.where(np.arange(_N) == 2, np.nan, _OK_SCORES), "1 NaN (first at index 2)"),
        (np.where(np.arange(_N) == 3, np.inf, _OK_SCORES), "1 +inf (first at index 3)"),
        (np.where(np.arange(_N) == 4, -np.inf, _OK_SCORES), "1 -inf (first at index 4)"),
        (np.array([np.nan, np.inf, -np.inf, 0.0, 0.0, 0.0]), "1 NaN, 1 +inf, 1 -inf"),
        (_OK_SCORES.astype(object), "dtype object is not integer or float"),
        (np.ones(_N, dtype=bool), "dtype bool is not integer or float"),
        (np.ones(_N, dtype=complex), "dtype complex128 is not integer or float"),
        (np.array(list("abcdef")), "is not integer or float"),
        (list(_OK_SCORES), "not a numpy ndarray"),
        (None, "not a numpy ndarray"),
    ],
)
def test_validate_scores_rejects_malformed_output(bad, condition):
    with pytest.raises(DetectorError) as exc:
        validate_scores("my_det", bad, _N)
    msg = str(exc.value)
    assert "'my_det'" in msg
    assert "score_samples()" in msg
    assert condition in msg
    assert f"shape ({_N},)" in msg  # the expected contract
    assert "got " in msg  # the actual type/dtype/shape


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.uint8])
def test_validate_scores_accepts_real_numeric_and_returns_float64(dtype):
    out = validate_scores("d", np.arange(_N).astype(dtype), _N)
    assert out.dtype == np.float64
    assert out.shape == (_N,)


def test_validate_scores_accepts_empty_input():
    assert validate_scores("d", np.empty(0), 0).shape == (0,)


@pytest.mark.parametrize(
    ("bad", "condition"),
    [
        (None, "not a numpy ndarray"),
        (list(_OK_FLAGS), "not a numpy ndarray"),
        (_OK_FLAGS[:-1], "wrong length 5 for 6 input rows"),
        (_OK_FLAGS.reshape(-1, 1), "2-dimensional, not 1-D"),
        (_OK_FLAGS.astype(np.int64), "dtype int64 is not bool"),
        (_OK_FLAGS.astype(np.uint8), "dtype uint8 is not bool"),
        (_OK_FLAGS.astype(np.float64), "dtype float64 is not bool"),
        (_OK_FLAGS.astype(object), "dtype object is not bool"),
        (np.array(["yes", "no", "no", "no", "no", "no"]), "is not bool"),
    ],
)
def test_validate_flags_rejects_malformed_output(bad, condition):
    with pytest.raises(DetectorError) as exc:
        validate_flags("my_det", bad, _N)
    msg = str(exc.value)
    assert "'my_det'" in msg
    assert "natural_flag()" in msg
    assert condition in msg
    assert "bool ndarray" in msg
    assert "got " in msg


def test_validate_flags_accepts_bool_and_returns_it_unchanged():
    assert validate_flags("d", _OK_FLAGS, _N) is _OK_FLAGS


def test_score_and_flag_wraps_detector_exceptions_with_the_detector_name():
    class _Raises(_GoodDetector):
        name = "raises"

        def score_samples(self, X):
            raise RuntimeError("boom")

    with pytest.raises(DetectorError, match=r"'raises' score_samples\(\) raised RuntimeError: boom"):
        score_and_flag(_Raises(), np.zeros((4, 2)))


def test_score_and_flag_reports_natural_flag_exceptions_and_unfitted_state():
    class _Unfitted(_GoodDetector):
        name = "unfitted"

        def natural_flag(self, scores):
            raise AssertionError("fit() must be called before natural_flag()")

    with pytest.raises(DetectorError, match=r"'unfitted' natural_flag\(\) raised AssertionError"):
        score_and_flag(_Unfitted(), np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# Output contract through the pipeline: a malformed plugin is a clear
# DetectorError on a fresh run, a reuse_models run and `score --from-run`
# ---------------------------------------------------------------------------

_PLUGIN = "cfg_plugin"
_runner = CliRunner()

# (mode, substring the stored group error must contain)
_SCORE_MODES = [
    ("short", "wrong length"),
    ("two_d", "2-dimensional, not 1-D"),
    ("nan", "NaN"),
    ("posinf", "+inf"),
    ("neginf", "-inf"),
    ("object", "dtype object is not integer or float"),
    ("as_list", "not a numpy ndarray"),
]
_FLAG_MODES = [
    ("flags_none", "not a numpy ndarray"),
    ("flags_short", "wrong length"),
    ("flags_two_d", "2-dimensional, not 1-D"),
    ("flags_int", "dtype int64 is not bool"),
    ("flags_float", "dtype float64 is not bool"),
    ("flags_object", "dtype object is not bool"),
]
_ALL_MODES = _SCORE_MODES + _FLAG_MODES


class _ConfigurableDetector:
    """Module-level (so it pickles) detector whose output a test can corrupt."""

    name: ClassVar[str] = _PLUGIN
    supports_tree_shap: ClassVar[bool] = False
    default_train_row_cap: ClassVar[int] = 10_000
    mode: ClassVar[str] = "good"

    def fit(self, X, *, seed):
        self._thr = float(np.percentile(-np.abs(X[:, 0]), 10))

    def score_samples(self, X):
        if self._thr is None:
            raise RuntimeError("fit() must be called first")
        good = -np.abs(np.asarray(X, dtype=np.float64)[:, 0]) + 1e-9 * np.arange(len(X))
        m = type(self).mode
        if m == "short":
            return good[:-1]
        if m == "two_d":
            return good.reshape(-1, 1)
        if m in {"nan", "posinf", "neginf"}:
            bad = good.copy()
            bad[1] = {"nan": np.nan, "posinf": np.inf, "neginf": -np.inf}[m]
            return bad
        if m == "object":
            return good.astype(object)
        if m == "as_list":
            return list(good)
        return good

    def natural_flag(self, scores):
        flags = np.asarray(scores, dtype=np.float64) < self._thr
        return {
            "flags_none": None,
            "flags_short": flags[:-1],
            "flags_two_d": flags.reshape(-1, 1),
            "flags_int": flags.astype(np.int64),
            "flags_float": flags.astype(np.float64),
            "flags_object": flags.astype(object),
        }.get(type(self).mode, flags)

    def get_params(self):
        return {}


@pytest.fixture
def plugin(monkeypatch):
    from sorethumb_ml.detectors import register

    monkeypatch.setattr(_ConfigurableDetector, "mode", "good")
    register(_ConfigurableDetector)
    yield _ConfigurableDetector
    registry.pop(_PLUGIN, None)


def _write_workspace(tmp_path: Path, *, reuse_models: bool = False) -> tuple[Path, Path]:
    csv_path = tmp_path / "data.csv"
    write_grouped_csv(csv_path, n_rows=300)
    workdir = tmp_path / "ws"
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        f"""\
[source]
uri = {json.dumps(str(csv_path))}
format = "csv"

[run]
workdir = {json.dumps(str(workdir))}
seed = 0
reuse_models = {str(reuse_models).lower()}

[columns]
id_column = "id"

[scoring]
contamination = 0.1
min_records = 10

[[detectors]]
name = "{_PLUGIN}"
enabled = true

[explain]
enabled = false
""",
        encoding="utf-8",
    )
    return toml, workdir


def _group_errors_and_models(workdir: Path, run_id: str) -> tuple[list[str], list[dict]]:
    from sorethumb_ml import Workspace

    with Workspace.open(workdir) as ws:
        groups = ws.store.all_run_groups(run_id)
        models = [m for g in groups for m in ws.store.models_for_run_group(run_id, g["group_key"])]
    return [g["error"] or "" for g in groups if g["status"] == "failed"], models


def _latest_run_id(workdir: Path) -> str:
    from sorethumb_ml import Workspace

    with Workspace.open(workdir) as ws:
        return ws.store.list_runs(limit=1)[0]["run_id"]


def _assert_clear_detector_error(error: str, expect: str) -> None:
    assert error.startswith("DetectorError: Detector 'cfg_plugin' "), error
    assert expect in error
    assert "Expected " in error
    assert "got " in error


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "expect"), _ALL_MODES)
def test_fresh_run_reports_malformed_output_as_a_detector_error(plugin, tmp_path, monkeypatch, mode, expect):
    toml, workdir = _write_workspace(tmp_path)
    monkeypatch.setattr(plugin, "mode", mode)

    result = _runner.invoke(app, ["run", "--config", str(toml), "--no-report"])

    assert result.exit_code in {1, 4}, result.stdout
    errors, models = _group_errors_and_models(workdir, _latest_run_id(workdir))
    assert len(errors) == 1
    _assert_clear_detector_error(errors[0], expect)
    assert models == [], "a malformed detector must never be persisted"


@pytest.mark.integration
@pytest.mark.usefixtures("plugin")
def test_fresh_run_with_good_plugin_succeeds(tmp_path):
    toml, workdir = _write_workspace(tmp_path)
    result = _runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert result.exit_code == 0, result.stdout
    errors, models = _group_errors_and_models(workdir, _latest_run_id(workdir))
    assert errors == []
    assert len(models) == 1


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "expect"), _ALL_MODES)
def test_score_from_run_reports_malformed_output_as_a_detector_error(
    plugin, tmp_path, monkeypatch, mode, expect
):
    toml, workdir = _write_workspace(tmp_path)
    good = _runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert good.exit_code == 0, good.stdout
    source_run = _latest_run_id(workdir)

    monkeypatch.setattr(plugin, "mode", mode)
    result = _runner.invoke(app, ["score", "--from-run", source_run, "--config", str(toml), "--no-report"])

    assert result.exit_code in {1, 4}, result.stdout
    run_id = _latest_run_id(workdir)
    assert run_id != source_run
    errors, _ = _group_errors_and_models(workdir, run_id)
    assert len(errors) == 1
    _assert_clear_detector_error(errors[0], expect)


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "expect"), [_SCORE_MODES[0], _SCORE_MODES[2], _FLAG_MODES[3]])
def test_reuse_models_reports_malformed_output_as_a_detector_error(
    plugin, tmp_path, monkeypatch, mode, expect
):
    toml, workdir = _write_workspace(tmp_path, reuse_models=True)
    good = _runner.invoke(app, ["run", "--config", str(toml), "--no-report"])
    assert good.exit_code == 0, good.stdout

    monkeypatch.setattr(plugin, "mode", mode)
    result = _runner.invoke(app, ["run", "--config", str(toml), "--no-report", "--force"])

    assert result.exit_code in {1, 4}, result.stdout
    errors, _ = _group_errors_and_models(workdir, _latest_run_id(workdir))
    assert len(errors) == 1
    _assert_clear_detector_error(errors[0], expect)
