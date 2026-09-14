"""Model manifest contract: save_model/load_model round-trip each detector's
estimator, calibrator and manifest independently (no cross-detector
clobbering), manifests record library versions and per-file SHA-256 digests,
and load_model fails closed on anything that doesn't match what was recorded
-- a missing manifest/calibrator, a corrupted file, a swapped manifest, or a
FeaturePlan digest mismatch (P0-5).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from sorethumb.errors import (
    ModelIntegrityError,
    ModelVersionMismatchError,
    ModelVersionMismatchWarning,
    StoreError,
)
from sorethumb.store.models import load_model, plan_digest, save_model
from sorethumb.store.workspace import make_group_key
from tests.factories.workspaces import open_workspace as _open_ws

pytestmark = pytest.mark.contract


def _fit_detector(n: int = 100, d: int = 4, seed: int = 0):
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    det = IsolationForestDetector(n_estimators=20)
    det.fit(X, seed=seed)
    return det, X


def _fitted_calibrator(det, X: np.ndarray):
    from sorethumb.scoring.calibrate import Calibrator

    c = Calibrator()
    scores = det.score_samples(X)
    c.fit(scores)
    return c


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


def test_save_load_model_roundtrip(tmp_path):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, '{"plan":"json"}', "hash_abc", 100, 42)
        det2, cal2, manifest = load_model(ws, "run1", gk, "isolation_forest")
    assert det2.name == "isolation_forest"
    assert cal2._quantile_values is not None
    assert manifest["feature_schema_hash"] == "hash_abc"
    assert manifest["seed"] == 42


def test_save_model_registers_db_row(tmp_path):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, '{"plan":"json"}', "hash_abc", 100, 42)
        rows = ws.store.models_for_run_group("run1", gk)
        assert len(rows) == 1
        assert rows[0]["detector_name"] == "isolation_forest"


def test_load_model_missing_raises(tmp_path):
    with _open_ws(tmp_path) as ws, pytest.raises(StoreError, match="not found"):
        load_model(ws, "norun", "nogroup", "isolation_forest")


def test_three_detector_group_round_trips_per_detector(tmp_path):
    """The default 3-detector ensemble in one group: every detector's calibrator,
    manifest and estimator must round-trip independently — the per-detector files
    must not clobber each other, and the writes leave no ``.tmp`` debris."""
    from sorethumb.detectors.kmeans_distance import KMeansDetector
    from sorethumb.detectors.one_class_svm import OneClassSVMDetector

    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))

    det_if, _ = _fit_detector(n=150)
    det_km = KMeansDetector(k=3)
    det_km.fit(X, seed=0)
    det_oc = OneClassSVMDetector(nu=0.1)
    det_oc.fit(X, seed=0)

    saved = {
        "isolation_forest": (det_if, _fitted_calibrator(det_if, X)),
        "kmeans_distance": (det_km, _fitted_calibrator(det_km, X)),
        "one_class_svm": (det_oc, _fitted_calibrator(det_oc, X)),
    }

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 150, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        for det, cal in saved.values():
            save_model(ws, "run1", gk, det, cal, "{}", "hash_abc", 150, 0)

        out_dir = ws.models_dir("run1", gk)
        # No cross-detector clobbering: each namespaced trio is present, the
        # legacy shared names are not, and no atomic-write temp files remain.
        for name in saved:
            for suffix in ("joblib", "calibrator.json", "manifest.json"):
                assert (out_dir / f"{name}.{suffix}").exists()
        assert not (out_dir / "calibrator.json").exists()
        assert not (out_dir / "manifest.json").exists()
        assert list(out_dir.glob(".*.tmp")) == []
        assert list(out_dir.glob("*.tmp")) == []

        for name, (_, cal) in saved.items():
            det_loaded, cal_loaded, manifest = load_model(ws, "run1", gk, name)
            assert det_loaded.name == name
            assert manifest["detector_name"] == name
            np.testing.assert_array_equal(cal_loaded._quantile_values, cal._quantile_values)

        # DB rows: one model per detector.
        rows = ws.store.models_for_run_group("run1", gk)
        assert {r["detector_name"] for r in rows} == set(saved)


def test_atomic_write_text_leaves_original_on_failure(tmp_path, monkeypatch):
    from sorethumb import _atomic

    target = tmp_path / "f.json"
    target.write_text("original", encoding="utf-8")

    def _boom(*_a):
        raise OSError("disk full")

    monkeypatch.setattr(_atomic.os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        _atomic.atomic_write_text(target, "new content")
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == "original"  # untouched
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]  # temp cleaned up


def test_save_load_plan_roundtrip(tmp_path):
    from sorethumb.store.models import load_plan, save_plan

    with _open_ws(tmp_path) as ws:
        plan_json = '{"chosen_time_column": null, "scaler_params": {"a": {"center": 0.0, "scale": 1.0}}}'
        digest = save_plan(ws, "run1", plan_json)
        assert len(digest) == 32
        assert (ws.run_dir("run1") / "plan.json").exists()

        with pytest.raises(StoreError, match="No persisted FeaturePlan"):
            load_plan(ws, "run_without_plan")


# ---------------------------------------------------------------------------
# Model library-version recording / verification
# ---------------------------------------------------------------------------


def _save_one_model(ws, run_id: str = "run1"):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)
    ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
    ws.store.insert_run(run_id, "fp1", "{}", 0)
    gk = make_group_key({"g": "A"})
    save_model(ws, run_id, gk, det, cal, "{}", "hash_abc", 100, 42)
    return gk, ws.models_dir(run_id, gk) / f"{det.name}.manifest.json"


def test_manifest_records_library_versions(tmp_path):
    with _open_ws(tmp_path) as ws:
        _gk, manifest_path = _save_one_model(ws)
        versions = json.loads(manifest_path.read_text())["library_versions"]
    assert "python" in versions
    assert "numpy" in versions
    assert "scikit-learn" in versions
    assert all(isinstance(v, str) and v for v in versions.values())


def test_load_model_version_mismatch_warns(tmp_path):
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        manifest = json.loads(manifest_path.read_text())
        manifest["library_versions"]["numpy"] = "0.0.0-fake"
        manifest_path.write_text(json.dumps(manifest))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_model(ws, "run1", gk, "isolation_forest")

    matches = [x for x in w if issubclass(x.category, ModelVersionMismatchWarning)]
    assert matches
    assert "numpy" in str(matches[0].message)


def test_load_model_version_mismatch_strict_raises(tmp_path):
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        manifest = json.loads(manifest_path.read_text())
        manifest["library_versions"]["scikit-learn"] = "0.0.0-fake"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ModelVersionMismatchError, match="scikit-learn"):
            load_model(ws, "run1", gk, "isolation_forest", strict=True)


def test_load_model_without_version_block_is_silent(tmp_path):
    """A manifest predating version recording must not warn."""
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        manifest = json.loads(manifest_path.read_text())
        del manifest["library_versions"]
        manifest_path.write_text(json.dumps(manifest))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_model(ws, "run1", gk, "isolation_forest", strict=True)

    assert not [x for x in w if issubclass(x.category, ModelVersionMismatchWarning)]


# ---------------------------------------------------------------------------
# load_model fails closed: missing manifest/calibrator, corruption, swapped
# manifest, wrong plan (P0-5)
# ---------------------------------------------------------------------------


def test_save_model_manifest_records_file_digests(tmp_path):
    with _open_ws(tmp_path) as ws:
        _gk, manifest_path = _save_one_model(ws)
        manifest = json.loads(manifest_path.read_text())
    digests = manifest["file_digests"]
    assert len(digests["estimator"]) == 64  # sha256 hex
    assert len(digests["calibrator"]) == 64


def test_load_model_missing_manifest_raises(tmp_path):
    """A manifest deleted out from under an estimator file must fail, never
    silently proceed with an empty manifest dict."""
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        manifest_path.unlink()

        with pytest.raises(StoreError, match="Manifest file not found"):
            load_model(ws, "run1", gk, "isolation_forest")


def test_load_model_missing_calibrator_raises(tmp_path):
    """A calibrator file deleted out from under a model must fail, never
    silently fall back to an unfitted Calibrator()."""
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        calibrator_path = manifest_path.parent / "isolation_forest.calibrator.json"
        calibrator_path.unlink()

        with pytest.raises(StoreError, match="Calibrator file not found"):
            load_model(ws, "run1", gk, "isolation_forest")


def test_load_model_corrupt_estimator_raises_integrity_error(tmp_path):
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        estimator_path = manifest_path.parent / "isolation_forest.joblib"
        estimator_path.write_bytes(b"not a joblib file at all")

        with pytest.raises(ModelIntegrityError, match="digest mismatch"):
            load_model(ws, "run1", gk, "isolation_forest")


def test_load_model_corrupt_calibrator_raises_integrity_error(tmp_path):
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        calibrator_path = manifest_path.parent / "isolation_forest.calibrator.json"
        calibrator_path.write_text('{"tampered": true}')

        with pytest.raises(ModelIntegrityError, match="digest mismatch"):
            load_model(ws, "run1", gk, "isolation_forest")


def test_load_model_swapped_manifest_raises_integrity_error(tmp_path):
    """A manifest whose recorded identity doesn't match what's being requested
    (e.g. copied from another detector/group/run) must be rejected outright."""
    with _open_ws(tmp_path) as ws:
        gk, manifest_path = _save_one_model(ws)
        manifest = json.loads(manifest_path.read_text())
        manifest["detector_name"] = "kmeans_distance"  # doesn't match the file it names
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ModelIntegrityError, match="identity mismatch"):
            load_model(ws, "run1", gk, "isolation_forest")


def test_load_model_wrong_plan_digest_raises_integrity_error(tmp_path):
    """Scoring against a FeaturePlan other than the one a model was fitted with
    must be rejected, not silently applied."""
    with _open_ws(tmp_path) as ws:
        gk, _manifest_path = _save_one_model(ws)  # saved with plan_json="{}"

        with pytest.raises(ModelIntegrityError, match="plan_digest mismatch"):
            load_model(
                ws,
                "run1",
                gk,
                "isolation_forest",
                expected_plan_digest=plan_digest('{"different": "plan"}'),
            )

        # The correct plan digest still loads fine.
        det, cal, _manifest = load_model(
            ws, "run1", gk, "isolation_forest", expected_plan_digest=plan_digest("{}")
        )
        assert det.name == "isolation_forest"
        assert cal._quantile_values is not None
