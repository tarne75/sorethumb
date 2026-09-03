"""Detector and calibrator persistence, plus score-forward.

Each (run_id, group_key, detector_name) triple has three files:
  - <detector_name>.joblib   — the fitted sklearn estimator
  - calibrator.json          — Calibrator quantile points
  - manifest.json            — feature_schema_hash, plan digest, params, seed, ...

score_with_existing() loads a source run's plan and fitted models, applies them
to new data without re-fitting, and compares feature_schema_hash to detect drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from typing import Any

import joblib
import numpy as np

from sorethumb.errors import ModelSchemaDriftError, ModelSchemaDriftWarning, StoreError
from sorethumb.scoring.calibrate import Calibrator
from sorethumb.store.workspace import Workspace

logger = logging.getLogger(__name__)


def _plan_digest(plan_json: str) -> str:
    return hashlib.sha256(plan_json.encode()).hexdigest()[:16]


def save_model(
    workspace: Workspace,
    run_id: str,
    group_key: str,
    detector: Any,
    calibrator: Calibrator,
    plan_json: str,
    feature_schema_hash: str,
    train_row_count: int,
    seed: int,
) -> str:
    """Persist a fitted detector and its calibrator to the workspace.

    Returns the model_id (a digest-based string).
    """
    detector_name: str = detector.name
    model_id = f"{run_id}_{group_key}_{detector_name}"

    out_dir = workspace.models_dir(run_id, group_key)

    # Write estimator
    estimator_path = out_dir / f"{detector_name}.joblib"
    joblib.dump(detector, str(estimator_path))

    # Write calibrator
    calibrator_path = out_dir / "calibrator.json"
    calibrator_d = calibrator.to_dict()
    calibrator_path.write_text(json.dumps(calibrator_d), encoding="utf-8")

    # Write manifest
    params = detector.get_params()
    manifest = {
        "model_id": model_id,
        "run_id": run_id,
        "group_key": group_key,
        "detector_name": detector_name,
        "feature_schema_hash": feature_schema_hash,
        "plan_digest": _plan_digest(plan_json),
        "train_row_count": train_row_count,
        "params": params,
        "seed": seed,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, default=str), encoding="utf-8")

    # Register with the database
    params_json = json.dumps(params, default=str)
    workspace.store.upsert_model(
        model_id=model_id,
        run_id=run_id,
        group_key=group_key,
        detector_name=detector_name,
        artifact_path=str(estimator_path),
        feature_schema_hash=feature_schema_hash,
        train_row_count=train_row_count,
        params_json=params_json,
    )
    workspace.store.upsert_calibrator(
        model_id=model_id,
        quantile_values_json=json.dumps(calibrator_d.get("quantile_values")),
    )

    # Register artifacts
    for fpath in (estimator_path, calibrator_path, manifest_path):
        workspace.store.register_artifact(
            artifact_id=f"{model_id}_{fpath.name}",
            path=str(fpath),
            kind="model",
            byte_size=fpath.stat().st_size,
            regenerable=False,
        )

    logger.info("Saved model %s to %s.", model_id, out_dir)
    return model_id


def load_model(
    workspace: Workspace,
    run_id: str,
    group_key: str,
    detector_name: str,
) -> tuple[Any, Calibrator, dict[str, Any]]:
    """Load a fitted detector, its calibrator, and the manifest dict.

    Returns (detector, calibrator, manifest).
    Raises StoreError if the model files are absent.
    """
    out_dir = workspace.models_dir(run_id, group_key)

    estimator_path = out_dir / f"{detector_name}.joblib"
    if not estimator_path.exists():
        msg = f"Model file not found: {estimator_path}"
        raise StoreError(msg)

    detector = joblib.load(str(estimator_path))

    calibrator_path = out_dir / "calibrator.json"
    if calibrator_path.exists():
        calibrator_d = json.loads(calibrator_path.read_text(encoding="utf-8"))
        calibrator = Calibrator.from_dict(calibrator_d)
    else:
        calibrator = Calibrator()

    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return detector, calibrator, manifest


def score_with_existing(
    workspace: Workspace,
    source_run_id: str,
    group_key: str,
    new_feature_matrix: np.ndarray,
    feature_schema_hash: str,
    detector_names: list[str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Score new data using models from a previous run without re-fitting.

    Compares feature_schema_hash to detect schema drift:
    - strict=True: raises ModelSchemaDriftError on mismatch.
    - strict=False: emits ModelSchemaDriftWarning; the caller must decide whether
      to refit and mark the result as drift-refitted.

    Parameters
    ----------
    workspace:
        Active workspace.
    source_run_id:
        The run whose models to load.
    group_key:
        Group digest (filesystem path segment).
    new_feature_matrix:
        Already-encoded feature matrix for the new data.
    feature_schema_hash:
        Hash of the new data's feature schema (from FeatureSpace).
    detector_names:
        Which detectors to score with. Must match what was saved.
    strict:
        If True, raise on schema drift instead of warning.

    Returns
    -------
    dict with keys:
        "scores": dict[detector_name -> np.ndarray]
        "calibrated": dict[detector_name -> np.ndarray]
        "drifted": bool

    """
    scores: dict[str, np.ndarray] = {}
    calibrated: dict[str, np.ndarray] = {}
    drifted = False

    for det_name in detector_names:
        try:
            detector, calibrator, manifest = load_model(workspace, source_run_id, group_key, det_name)
        except StoreError:
            logger.warning(
                "No saved model for detector=%s group=%s run=%s; skipping.",
                det_name,
                group_key,
                source_run_id,
            )
            continue

        saved_hash = manifest.get("feature_schema_hash", "")
        if saved_hash and saved_hash != feature_schema_hash:
            msg = (
                f"Feature schema drift detected for group={group_key} detector={det_name}: "
                f"saved={saved_hash!r} new={feature_schema_hash!r}."
            )
            if strict:
                raise ModelSchemaDriftError(msg)
            warnings.warn(msg, ModelSchemaDriftWarning, stacklevel=2)
            drifted = True

        raw_scores = detector.score_samples(new_feature_matrix)
        cal_scores = calibrator.transform(raw_scores)
        scores[det_name] = raw_scores
        calibrated[det_name] = cal_scores

    return {"scores": scores, "calibrated": calibrated, "drifted": drifted}
