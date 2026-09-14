"""Detector and calibrator persistence, plus score-forward.

Each (run_id, group_key, detector_name) triple has three files, all namespaced
by detector so several detectors can share a group directory:
  - <detector_name>.joblib          — the fitted sklearn estimator
  - <detector_name>.calibrator.json — Calibrator quantile points
  - <detector_name>.manifest.json   — feature_schema_hash, plan digest, params,
                                      seed, library_versions, ...

The run's fitted FeaturePlan is written once per run at models/<run_id>/plan.json
(save_plan / load_plan) for score-forward reuse.

Every write is atomic (sibling temp file + fsync + os.replace, see
sorethumb._atomic), so a crash mid write never leaves a half-written model
file for a later score-forward run.

score_with_existing() loads a source run's plan and fitted models, applies them
to new data without re-fitting, compares feature_schema_hash to detect drift,
and checks the fit-time library versions against the current environment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sorethumb._atomic import atomic_write, atomic_write_text
from sorethumb.errors import (
    ModelIntegrityError,
    ModelSchemaDriftError,
    ModelSchemaDriftWarning,
    ModelVersionMismatchError,
    ModelVersionMismatchWarning,
    StoreError,
)
from sorethumb.scoring.calibrate import Calibrator
from sorethumb.store.workspace import Workspace

logger = logging.getLogger(__name__)

# Libraries whose version changes can silently change a pickled estimator's
# scores. Recorded at fit time and checked when the model is reloaded.
_TRACKED_LIBRARIES = ("sorethumb", "scikit-learn", "numpy", "scipy", "joblib")


def plan_digest(plan_json: str) -> str:
    """Digest a serialised FeaturePlan for manifest/round-trip identity checks."""
    return hashlib.sha256(plan_json.encode()).hexdigest()[:32]


def _atomic_joblib_dump(obj: Any, path: Path) -> None:
    """``joblib.dump`` via a sibling temp file, then ``os.replace``."""
    with atomic_write(path) as tmp:
        joblib.dump(obj, tmp)


def _sha256_file(path: Path) -> str:
    """Full SHA-256 hex digest of a file's on-disk bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_PLAN_FILENAME = "plan.json"


def save_plan(workspace: Workspace, run_id: str, plan_json: str) -> str:
    """Persist the run's fitted FeaturePlan JSON. Returns the digest.

    Written once per run (after ``fit_features``) so a later score-forward run
    (``sorethumb score --from-run``) can reload the exact fitted plan —
    frequency maps, scaler params, PCA components, correlation-drop list — and
    apply it to new data without re-fitting.
    """
    path = workspace.run_dir(run_id) / _PLAN_FILENAME
    atomic_write_text(path, plan_json)
    digest = plan_digest(plan_json)
    workspace.store.register_artifact(
        artifact_id=f"{run_id}_plan",
        path=str(path),
        kind="plan",
        byte_size=path.stat().st_size,
        regenerable=False,
        run_id=run_id,
    )
    logger.info("Saved FeaturePlan for run %s (digest=%s).", run_id, digest[:8])
    return digest


def load_plan(workspace: Workspace, run_id: str) -> Any:
    """Load the fitted FeaturePlan persisted for *run_id*.

    Raises StoreError if the run predates plan persistence or the file is gone.
    """
    from sorethumb.profiling.plan import FeaturePlan  # noqa: PLC0415

    path = workspace.run_dir(run_id) / _PLAN_FILENAME
    if not path.exists():
        msg = (
            f"No persisted FeaturePlan for run {run_id!r} at {path}. "
            "The run may predate plan persistence; re-run it to enable score-forward."
        )
        raise StoreError(msg)
    return FeaturePlan.from_json(path.read_text(encoding="utf-8"))


def _library_versions() -> dict[str, str]:
    """Return {distribution -> version} for the libraries that affect model scores."""
    versions = {"python": platform.python_version()}
    for dist in _TRACKED_LIBRARIES:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _check_library_versions(manifest: dict[str, Any], *, strict: bool) -> None:
    """Compare the manifest's fit-time library versions against the current env.

    No-op when the manifest predates version recording. On mismatch: raise
    ModelVersionMismatchError if *strict*, else emit ModelVersionMismatchWarning.
    """
    saved = manifest.get("library_versions")
    if not saved:
        return
    current = _library_versions()
    drifted = {
        name: (ver, current.get(name, "<absent>")) for name, ver in saved.items() if current.get(name) != ver
    }
    if not drifted:
        return
    detail = ", ".join(f"{name}: fitted={was!r} now={now!r}" for name, (was, now) in sorted(drifted.items()))
    msg = (
        f"Model {manifest.get('model_id', '<unknown>')} was fitted under different "
        f"library versions; scores may not be reproducible ({detail})."
    )
    if strict:
        raise ModelVersionMismatchError(msg)
    warnings.warn(msg, ModelVersionMismatchWarning, stacklevel=3)


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

    # Every file is namespaced by detector — a group can hold several detectors
    # and they must not clobber each other's calibrator / manifest. Writes are
    # atomic (temp file + rename) so a crash never leaves a half-written file.
    estimator_path = out_dir / f"{detector_name}.joblib"
    _atomic_joblib_dump(detector, estimator_path)
    estimator_digest = _sha256_file(estimator_path)

    calibrator_path = out_dir / f"{detector_name}.calibrator.json"
    calibrator_d = calibrator.to_dict()
    atomic_write_text(calibrator_path, json.dumps(calibrator_d))
    calibrator_digest = _sha256_file(calibrator_path)

    # Write manifest. file_digests lets a later load_model verify the estimator
    # and calibrator files it's about to deserialise haven't been corrupted,
    # truncated, or swapped with another model's files since this write.
    params = detector.get_params()
    manifest = {
        "model_id": model_id,
        "run_id": run_id,
        "group_key": group_key,
        "detector_name": detector_name,
        "feature_schema_hash": feature_schema_hash,
        "plan_digest": plan_digest(plan_json),
        "train_row_count": train_row_count,
        "params": params,
        "seed": seed,
        "library_versions": _library_versions(),
        "file_digests": {"estimator": estimator_digest, "calibrator": calibrator_digest},
    }
    manifest_path = out_dir / f"{detector_name}.manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, default=str))

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
            run_id=run_id,
        )

    logger.info("Saved model %s to %s.", model_id, out_dir)
    return model_id


def load_model(
    workspace: Workspace,
    run_id: str,
    group_key: str,
    detector_name: str,
    *,
    strict: bool = False,
    expected_plan_digest: str | None = None,
) -> tuple[Any, Calibrator, dict[str, Any]]:
    """Load a fitted detector, its calibrator, and the manifest dict — fail closed.

    Returns (detector, calibrator, manifest).

    Raises StoreError if the estimator, calibrator, or manifest file is simply
    absent (e.g. this detector was never fitted for this run/group — a caller
    such as :func:`score_with_existing` may treat that as "no model to score
    with" and continue with other detectors).

    Raises ModelIntegrityError -- never silently degraded, regardless of
    *strict* -- if a file exists but fails an integrity check: the manifest's
    recorded (run_id, group_key, detector_name) doesn't match what was
    requested (a swapped/misplaced manifest), *expected_plan_digest* is given
    and doesn't match the manifest's ``plan_digest`` (scoring against the wrong
    source plan), or the estimator/calibrator file's content no longer matches
    the digest recorded at save time (corruption or a swapped file). There is
    no legacy/un-namespaced filename fallback: this is the only supported
    on-disk format.

    The fit-time library versions recorded in the manifest are compared against
    the current environment: a mismatch raises ModelVersionMismatchError when
    *strict*, otherwise emits ModelVersionMismatchWarning.

    The digest checks above are integrity checks, not a security boundary: the
    estimator file is unpickled via ``joblib.load``, which executes arbitrary
    code, and a maliciously crafted file carries its own matching digest. Only
    call this against a workspace you created yourself or fully trust.
    """
    out_dir = workspace.models_dir(run_id, group_key)

    estimator_path = out_dir / f"{detector_name}.joblib"
    if not estimator_path.exists():
        msg = f"Model file not found: {estimator_path}"
        raise StoreError(msg)

    manifest_path = out_dir / f"{detector_name}.manifest.json"
    if not manifest_path.exists():
        msg = f"Manifest file not found: {manifest_path}"
        raise StoreError(msg)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    identity = {"run_id": run_id, "group_key": group_key, "detector_name": detector_name}
    mismatched = {k: (v, manifest.get(k)) for k, v in identity.items() if manifest.get(k) != v}
    if mismatched:
        detail = ", ".join(f"{k}: expected={want!r} manifest={got!r}" for k, (want, got) in mismatched.items())
        msg = f"Manifest identity mismatch at {manifest_path} ({detail}); refusing to load a swapped model."
        raise ModelIntegrityError(msg)

    if expected_plan_digest is not None and manifest.get("plan_digest") != expected_plan_digest:
        msg = (
            f"Manifest plan_digest mismatch at {manifest_path}: "
            f"expected={expected_plan_digest!r} manifest={manifest.get('plan_digest')!r}. "
            "This model was fitted against a different FeaturePlan than the one loaded now."
        )
        raise ModelIntegrityError(msg)

    digests = manifest.get("file_digests") or {}
    estimator_digest = digests.get("estimator")
    if not estimator_digest or _sha256_file(estimator_path) != estimator_digest:
        msg = f"Estimator file digest mismatch or missing for {estimator_path}; file may be corrupt."
        raise ModelIntegrityError(msg)

    calibrator_path = out_dir / f"{detector_name}.calibrator.json"
    if not calibrator_path.exists():
        msg = f"Calibrator file not found: {calibrator_path}"
        raise StoreError(msg)
    calibrator_digest = digests.get("calibrator")
    if not calibrator_digest or _sha256_file(calibrator_path) != calibrator_digest:
        msg = f"Calibrator file digest mismatch or missing for {calibrator_path}; file may be corrupt."
        raise ModelIntegrityError(msg)

    _check_library_versions(manifest, strict=strict)

    detector = joblib.load(str(estimator_path))
    calibrator_d = json.loads(calibrator_path.read_text(encoding="utf-8"))
    calibrator = Calibrator.from_dict(calibrator_d)

    return detector, calibrator, manifest


def score_with_existing(
    workspace: Workspace,
    source_run_id: str,
    group_key: str,
    new_feature_matrix: np.ndarray,
    feature_schema_hash: str,
    detector_names: list[str],
    plan_digest: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Score new data using models from a previous run without re-fitting.

    Compares feature_schema_hash to detect schema drift:
    - strict=True: raises ModelSchemaDriftError on mismatch.
    - strict=False: emits ModelSchemaDriftWarning; the caller must decide whether
      to refit and mark the result as drift-refitted.

    Also compares the fit-time library versions recorded in each manifest against
    the current environment: strict=True raises ModelVersionMismatchError, else
    ModelVersionMismatchWarning is emitted.

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
    plan_digest:
        Digest of the FeaturePlan being applied to the new data (see
        :func:`plan_digest`). Verified against each manifest's ``plan_digest`` --
        a mismatch means these models were fitted against a different plan and
        raises ModelIntegrityError unconditionally (not gated on *strict*).
    strict:
        If True, raise on schema drift or library-version mismatch instead of warning.

    Returns
    -------
    dict with keys:
        "scores":        dict[detector_name -> np.ndarray]  (raw, higher = more normal)
        "calibrated":    dict[detector_name -> np.ndarray]  (via the persisted calibrator)
        "natural_flags": dict[detector_name -> np.ndarray[bool]]
        "detectors":     dict[detector_name -> loaded detector instance]
        "missing":       list[detector_name]  (requested but no persisted model)
        "drifted":       bool

    No detector is re-fitted: each is unpickled from the source run and only
    ``score_samples`` / ``natural_flag`` are called.

    A detector with no persisted model at all (never fitted in the source run)
    is recorded in "missing" and skipped, so the group can still score with the
    rest. A detector whose persisted files exist but fail an integrity check
    (swapped manifest, wrong plan, corrupt file) raises ModelIntegrityError
    instead -- that is never treated as "just missing".
    """
    scores: dict[str, np.ndarray] = {}
    calibrated: dict[str, np.ndarray] = {}
    natural_flags: dict[str, np.ndarray] = {}
    detectors: dict[str, Any] = {}
    missing: list[str] = []
    drifted = False

    for det_name in detector_names:
        try:
            detector, calibrator, manifest = load_model(
                workspace,
                source_run_id,
                group_key,
                det_name,
                strict=strict,
                expected_plan_digest=plan_digest,
            )
        except ModelIntegrityError:
            raise
        except StoreError:
            logger.warning(
                "No saved model for detector=%s group=%s run=%s; skipping.",
                det_name,
                group_key,
                source_run_id,
            )
            missing.append(det_name)
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
        scores[det_name] = raw_scores
        calibrated[det_name] = calibrator.transform(raw_scores)
        natural_flags[det_name] = detector.natural_flag(raw_scores)
        detectors[det_name] = detector

    return {
        "scores": scores,
        "calibrated": calibrated,
        "natural_flags": natural_flags,
        "detectors": detectors,
        "missing": missing,
        "drifted": drifted,
    }
