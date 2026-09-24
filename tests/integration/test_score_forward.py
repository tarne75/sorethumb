"""`sorethumb score --from-run` / ``score_forward`` — reuse a prior run's fitted
FeaturePlan, per-detector models and calibrators, and re-fit *nothing*.

No network; all data generated in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb import Config, score_forward
from sorethumb._pipeline import run_detection
from sorethumb.config import DetectorConfig, ExplainConfig, FeaturesConfig, ProfilingConfig, ReportConfig
from sorethumb.errors import StoreError
from sorethumb.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.detectors import ban_all_fitting as _ban_all_fitting
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration


def _planted_csv(path: Path, *, n_normal: int = 200, n_anomaly: int = 6, seed: int = 0) -> list[int]:
    return write_planted_csv(path, n_normal=n_normal, n_anomaly=n_anomaly, seed=seed)


def _cfg(csv: Path, ws: Path) -> Config:
    return make_config(
        csv,
        ws,
        combination="composite",
        detectors=[
            DetectorConfig(name="isolation_forest"),
            DetectorConfig(name="kmeans_distance"),
        ],
    )


def test_score_forward_does_not_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    src_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    _planted_csv(src_csv, seed=0)
    _planted_csv(new_csv, seed=1)  # same schema, fresh draw, same 6 planted anomalies

    src = run_detection(_cfg(src_csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    _ban_all_fitting(monkeypatch)  # any fit from here raises _FitAttempted

    fwd = score_forward(_cfg(new_csv, ws), src.run_id, no_report=True)

    # It ran, wrote a distinct run, and touched no fitting path.
    assert fwd.run_id != src.run_id
    assert fwd.run_id.startswith("score_")
    assert fwd.n_failed == 0, [g.error for g in fwd.groups if g.error]
    assert fwd.n_succeeded >= 1
    good = next(g for g in fwd.groups if g.status == "success")
    assert good.results_path is not None
    assert good.results_path.exists()

    # Provenance is recorded on the run row.
    with Workspace.open(ws) as w:
        assert w.store.get_run(fwd.run_id)["source_run_id"] == src.run_id
        # No new model dirs were created for the score-forward run.
        assert not (ws / "models" / fwd.run_id).exists() or not any(
            (ws / "models" / fwd.run_id).glob("*/*.joblib")
        )


def test_score_forward_reuses_reference_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Score-forward on the *same* data as the source run must reproduce the
    source run's calibrated scores (same models, same calibrator)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)

    src = run_detection(_cfg(csv, ws), no_report=True)
    src_group = next(g for g in src.groups if g.status == "success")
    src_scores = pl.read_parquet(src_group.results_path).sort("row_id")

    _ban_all_fitting(monkeypatch)
    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    fwd_group = next(g for g in fwd.groups if g.status == "success")
    fwd_scores = pl.read_parquet(fwd_group.results_path).sort("row_id")

    assert src_scores["row_id"].to_list() == fwd_scores["row_id"].to_list()
    np.testing.assert_allclose(
        src_scores["composite_score"].to_numpy(),
        fwd_scores["composite_score"].to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_score_forward_missing_source_run_raises(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    run_detection(_cfg(csv, ws), no_report=True)

    with pytest.raises(StoreError, match="not found"):
        score_forward(_cfg(csv, ws), "run_does_not_exist", no_report=True)


def test_score_forward_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)

    _ban_all_fitting(monkeypatch)
    a = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    b = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert a.run_id == b.run_id  # deterministic id
    # second call skips the already-complete group
    assert any(g.status == "skipped" for g in b.groups)


def test_score_forward_missing_model_is_failed_not_complete_and_is_retried(tmp_path: Path) -> None:
    """A score-forward group that requests a detector never fitted in the
    source run must be recorded as failed -- both in the RunResult and the
    ledger -- and a retry with the same inputs must re-execute it rather than
    treating it as already complete (P0-1).
    """
    ws = tmp_path / "ws"
    src_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    _planted_csv(src_csv, seed=0)
    _planted_csv(new_csv, seed=1)

    # Source run only ever fits isolation_forest + kmeans_distance.
    src = run_detection(_cfg(src_csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    # Request a detector the source run never persisted a model for.
    fwd_cfg = make_config(
        new_csv, ws, combination="composite", detectors=[DetectorConfig(name="one_class_svm")]
    )

    fwd1 = score_forward(fwd_cfg, src.run_id, no_report=True)
    assert fwd1.n_failed == 1
    assert fwd1.n_succeeded == 0
    g1 = fwd1.groups[0]
    assert g1.status == "failed"
    assert "No persisted models" in (g1.error or "")

    with Workspace.open(ws) as w:
        assert w.store.run_status(fwd1.run_id) == "failed"
        assert w.store.group_status(fwd1.run_id, g1.group_key) == "failed"

    # Same inputs -> same deterministic run_id; must be retried, not skipped.
    fwd2 = score_forward(fwd_cfg, src.run_id, no_report=True)
    assert fwd2.run_id == fwd1.run_id
    assert fwd2.n_failed == 1
    assert fwd2.groups[0].status == "failed"

    with Workspace.open(ws) as w:
        assert w.store.run_status(fwd2.run_id) == "failed"


def test_score_forward_rejects_incomplete_source_run(tmp_path: Path) -> None:
    """A source run that is still 'running' (crashed before completion) or
    'failed' has no trustworthy persisted models and must be rejected outright,
    not scored against (P0-5)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    with Workspace.open(ws) as w:
        w.store.mark_run_failed(src.run_id, "simulated crash")

    with pytest.raises(StoreError, match="status"):
        score_forward(_cfg(csv, ws), src.run_id, no_report=True)


def test_score_forward_rejects_score_forward_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A score-forward run never persists its own models -- it must be
    rejected as a source for a further score-forward run (P0-5)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)

    _ban_all_fitting(monkeypatch)
    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert fwd.n_succeeded >= 1

    with pytest.raises(StoreError, match="score-forward run"):
        score_forward(_cfg(csv, ws), fwd.run_id, no_report=True)


def test_score_forward_rejects_corrupted_model_file(tmp_path: Path) -> None:
    """A corrupted persisted estimator file must fail the group loudly, not be
    silently unpickled or skipped as though it were never fitted (P0-5)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    good_group = next(g for g in src.groups if g.status == "success")

    with Workspace.open(ws) as w:
        model_dir = w.models_dir(src.run_id, good_group.group_key)
        (model_dir / "isolation_forest.joblib").write_bytes(b"not actually a joblib file")

    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert fwd.n_failed == 1
    assert fwd.n_succeeded == 0
    err = fwd.groups[0].error or ""
    assert "ModelIntegrityError" in err


def test_score_forward_rejects_swapped_manifest(tmp_path: Path) -> None:
    """A manifest copied onto a different detector's files must be rejected,
    not loaded as though it described the file it's sitting next to (P0-5)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    good_group = next(g for g in src.groups if g.status == "success")

    with Workspace.open(ws) as w:
        model_dir = w.models_dir(src.run_id, good_group.group_key)
        manifest_path = model_dir / "isolation_forest.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["detector_name"] = "kmeans_distance"  # no longer matches the file it's next to
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert fwd.n_failed == 1
    err = fwd.groups[0].error or ""
    assert "ModelIntegrityError" in err


def test_score_forward_multi_detector_round_trip_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean, uncorrupted multi-detector source run must score forward
    successfully end to end (P0-5's positive case)."""
    ws = tmp_path / "ws"
    src_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    _planted_csv(src_csv, seed=0)
    _planted_csv(new_csv, seed=1)

    src = run_detection(_cfg(src_csv, ws), no_report=True)  # isolation_forest + kmeans_distance
    assert src.n_succeeded >= 1

    _ban_all_fitting(monkeypatch)
    fwd = score_forward(_cfg(new_csv, ws), src.run_id, no_report=True)

    assert fwd.n_failed == 0, [g.error for g in fwd.groups if g.error]
    assert fwd.n_succeeded >= 1
    good = next(g for g in fwd.groups if g.status == "success")
    scores = pl.read_parquet(good.results_path)
    for det_name in ("isolation_forest", "kmeans_distance"):
        assert f"score_raw_{det_name}" in scores.columns
        assert f"score_cal_{det_name}" in scores.columns


def test_score_forward_rejects_drifted_schema(tmp_path: Path) -> None:
    """New data whose raw schema no longer matches the fitted plan must fail the
    group loudly (PlanError) -- not silently mis-encode a column or leave one
    unscaled in the distance matrix.
    """
    ws = tmp_path / "ws"
    src_csv = tmp_path / "train.csv"
    _planted_csv(src_csv, seed=0)
    src = run_detection(_cfg(src_csv, ws), no_report=True)
    assert src.n_succeeded == 1

    # Same rows, but num_b now arrives as non-numeric text — CSV type inference
    # would otherwise coerce a still-numeric-looking string straight back to
    # Float64, masking the drift, so use values that can't parse as numbers.
    # This is the exact "drifted input" scenario: apply_feature_plan must catch
    # it before apply_scaler ever gets a chance to leave the column unscaled.
    drifted_csv = tmp_path / "drifted.csv"
    df = pl.read_csv(src_csv)
    df.with_columns(pl.Series("num_b", [f"txt_{i}" for i in range(len(df))])).write_csv(str(drifted_csv))

    result = score_forward(_cfg(drifted_csv, ws), src.run_id, no_report=True)
    assert result.n_succeeded == 0
    assert result.n_failed == 1
    err = result.groups[0].error or ""
    assert "PlanError" in err
    assert "schema fingerprint" in err


# ---------------------------------------------------------------------------
# Fit-time vs. score-time config validation (P0-6)
# ---------------------------------------------------------------------------


def test_score_forward_rejects_different_detector_params(tmp_path: Path) -> None:
    """A score-forward config that requests different hyperparameters for a
    detector the source run actually fit must be rejected outright -- the
    persisted model was fit with the source's params, not these, and
    score-forward never re-fits."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)  # isolation_forest, default params
    assert src.n_succeeded >= 1

    fwd_cfg = make_config(
        csv,
        ws,
        combination="composite",
        detectors=[DetectorConfig(name="isolation_forest", params={"n_estimators": 17})],
    )
    with pytest.raises(StoreError, match="isolation_forest"):
        score_forward(fwd_cfg, src.run_id, no_report=True)


def test_score_forward_rejects_different_train_row_cap(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    fwd_cfg = make_config(
        csv,
        ws,
        combination="composite",
        detectors=[DetectorConfig(name="isolation_forest", train_row_cap=3)],
    )
    with pytest.raises(StoreError, match="train_row_cap"):
        score_forward(fwd_cfg, src.run_id, no_report=True)


def test_score_forward_rejects_different_columns_config(tmp_path: Path) -> None:
    """columns is feature-plan-affecting: apply_feature_plan never reads it,
    so a different id_column here would silently not be what actually ran."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)  # id_column="id" (make_config's default)
    assert src.n_succeeded >= 1

    fwd_cfg = make_config(csv, ws, combination="composite", id_column="row_id_that_does_not_exist")
    with pytest.raises(StoreError, match="columns"):
        score_forward(fwd_cfg, src.run_id, no_report=True)


def test_score_forward_rejects_different_features_config(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    fwd_cfg = _cfg(csv, ws).model_copy(update={"features": FeaturesConfig(scaler="standard")})
    with pytest.raises(StoreError, match="features"):
        score_forward(fwd_cfg, src.run_id, no_report=True)


def test_score_forward_rejects_different_profiling_config(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    fwd_cfg = _cfg(csv, ws).model_copy(update={"profiling": ProfilingConfig(null_ratio_drop=0.42)})
    with pytest.raises(StoreError, match="profiling"):
        score_forward(fwd_cfg, src.run_id, no_report=True)


def test_score_forward_config_mismatch_rejected_regardless_of_strict(tmp_path: Path) -> None:
    """The fit-time config guard is a correctness check, not a data-drift
    warning -- it must reject a mismatch the same way whether strict=True or
    strict=False (unlike schema/library-version drift, which strict does
    genuinely gate)."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.n_succeeded >= 1

    fwd_cfg = make_config(
        csv,
        ws,
        combination="composite",
        detectors=[DetectorConfig(name="isolation_forest", params={"n_estimators": 17})],
    )
    with pytest.raises(StoreError, match="isolation_forest"):
        score_forward(fwd_cfg, src.run_id, no_report=True, strict=False)
    with pytest.raises(StoreError, match="isolation_forest"):
        score_forward(fwd_cfg, src.run_id, no_report=True, strict=True)


def test_score_forward_ignores_param_mismatch_for_a_detector_never_fit(tmp_path: Path) -> None:
    """A detector the source run never enabled (so never fit, no persisted
    model exists for it either way) needs no params validation -- it's
    caught downstream as a "missing model", a separate, already-lenient
    path this guard must not disturb."""
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)  # isolation_forest + kmeans_distance only
    assert src.n_succeeded >= 1

    fwd_cfg = make_config(
        csv,
        ws,
        combination="composite",
        detectors=[DetectorConfig(name="one_class_svm", params={"nu": 0.2})],
    )
    fwd = score_forward(fwd_cfg, src.run_id, no_report=True)
    assert fwd.n_failed == 1
    assert "No persisted models" in (fwd.groups[0].error or "")


def test_score_forward_allows_legal_scoring_explain_report_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scoring/explain/report/run genuinely are recomputed fresh at score
    time -- overriding them is not a lie about what ran, it's the honest,
    real behaviour, so it must not be rejected."""
    ws = tmp_path / "ws"
    src_csv, new_csv = tmp_path / "train.csv", tmp_path / "new.csv"
    _planted_csv(src_csv, seed=0)
    _planted_csv(new_csv, seed=1)
    src = run_detection(_cfg(src_csv, ws), no_report=True)  # combination="composite"
    assert src.n_succeeded >= 1

    _ban_all_fitting(monkeypatch)
    fwd_cfg = make_config(
        new_csv,
        ws,
        combination="composite",
        weighting="agreement",
        contamination=0.2,
        detectors=[DetectorConfig(name="isolation_forest"), DetectorConfig(name="kmeans_distance")],
    ).model_copy(
        update={
            "explain": ExplainConfig(top_n=1),
            "report": ReportConfig(formats=["json"]),
        }
    )
    fwd = score_forward(fwd_cfg, src.run_id, no_report=True)
    assert fwd.n_failed == 0, [g.error for g in fwd.groups if g.error]
    assert fwd.n_succeeded >= 1


# ---------------------------------------------------------------------------
# Provenance surfacing (P0-6)
# ---------------------------------------------------------------------------


def test_score_forward_run_result_carries_source_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)
    assert src.source_run_id is None  # an ordinary fit has no source

    _ban_all_fitting(monkeypatch)
    fwd = score_forward(_cfg(csv, ws), src.run_id, no_report=True)
    assert fwd.source_run_id == src.run_id


def test_score_forward_report_provenance_includes_source_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    csv = tmp_path / "data.csv"
    _planted_csv(csv, seed=0)
    src = run_detection(_cfg(csv, ws), no_report=True)

    _ban_all_fitting(monkeypatch)
    fwd_cfg = _cfg(csv, ws).model_copy(update={"report": ReportConfig(formats=["json"])})
    fwd = score_forward(fwd_cfg, src.run_id, no_report=False)
    assert fwd.report_path is not None
    assert fwd.report_path.name == "index.json"  # formats=["json"] only -> render_report returns it directly

    payload = json.loads(fwd.report_path.read_text(encoding="utf-8"))
    assert payload["source_run_id"] == src.run_id
