"""Store-backed workflows spanning more than one persistence contract: an
interrupted multi-group run's completed groups survive, score-forward's
score_with_existing reuses persisted models correctly (or fails loudly on
drift/corruption), and retention pruning deletes only what it should.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from sorethumb_ml.errors import (
    ModelIntegrityError,
    ModelSchemaDriftError,
    ModelSchemaDriftWarning,
    StoreError,
)
from sorethumb_ml.store.models import plan_digest, save_model, score_with_existing
from sorethumb_ml.store.results import read_results, write_results
from sorethumb_ml.store.workspace import make_group_key
from tests.factories.workspaces import open_workspace as _open_ws

pytestmark = pytest.mark.integration


def _fit_detector(n: int = 100, d: int = 4, seed: int = 0):
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    det = IsolationForestDetector(n_estimators=20)
    det.fit(X, seed=seed)
    return det, X


def _fitted_calibrator(det, X: np.ndarray):
    from sorethumb_ml.scoring.calibrate import Calibrator

    c = Calibrator()
    scores = det.score_samples(X)
    c.fit(scores)
    return c


# ---------------------------------------------------------------------------
# Interrupted run: completed groups survive
# ---------------------------------------------------------------------------


def test_interrupted_run_completed_groups_survive(tmp_path):
    import polars as pl

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        ws.store.insert_run("run1", "fp1", "{}", 0)

        groups = [make_group_key({"g": str(i)}) for i in range(5)]
        for i, gk in enumerate(groups):
            df = pl.DataFrame({"row_id": [i], "score": [float(i) / 5]})
            write_results(ws, "run1", gk, df)
            ws.store.upsert_run_group("run1", gk, json.dumps({"g": str(i)}), str(i), status="complete")
            if i == 1:
                # Simulate interrupt after group 1 (0-indexed, so groups 0 and 1 done)
                break

        completed = ws.store.completed_groups("run1")
        assert len(completed) == 2  # groups 0 and 1

        # Groups 2-4 were never written
        for gk in groups[2:]:
            assert read_results(ws, "run1", gk) is None


# ---------------------------------------------------------------------------
# score_with_existing
# ---------------------------------------------------------------------------


def test_score_with_existing_identical_record(tmp_path):
    """An identical record scored forward must produce the same calibrated score."""
    det, X_train = _fit_detector(n=200, seed=0)
    cal = _fitted_calibrator(det, X_train)
    schema_hash = "abc123def456"

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 200, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", schema_hash, 200, 0)

        # Score same training data forward (must match)
        result = score_with_existing(
            ws, "run1", gk, X_train, schema_hash, ["isolation_forest"], plan_digest("{}")
        )

    assert not result["drifted"]
    cal_scores = result["calibrated"]["isolation_forest"]
    # Compare against direct calibration
    direct = cal.transform(det.score_samples(X_train))
    np.testing.assert_allclose(cal_scores, direct, rtol=1e-6)


def test_score_with_existing_drift_strict_raises(tmp_path):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", "old_hash", 100, 0)

        with pytest.raises(ModelSchemaDriftError, match="drift"):
            score_with_existing(
                ws, "run1", gk, X, "new_hash", ["isolation_forest"], plan_digest("{}"), strict=True
            )


def test_score_with_existing_drift_warning(tmp_path):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", "old_hash", 100, 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = score_with_existing(
                ws, "run1", gk, X, "new_hash", ["isolation_forest"], plan_digest("{}"), strict=False
            )

    assert result["drifted"]
    assert any(issubclass(x.category, ModelSchemaDriftWarning) for x in w)


def test_score_with_existing_missing_model_skips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))

    with _open_ws(tmp_path) as ws:
        result = score_with_existing(
            ws, "norun", "nogroup", X, "hash", ["isolation_forest"], plan_digest("{}")
        )

    assert result["scores"] == {}
    assert result["calibrated"] == {}
    assert not result["drifted"]


def test_score_with_existing_corrupt_model_raises_not_skips(tmp_path):
    """A detector with genuinely no persisted model is lenient (skipped, in
    "missing"). A detector whose persisted files exist but fail an integrity
    check must never be treated the same way -- it has to raise."""
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        save_model(ws, "run1", gk, det, cal, "{}", "hash_abc", 100, 0)
        estimator_path = ws.models_dir("run1", gk) / "isolation_forest.joblib"
        estimator_path.write_bytes(b"corrupted")

        with pytest.raises(ModelIntegrityError, match="digest mismatch"):
            score_with_existing(ws, "run1", gk, X, "hash_abc", ["isolation_forest"], plan_digest("{}"))


# ---------------------------------------------------------------------------
# Retention / pruning
# ---------------------------------------------------------------------------


def test_workspace_list_prunable(tmp_path):
    with _open_ws(tmp_path) as ws:
        prunable = ws.list_prunable(retention_days=0)
        assert isinstance(prunable, list)


def test_prune_rejects_negative_retention_days(tmp_path):
    """P2-6: a negative retention_days would match julianday(now) - created_at
    > (a negative number), i.e. every artifact regardless of age, including
    ones just written. Must be refused, not silently treated as "prune
    everything"."""
    with _open_ws(tmp_path) as ws:
        with pytest.raises(StoreError, match="retention_days"):
            ws.prune(-1)
        with pytest.raises(StoreError, match="retention_days"):
            ws.list_prunable(-1)


def test_prune_refuses_a_path_outside_the_workspace_root(tmp_path):
    """P2-6: the artifact index is a database file, not a signed record. A
    corrupted or hand-edited row pointing outside the workspace root must
    not turn a routine prune into deleting an arbitrary path."""
    outside_path = tmp_path / "outside_the_workspace.parquet"
    outside_path.write_text("do not touch")

    with _open_ws(tmp_path) as ws:
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 4, 1, '2020-01-01T00:00:00Z')",
            ("escaped_art", str(outside_path)),
        )
        ws.store._conn.commit()

        deleted = ws.prune(retention_days=1, dry_run=False)
        assert str(outside_path) not in deleted

        row = ws.store._conn.execute("SELECT * FROM artifact WHERE artifact_id='escaped_art'").fetchone()
        assert row is not None, "the escaping row must not be removed from the index either"

    assert outside_path.read_text() == "do not touch"


def test_workspace_prune_deletes_missing_file(tmp_path):
    """prune() must tolerate a regenerable artifact whose file is already gone."""
    with _open_ws(tmp_path) as ws:
        ghost_path = str(tmp_path / "ws" / "ghost.parquet")
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now', '-400 days'))",
            ("ghost_art", ghost_path, "results", 0, 1),
        )
        ws.store._conn.commit()
        deleted = ws.prune(retention_days=1, dry_run=False)
        assert ghost_path in deleted


def test_prune_dry_run_lists_eligible(tmp_path):
    with _open_ws(tmp_path) as ws:
        # Register a regenerable artifact with a very old created_at (simulate old file)
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 0, 1, '2020-01-01T00:00:00Z')",
            ("art1", str(ws.root / "old_file.parquet")),
        )
        ws.store._conn.commit()
        deleted = ws.prune(retention_days=1, dry_run=True)
        assert any("old_file.parquet" in p for p in deleted)


def test_prune_dry_run_does_not_delete(tmp_path):
    with _open_ws(tmp_path) as ws:
        old_file_path = ws.root / "old_cache.parquet"
        old_file_path.write_text("dummy")
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 5, 1, '2020-01-01T00:00:00Z')",
            ("art1", str(old_file_path)),
        )
        ws.store._conn.commit()
        ws.prune(retention_days=1, dry_run=True)
    # File still present after dry run
    assert old_file_path.exists()


def test_prune_real_removes_file_and_row(tmp_path):
    with _open_ws(tmp_path) as ws:
        old_file_path = ws.root / "old_cache.parquet"
        old_file_path.write_text("dummy")
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 5, 1, '2020-01-01T00:00:00Z')",
            ("art1", str(old_file_path)),
        )
        ws.store._conn.commit()
        deleted = ws.prune(retention_days=1, dry_run=False)
        assert len(deleted) >= 1
    assert not old_file_path.exists()
    with _open_ws(tmp_path) as ws2:
        row = ws2.store._conn.execute("SELECT * FROM artifact WHERE artifact_id='art1'").fetchone()
        assert row is None


def test_prune_dry_run_same_list_as_real(tmp_path):
    with _open_ws(tmp_path) as ws:
        old_file_path = ws.root / "old_cache_2.parquet"
        old_file_path.write_text("dummy")
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 5, 1, '2020-01-01T00:00:00Z')",
            ("art2", str(old_file_path)),
        )
        ws.store._conn.commit()
        dry = ws.prune(retention_days=1, dry_run=True)
        real = ws.prune(retention_days=1, dry_run=False)
    assert set(dry) == set(real)


def test_register_artifact_records_run_id(tmp_path):
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        ws.store.insert_run("run_abc", "fp1", "{}", 0)
        ws.store.register_artifact("art1", str(tmp_path / "f.parquet"), "results", 1, False, run_id="run_abc")
        row = ws.store._conn.execute("SELECT run_id FROM artifact WHERE artifact_id='art1'").fetchone()
    assert row["run_id"] == "run_abc"


def test_prune_failed_run_matches_on_run_id_not_path_substring(tmp_path):
    """A failed run must not drag in another run's artifact just because its
    run_id is a substring of that artifact's path."""
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        # "run_1" is a substring of "run_10" and of the sibling run's path.
        ws.store.insert_run("run_1", "fp1", "{}", 0)
        ws.store.insert_run("run_10", "fp1", "{}", 0)
        ws.store.mark_run_failed("run_1", "boom")
        # Age the failed run past the retention window.
        ws.store._conn.execute("UPDATE run SET started_at='2020-01-01T00:00:00Z' WHERE run_id='run_1'")
        # Artifact belongs to the healthy run_10; its path contains "run_1".
        good_file = ws.root / "models" / "run_10" / "iso.joblib"
        good_file.parent.mkdir(parents=True, exist_ok=True)
        good_file.write_text("keep me")
        ws.store.register_artifact("art_run10", str(good_file), "model", 7, False, run_id="run_10")
        ws.store._conn.commit()

        deleted = ws.prune(retention_days=1, dry_run=False)

    assert str(good_file) not in deleted
    assert good_file.exists()


def test_prune_failed_run_removes_its_own_artifacts(tmp_path):
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        ws.store.insert_run("run_x", "fp1", "{}", 0)
        ws.store.mark_run_failed("run_x", "boom")
        ws.store._conn.execute("UPDATE run SET started_at='2020-01-01T00:00:00Z' WHERE run_id='run_x'")
        doomed = ws.root / "models" / "run_x" / "iso.joblib"
        doomed.parent.mkdir(parents=True, exist_ok=True)
        doomed.write_text("bye")
        ws.store.register_artifact("art_x", str(doomed), "model", 3, False, run_id="run_x")
        ws.store._conn.commit()

        deleted = ws.prune(retention_days=1, dry_run=False)

    assert str(doomed) in deleted


# ---------------------------------------------------------------------------
# Concurrency: parallel writers wait or fail explicitly, never corrupt
# ---------------------------------------------------------------------------


def test_concurrent_write_results_does_not_corrupt_the_workspace(tmp_path):
    """Several threads writing results for the same (run_id, group_key) at
    once must each either succeed cleanly or raise a real exception -- never
    leave a half-written file or a database only some of them can read."""
    import threading

    import polars as pl

    n_threads = 8
    errors: list[BaseException] = []

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})

        def _write(i: int) -> None:
            df = pl.DataFrame({"row_id": [i], "score": [float(i)]})
            try:
                write_results(ws, "run1", gk, df)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Any failure must be a real, catchable exception -- not a hang or a
        # silently-swallowed corruption.
        for exc in errors:
            assert isinstance(exc, Exception)

        # The workspace must still be fully usable: exactly one writer's rows
        # are the final state (atomic_write's rename means the last one to
        # replace() wins wholesale, never an interleaved mix of two writes).
        final = read_results(ws, "run1", gk)
        assert final is not None
        assert len(final) == 1
        assert final["row_id"][0] in range(n_threads)

        # And the database connection itself is unharmed by the contention.
        assert ws.store.run_status("run1") == "running"


# ---------------------------------------------------------------------------
# register_artifact is an UPSERT keyed by artifact_id (P1-5)
# ---------------------------------------------------------------------------


def _artifact_row(ws, artifact_id):
    row = ws.store._conn.execute("SELECT * FROM artifact WHERE artifact_id=?", (artifact_id,)).fetchone()
    return None if row is None else dict(row)


def _age_artifact(ws, artifact_id, created_at="2020-01-01T00:00:00Z"):
    ws.store._conn.execute("UPDATE artifact SET created_at=? WHERE artifact_id=?", (created_at, artifact_id))
    ws.store._conn.commit()


def test_reregistering_an_artifact_preserves_created_at_and_updates_mutable_fields(tmp_path):
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        ws.store.insert_run("run_a", "fp1", "{}", 0)
        ws.store.register_artifact("art", str(tmp_path / "f1"), "model", 10, False, run_id="run_a")
        _age_artifact(ws, "art")

        for size in (20, 30, 40):  # repeated registration, as a resumed/forced run does
            ws.store.register_artifact("art", str(tmp_path / "f2"), "results", size, True, run_id="run_a")

        row = _artifact_row(ws, "art")
    assert row["created_at"] == "2020-01-01T00:00:00Z"  # age survives every re-registration
    assert (row["path"], row["kind"], row["byte_size"], row["regenerable"]) == (
        str(tmp_path / "f2"),
        "results",
        40,
        1,
    )
    assert row["run_id"] == "run_a"


def test_reregistering_without_a_run_id_keeps_the_existing_owner(tmp_path):
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 10, 2)
        ws.store.insert_run("run_a", "fp1", "{}", 0)
        ws.store.register_artifact("art", str(tmp_path / "f"), "model", 1, False, run_id="run_a")
        ws.store.register_artifact("art", str(tmp_path / "f"), "model", 2, False)
        assert _artifact_row(ws, "art")["run_id"] == "run_a"


def test_repeated_registration_does_not_reset_the_prune_clock(tmp_path):
    """A regenerable artifact older than the retention window stays prunable after it is re-registered;
    INSERT OR REPLACE reset created_at to 'now' and made it immortal on every resume."""
    with _open_ws(tmp_path) as ws:
        ws.store.register_artifact("old", str(tmp_path / "old.html"), "report", 5, True)
        _age_artifact(ws, "old")
        ws.store.register_artifact("old", str(tmp_path / "old.html"), "report", 6, True)
        assert [r["artifact_id"] for r in ws.store.artifacts_for_prune(retention_days=30)] == ["old"]


def test_a_path_owned_by_another_artifact_is_an_error_and_changes_nothing(tmp_path):
    from sorethumb_ml.errors import StoreError

    shared = str(tmp_path / "shared.bin")
    with _open_ws(tmp_path) as ws:
        ws.store.register_artifact("first", shared, "model", 1, False)
        _age_artifact(ws, "first")
        with pytest.raises(StoreError, match="first"):
            ws.store.register_artifact("second", shared, "model", 2, False)

        assert _artifact_row(ws, "second") is None
        kept = _artifact_row(ws, "first")
        # The original registration is untouched (INSERT OR REPLACE deleted it).
        assert kept is not None
        assert kept["created_at"] == "2020-01-01T00:00:00Z"
        assert kept["byte_size"] == 1
        # The connection is still usable after the refusal.
        ws.store.register_artifact("third", str(tmp_path / "other.bin"), "model", 3, False)
        assert _artifact_row(ws, "third") is not None


def test_a_path_can_change_hands_only_by_deleting_the_old_registration(tmp_path):
    shared = str(tmp_path / "shared.bin")
    with _open_ws(tmp_path) as ws:
        ws.store.register_artifact("first", shared, "model", 1, False)
        ws.store.delete_artifact("first")
        ws.store.register_artifact("second", shared, "model", 2, False)
        assert _artifact_row(ws, "first") is None
        assert _artifact_row(ws, "second")["path"] == shared


def test_a_forced_rerun_keeps_every_artifact_age(tmp_path):
    """End to end through the real writers (plan, models, results): re-running the same run with
    force=True re-registers every artifact id; none may lose its age or its row."""
    from sorethumb_ml import Workspace
    from sorethumb_ml._pipeline import run_detection
    from tests.factories.configs import make_config
    from tests.factories.frames import write_leading_anomaly_csv

    csv = write_leading_anomaly_csv(tmp_path / "d.csv")
    cfg = make_config(csv, tmp_path / "ws", contamination=0.05, combination="composite")
    run_detection(cfg, no_report=True)

    with Workspace.open(tmp_path / "ws") as ws:
        ids = [
            r["artifact_id"] for r in ws.store._conn.execute("SELECT artifact_id FROM artifact").fetchall()
        ]
        assert any(i.endswith("_plan") for i in ids)
        assert len(ids) >= 4
        ws.store._conn.execute("UPDATE artifact SET created_at='2020-01-01T00:00:00Z'")
        ws.store._conn.commit()

    run_detection(cfg, no_report=True, force=True)

    with Workspace.open(tmp_path / "ws") as ws:
        rows = ws.store._conn.execute("SELECT artifact_id, created_at FROM artifact").fetchall()
    assert sorted(r["artifact_id"] for r in rows) == sorted(ids)
    assert {r["created_at"] for r in rows} == {"2020-01-01T00:00:00Z"}
