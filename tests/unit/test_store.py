"""Unit tests for M5: workspace, store, model persistence, score-forward."""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb.errors import ModelSchemaDriftError, ModelSchemaDriftWarning, StoreError
from sorethumb.store.db import Store
from sorethumb.store.identifiers import validate_identifier
from sorethumb.store.models import load_model, save_model, score_with_existing
from sorethumb.store.results import read_results, write_results
from sorethumb.store.workspace import Workspace, make_group_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_ws(tmp_path: Path) -> Workspace:
    return Workspace.init(tmp_path / "ws")


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
# make_group_key
# ---------------------------------------------------------------------------


def test_group_key_is_hex():
    k = make_group_key({"country": "US", "cat": "A"})
    assert all(c in "0123456789abcdef" for c in k)


def test_group_key_length():
    k = make_group_key({"a": "1"})
    assert len(k) == 16


def test_group_key_stable():
    a = make_group_key({"country": "US", "cat": "A"})
    b = make_group_key({"cat": "A", "country": "US"})  # key order doesn't matter
    assert a == b


def test_group_key_different_for_different_values():
    a = make_group_key({"country": "US"})
    b = make_group_key({"country": "AU"})
    assert a != b


def test_group_key_special_chars_round_trip():
    # Special chars: quotes, semicolon, slash, newline
    group = {"val": 'it\'s a "test"; /path\nnewline'}
    k = make_group_key(group)
    assert len(k) == 16
    # Same key reconstructed from the same values
    assert k == make_group_key(group)


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_valid_identifier():
    assert validate_identifier("my_table") == "my_table"
    assert validate_identifier("Column1") == "Column1"
    assert validate_identifier("_private") == "_private"


def test_invalid_identifier_raises():
    with pytest.raises(StoreError, match="Invalid SQL"):
        validate_identifier("1bad")


def test_identifier_rejects_spaces():
    with pytest.raises(StoreError):
        validate_identifier("my column")


def test_identifier_rejects_dash():
    with pytest.raises(StoreError):
        validate_identifier("my-col")


def test_identifier_rejects_dot():
    with pytest.raises(StoreError):
        validate_identifier("schema.table")


# ---------------------------------------------------------------------------
# Workspace.init / open
# ---------------------------------------------------------------------------


def test_workspace_init_creates_db(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    assert (tmp_path / "ws" / "sorethumb.db").exists()
    ws.close()


def test_workspace_init_creates_subdirs(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    root = tmp_path / "ws"
    for sub in ("cache", "models", "results", "reports", "logs", "tmp"):
        assert (root / sub).is_dir()
    ws.close()


def test_workspace_init_idempotent(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    ws.close()
    ws2 = Workspace.init(tmp_path / "ws")  # must not raise
    ws2.close()


def test_workspace_open_existing(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    ws.close()
    ws2 = Workspace.open(tmp_path / "ws")
    ws2.close()


def test_workspace_open_missing_dir_raises(tmp_path):
    with pytest.raises(StoreError, match="not exist"):
        Workspace.open(tmp_path / "nonexistent")


def test_workspace_open_non_workspace_raises(tmp_path):
    (tmp_path / "notws").mkdir()
    with pytest.raises(StoreError, match="not a sorethumb workspace"):
        Workspace.open(tmp_path / "notws")


def test_workspace_context_manager(tmp_path):
    with Workspace.init(tmp_path / "ws") as ws:
        assert ws.root.is_dir()


# ---------------------------------------------------------------------------
# Store: migrations and basic ops
# ---------------------------------------------------------------------------


def test_store_migrations_applied(tmp_path):
    db = tmp_path / "test.db"
    with Store(db) as store:
        rows = store._conn.execute("SELECT version FROM schema_migration").fetchall()
        assert any(r[0] == 1 for r in rows)


def test_store_second_open_no_duplicate_migration(tmp_path):
    db = tmp_path / "test.db"
    Store(db).close()
    with Store(db) as store:
        rows = store._conn.execute("SELECT version FROM schema_migration ORDER BY version").fetchall()
        versions = [r[0] for r in rows]
        assert len(versions) == len(set(versions)), "duplicate migration versions"


def test_store_dataset_upsert(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "http://example.com", "sfp", "cfp", 1000, 10)
        s.upsert_dataset("fp1", "http://example.com", "sfp2", "cfp2", 2000, 12)
        row = s._conn.execute("SELECT * FROM dataset WHERE dataset_fp='fp1'").fetchone()
        assert row["n_rows"] == 2000  # updated


def test_store_run_insert_and_status(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 42)
        assert s.run_status("run1") == "running"
        s.mark_run_complete("run1")
        assert s.run_status("run1") == "complete"


def test_store_run_failed(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 0)
        s.mark_run_failed("run1", "OOM error")
        assert s.run_status("run1") == "failed"


def test_store_run_insert_idempotent(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 42)
        s.insert_run("run1", "fp1", "{}", 42)  # second insert must be no-op
        rows = s._conn.execute("SELECT COUNT(*) AS n FROM run WHERE run_id='run1'").fetchone()
        assert rows["n"] == 1


def test_store_run_group_upsert_no_duplicate_rows(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"country": "US"})
        s.upsert_run_group("run1", gk, '{"country":"US"}', "US", status="running")
        s.upsert_run_group(
            "run1", gk, '{"country":"US"}', "US", status="complete", record_count=50, anomaly_count=5
        )
        rows = s._conn.execute(
            "SELECT COUNT(*) AS n FROM run_group WHERE run_id='run1' AND group_key=?", (gk,)
        ).fetchone()
        assert rows["n"] == 1
        row = s._conn.execute(
            "SELECT status, anomaly_count FROM run_group WHERE run_id='run1' AND group_key=?", (gk,)
        ).fetchone()
        assert row["status"] == "complete"
        assert row["anomaly_count"] == 5


def test_store_completed_groups(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 0)
        gk_a = make_group_key({"g": "A"})
        gk_b = make_group_key({"g": "B"})
        s.upsert_run_group("run1", gk_a, '{"g":"A"}', "A", status="complete")
        s.upsert_run_group("run1", gk_b, '{"g":"B"}', "B", status="running")
        completed = s.completed_groups("run1")
        assert gk_a in completed
        assert gk_b not in completed


# ---------------------------------------------------------------------------
# write_results / read_results
# ---------------------------------------------------------------------------


def test_write_read_results_roundtrip(tmp_path):
    with _open_ws(tmp_path) as ws:
        df = pl.DataFrame({"row_id": [0, 1, 2], "score": [0.1, 0.9, 0.5], "flagged": [False, True, False]})
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 3, 3)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        path = write_results(ws, "run1", "gk01", df)
        assert path.exists()
        df2 = read_results(ws, "run1", "gk01")
        assert df2 is not None
        assert len(df2) == 3
        assert list(df2.columns) == list(df.columns)


def test_write_results_registers_artifact(tmp_path):
    with _open_ws(tmp_path) as ws:
        df = pl.DataFrame({"row_id": [0], "score": [0.5]})
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 1, 2)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        write_results(ws, "run1", "gk01", df)
        arts = ws.store._conn.execute("SELECT * FROM artifact").fetchall()
        assert len(arts) >= 1


def test_read_results_missing_returns_none(tmp_path):
    with _open_ws(tmp_path) as ws:
        result = read_results(ws, "norun", "nogroup")
        assert result is None


# ---------------------------------------------------------------------------
# Interrupted run: completed groups survive
# ---------------------------------------------------------------------------


def test_interrupted_run_completed_groups_survive(tmp_path):
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
# save_model / load_model
# ---------------------------------------------------------------------------


def test_save_load_model_roundtrip(tmp_path):
    det, X = _fit_detector()
    cal = _fitted_calibrator(det, X)
    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 4)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"g": "A"})
        model_id = save_model(ws, "run1", gk, det, cal, '{"plan":"json"}', "hash_abc", 100, 42)
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
        result = score_with_existing(ws, "run1", gk, X_train, schema_hash, ["isolation_forest"])

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
            score_with_existing(ws, "run1", gk, X, "new_hash", ["isolation_forest"], strict=True)


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
            result = score_with_existing(ws, "run1", gk, X, "new_hash", ["isolation_forest"], strict=False)

    assert result["drifted"]
    assert any(issubclass(x.category, ModelSchemaDriftWarning) for x in w)


def test_score_with_existing_missing_model_skips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))

    with _open_ws(tmp_path) as ws:
        result = score_with_existing(ws, "norun", "nogroup", X, "hash", ["isolation_forest"])

    assert result["scores"] == {}
    assert result["calibrated"] == {}
    assert not result["drifted"]


# ---------------------------------------------------------------------------
# Retention / pruning
# ---------------------------------------------------------------------------


def test_prune_dry_run_lists_eligible(tmp_path):
    with _open_ws(tmp_path) as ws:
        # Register a regenerable artifact with a very old created_at (simulate old file)
        ws.store._conn.execute(
            "INSERT INTO artifact (artifact_id, path, kind, byte_size, regenerable, created_at) "
            "VALUES (?, ?, 'cache', 0, 1, '2020-01-01T00:00:00Z')",
            ("art1", str(tmp_path / "old_file.parquet")),
        )
        ws.store._conn.commit()
        deleted = ws.prune(retention_days=1, dry_run=True)
        assert any("old_file.parquet" in p for p in deleted)


def test_prune_dry_run_does_not_delete(tmp_path):
    old_file = tmp_path / "ws" / "old_cache.parquet"
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
