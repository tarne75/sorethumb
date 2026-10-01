"""Database/workspace schema contract: Workspace.init/open create and reopen
the expected SQLite database and directory layout, every bundled migration
applies cleanly and is checksummed, a tampered checksum or a newer-than-known
schema is rejected, and the Store's dataset/run/group CRUD methods behave
correctly against that schema.

See tests/integration/test_store_workflows.py for retention/pruning and
score-forward-adjacent workflows built on top of this schema.
"""

from __future__ import annotations

import json
import sqlite3

import polars as pl
import pytest

from sorethumb_ml.errors import StoreError
from sorethumb_ml.store.db import Store, _iter_sql_statements
from sorethumb_ml.store.workspace import Workspace, make_group_key
from tests.factories.workspaces import open_workspace as _open_ws

pytestmark = pytest.mark.contract


def _bundled_versions() -> list[int]:
    """Versions of the shipped migration files, read from the package itself, so a new
    migration does not require editing every test that asserts "all migrations applied"."""
    from importlib.resources import files

    return sorted(
        int(entry.name[:3])
        for entry in files("sorethumb_ml.store.migrations").iterdir()
        if entry.name.endswith(".sql") and entry.name[:3].isdigit()
    )


_BUNDLED_VERSIONS = _bundled_versions()

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


def test_workspace_property_accessors(tmp_path):
    with _open_ws(tmp_path) as ws:
        assert ws.root == (tmp_path / "ws").resolve()
        assert ws.store is not None
        assert ws.db_path().name == "sorethumb.db"


def test_workspace_directory_methods(tmp_path):
    with _open_ws(tmp_path) as ws:
        assert ws.models_dir("run1", "grp1").is_dir()
        assert ws.features_dir("run1", "grp1").is_dir()
        assert ws.logs_dir().is_dir()
        assert ws.tmp_dir().is_dir()


# ---------------------------------------------------------------------------
# Store: migrations and basic ops
# ---------------------------------------------------------------------------


def test_store_migrations_applied(tmp_path):
    db = tmp_path / "test.db"
    with Store(db) as store:
        rows = store._conn.execute("SELECT version FROM schema_migration").fetchall()
        assert any(r[0] == 1 for r in rows)


def test_iter_sql_statements_ignores_semicolons_in_comments():
    sql = (
        "-- a comment with a semicolon; and more prose after it\n"
        "CREATE TABLE t (a INT);\n"
        "-- another; comment\n"
        "CREATE INDEX ix ON t(a);\n"
        "INSERT INTO schema_migration (version) VALUES (9);\n"
    )
    stmts = list(_iter_sql_statements(sql))
    assert stmts == ["CREATE TABLE t (a INT)", "CREATE INDEX ix ON t(a)"]


def test_all_bundled_migrations_apply_cleanly(tmp_path):
    """Every shipped migration file applies without error and is recorded."""
    with Store(tmp_path / "m.db") as store:
        versions = {r[0] for r in store._conn.execute("SELECT version FROM schema_migration")}
    assert versions == set(_BUNDLED_VERSIONS)


def test_migration_003_adds_artifact_run_id_column(tmp_path):
    with Store(tmp_path / "m.db") as store:
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(artifact)")}
    assert "run_id" in cols


def test_migration_005_adds_snapshot_table_and_column(tmp_path):
    with Store(tmp_path / "m.db") as store:
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(dataset)")}
        tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "snapshot_fp" in cols
    assert "dataset_snapshot" in tables


def test_migration_006_adds_period_execution_table(tmp_path):
    with Store(tmp_path / "m.db") as store:
        tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(period_execution)")}
    assert "period_execution" in tables
    assert {
        "dataset_fp",
        "period_label",
        "config_hash",
        "run_id",
        "group_count",
        "failed_count",
        "complete",
    } <= cols


# --- query plans and required indexes (P2-3) ---------------------------------------------

# index name -> (table, ordered columns). Dropping or reshaping one of these makes a
# Store query scan a table that grows with the workspace's history.
_REQUIRED_INDEXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "idx_model_run_group": ("model", ("run_id", "group_key")),
    "idx_run_started_at": ("run", ("started_at",)),
    "idx_run_status_started_at": ("run", ("status", "started_at")),
    "idx_run_source": ("run", ("source_run_id",)),
    "idx_artifact_run_id": ("artifact", ("run_id",)),
    "idx_dataset_snapshot_fp": ("dataset_snapshot", ("snapshot_fp",)),
    "idx_totals_period": ("totals", ("dataset_fp", "period_label", "config_hash", "group_key")),
    "idx_period_execution_last_complete": (
        "period_execution",
        ("dataset_fp", "config_hash", "complete", "period_label"),
    ),
}


def _index_columns(conn, name: str) -> tuple[str, ...]:
    return tuple(
        r["name"] for r in sorted(conn.execute(f"PRAGMA index_info({name})"), key=lambda r: r["seqno"])
    )


def test_required_indexes_exist_with_the_expected_columns(tmp_path):
    with Store(tmp_path / "m.db") as store:
        conn = store._conn
        found = {
            r["name"]: r["tbl_name"]
            for r in conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'")
        }
        for name, (table, columns) in _REQUIRED_INDEXES.items():
            assert found.get(name) == table, f"missing index {name} on {table}"
            assert _index_columns(conn, name) == columns, f"index {name} has the wrong columns"
        assert "idx_period_execution_dataset" not in found  # superseded by ..._last_complete


# (query, parameter count, table that must not be full-scanned). Only queries on tables that
# grow with history are listed; the artifact prune's regenerable branch is a documented,
# deliberate scan (see migration 008).
_PLANNED_QUERIES: dict[str, tuple[str, int, str, str | None]] = {
    "models_for_run_group": (
        "SELECT * FROM model WHERE run_id=? AND group_key=?",
        2,
        "model",
        "idx_model_run_group",
    ),
    "runs_list": (
        "SELECT * FROM run ORDER BY started_at DESC LIMIT ?",
        1,
        "run",
        "idx_run_started_at",
    ),
    "run_by_id": (
        "SELECT * FROM run WHERE run_id=?",
        1,
        "run",
        None,
    ),
    "run_group_for_run": (
        "SELECT * FROM run_group WHERE run_id=?",
        1,
        "run_group",
        None,
    ),
    "run_group_one": (
        "SELECT status FROM run_group WHERE run_id=? AND group_key=?",
        2,
        "run_group",
        None,
    ),
    "failed_runs_for_prune": (
        "SELECT * FROM run WHERE status='failed' AND julianday('now') - julianday(started_at) > ?",
        1,
        "run",
        "idx_run_status_started_at",
    ),
    "artifacts_of_failed_runs": (
        "SELECT a.* FROM artifact a JOIN run r ON a.run_id = r.run_id WHERE r.run_id = ?",
        1,
        "artifact",
        None,
    ),
    "artifact_path_conflict": (
        "SELECT artifact_id FROM artifact WHERE path=? AND artifact_id != ?",
        2,
        "artifact",
        None,
    ),
    "period_complete": (
        "SELECT complete FROM period_execution WHERE dataset_fp=? AND period_label=? AND config_hash=?",
        3,
        "period_execution",
        None,
    ),
    "last_complete_period": (
        "SELECT MAX(period_label) FROM period_execution WHERE dataset_fp=? AND config_hash=? AND complete=1",
        2,
        "period_execution",
        "idx_period_execution_last_complete",
    ),
    "completed_group_keys": (
        "SELECT DISTINCT group_key FROM totals WHERE dataset_fp=? AND period_label=? AND config_hash=?",
        3,
        "totals",
        "idx_totals_period",
    ),
    "totals_for_periods": (
        "SELECT * FROM totals WHERE dataset_fp=? AND config_hash=? AND period_label IN (?, ?)",
        4,
        "totals",
        "idx_totals_period",
    ),
    "other_config_hashes": (
        "SELECT DISTINCT config_hash FROM totals WHERE dataset_fp=? AND period_label IN (?, ?) AND config_hash != ?",
        4,
        "totals",
        "idx_totals_period",
    ),
    "delete_totals_for_period": (
        "DELETE FROM totals WHERE dataset_fp=? AND period_label=? AND config_hash=?",
        3,
        "totals",
        "idx_totals_period",
    ),
}


@pytest.mark.parametrize("name", sorted(_PLANNED_QUERIES))
def test_store_queries_do_not_scan_growing_tables(tmp_path, name):
    sql, n_params, table, required_index = _PLANNED_QUERIES[name]
    with Store(tmp_path / "m.db") as store:
        plan = [r["detail"] for r in store._conn.execute("EXPLAIN QUERY PLAN " + sql, ["x"] * n_params)]
    # "SCAN t USING INDEX ..." is an ordered index walk; a bare "SCAN t" reads the whole table.
    scans = [d for d in plan if d.split()[:2] == ["SCAN", table] and "USING" not in d]
    assert not scans, f"{name} full-scans {table}: {plan}"
    if required_index is not None:
        assert any(required_index in d for d in plan), f"{name} should use {required_index}: {plan}"
    # The runs listing sorts by an indexed column, so no sort step either.
    if name == "runs_list":
        assert not any("TEMP B-TREE" in d for d in plan), plan


def test_the_stores_real_read_and_delete_statements_have_no_unexpected_full_scans(tmp_path):
    """The table above copies SQL; this one captures what the Store methods actually run
    (sqlite's trace callback), so a new or changed query is checked without editing a list."""
    captured: list[str] = []
    with Store(tmp_path / "m.db") as store:
        store._conn.set_trace_callback(captured.append)
        store.dataset_snapshots("fp")
        store.list_runs(10)
        store.get_run("r")
        store.run_status("r")
        store.group_status("r", "g")
        store.get_run_group("r", "g")
        store.completed_groups("r")
        store.all_run_groups("r")
        store.models_for_run_group("r", "g")
        store.calibrator_for_model("m")
        store.artifacts_for_prune(30)
        store.period_is_complete("fp", "p", "c")
        store.last_complete_period_label("fp", "c")
        store.completed_group_keys("fp", "p", "c")
        store.totals_for_periods("fp", ["p1", "p2"], "c", ["g1"])
        store.totals_for_periods("fp", ["p1", "p2"], "c")
        store.other_config_hashes_for_periods("fp", ["p1", "p2"], "c")
        store.delete_totals_for_period("fp", "p", "c")
        store._conn.set_trace_callback(None)

        statements = [
            q
            for q in captured
            if q.lstrip().upper().startswith(("SELECT", "DELETE")) and "EXPLAIN" not in q.upper()
        ]
        assert len(statements) >= 15, statements  # the capture itself works
        # The artifact prune's regenerable branch is a documented, deliberate scan (migration 008)
        # and is the only bare table scan allowed.
        allowed = {"SCAN a"}
        offenders = []
        for sql in statements:
            for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql):
                detail = row["detail"]
                if detail.startswith("SCAN ") and "USING" not in detail and detail not in allowed:
                    offenders.append((sql, detail))
    assert not offenders, offenders


def test_a_workspace_at_the_previous_schema_gains_the_new_indexes_on_open(tmp_path):
    """The indexes arrive through a numbered migration, not by editing an applied one: a
    workspace already at migration 007 picks them up (and keeps its data) when reopened."""
    db = tmp_path / "old.db"
    with Store(db) as store:
        store._conn.execute("DROP INDEX idx_model_run_group")
        store._conn.execute("DROP INDEX idx_totals_period")
        store._conn.execute("DROP INDEX idx_run_started_at")
        store._conn.execute("DROP INDEX idx_run_status_started_at")
        store._conn.execute("DROP INDEX idx_period_execution_last_complete")
        store._conn.execute(
            "CREATE INDEX idx_period_execution_dataset ON period_execution(dataset_fp, config_hash)"
        )
        store._conn.execute("DELETE FROM schema_migration WHERE version=8")
        store._conn.commit()
    with Store(db) as store:
        names = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        applied = {r[0] for r in store._conn.execute("SELECT version FROM schema_migration")}
    assert 8 in applied
    assert set(_REQUIRED_INDEXES) <= names
    assert "idx_period_execution_dataset" not in names


def test_execute_ddl_rejects_alter_table_identifier_starting_with_digit(tmp_path):
    """P3-6: _execute_ddl's table/column names now route through
    validate_identifier (store/identifiers.py) before being spliced into a
    PRAGMA string. _ALTER_ADD_COLUMN_RE's own \\w+ groups already exclude
    SQL syntax characters, but \\w+ still matches a leading digit, which
    validate_identifier's stricter ^[A-Za-z_][A-Za-z0-9_]*$ does not --
    exercising that extra check requires a statement bypassing the regex
    only at the character-class boundary, not a full injection payload."""
    store = Store(tmp_path / "test.db")
    with pytest.raises(StoreError, match="Invalid SQL table name"):
        store._execute_ddl("ALTER TABLE 1bad ADD COLUMN x TEXT")


def test_a_migration_failing_partway_rolls_back_and_is_safely_replayable(tmp_path, monkeypatch):
    """A migration's DDL runs inside BEGIN IMMEDIATE (see _apply_one_migration),
    so a statement failing partway through must roll back its own DDL and
    leave no partial version row -- and a fresh, unpatched open afterward
    must still apply every migration cleanly, exactly once."""
    db = tmp_path / "test.db"
    real_execute_ddl = Store._execute_ddl
    call_count = 0

    def _flaky_execute_ddl(self, stmt: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # fail partway through migration 001's statements
            raise sqlite3.OperationalError("simulated failure mid-migration")
        real_execute_ddl(self, stmt)

    monkeypatch.setattr(Store, "_execute_ddl", _flaky_execute_ddl)
    with pytest.raises(sqlite3.OperationalError, match="simulated failure"):
        Store(db)
    monkeypatch.undo()

    # The failed attempt must not have recorded version 1 as applied, and
    # must not have left the DB in a state a plain reopen can't handle.
    with Store(db) as store:
        versions = {r[0] for r in store._conn.execute("SELECT version FROM schema_migration")}
    assert versions == set(_BUNDLED_VERSIONS), "a clean reopen must fully migrate after a rolled-back failure"


def test_store_second_open_no_duplicate_migration(tmp_path):
    db = tmp_path / "test.db"
    Store(db).close()
    with Store(db) as store:
        rows = store._conn.execute("SELECT version FROM schema_migration ORDER BY version").fetchall()
        versions = [r[0] for r in rows]
        assert len(versions) == len(set(versions)), "duplicate migration versions"


# ---------------------------------------------------------------------------
# Store hardening: busy_timeout, migration checksums, schema ceiling
# ---------------------------------------------------------------------------


def test_busy_timeout_is_set(tmp_path):
    with Store(tmp_path / "m.db") as store:
        (timeout_ms,) = store._conn.execute("PRAGMA busy_timeout").fetchone()
    assert timeout_ms > 0, "absent busy_timeout: a concurrent writer fails instantly instead of waiting"


def test_concurrent_opens_all_succeed_and_migrate_once(tmp_path):
    """Several threads racing Store(db_path) on a fresh workspace must all
    succeed, and every migration must be recorded exactly once -- this is
    what busy_timeout plus the lock-then-recheck in _apply_one_migration are
    for."""
    import threading

    db = tmp_path / "race.db"
    errors: list[Exception] = []

    def _open() -> None:
        try:
            Store(db).close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_open) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent Store() opens raised: {errors}"
    with Store(db) as store:
        versions = [
            r[0] for r in store._conn.execute("SELECT version FROM schema_migration ORDER BY version")
        ]
    assert versions == _BUNDLED_VERSIONS


def test_migrations_record_a_checksum(tmp_path):
    with Store(tmp_path / "m.db") as store:
        rows = store._conn.execute("SELECT version, checksum FROM schema_migration").fetchall()
    assert rows
    for version, checksum in rows:
        assert checksum, f"migration {version} recorded no checksum"


def test_tampered_checksum_raises_on_reopen(tmp_path):
    db = tmp_path / "m.db"
    Store(db).close()

    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("UPDATE schema_migration SET checksum='not-the-real-checksum' WHERE version=1")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError, match="checksum"):
        Store(db)


def test_schema_ceiling_rejects_newer_workspace(tmp_path):
    """A workspace already migrated past what this code knows must not be
    silently adopted -- an older client writing against an unknown schema
    risks corrupting it."""
    db = tmp_path / "m.db"
    Store(db).close()

    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO schema_migration (version, checksum) VALUES (999, 'x')")
    conn.commit()
    conn.close()

    with pytest.raises(StoreError, match="newer than this sorethumb"):
        Store(db)


def test_alter_add_column_replay_is_idempotent(tmp_path):
    """Simulates a partial-crash retry: the column from an ALTER TABLE ADD
    COLUMN migration is already there, but its schema_migration row is
    missing. Re-applying must skip the ALTER, not raise "duplicate column
    name"."""
    db = tmp_path / "m.db"
    with Store(db) as store:
        store._conn.execute("DELETE FROM schema_migration WHERE version=3")
        store._conn.commit()

        files = {v: (sql, checksum) for v, sql, checksum in store._discover_migration_files()}
        sql, checksum = files[3]
        store._apply_one_migration(3, sql, checksum)  # must not raise

        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(artifact)")}
        assert "run_id" in cols
        row = store._conn.execute("SELECT version FROM schema_migration WHERE version=3").fetchone()
        assert row is not None


def test_store_dataset_upsert(tmp_path):
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "http://example.com", "sfp", "cfp", 1000, 10)
        s.upsert_dataset("fp1", "http://example.com", "sfp2", "cfp2", 2000, 12)
        row = s._conn.execute("SELECT * FROM dataset WHERE dataset_fp='fp1'").fetchone()
        assert row["n_rows"] == 2000  # updated


def test_store_dataset_upsert_records_each_snapshot(tmp_path):
    """A stable dataset_fp accumulates one dataset_snapshot row per content/schema version."""
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("sales", "file:///d.parquet", "schemaA", "contentA", 1000, 10, snapshot_fp="snapA")
        s.upsert_dataset("sales", "file:///d.parquet", "schemaA", "contentB", 1400, 10, snapshot_fp="snapB")
        s.upsert_dataset("sales", "file:///d.parquet", "schemaA", "contentB", 1450, 10, snapshot_fp="snapB")

        # one logical dataset row, pointing at the most recent snapshot
        datasets = s._conn.execute("SELECT dataset_fp, snapshot_fp, n_rows FROM dataset").fetchall()
        assert [tuple(r) for r in datasets] == [("sales", "snapB", 1450)]

        snaps = s.dataset_snapshots("sales")
        assert [(x["snapshot_fp"], x["n_rows"]) for x in snaps] == [("snapA", 1000), ("snapB", 1450)]


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


def test_run_group_warnings_json_round_trips(tmp_path):
    """P2-4: group-level warnings (e.g. ZeroAnomalyWarning) must be
    persisted, not only returned in the in-memory RunResult -- otherwise
    render_report_for_run, which always renders from persisted state, could
    never show them."""
    with _open_ws(tmp_path) as ws:
        s = ws.store
        s.upsert_dataset("fp1", "uri", "sfp", "cfp", 100, 5)
        s.insert_run("run1", "fp1", "{}", 0)
        gk = make_group_key({"country": "US"})
        payload = json.dumps(["Three-way intersection flagged zero rows: ..."])
        s.upsert_run_group("run1", gk, '{"country":"US"}', "US", status="complete", warnings_json=payload)
        row = s.get_run_group("run1", gk)
        assert row is not None
        assert json.loads(row["warnings_json"]) == ["Three-way intersection flagged zero rows: ..."]

        all_rows = s.all_run_groups("run1")
        assert any(json.loads(r["warnings_json"] or "[]") for r in all_rows)


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
    from sorethumb_ml.store.results import read_results, write_results

    with _open_ws(tmp_path) as ws:
        df = pl.DataFrame({"row_id": [0, 1, 2], "score": [0.1, 0.9, 0.5], "flagged": [False, True, False]})
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 3, 3)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        path = write_results(ws, "run1", "gk01", df)
        assert path.exists()
        assert path.name == "anomalies.parquet"
        df2 = read_results(ws, "run1", "gk01")
        assert df2 is not None
        assert len(df2) == 3
        assert list(df2.columns) == list(df.columns)


def test_write_results_registers_artifact(tmp_path):
    from sorethumb_ml.store.results import write_results

    with _open_ws(tmp_path) as ws:
        df = pl.DataFrame({"row_id": [0], "score": [0.5]})
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 1, 2)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        write_results(ws, "run1", "gk01", df)
        arts = ws.store._conn.execute("SELECT * FROM artifact").fetchall()
        assert len(arts) >= 1


def test_read_results_missing_returns_none(tmp_path):
    from sorethumb_ml.store.results import read_results

    with _open_ws(tmp_path) as ws:
        result = read_results(ws, "norun", "nogroup")
        assert result is None


def test_write_results_leaves_existing_file_on_failure(tmp_path, monkeypatch):
    """write_results goes through atomic_write (temp file + rename); a failure
    during the rename must leave the previous Parquet file (and any prior
    reader of it) untouched, not a half-written replacement."""
    from sorethumb_ml import _atomic
    from sorethumb_ml.store.results import write_results

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 3, 3)
        ws.store.insert_run("run1", "fp1", "{}", 0)

        df1 = pl.DataFrame({"row_id": [0, 1, 2]})
        path = write_results(ws, "run1", "gk01", df1)
        original_bytes = path.read_bytes()

        def _boom(*_a):
            raise OSError("disk full")

        monkeypatch.setattr(_atomic.os, "replace", _boom)
        df2 = pl.DataFrame({"row_id": [9, 10, 11, 12]})
        with pytest.raises(OSError, match="disk full"):
            write_results(ws, "run1", "gk01", df2)
        monkeypatch.undo()

        assert path.read_bytes() == original_bytes  # untouched, not truncated/replaced
        assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize(
    ("df", "match"),
    [
        (pl.DataFrame({"score": [0.1, 0.2]}), "missing"),
        (pl.DataFrame({"row_id": [0, None]}), "null"),
        (pl.DataFrame({"row_id": [0, 0]}), "duplicate"),
    ],
    ids=["missing_row_id", "null_row_id", "duplicate_row_id"],
)
def test_write_results_rejects_an_unsafe_results_frame(tmp_path, df, match):
    """P2-6: row_id is the only column every downstream reader relies on to
    join a result row back to its source; write_results must fail closed
    before writing (and registering) a Parquet file it would be dangerous to
    trust."""
    from sorethumb_ml.store.results import write_results

    with _open_ws(tmp_path) as ws:
        ws.store.upsert_dataset("fp1", "uri", "sfp", "cfp", 2, 2)
        ws.store.insert_run("run1", "fp1", "{}", 0)
        with pytest.raises(StoreError, match=match):
            write_results(ws, "run1", "gk01", df)
        assert ws.store._conn.execute("SELECT * FROM artifact").fetchall() == []
