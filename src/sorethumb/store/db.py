"""SQLite store — single connection, WAL mode, foreign keys, numbered migrations.

All writes go through parameterised statements. The only non-parameterised SQL
is in migration files, which are reviewed and committed as source.

One Store owns one connection. Nothing else in the library should open the database.

Concurrency: ``PRAGMA busy_timeout`` makes a second process's writer wait for
the first to finish instead of failing instantly with "database is locked" —
this matters most while migrating, so each migration runs under an explicit
``BEGIN IMMEDIATE`` (see ``_apply_one_migration``): that takes the write lock
up front, making the "is this version already applied?" re-check race-free
against a concurrent process that got there first. Note this also means
migration DDL genuinely rolls back on failure, unlike a bare ``with conn:``
around DDL — Python's sqlite3 module only auto-opens a transaction ahead of
DML (INSERT/UPDATE/DELETE), not DDL, so without an explicit BEGIN each
CREATE/ALTER/DROP would commit individually as it runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.resources
import logging
import re
import sqlite3
import sys
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from sorethumb.errors import StoreError

logger = logging.getLogger(__name__)

_MIGRATIONS_PACKAGE = "sorethumb.store.migrations"

# How long a writer waits for another connection's lock before raising
# "database is locked". Absent, concurrent runs against the same workspace
# fail instantly rather than simply queuing.
_BUSY_TIMEOUT_MS = 30_000

# SQLite has no ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``; this is the one
# DDL shape in the bundled migrations that a plain ``IF NOT EXISTS`` can't
# cover, so a replayed migration would otherwise fail with "duplicate column
# name" on retry. Matched against each statement in ``_execute_ddl``.
_ALTER_ADD_COLUMN_RE = re.compile(r"^ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)", re.IGNORECASE)

_UPSERT_TOTAL_SQL = """
    INSERT INTO totals
        (dataset_fp, group_key, period_label, config_hash,
         anomaly_count, population, rate, run_id, computed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(dataset_fp, group_key, period_label, config_hash) DO UPDATE SET
        anomaly_count = excluded.anomaly_count,
        population    = excluded.population,
        rate          = excluded.rate,
        run_id        = excluded.run_id,
        computed_at   = excluded.computed_at
"""


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _config_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode()).hexdigest()[:32]


def _iter_sql_statements(sql: str) -> Generator[str, None, None]:
    """Yield non-empty SQL statements from a migration file.

    Full-line ``--`` comments are dropped first (so a semicolon inside prose in a
    comment cannot split a statement), then the remainder is split on ';' and any
    ``INSERT INTO schema_migration`` statement is skipped — the migration runner
    records the version itself, atomically with the rest of the migration.
    """
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    for raw in code.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        if stmt.lower().startswith("insert into schema_migration"):
            continue
        yield stmt


class Store:
    """Owns the SQLite connection for a single workspace."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the database at db_path and apply pending migrations."""
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Set before anything else: switching journal mode can itself need the
        # lock, and this is what turns a concurrent run's "database is locked"
        # into a bounded wait instead of an instant failure.
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._apply_migrations()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        """Support context-manager usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close on exit."""
        self.close()

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _apply_migrations(self) -> None:
        self._bootstrap_schema_migration_table()

        migration_files = self._discover_migration_files()
        self._check_schema_ceiling(migration_files)

        for version, sql, checksum in migration_files:
            self._apply_one_migration(version, sql, checksum)

    def _bootstrap_schema_migration_table(self) -> None:
        """Create ``schema_migration`` and backfill its ``checksum`` column.

        (Added to a workspace whose table predates it -- those rows keep
        ``checksum=NULL``, nothing to verify them against.) Runs under the
        same lock-then-recheck discipline as ``_apply_one_migration``.

        This used to run as two bare statements with no explicit transaction
        around them -- harmless for one process, but under real concurrency
        (several threads/processes racing to open the same brand-new
        workspace) that gap could still raise "database is locked" even with
        busy_timeout set, because no lock was held across the two statements
        to serialise the racers. ``BEGIN IMMEDIATE`` closes that: every
        opener queues through this bootstrap step one at a time, and
        busy_timeout governs how long the others wait rather than failing.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migration "
                "(version INTEGER PRIMARY KEY, "
                "applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')), "
                "checksum TEXT)"
            )
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(schema_migration)")}
            if "checksum" not in cols:
                self._conn.execute("ALTER TABLE schema_migration ADD COLUMN checksum TEXT")
            self._conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ROLLBACK")
            raise

    def _discover_migration_files(self) -> list[tuple[int, str, str]]:
        """Return (version, sql, sha256-of-sql) for every bundled migration, sorted."""
        migration_files: list[tuple[int, str, str]] = []
        try:
            pkg = importlib.resources.files(_MIGRATIONS_PACKAGE)
            for entry in pkg.iterdir():
                name = entry.name
                if name.endswith(".sql") and name[:3].isdigit():
                    version = int(name[:3])
                    sql = entry.read_text(encoding="utf-8")
                    checksum = hashlib.sha256(sql.encode()).hexdigest()
                    migration_files.append((version, sql, checksum))
        except (FileNotFoundError, AttributeError, TypeError) as exc:
            raise StoreError(f"Cannot load migration files from {_MIGRATIONS_PACKAGE}: {exc}") from exc
        migration_files.sort(key=lambda x: x[0])
        return migration_files

    def _check_schema_ceiling(self, migration_files: list[tuple[int, str, str]]) -> None:
        """Refuse to open a workspace whose schema is newer than this code knows.

        A workspace migrated by a newer sorethumb has tables/columns this
        version has never heard of; writing against it with a stale
        understanding of the schema risks corrupting data rather than just
        under-using it, so this fails closed instead of limping on.
        """
        latest_known = migration_files[-1][0] if migration_files else 0
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_migration").fetchone()
        max_applied = row["v"] if row else None
        if max_applied is not None and max_applied > latest_known:
            raise StoreError(
                f"This workspace's schema is at migration {max_applied:03d}, newer than "
                f"this sorethumb installation understands (up to {latest_known:03d}). It "
                "was likely created or migrated by a newer sorethumb version. Upgrade "
                "sorethumb before opening this workspace."
            )

    def _apply_one_migration(self, version: int, sql: str, checksum: str) -> None:
        """Apply one migration file as a real atomic transaction, if not already applied.

        ``BEGIN IMMEDIATE`` takes the write lock before the "already applied?"
        check, so that check is race-free against another process that is
        concurrently migrating this same database (``busy_timeout`` governs
        how long this waits for that lock rather than failing instantly).
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT checksum FROM schema_migration WHERE version=?", (version,)
            ).fetchone()
            if row is not None:
                self._conn.execute("ROLLBACK")
                self._verify_checksum(version, row["checksum"], checksum)
                return
            logger.info("Applying migration %03d.", version)
            for stmt in _iter_sql_statements(sql):
                self._execute_ddl(stmt)
            self._conn.execute(
                "INSERT INTO schema_migration (version, checksum) VALUES (?, ?)",
                (version, checksum),
            )
            self._conn.execute("COMMIT")
            logger.info("Migration %03d applied.", version)
        except BaseException:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ROLLBACK")
            raise

    def _verify_checksum(self, version: int, recorded: str | None, current: str) -> None:
        """Raise if a migration already on record doesn't match the bundled file.

        ``recorded is None`` means this workspace's version row predates
        checksum tracking — nothing to compare against, so that's not a
        mismatch.
        """
        if recorded is not None and recorded != current:
            raise StoreError(
                f"Migration {version:03d}'s checksum does not match what was recorded "
                "when it was applied to this workspace. The migration file bundled with "
                f"this sorethumb install has changed since (recorded={recorded[:12]}… "
                f"now={current[:12]}…); refusing to trust the schema."
            )

    def _execute_ddl(self, stmt: str) -> None:
        """Execute one migration statement, replay-safe.

        ``CREATE TABLE``/``CREATE INDEX`` in the migration files already say
        ``IF NOT EXISTS``. ``ALTER TABLE ... ADD COLUMN`` has no such clause in
        SQLite, so it is special-cased here: skip it if the column is already
        there instead of failing with "duplicate column name".
        """
        match = _ALTER_ADD_COLUMN_RE.match(stmt.strip())
        if match:
            table, column = match.group(1), match.group(2)
            existing_cols = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column in existing_cols:
                logger.info("Column %s.%s already present; skipping.", table, column)
                return
        self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------

    def upsert_dataset(
        self,
        dataset_fp: str,
        source_uri: str,
        schema_fingerprint: str,
        content_fingerprint: str,
        n_rows: int,
        n_cols: int,
        snapshot_fp: str | None = None,
    ) -> None:
        """Insert or update a dataset row and record the observed snapshot.

        *dataset_fp* is the stable logical id. *snapshot_fp* identifies the
        content+schema version seen on this call; the dataset row's
        schema/content/n_rows/n_cols columns track the latest snapshot, while
        ``dataset_snapshot`` keeps one row per version (``first_seen`` preserved).
        Defaults to ``content[:32]_schema[:16]`` when not supplied.
        """
        now = _now_utc()
        snap = snapshot_fp or f"{content_fingerprint[:32]}_{schema_fingerprint[:16]}"
        self._conn.execute(
            """
            INSERT INTO dataset
                (dataset_fp, source_uri, schema_fingerprint, content_fingerprint,
                 n_rows, n_cols, snapshot_fp, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_fp) DO UPDATE SET
                source_uri = excluded.source_uri,
                schema_fingerprint = excluded.schema_fingerprint,
                content_fingerprint = excluded.content_fingerprint,
                n_rows = excluded.n_rows,
                n_cols = excluded.n_cols,
                snapshot_fp = excluded.snapshot_fp,
                last_seen = excluded.last_seen
            """,
            (dataset_fp, source_uri, schema_fingerprint, content_fingerprint, n_rows, n_cols, snap, now, now),
        )
        self._conn.execute(
            """
            INSERT INTO dataset_snapshot
                (dataset_fp, snapshot_fp, schema_fingerprint, content_fingerprint,
                 n_rows, n_cols, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_fp, snapshot_fp) DO UPDATE SET
                n_rows = excluded.n_rows,
                n_cols = excluded.n_cols,
                last_seen = excluded.last_seen
            """,
            (dataset_fp, snap, schema_fingerprint, content_fingerprint, n_rows, n_cols, now, now),
        )
        self._conn.commit()

    def dataset_snapshots(self, dataset_fp: str) -> list[dict[str, Any]]:
        """All recorded snapshots for a logical dataset, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM dataset_snapshot WHERE dataset_fp=? ORDER BY first_seen, snapshot_fp",
            (dataset_fp,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def insert_run(
        self,
        run_id: str,
        dataset_fp: str,
        config_json: str,
        seed: int,
        library_version: str = "",
        python_version: str = "",
        source_run_id: str | None = None,
    ) -> None:
        """Record a new run in status 'running'.

        *source_run_id* is set for score-forward runs (``sorethumb score
        --from-run``) and NULL for ordinary fitted runs.
        """
        now = _now_utc()
        cfg_hash = _config_hash(config_json)
        lib_ver = library_version or "0.1.0"
        py_ver = (
            python_version or f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO run
                (run_id, dataset_fp, config_hash, config_json, seed,
                 library_version, python_version, started_at, status, source_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (run_id, dataset_fp, cfg_hash, config_json, seed, lib_ver, py_ver, now, source_run_id),
        )
        self._conn.commit()

    def mark_run_complete(self, run_id: str) -> None:
        """Transition run to complete."""
        self._conn.execute(
            "UPDATE run SET status='complete', ended_at=? WHERE run_id=?",
            (_now_utc(), run_id),
        )
        self._conn.commit()

    def mark_run_failed(self, run_id: str, error: str) -> None:
        """Transition run to failed."""
        self._conn.execute(
            "UPDATE run SET status='failed', ended_at=?, error_text=? WHERE run_id=?",
            (_now_utc(), error, run_id),
        )
        self._conn.commit()

    def run_status(self, run_id: str) -> str | None:
        """Return the run's status string, or None if the run does not exist."""
        row = self._conn.execute("SELECT status FROM run WHERE run_id=?", (run_id,)).fetchone()
        return str(row["status"]) if row else None

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent run rows ordered by started_at descending."""
        rows = self._conn.execute("SELECT * FROM run ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run row or None if not found."""
        row = self._conn.execute("SELECT * FROM run WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # run_group
    # ------------------------------------------------------------------

    def upsert_run_group(
        self,
        run_id: str,
        group_key: str,
        group_values_json: str,
        group_label: str,
        status: str = "running",
        record_count: int | None = None,
        anomaly_count: int | None = None,
        rate: float | None = None,
        timing_seconds: float | None = None,
        error: str | None = None,
    ) -> None:
        """Insert or replace a run_group row."""
        self._conn.execute(
            """
            INSERT INTO run_group
                (run_id, group_key, group_values_json, group_label, status,
                 record_count, anomaly_count, rate, timing_seconds, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, group_key) DO UPDATE SET
                status = excluded.status,
                record_count = excluded.record_count,
                anomaly_count = excluded.anomaly_count,
                rate = excluded.rate,
                timing_seconds = excluded.timing_seconds,
                error = excluded.error
            """,
            (
                run_id,
                group_key,
                group_values_json,
                group_label,
                status,
                record_count,
                anomaly_count,
                rate,
                timing_seconds,
                error,
            ),
        )
        self._conn.commit()

    def group_status(self, run_id: str, group_key: str) -> str | None:
        """Return the group's status, or None if not recorded."""
        row = self._conn.execute(
            "SELECT status FROM run_group WHERE run_id=? AND group_key=?",
            (run_id, group_key),
        ).fetchone()
        return str(row["status"]) if row else None

    def get_run_group(self, run_id: str, group_key: str) -> dict[str, Any] | None:
        """Return one run_group row, or None if not recorded."""
        row = self._conn.execute(
            "SELECT * FROM run_group WHERE run_id=? AND group_key=?",
            (run_id, group_key),
        ).fetchone()
        return dict(row) if row else None

    def completed_groups(self, run_id: str) -> list[str]:
        """Return group_key values that reached status='complete' for this run."""
        rows = self._conn.execute(
            "SELECT group_key FROM run_group WHERE run_id=? AND status='complete'",
            (run_id,),
        ).fetchall()
        return [str(r["group_key"]) for r in rows]

    def all_run_groups(self, run_id: str) -> list[dict[str, Any]]:
        """Return all run_group rows for a run."""
        rows = self._conn.execute("SELECT * FROM run_group WHERE run_id=?", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # model + calibrator
    # ------------------------------------------------------------------

    def upsert_model(
        self,
        model_id: str,
        run_id: str,
        group_key: str,
        detector_name: str,
        artifact_path: str,
        feature_schema_hash: str,
        train_row_count: int,
        params_json: str,
    ) -> None:
        """Insert or update a model record."""
        now = _now_utc()
        self._conn.execute(
            """
            INSERT INTO model
                (model_id, run_id, group_key, detector_name, artifact_path,
                 feature_schema_hash, train_row_count, params_json, fitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                artifact_path = excluded.artifact_path,
                feature_schema_hash = excluded.feature_schema_hash,
                train_row_count = excluded.train_row_count,
                params_json = excluded.params_json,
                fitted_at = excluded.fitted_at
            """,
            (
                model_id,
                run_id,
                group_key,
                detector_name,
                artifact_path,
                feature_schema_hash,
                train_row_count,
                params_json,
                now,
            ),
        )
        self._conn.commit()

    def upsert_calibrator(self, model_id: str, quantile_values_json: str) -> None:
        """Insert or update calibrator quantiles for a model."""
        self._conn.execute(
            """
            INSERT INTO calibrator (model_id, quantile_values_json)
            VALUES (?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                quantile_values_json = excluded.quantile_values_json
            """,
            (model_id, quantile_values_json),
        )
        self._conn.commit()

    def models_for_run_group(self, run_id: str, group_key: str) -> list[dict[str, Any]]:
        """Return model rows for a (run_id, group_key)."""
        rows = self._conn.execute(
            "SELECT * FROM model WHERE run_id=? AND group_key=?",
            (run_id, group_key),
        ).fetchall()
        return [dict(r) for r in rows]

    def calibrator_for_model(self, model_id: str) -> str | None:
        """Return quantile_values_json for a model, or None."""
        row = self._conn.execute(
            "SELECT quantile_values_json FROM calibrator WHERE model_id=?",
            (model_id,),
        ).fetchone()
        return str(row["quantile_values_json"]) if row else None

    # ------------------------------------------------------------------
    # artifact
    # ------------------------------------------------------------------

    def register_artifact(
        self,
        artifact_id: str,
        path: str,
        kind: str,
        byte_size: int,
        regenerable: bool,
        run_id: str | None = None,
    ) -> None:
        """Register a file artifact in the index.

        *run_id* is the run that produced the artifact; it lets the prune query
        join failed runs to their artifacts by equality rather than a path
        substring match. Pass it whenever the artifact belongs to a run.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO artifact (artifact_id, path, kind, byte_size, regenerable, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, path, kind, byte_size, 1 if regenerable else 0, run_id),
        )
        self._conn.commit()

    def artifacts_for_prune(self, retention_days: int) -> list[dict[str, Any]]:
        """Return artifact rows eligible for pruning.

        Regenerable artifacts older than retention_days, plus artifacts whose
        run has status='failed' and is older than retention_days.
        """
        rows = self._conn.execute(
            """
            SELECT a.*
            FROM artifact a
            WHERE a.regenerable = 1
              AND julianday('now') - julianday(a.created_at) > ?
            UNION
            SELECT a.*
            FROM artifact a
            JOIN run r ON a.run_id = r.run_id
            WHERE r.status = 'failed'
              AND julianday('now') - julianday(r.started_at) > ?
            """,
            (retention_days, retention_days),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_artifact(self, artifact_id: str) -> None:
        """Remove an artifact row from the index."""
        self._conn.execute("DELETE FROM artifact WHERE artifact_id=?", (artifact_id,))
        self._conn.commit()

    def vacuum(self) -> None:
        """Run SQLite VACUUM."""
        self._conn.execute("VACUUM")

    # ------------------------------------------------------------------
    # period
    # ------------------------------------------------------------------

    def upsert_period(
        self,
        dataset_fp: str,
        period_label: str,
        period_from: str,
        period_to: str,
    ) -> None:
        """Insert or replace a period row."""
        self._conn.execute(
            """
            INSERT INTO period (dataset_fp, period_label, period_from, period_to)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dataset_fp, period_label) DO UPDATE SET
                period_from = excluded.period_from,
                period_to   = excluded.period_to
            """,
            (dataset_fp, period_label, period_from, period_to),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # totals + period_execution
    # ------------------------------------------------------------------

    def upsert_total(
        self,
        dataset_fp: str,
        group_key: str,
        period_label: str,
        anomaly_count: int,
        population: int,
        rate: float | None,
        run_id: str,
        config_hash: str = "",
    ) -> None:
        """Insert or replace a totals row on its natural key (dataset, group, period, config)."""
        self._conn.execute(
            _UPSERT_TOTAL_SQL,
            (
                dataset_fp,
                group_key,
                period_label,
                config_hash,
                anomaly_count,
                population,
                rate,
                run_id,
                _now_utc(),
            ),
        )
        self._conn.commit()

    def record_period_completion(
        self,
        *,
        dataset_fp: str,
        period_label: str,
        period_from: str,
        period_to: str,
        config_hash: str,
        run_id: str,
        totals: list[tuple[str, int, int, float | None]],
        failed_count: int,
    ) -> None:
        """Write the period row, every group's totals row, and this attempt's completion record atomically.

        ``totals`` is ``(group_key, anomaly_count, population, rate)`` for every
        group this run has a value for (success, too-few-records, *and* resumed
        ``skipped`` groups -- re-upserting an already-recorded skipped group is
        a harmless no-op, and is what lets a period recover if a previous
        attempt crashed after ``run_group`` marked a group complete but before
        this method ever ran). ``complete`` is derived from ``failed_count``:
        a period with any failed group -- or with zero groups discovered -- is
        not complete and must be retried.

        The whole write is wrapped in ``BEGIN IMMEDIATE``/``COMMIT`` (see
        ``_apply_one_migration`` for the same pattern) so a crash between the
        period row and the completion row can never leave one without the
        other: a reader either sees the previous attempt's state in full, or
        this one's, never a mix.
        """
        complete = failed_count == 0 and len(totals) > 0
        now = _now_utc()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                """
                INSERT INTO period (dataset_fp, period_label, period_from, period_to)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dataset_fp, period_label) DO UPDATE SET
                    period_from = excluded.period_from,
                    period_to   = excluded.period_to
                """,
                (dataset_fp, period_label, period_from, period_to),
            )
            for group_key, anomaly_count, population, rate in totals:
                self._conn.execute(
                    _UPSERT_TOTAL_SQL,
                    (
                        dataset_fp,
                        group_key,
                        period_label,
                        config_hash,
                        anomaly_count,
                        population,
                        rate,
                        run_id,
                        now,
                    ),
                )
            self._conn.execute(
                """
                INSERT INTO period_execution
                    (dataset_fp, period_label, config_hash, run_id, group_count, failed_count, complete, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_fp, period_label, config_hash) DO UPDATE SET
                    run_id       = excluded.run_id,
                    group_count  = excluded.group_count,
                    failed_count = excluded.failed_count,
                    complete     = excluded.complete,
                    updated_at   = excluded.updated_at
                """,
                (
                    dataset_fp,
                    period_label,
                    config_hash,
                    run_id,
                    len(totals),
                    failed_count,
                    int(complete),
                    now,
                ),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ROLLBACK")
            raise

    def period_is_complete(self, dataset_fp: str, period_label: str, config_hash: str) -> bool:
        """Check whether a run against *config_hash* has completed this period with no failed groups.

        This is the sole definition of "done" for backfill/resume purposes --
        a period with only a partial totals row (some groups failed, or the
        process was interrupted before ``record_period_completion`` ran) is
        not complete and must be retried.
        """
        row = self._conn.execute(
            "SELECT complete FROM period_execution WHERE dataset_fp=? AND period_label=? AND config_hash=?",
            (dataset_fp, period_label, config_hash),
        ).fetchone()
        return bool(row is not None and row["complete"])

    def last_complete_period_label(self, dataset_fp: str, config_hash: str) -> str | None:
        """Return the most recent period_label completed (no failed groups) under *config_hash*."""
        row = self._conn.execute(
            "SELECT MAX(period_label) AS pl FROM period_execution "
            "WHERE dataset_fp=? AND config_hash=? AND complete=1",
            (dataset_fp, config_hash),
        ).fetchone()
        val = row["pl"] if row else None
        return str(val) if val is not None else None

    def completed_group_keys(self, dataset_fp: str, period_label: str, config_hash: str) -> list[str]:
        """Return group_keys with a totals row for this (dataset_fp, period_label, config_hash).

        Reflects whatever totals exist even for an incomplete period (some
        groups may have data while others failed) -- use ``period_is_complete``
        to ask whether the period as a whole is done.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT group_key FROM totals WHERE dataset_fp=? AND period_label=? AND config_hash=?",
            (dataset_fp, period_label, config_hash),
        ).fetchall()
        return [str(r["group_key"]) for r in rows]

    def groups_seen_for_dataset(self, dataset_fp: str, config_hash: str) -> set[str]:
        """Return all group_keys ever seen in totals for a dataset under *config_hash*.

        Scoped per config because different configurations can define
        entirely different group_by dimensions -- a group_key meaningful under
        one configuration is not necessarily meaningful, or even comparable,
        under another.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT group_key FROM totals WHERE dataset_fp=? AND config_hash=?",
            (dataset_fp, config_hash),
        ).fetchall()
        return {str(r["group_key"]) for r in rows}

    def totals_for_periods(
        self,
        dataset_fp: str,
        period_labels: list[str],
        config_hash: str,
        group_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return totals rows for the given period labels under one config_hash.

        Filtering by config_hash is mandatory: a dataset touched by two
        different configurations has independent totals rows for the same
        (dataset_fp, group_key, period_label) triple (see migration 002), and
        summing across both would double-count.
        """
        if not period_labels:
            return []
        ph = ",".join("?" * len(period_labels))
        params: list[Any] = [dataset_fp, config_hash, *period_labels]
        gk_clause = ""
        if group_keys is not None:
            gk_ph = ",".join("?" * len(group_keys))
            gk_clause = f"AND group_key IN ({gk_ph})"
            params.extend(group_keys)
        rows = self._conn.execute(
            f"SELECT * FROM totals WHERE dataset_fp=? AND config_hash=? AND period_label IN ({ph}) {gk_clause}",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def other_config_hashes_for_periods(
        self, dataset_fp: str, period_labels: list[str], config_hash: str
    ) -> set[str]:
        """Return config_hash values (other than *config_hash*) that also have totals here.

        Replaces a dead lookup that diffed a ``calibration_mode`` config field
        removed when self-calibration became the only supported mode (so it
        always returned an empty set). This is real provenance instead: a
        non-empty result means part of this dataset's history in this window
        was computed under a different configuration and is excluded from the
        aggregate below -- the trend may be comparing an incomplete picture.
        """
        if not period_labels:
            return set()
        ph = ",".join("?" * len(period_labels))
        rows = self._conn.execute(
            f"SELECT DISTINCT config_hash FROM totals "
            f"WHERE dataset_fp=? AND period_label IN ({ph}) AND config_hash != ?",
            [dataset_fp, *period_labels, config_hash],
        ).fetchall()
        return {str(r["config_hash"]) for r in rows}

    def delete_totals_for_period(self, dataset_fp: str, period_label: str, config_hash: str) -> None:
        """Remove totals + the completion record for (dataset_fp, period_label, config_hash)."""
        self._conn.execute(
            "DELETE FROM totals WHERE dataset_fp=? AND period_label=? AND config_hash=?",
            (dataset_fp, period_label, config_hash),
        )
        self._conn.execute(
            "DELETE FROM period_execution WHERE dataset_fp=? AND period_label=? AND config_hash=?",
            (dataset_fp, period_label, config_hash),
        )
        self._conn.commit()
