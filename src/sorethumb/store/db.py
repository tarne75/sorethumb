"""SQLite store — single connection, WAL mode, foreign keys, numbered migrations.

All writes go through parameterised statements. The only non-parameterised SQL
is in migration files, which are reviewed and committed as source.

One Store owns one connection. Nothing else in the library should open the database.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from sorethumb.errors import StoreError

logger = logging.getLogger(__name__)

_MIGRATIONS_PACKAGE = "sorethumb.store.migrations"


def _now_utc() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _config_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


class Store:
    """Owns the SQLite connection for a single workspace."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the database at db_path and apply pending migrations."""
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration "
            "(version INTEGER PRIMARY KEY, "
            "applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')))"
        )
        self._conn.commit()

        applied: set[int] = {
            row[0] for row in self._conn.execute("SELECT version FROM schema_migration ORDER BY version")
        }

        # Discover migration SQL files bundled with the package
        migration_files: list[tuple[int, str]] = []
        try:
            pkg = importlib.resources.files(_MIGRATIONS_PACKAGE)
            for entry in pkg.iterdir():
                name = entry.name
                if name.endswith(".sql") and name[:3].isdigit():
                    version = int(name[:3])
                    migration_files.append((version, entry.read_text(encoding="utf-8")))
        except (FileNotFoundError, AttributeError, TypeError) as exc:
            raise StoreError(f"Cannot load migration files from {_MIGRATIONS_PACKAGE}: {exc}") from exc

        migration_files.sort(key=lambda x: x[0])

        for version, sql in migration_files:
            if version in applied:
                continue
            logger.info("Applying migration %03d.", version)
            # Migrations run as a single transaction each
            with self._conn:
                # Migration 001 already creates schema_migration; strip that CREATE to avoid conflict
                if version == 1:
                    # The migration file manages its own schema_migration table, skip CREATE
                    pass
                self._conn.executescript(sql)
            logger.info("Migration %03d applied.", version)

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
    ) -> None:
        """Insert or update a dataset row, preserving first_seen."""
        now = _now_utc()
        self._conn.execute(
            """
            INSERT INTO dataset
                (dataset_fp, source_uri, schema_fingerprint, content_fingerprint,
                 n_rows, n_cols, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_fp) DO UPDATE SET
                source_uri = excluded.source_uri,
                schema_fingerprint = excluded.schema_fingerprint,
                content_fingerprint = excluded.content_fingerprint,
                n_rows = excluded.n_rows,
                n_cols = excluded.n_cols,
                last_seen = excluded.last_seen
            """,
            (dataset_fp, source_uri, schema_fingerprint, content_fingerprint, n_rows, n_cols, now, now),
        )
        self._conn.commit()

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
    ) -> None:
        """Record a new run in status 'running'."""
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
                 library_version, python_version, started_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (run_id, dataset_fp, cfg_hash, config_json, seed, lib_ver, py_ver, now),
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
    ) -> None:
        """Register a file artifact in the index."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO artifact (artifact_id, path, kind, byte_size, regenerable)
            VALUES (?, ?, ?, ?, ?)
            """,
            (artifact_id, path, kind, byte_size, 1 if regenerable else 0),
        )
        self._conn.commit()

    def artifacts_for_prune(self, retention_days: int) -> list[dict[str, Any]]:
        """Return artifact rows eligible for pruning.

        Regenerable artifacts older than retention_days, plus artifacts whose
        run has status='failed' and is older than retention_days.
        """
        cutoff = datetime.now(tz=UTC)
        # Use raw SQL date arithmetic with the retention threshold
        rows = self._conn.execute(
            """
            SELECT a.*
            FROM artifact a
            WHERE a.regenerable = 1
              AND julianday('now') - julianday(a.created_at) > ?
            UNION
            SELECT a.*
            FROM artifact a
            JOIN run r ON instr(a.path, r.run_id) > 0
            WHERE r.status = 'failed'
              AND julianday('now') - julianday(r.started_at) > ?
            """,
            (retention_days, retention_days),
        ).fetchall()
        del cutoff  # used for intent documentation only
        return [dict(r) for r in rows]

    def delete_artifact(self, artifact_id: str) -> None:
        """Remove an artifact row from the index."""
        self._conn.execute("DELETE FROM artifact WHERE artifact_id=?", (artifact_id,))
        self._conn.commit()

    def vacuum(self) -> None:
        """Run SQLite VACUUM."""
        self._conn.execute("VACUUM")
