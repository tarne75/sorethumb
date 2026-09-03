-- Migration 001: initial schema

CREATE TABLE IF NOT EXISTS schema_migration (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE dataset (
    dataset_fp          TEXT PRIMARY KEY,
    source_uri          TEXT NOT NULL,
    schema_fingerprint  TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    n_rows              INTEGER NOT NULL,
    n_cols              INTEGER NOT NULL,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL
);

CREATE TABLE run (
    run_id          TEXT PRIMARY KEY,
    dataset_fp      TEXT NOT NULL REFERENCES dataset(dataset_fp),
    config_hash     TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    library_version TEXT NOT NULL,
    python_version  TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running', 'complete', 'failed')),
    error_text      TEXT
);

CREATE TABLE run_group (
    run_id          TEXT NOT NULL REFERENCES run(run_id),
    group_key       TEXT NOT NULL,
    group_values_json TEXT NOT NULL,
    group_label     TEXT NOT NULL,
    record_count    INTEGER,
    anomaly_count   INTEGER,
    rate            REAL,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK(status IN ('running', 'complete', 'skipped', 'failed')),
    timing_seconds  REAL,
    error           TEXT,
    PRIMARY KEY (run_id, group_key)
);

CREATE TABLE period (
    dataset_fp   TEXT NOT NULL REFERENCES dataset(dataset_fp),
    period_label TEXT NOT NULL,
    period_from  TEXT NOT NULL,
    period_to    TEXT NOT NULL,
    PRIMARY KEY (dataset_fp, period_label)
);

CREATE TABLE totals (
    dataset_fp    TEXT NOT NULL,
    group_key     TEXT NOT NULL,
    period_label  TEXT NOT NULL,
    anomaly_count INTEGER NOT NULL,
    population    INTEGER NOT NULL,
    rate          REAL,
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    computed_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (dataset_fp, group_key, period_label)
);

CREATE TABLE model (
    model_id            TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES run(run_id),
    group_key           TEXT NOT NULL,
    detector_name       TEXT NOT NULL,
    artifact_path       TEXT NOT NULL,
    feature_schema_hash TEXT NOT NULL,
    train_row_count     INTEGER NOT NULL,
    params_json         TEXT NOT NULL,
    fitted_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE calibrator (
    model_id            TEXT PRIMARY KEY REFERENCES model(model_id),
    quantile_values_json TEXT NOT NULL
);

CREATE TABLE artifact (
    artifact_id  TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,
    byte_size    INTEGER NOT NULL,
    regenerable  INTEGER NOT NULL CHECK(regenerable IN (0, 1)),
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

INSERT INTO schema_migration (version) VALUES (1);
