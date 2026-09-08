-- Migration 002: add config_hash to totals primary key
--
-- Previously totals were keyed on (dataset_fp, group_key, period_label).
-- A config change would silently overwrite the prior trend point for the same
-- period. Adding config_hash means each (dataset, group, period, config) tuple
-- is a separate row, preserving the history across config changes.
--
-- Existing rows are migrated with config_hash = '' so they remain queryable.

CREATE TABLE totals_new (
    dataset_fp    TEXT NOT NULL,
    group_key     TEXT NOT NULL,
    period_label  TEXT NOT NULL,
    config_hash   TEXT NOT NULL DEFAULT '',
    anomaly_count INTEGER NOT NULL,
    population    INTEGER NOT NULL,
    rate          REAL,
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    computed_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (dataset_fp, group_key, period_label, config_hash)
);

INSERT INTO totals_new
    (dataset_fp, group_key, period_label, config_hash,
     anomaly_count, population, rate, run_id, computed_at)
SELECT
    dataset_fp, group_key, period_label, '',
    anomaly_count, population, rate, run_id, computed_at
FROM totals;

DROP TABLE totals;

ALTER TABLE totals_new RENAME TO totals;
