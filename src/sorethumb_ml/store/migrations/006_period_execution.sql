-- Migration 006: explicit, config-scoped period completion record
--
-- Before this migration, "is this period done?" was answered by "does the
-- totals table have any row for (dataset_fp, period_label)?" -- ignoring
-- config_hash entirely. That made two things possible:
--   1. A period with 3 groups where only 1 succeeded (2 failed) still had a
--      totals row for the 1 that succeeded, so it read as fully complete and
--      was never retried.
--   2. Two different configurations processing the same period_label wrote
--      to the same natural key space; a reader that didn't filter by
--      config_hash would see "this period has data" as soon as *either*
--      configuration touched it, even if the other configuration's groups
--      were never computed.
--
-- period_execution is written once per (dataset_fp, period_label,
-- config_hash) atomically alongside that attempt's totals rows (see
-- Store.record_period_completion): complete=1 only when every group
-- discovered in that run reached a non-failed terminal status. A period
-- with no row here -- or a row with complete=0 -- must be retried.

CREATE TABLE IF NOT EXISTS period_execution (
    dataset_fp    TEXT NOT NULL REFERENCES dataset(dataset_fp),
    period_label  TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    group_count   INTEGER NOT NULL,
    failed_count  INTEGER NOT NULL,
    complete      INTEGER NOT NULL CHECK(complete IN (0, 1)),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (dataset_fp, period_label, config_hash)
);

CREATE INDEX IF NOT EXISTS idx_period_execution_dataset ON period_execution(dataset_fp, config_hash);
