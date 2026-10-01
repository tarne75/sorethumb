-- Migration 008: indexes for the predicates the tables that grow without bound are queried by
--
-- EXPLAIN QUERY PLAN on the Store's real queries (see
-- tests/contract/test_database_schema.py::test_store_queries_do_not_scan_growing_tables)
-- showed full-table scans where a workspace's history accumulates one row per run,
-- per group x period or per fitted model:
--
--   model         WHERE run_id=? AND group_key=?   SCAN: the primary key is model_id only
--   run           ORDER BY started_at DESC LIMIT   SCAN + sort: `sorethumb runs`
--   run           WHERE status='failed' AND ...    SCAN: failed-run branch of artifact pruning
--   totals        WHERE dataset_fp=? AND period_label [IN ...] AND config_hash=?
--                 only the dataset_fp prefix of the (dataset_fp, group_key, period_label,
--                 config_hash) primary key could be used, so every group x period x config
--                 row of the dataset was read to find one period
--   period_execution  MAX(period_label) WHERE dataset_fp=? AND config_hash=? AND complete=1
--                 walked every period of the dataset
--
-- Reviewed and deliberately left alone (small or bounded, or index would not be used):
--   artifact prune, regenerable branch: compares julianday(created_at), an expression an
--     index cannot serve, and rewriting it to a text comparison would change the retention
--     boundary; the table is trimmed by `workspace prune` itself.
--   dataset_snapshot by dataset_fp ORDER BY first_seen: a handful of snapshots per dataset.
--   run_group / calibrator / artifact path / run.source_run_id: already index-backed.

CREATE INDEX IF NOT EXISTS idx_model_run_group ON model(run_id, group_key);

CREATE INDEX IF NOT EXISTS idx_run_started_at ON run(started_at);

CREATE INDEX IF NOT EXISTS idx_run_status_started_at ON run(status, started_at);

-- group_key last makes the (dataset_fp, period_label, config_hash) lookups covering for
-- completed_group_keys, and lets `period_label IN (...)` seek per label.
CREATE INDEX IF NOT EXISTS idx_totals_period ON totals(dataset_fp, period_label, config_hash, group_key);

-- Supersedes idx_period_execution_dataset(dataset_fp, config_hash): same prefix, and the
-- MAX(period_label) over complete rows becomes a single index seek.
CREATE INDEX IF NOT EXISTS idx_period_execution_last_complete
    ON period_execution(dataset_fp, config_hash, complete, period_label);

DROP INDEX IF EXISTS idx_period_execution_dataset;
