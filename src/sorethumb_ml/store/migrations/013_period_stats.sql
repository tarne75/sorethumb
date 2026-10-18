-- Migration 013: per-period statistics for drift
--
-- One row per (dataset, group, period, configuration), written in the same
-- transaction as that group's totals row (Store.record_period_completion), so
-- the two always describe the same run. stats_json is the versioned summary
-- built by sorethumb_ml.history.period_stats: the population (row count, and
-- per original field its missing count plus percentiles or top values), each
-- detector's raw-score percentiles and natural flag count on that period's
-- self-fitted scale, and the composite threshold, contamination and flag count.
CREATE TABLE IF NOT EXISTS period_stats (
    dataset_fp    TEXT NOT NULL,
    group_key     TEXT NOT NULL,
    period_label  TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    stats_version INTEGER NOT NULL,
    stats_json    TEXT NOT NULL,
    computed_at   TEXT NOT NULL,
    PRIMARY KEY (dataset_fp, group_key, period_label, config_hash)
);
