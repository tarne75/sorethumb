-- Migration 012: detection and explanation recorded separately per group
--
-- A group's detection output (scores, flags, ranks; anomalies.parquet) is
-- written and digested before any explanation is computed, and explanation
-- output goes to its own file (explanations.parquet). These columns record the
-- explanation stage's outcome without touching the detection columns:
--   explain_status         'ok' | 'partial' | 'failed' | 'not_requested' | 'no_flagged_rows'
--   n_explained            flagged rows with an explanation
--   n_explain_unavailable  flagged rows marked unavailable (e.g. beyond explain.max_rows)
--   detection_digest       sha256 of anomalies.parquet as written by the detection stage
-- NULL for groups recorded before this migration.
ALTER TABLE run_group ADD COLUMN explain_status TEXT;
ALTER TABLE run_group ADD COLUMN n_explained INTEGER;
ALTER TABLE run_group ADD COLUMN n_explain_unavailable INTEGER;
ALTER TABLE run_group ADD COLUMN detection_digest TEXT;
