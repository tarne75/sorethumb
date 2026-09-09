-- Migration 004: record the source run for score-forward runs
--
-- `sorethumb score --from-run SRC` writes a new, distinct run that reused SRC's
-- FeaturePlan and per-detector models/calibrators without refitting.
-- source_run_id records that link so `sorethumb runs` and history can tell a
-- score-forward run from an ordinary fitted one. Plain nullable column: NULL for
-- ordinary runs.

ALTER TABLE run ADD COLUMN source_run_id TEXT;

CREATE INDEX idx_run_source ON run(source_run_id);
