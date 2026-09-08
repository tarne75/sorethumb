-- Migration 003: add run_id to artifact for exact prune joins
--
-- artifacts_for_prune() previously joined failed runs to their artifacts with
--   JOIN run r ON instr(a.path, r.run_id) > 0
-- a substring match on the file path. One run_id that is a substring of another
-- run_id (or that appears anywhere in an unrelated path) would drag the wrong
-- files into a failed-run prune. Recording the owning run_id on the artifact
-- row lets the prune query join on equality instead.
--
-- Existing rows keep a NULL run_id and are simply not matched by the failed-run
-- branch of the prune query. The regenerable branch is unaffected.
--
-- run_id is a plain column, not a foreign key: artifacts may be registered
-- before the run row is committed, and a dangling run_id is harmless to the
-- prune join (it just never matches a failed run).

ALTER TABLE artifact ADD COLUMN run_id TEXT;

CREATE INDEX idx_artifact_run_id ON artifact(run_id);
