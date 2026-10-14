-- Migration 010: record the ensemble policy each run used
--
-- run.ensemble_policy is the versioned id (name@version, e.g. intersection@1) of
-- the rule that turned detector votes into flags -- see scoring/policies.py and
-- docs/ensemble-policy.md. A policy's version identifies its behaviour, so a
-- stored id keeps meaning what it meant when the run was made. NULL for runs
-- recorded before this migration: their rule is only recoverable from
-- config_json, and from the library version that ran it.
ALTER TABLE run ADD COLUMN ensemble_policy TEXT;
