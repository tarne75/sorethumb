-- Migration 009: source provenance without query values
--
-- dataset.source_uri and run.config_json used to hold the source URI with only a
-- fixed list of "sensitive" query parameters redacted, so any other name carrying
-- a secret (client_secret, jwt, accessKey, ...) was stored in the clear. From this
-- version every query VALUE is replaced by REDACTED (key names are kept) and the
-- full URI never reaches the database.
--
-- source_digest is a one-way SHA-256 of the canonical full URI, stored next to the
-- redacted display URI so two sources that differ only in a query value stay
-- distinguishable without recording either value. It is NULL for rows written
-- before this migration: the full URI was never stored, so it cannot be computed.
--
-- This migration also re-redacts dataset.source_uri and the source.uri inside every
-- stored run.config_json in place (Store._scrub_stored_source_uris, run in the same
-- transaction), so values stored by an earlier version are removed, not just
-- stopped. The in-place rewrite is not reversible.

ALTER TABLE dataset ADD COLUMN source_digest TEXT;

ALTER TABLE run ADD COLUMN source_digest TEXT;
