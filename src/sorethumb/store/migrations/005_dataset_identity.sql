-- Migration 005: separate logical dataset identity from snapshot identity
--
-- dataset_fp was derived from the full file content + schema
-- (content_fingerprint[:32] + "_" + schema_fingerprint[:16]). Appending a day
-- of rows changed the content fingerprint, so run_detection wrote a brand-new
-- dataset row and every prior period / totals / run row -- keyed on the old
-- dataset_fp -- was orphaned. Backfill then re-bootstrapped from scratch.
--
-- dataset_fp is now a stable logical id: source.dataset_id when configured,
-- otherwise derived from the source URI. The content+schema fingerprint becomes
-- a per-snapshot version, recorded in dataset_snapshot and pointed at by
-- dataset.snapshot_fp (the most recently seen snapshot).
--
-- Existing rows: their dataset_fp values ARE the old content+schema fingerprint,
-- so they already double as that dataset's first snapshot. We seed
-- dataset_snapshot from them and leave dataset_fp untouched -- a workspace
-- carrying pre-005 history keeps it under the old key, and the next run against
-- an unchanged source re-registers it under the new logical id (a one-time
-- re-baseline; see CHANGELOG).

ALTER TABLE dataset ADD COLUMN snapshot_fp TEXT;

CREATE TABLE dataset_snapshot (
    dataset_fp          TEXT NOT NULL REFERENCES dataset(dataset_fp),
    snapshot_fp         TEXT NOT NULL,
    schema_fingerprint  TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    n_rows              INTEGER NOT NULL,
    n_cols              INTEGER NOT NULL,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    PRIMARY KEY (dataset_fp, snapshot_fp)
);

CREATE INDEX idx_dataset_snapshot_fp ON dataset_snapshot(snapshot_fp);

INSERT INTO dataset_snapshot
    (dataset_fp, snapshot_fp, schema_fingerprint, content_fingerprint,
     n_rows, n_cols, first_seen, last_seen)
SELECT dataset_fp, dataset_fp, schema_fingerprint, content_fingerprint,
       n_rows, n_cols, first_seen, last_seen
FROM dataset;

UPDATE dataset SET snapshot_fp = dataset_fp WHERE snapshot_fp IS NULL;
