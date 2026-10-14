-- Migration 011: backtest evidence for ensemble policies, and the evidence a run relied on
--
-- One row per (candidate policy, baseline policy) comparison made by
-- `sorethumb backtest` on a dataset (see evaluate/user_backtest.py and
-- docs/ensemble-policy.md). dataset_fp is the stable logical dataset id;
-- snapshot_fp the bytes the backtest actually saw. scope_hash fingerprints the
-- configuration apart from the policy itself (and the explain settings, which
-- do not change flags): evidence only counts for a run whose scope matches.
-- verdict is 'outperforms', 'does_not_outperform' or 'not_assessable' (no
-- labels). details_json holds the per-replicate values the verdict rests on.
CREATE TABLE IF NOT EXISTS policy_evidence (
    evidence_id       TEXT PRIMARY KEY,
    dataset_fp        TEXT NOT NULL REFERENCES dataset(dataset_fp),
    snapshot_fp       TEXT,
    policy_id         TEXT NOT NULL,
    baseline_policy_id TEXT NOT NULL,
    scope_hash        TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    labelled          INTEGER NOT NULL,
    primary_metric    TEXT,
    mean_difference   REAL,
    ci_low            REAL,
    ci_high           REAL,
    p_holm            REAL,
    n_replicates      INTEGER NOT NULL,
    review_budget     REAL NOT NULL,
    library_version   TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    details_json      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_evidence_lookup
    ON policy_evidence (dataset_fp, policy_id, created_at);

-- The evidence record a run with a non-default policy relied on: NULL for the
-- default policy, for an unevidenced non-default policy (the run warned, or
-- refused under run.strict), and for runs recorded before this migration.
ALTER TABLE run ADD COLUMN policy_evidence_id TEXT;
