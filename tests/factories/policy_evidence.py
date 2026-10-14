"""Record backtest evidence directly, for tests that need a non-default policy under run.strict.

``run.strict`` refuses a non-default ensemble policy with no ``sorethumb backtest``
evidence (see ``_pipeline._policy_evidence_gate``). Tests about something else
that happen to use a composite or union policy record an ``outperforms`` record
for their configuration first, instead of switching policy and changing what
they exercise.
"""

from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

from sorethumb_ml.config import Config
from sorethumb_ml.io.fingerprint import logical_dataset_id
from sorethumb_ml.scoring.policies import DEFAULT_POLICY_ID, policy_scope_hash, resolve_policy
from sorethumb_ml.store.workspace import Workspace


def config_from_toml(path: Path) -> Config:
    """Validate a sorethumb.toml into a Config, without CLI flag overrides."""
    with path.open("rb") as fh:
        return Config.model_validate(tomllib.load(fh))


def record_evidence(config: Config, verdict: str = "outperforms", *, scope_hash: str | None = None) -> str:
    """Record a backtest verdict for *config*'s policy on its dataset; return the evidence id."""
    root = Path(config.run.workdir)
    ws = Workspace.open(root) if (root / "sorethumb.db").exists() else Workspace.init(root)
    dataset_fp = logical_dataset_id(config.source.dataset_id, config.source.uri)
    evidence_id = f"ev_{uuid.uuid4().hex[:16]}"
    with ws:
        if ws.store.get_dataset(dataset_fp) is None:
            ws.store.upsert_dataset(
                dataset_fp=dataset_fp,
                source_uri="test",
                source_digest=None,
                schema_fingerprint="test",
                content_fingerprint="test",
                n_rows=0,
                n_cols=0,
            )
        ws.store.insert_policy_evidence(
            {
                "evidence_id": evidence_id,
                "dataset_fp": dataset_fp,
                "snapshot_fp": None,
                "policy_id": resolve_policy(config.scoring.combination, config.scoring.weighting).id,
                "baseline_policy_id": DEFAULT_POLICY_ID,
                "scope_hash": scope_hash or policy_scope_hash(config),
                "verdict": verdict,
                "labelled": 1,
                "primary_metric": "flag_f1",
                "mean_difference": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
                "p_holm": 0.001,
                "n_replicates": 20,
                "review_budget": 0.05,
                "library_version": "test",
                "details_json": "{}",
            }
        )
    return evidence_id
