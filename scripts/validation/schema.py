"""Pure dataclasses for the validation matrix: dataset/combo specs and the result schema.

Nothing in this module does I/O, fits a model, or imports sorethumb's heavier
pipeline modules — it is safe to import and exercise in a fast unit test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

# Bump whenever a field is added/removed/redefined in CaseResult or the
# identity inputs below, so a results.json written by an older schema is
# never silently reused as if it meant the same thing under the new one.
CASE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset entry in the validation matrix."""

    name: str
    file: str
    ignore: list[str]
    source: str
    description: str
    rows: int
    cols: int
    # When set, this column (already excluded from features via `ignore`)
    # carries ground-truth anomaly labels usable for held-out ROC-AUC/AP.
    # `label_is_anomaly` maps its raw values to a boolean anomaly mask;
    # `None` means the dataset has no usable anomaly ground truth (most of
    # data-samples/*.parquet were built for other classification tasks, not
    # anomaly detection) and only operational/pipeline-smoke metrics apply.
    label_column: str | None = None
    label_is_anomaly: Callable[[pl.Series], pl.Series] | None = None


@dataclass(frozen=True)
class ComboSpec:
    """One named detector combination."""

    name: str
    detectors: tuple[str, ...]


@dataclass(frozen=True)
class CaseKey:
    """Identity of one planned matrix cell: which (dataset, pca, combo, seed) to run."""

    dataset: str
    pca: bool
    combo: str
    seed: int


@dataclass(frozen=True)
class RunIdentity:
    """Everything that must match for a stored result to be safely reused.

    A key match alone (dataset/pca/combo/seed) is not enough: the same case
    run against different code, a different dataset file, a different
    resolved config, or different dependency versions can legitimately
    produce a different result. Any mismatch here makes the stored result
    stale and forces a rerun.
    """

    schema_version: int
    code_revision: str
    data_fingerprint: str
    config_hash: str
    seed: int
    dependency_versions: dict[str, str]


@dataclass
class CaseResult:
    """One row of the validation matrix's persisted results."""

    dataset: str
    pca: bool
    combo: str
    detectors: list[str]
    seed: int

    identity: RunIdentity

    # "success": at least one group actually fit and scored.
    # "too_few_records": every group's train or holdout split was below
    #   scoring.min_records -- a benign, resumable outcome (see
    #   runner.run_case), not a failure. n_train/n_holdout are populated;
    #   every metric field stays at its default (None/0).
    # "error": a real failure (fit/scoring exception, a group actually
    #   failing). Always reruns on resume; counted for the process exit code.
    status: str
    elapsed_seconds: float

    n_train: int = 0
    n_holdout: int = 0
    # None when the dataset has no label_column (operational smoke test only).
    n_holdout_anomalies_true: int | None = None
    n_holdout_flagged: int = 0
    holdout_flag_rate: float = 0.0
    review_budget: float = 0.0

    # None when undefined (unlabelled dataset, or single-class holdout split).
    roc_auc: float | None = None
    average_precision: float | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    f1_at_contamination: float | None = None

    run_id: str | None = None
    score_run_id: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def key(self) -> CaseKey:
        """Return this result's planning identity (dataset, pca, combo, seed)."""
        return CaseKey(dataset=self.dataset, pca=self.pca, combo=self.combo, seed=self.seed)

    def as_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable dict (identity fields inlined)."""
        return {
            "dataset": self.dataset,
            "pca": self.pca,
            "combo": self.combo,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "schema_version": self.identity.schema_version,
            "code_revision": self.identity.code_revision,
            "data_fingerprint": self.identity.data_fingerprint,
            "config_hash": self.identity.config_hash,
            "dependency_versions": dict(self.identity.dependency_versions),
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "n_holdout_anomalies_true": self.n_holdout_anomalies_true,
            "n_holdout_flagged": self.n_holdout_flagged,
            "holdout_flag_rate": self.holdout_flag_rate,
            "review_budget": self.review_budget,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_contamination": self.f1_at_contamination,
            "run_id": self.run_id,
            "score_run_id": self.score_run_id,
            "error": self.error,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        """Reconstruct a CaseResult from the dict shape ``as_dict`` produces."""
        identity = RunIdentity(
            schema_version=d["schema_version"],
            code_revision=d["code_revision"],
            data_fingerprint=d["data_fingerprint"],
            config_hash=d["config_hash"],
            seed=d["seed"],
            dependency_versions=dict(d["dependency_versions"]),
        )
        return cls(
            dataset=d["dataset"],
            pca=d["pca"],
            combo=d["combo"],
            detectors=list(d["detectors"]),
            seed=d["seed"],
            identity=identity,
            status=d["status"],
            elapsed_seconds=d["elapsed_seconds"],
            n_train=d.get("n_train", 0),
            n_holdout=d.get("n_holdout", 0),
            n_holdout_anomalies_true=d.get("n_holdout_anomalies_true"),
            n_holdout_flagged=d.get("n_holdout_flagged", 0),
            holdout_flag_rate=d.get("holdout_flag_rate", 0.0),
            review_budget=d.get("review_budget", 0.0),
            roc_auc=d.get("roc_auc"),
            average_precision=d.get("average_precision"),
            precision_at_k=d.get("precision_at_k"),
            recall_at_k=d.get("recall_at_k"),
            f1_at_contamination=d.get("f1_at_contamination"),
            run_id=d.get("run_id"),
            score_run_id=d.get("score_run_id"),
            error=d.get("error"),
            warnings=list(d.get("warnings", [])),
        )
