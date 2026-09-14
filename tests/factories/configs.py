"""``Config`` builders shared across CLI, config-wiring, end-to-end and
score-forward tests.

``make_config`` defaults to ``combination="intersection"`` — the shipped
default — so tests exercise real production behaviour unless a test has a
specific reason to ask for something else (e.g. composite scoring, or an
explicit three-way intersection with named detectors).
"""

from __future__ import annotations

from pathlib import Path

from sorethumb.config import (
    ColumnsConfig,
    Config,
    DetectorConfig,
    ExplainConfig,
    HistoryConfig,
    ReportConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)


def make_config(
    source_path: Path,
    workdir: Path,
    *,
    source_format: str = "csv",
    contamination: str | float = "auto",
    combination: str = "intersection",
    detectors: list[DetectorConfig] | None = None,
    weighting: str = "equal",
    min_records: int = 5,
    id_column: str = "id",
    time_column: str | None = None,
    group_by: list[str] | None = None,
    seed: int = 42,
    run_kwargs: dict | None = None,
    explain_kwargs: dict | None = None,
    report_kwargs: dict | None = None,
    history_kwargs: dict | None = None,
) -> Config:
    """Build a ``Config`` pointed at ``source_path`` with sensible test defaults.

    Only ``source_path`` and ``workdir`` are required; every other field has
    a fast, deterministic default (single ``isolation_forest`` detector,
    ``contamination="auto"``, ``combination="intersection"``).
    """
    columns_kwargs: dict = {"id_column": id_column}
    if time_column is not None:
        columns_kwargs["time_column"] = time_column
    if group_by is not None:
        columns_kwargs["group_by"] = group_by

    cfg_kwargs: dict = {
        "source": SourceConfig(uri=str(source_path), format=source_format),
        "run": RunConfig(workdir=str(workdir), seed=seed, **(run_kwargs or {})),
        "columns": ColumnsConfig(**columns_kwargs),
        "detectors": detectors or [DetectorConfig(name="isolation_forest")],
        "scoring": ScoringConfig(
            combination=combination,
            contamination=contamination,
            weighting=weighting,
            min_records=min_records,
        ),
        "explain": ExplainConfig(**(explain_kwargs or {})),
        "report": ReportConfig(**(report_kwargs or {})),
    }
    if history_kwargs is not None:
        cfg_kwargs["history"] = HistoryConfig(**history_kwargs)
    return Config(**cfg_kwargs)
