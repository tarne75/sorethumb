"""Per-group anomaly result Parquet writer.

Results are written immediately after a group completes — never at the end of
the run — so an interrupted run leaves all completed groups intact and only the
in-progress group is lost.

The Parquet file is the authoritative record of which rows were flagged. The
database row_group table only stores aggregate counts; the per-row detail lives
in Parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from sorethumb.store.workspace import Workspace

logger = logging.getLogger(__name__)

_RESULT_FILENAME = "anomalies.parquet"


def write_results(
    workspace: Workspace,
    run_id: str,
    group_key: str,
    df: pl.DataFrame,
) -> Path:
    """Write a result DataFrame to Parquet and register the artifact.

    Parameters
    ----------
    workspace:
        The active Workspace.
    run_id:
        Current run identifier.
    group_key:
        Group digest (not raw group values).
    df:
        Result frame. At minimum must include a ``row_id`` column. Any subset
        of the documented columns is accepted; missing optional columns are not
        added — callers are responsible for assembling the full frame.

    Returns
    -------
    Path to the written Parquet file.

    """
    out_dir = workspace.results_dir(run_id, group_key)
    out_path = out_dir / _RESULT_FILENAME

    df.write_parquet(str(out_path))
    byte_size = out_path.stat().st_size
    logger.info(
        "Results written: run=%s group=%s rows=%d path=%s",
        run_id,
        group_key,
        len(df),
        out_path,
    )

    artifact_id = f"{run_id}_{group_key}_results"
    workspace.store.register_artifact(
        artifact_id=artifact_id,
        path=str(out_path),
        kind="results",
        byte_size=byte_size,
        regenerable=False,
    )
    return out_path


def read_results(
    workspace: Workspace,
    run_id: str,
    group_key: str,
) -> pl.DataFrame | None:
    """Load the result Parquet for a (run_id, group_key), or None if absent."""
    path = workspace.results_dir(run_id, group_key) / _RESULT_FILENAME
    if not path.exists():
        return None
    return pl.read_parquet(str(path))
