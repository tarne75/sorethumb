"""Per-group anomaly result Parquet writer.

Results are written immediately after a group completes — never at the end of
the run — so an interrupted run leaves all completed groups intact and only the
in-progress group is lost.

The Parquet file is the authoritative record of which rows were flagged. The
database row_group table only stores aggregate counts; the per-row detail lives
in Parquet.

Detection and explanation are separate files. ``anomalies.parquet`` holds the
detection output only (scores, flags, ranks) and is written -- and its sha256
digest recorded -- before any explanation is computed, so nothing the
explanation stage does, or fails to do, can change which rows were flagged or
how many. ``explanations.parquet`` holds the explanation for those rows.
:func:`read_results` joins the two for readers.

Written atomically (sibling temp file + rename, see ``sorethumb_ml._atomic``): a
crash mid-write, or a reader racing the writer, never sees a truncated file.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import polars as pl

from sorethumb_ml._atomic import atomic_write
from sorethumb_ml.errors import StoreError
from sorethumb_ml.store.workspace import Workspace

logger = logging.getLogger(__name__)

_RESULT_FILENAME = "anomalies.parquet"
_EXPLANATION_FILENAME = "explanations.parquet"


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

    Raises
    ------
    StoreError
        If ``row_id`` is missing, contains nulls, or has duplicates. This is
        the only column every reader (CLI, report, history) relies on to
        identify a row; writing (and registering) a Parquet file without it
        would surface as a confusing downstream failure far from its cause.

    """
    _validate_results_frame(df)

    out_dir = workspace.results_dir(run_id, group_key)
    out_path = out_dir / _RESULT_FILENAME

    # Temp file + rename: a reader (or a crash) never sees a half-written
    # Parquet file, and this is the only record of which rows were flagged.
    with atomic_write(out_path) as tmp:
        df.write_parquet(str(tmp))
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
        run_id=run_id,
    )
    return out_path


def _validate_results_frame(df: pl.DataFrame) -> None:
    """Reject a results frame that can't be safely joined back to source rows."""
    if "row_id" not in df.columns:
        raise StoreError("write_results: df is missing the required 'row_id' column.")
    if df["row_id"].null_count() > 0:
        raise StoreError("write_results: 'row_id' contains null values.")
    if df["row_id"].n_unique() != len(df):
        raise StoreError("write_results: 'row_id' contains duplicate values.")


def results_path(workspace: Workspace, run_id: str, group_key: str) -> Path:
    """Return the result Parquet path for a (run_id, group_key), existing or not."""
    return workspace.results_dir(run_id, group_key) / _RESULT_FILENAME


def file_digest(path: Path) -> str:
    """Return the sha256 hex digest of the file at *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def explanations_path(workspace: Workspace, run_id: str, group_key: str) -> Path:
    """Return the explanation Parquet path for a (run_id, group_key), existing or not."""
    return workspace.results_dir(run_id, group_key) / _EXPLANATION_FILENAME


def write_explanations(workspace: Workspace, run_id: str, group_key: str, df: pl.DataFrame) -> Path:
    """Write a group's explanation frame (one row per explained flagged row) and register it.

    Never touches ``anomalies.parquet``: the detection output is final before
    this is called.

    Raises
    ------
    StoreError
        If ``row_id`` is missing, null or duplicated.

    """
    _validate_results_frame(df)
    out_path = explanations_path(workspace, run_id, group_key)
    with atomic_write(out_path) as tmp:
        df.write_parquet(str(tmp))
    workspace.store.register_artifact(
        artifact_id=f"{run_id}_{group_key}_explanations",
        path=str(out_path),
        kind="explanations",
        byte_size=out_path.stat().st_size,
        regenerable=False,
        run_id=run_id,
    )
    return out_path


def read_explanations(workspace: Workspace, run_id: str, group_key: str) -> pl.DataFrame | None:
    """Load a group's explanation Parquet, or None if absent."""
    path = explanations_path(workspace, run_id, group_key)
    if not path.exists():
        return None
    return pl.read_parquet(str(path))


def read_results(
    workspace: Workspace,
    run_id: str,
    group_key: str,
    *,
    with_explanations: bool = True,
) -> pl.DataFrame | None:
    """Load a group's results, or None if absent.

    With *with_explanations* (the default), the explanation columns
    (``attribution_kind``, ``reason_<k>``, ...) are left-joined onto the
    detection rows by ``row_id``. A flagged row with no explanation keeps
    nulls there. Results written before detection and explanation were
    separated already carry those columns and are returned as stored.
    """
    path = results_path(workspace, run_id, group_key)
    if not path.exists():
        return None
    detection = pl.read_parquet(str(path))
    if not with_explanations:
        return detection
    explanation = read_explanations(workspace, run_id, group_key)
    if explanation is None:
        return detection
    extra = [c for c in explanation.columns if c == "row_id" or c not in detection.columns]
    return detection.join(explanation.select(extra), on="row_id", how="left")
