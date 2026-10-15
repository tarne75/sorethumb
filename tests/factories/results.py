"""Read a group's results as the readers see them: detection rows with explanations joined on.

Detection (``anomalies.parquet``) and explanation (``explanations.parquet``)
are separate files in the same directory; ``store.results.read_results`` joins
them for a workspace. Tests that hold only ``GroupSummary.results_path`` use
this path-based equivalent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def joined_results(results_path: Path | str) -> pl.DataFrame:
    """Return ``anomalies.parquet`` at *results_path*, left-joined with its ``explanations.parquet``."""
    path = Path(results_path)
    detection = pl.read_parquet(path)
    explanations = path.with_name("explanations.parquet")
    if not explanations.exists():
        return detection
    extra = pl.read_parquet(explanations)
    cols = [c for c in extra.columns if c == "row_id" or c not in detection.columns]
    return detection.join(extra.select(cols), on="row_id", how="left")
