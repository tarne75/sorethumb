"""CSV report writer.

Writes one sibling CSV per group alongside the HTML report. The HTML links to
these files with relative paths (./<group_key>.csv) so that moving the HTML
without its CSVs breaks the links — document this in the report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def write_group_csv(df: pl.DataFrame, out_dir: Path, group_key: str) -> Path:
    """Write *df* as a CSV sibling to the HTML report.

    Parameters
    ----------
    df:
        Result frame for one group.
    out_dir:
        The run's report directory (``reports/<run_id>/``).
    group_key:
        16-character group digest used as the file stem.

    Returns
    -------
    Path to the written CSV file.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{group_key}.csv"
    df.write_csv(str(path))
    logger.info("CSV written: %s (%d rows).", path, len(df))
    return path
