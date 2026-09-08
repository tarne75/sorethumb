"""CSV report writer.

Writes one sibling CSV per group alongside the HTML report. The HTML links to
these files with relative paths (./<group_key>.csv) so that moving the HTML
without its CSVs breaks the links — document this in the report.

Reports are meant to be shared and re-opened in spreadsheet software, so every
string cell and column name is neutralised against formula injection before the
frame is written: a leading =, +, -, @, or control character (tab/CR/LF) is the
trigger Excel, LibreOffice and Google Sheets use to evaluate a cell as a
formula, so such values are prefixed with a single quote.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# A leading one of these makes a spreadsheet evaluate the cell as a formula.
_TRIGGER_RE = r"^[=+\-@\t\r\n]"
_STRING_DTYPES = (pl.String, pl.Categorical, pl.Enum, pl.Object)


def _needs_quote(value: str) -> bool:
    return bool(value) and value[0] in "=+-@\t\r\n"


def _neutralize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return *df* with formula-injection-prone string cells and headers escaped."""
    string_cols = [name for name, dtype in zip(df.columns, df.dtypes, strict=True) if dtype in _STRING_DTYPES]
    if string_cols:
        df = df.with_columns(
            pl.when(pl.col(name).cast(pl.String).str.contains(_TRIGGER_RE))
            .then(pl.lit("'") + pl.col(name).cast(pl.String))
            .otherwise(pl.col(name).cast(pl.String))
            .alias(name)
            for name in string_cols
        )
    renames = {name: f"'{name}" for name in df.columns if _needs_quote(name)}
    if renames:
        df = df.rename(renames)
    return df


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
    _neutralize_frame(df).write_csv(str(path))
    logger.info("CSV written: %s (%d rows).", path, len(df))
    return path
