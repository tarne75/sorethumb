"""Format-detecting lazy readers.

Every reader returns a ``pl.LazyFrame`` so the caller controls when work
actually happens. Format is detected from the file extension when
``config.format == 'auto'``.

Ambiguous reads (single-column CSVs, all-string schemas after schema
inference) raise ``SchemaError`` rather than silently returning bad data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from sorethumb.config import SourceConfig
from sorethumb.errors import SchemaError, SourceError

logger = logging.getLogger(__name__)

_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "parquet": [".parquet"],
    "csv": [".csv", ".csv.gz"],
    "tsv": [".tsv", ".tsv.gz"],
    "json": [".json", ".json.gz"],
    "jsonl": [".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz"],
}


def read_frame(path: Path, config: SourceConfig) -> pl.LazyFrame:
    """Return a ``LazyFrame`` for *path* according to *config*.

    Raises:
        SourceError: Format cannot be determined or path does not exist.
        SchemaError: The inferred schema looks degenerate (single column,
            or all-string after infer_schema_length rows scanned).

    """
    fmt = _detect_format(path, config)
    logger.debug("Reading %s as '%s'", path, fmt)

    opts: dict[str, object] = dict(config.read_options)

    if fmt == "parquet":
        lf = pl.scan_parquet(path, **opts)  # type: ignore[arg-type]
    elif fmt in ("csv", "tsv"):
        if fmt == "tsv" and "separator" not in opts and "delimiter" not in opts:
            opts["separator"] = "\t"
        lf = pl.scan_csv(path, infer_schema_length=10_000, **opts)  # type: ignore[arg-type]
    elif fmt == "json":
        lf = pl.read_json(path, **opts).lazy()  # type: ignore[arg-type]
    elif fmt == "jsonl":
        lf = pl.scan_ndjson(path, infer_schema_length=10_000, **opts)  # type: ignore[arg-type]
    else:
        raise SourceError(f"Unrecognised format '{fmt}' for {path}")

    _assert_non_degenerate(lf, path)
    return lf


def _detect_format(path: Path, config: SourceConfig) -> str:
    if config.format != "auto":
        return config.format

    name = path.name.lower()
    for fmt, exts in _FORMAT_EXTENSIONS.items():
        if any(name.endswith(ext) for ext in exts):
            return fmt

    raise SourceError(
        f"Cannot auto-detect format for '{path}'. "
        "Set source.format explicitly (csv, tsv, parquet, json, jsonl)."
    )


def _assert_non_degenerate(lf: pl.LazyFrame, path: Path) -> None:
    schema = lf.collect_schema()
    n_cols = len(schema)

    if n_cols == 0:
        raise SchemaError(f"'{path}' produced a frame with no columns.")

    if n_cols == 1:
        raise SchemaError(
            f"'{path}' has only one column — this usually means the delimiter "
            "was not detected correctly. Check source.read_options['separator']."
        )

    n_string = sum(1 for dtype in schema.values() if dtype == pl.String)
    if n_string == n_cols:
        raise SchemaError(
            f"All {n_cols} columns in '{path}' were inferred as String. "
            "The file may have too many header rows, a wrong delimiter, "
            "or schema inference may need a larger infer_schema_length."
        )
