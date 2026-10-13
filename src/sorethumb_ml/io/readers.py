"""Format-detecting lazy readers.

Every reader returns a ``pl.LazyFrame`` so the caller controls when work
actually happens. Format is detected from the file extension when
``config.format == 'auto'``.

A single-column or all-string CSV/TSV is valid data on its own (one numeric
column, or several categorical columns) and is accepted. It is refused only
when there is concrete evidence something else is wrong:

- the header holds a delimiter character that was never applied (a
  character from ``_CANDIDATE_DELIMITERS`` other than the one configured,
  appearing in the first line); or
- (more than one column, all inferred String) a full-file re-scan finds a
  non-string column among them -- the sampled ``infer_schema_length`` rows
  just happened to look non-numeric (blank leading rows are the common case).

Parquet, JSON, JSONL and TSF have no delimiter to get wrong, so none of this
applies to them.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import polars as pl

from sorethumb_ml.config import SourceConfig
from sorethumb_ml.errors import SchemaError, SourceError

logger = logging.getLogger(__name__)

_CANDIDATE_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")

_FORMAT_EXTENSIONS: dict[str, list[str]] = {
    "parquet": [".parquet"],
    "csv": [".csv", ".csv.gz"],
    "tsv": [".tsv", ".tsv.gz"],
    "json": [".json", ".json.gz"],
    "jsonl": [".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz"],
    "tsf": [".tsf"],
}


def read_frame(path: Path, config: SourceConfig) -> pl.LazyFrame:
    """Return a ``LazyFrame`` for *path* according to *config*.

    Raises:
        SourceError: Format cannot be determined or path does not exist.
        SchemaError: The frame has no columns at all, or (CSV/TSV only) the
            header shows direct evidence of an untried delimiter, or a
            full-file re-scan of an all-string schema finds a numeric column
            the sampled ``infer_schema_length`` missed.

    """
    fmt = _detect_format(path, config)
    logger.debug("Reading %s as '%s'", path, fmt)

    opts: dict[str, object] = dict(config.read_options)

    if fmt == "parquet":
        lf = pl.scan_parquet(path, **opts)  # type: ignore[arg-type]
    elif fmt in ("csv", "tsv"):
        if fmt == "tsv" and "separator" not in opts and "delimiter" not in opts:
            opts["separator"] = "\t"
        opts.setdefault("infer_schema_length", 10_000)
        lf = pl.scan_csv(path, **opts)  # type: ignore[arg-type]
    elif fmt == "json":
        lf = pl.read_json(path, **opts).lazy()  # type: ignore[arg-type]
    elif fmt == "jsonl":
        opts.setdefault("infer_schema_length", 10_000)
        lf = pl.scan_ndjson(path, **opts)  # type: ignore[arg-type]
    elif fmt == "tsf":
        lf = _read_tsf(path)
    else:
        raise SourceError(f"Unrecognised format '{fmt}' for {path}")

    _assert_non_degenerate(lf, path, fmt, opts)
    return lf


def collect_frame(lf: pl.LazyFrame, path: Path) -> pl.DataFrame:
    """Collect a LazyFrame from :func:`read_frame`, explaining a non-UTF-8 source.

    Polars reads CSV/TSV/JSON as UTF-8 only, and a file in another encoding
    (cp1252 is what Excel's plain "CSV" format writes on Windows) only fails
    when the rows are read, as ``ComputeError: invalid utf-8 sequence``. That
    becomes a SourceError saying what's wrong and how to fix it.
    """
    try:
        return lf.collect()
    except pl.exceptions.ComputeError as exc:
        if "utf-8" not in str(exc).lower() and "utf8" not in str(exc).lower():
            raise
        msg = (
            f"'{path}' is not valid UTF-8 ({exc}). sorethumb reads text files as UTF-8; "
            "a file saved by Excel as plain 'CSV' on Windows is usually cp1252. Re-save it as "
            "'CSV UTF-8', convert it once (see the README's 'Supported file formats'), or set "
            'source.read_options = { encoding = "utf8-lossy" } to read it with undecodable '
            "characters replaced."
        )
        raise SourceError(msg) from exc


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


def _read_tsf(path: Path) -> pl.LazyFrame:
    """Parse a TSF (Time Series Forecasting) file into a LazyFrame.

    Each series becomes one row. Declared @attribute columns come first,
    followed by value_0, value_1, … (the series observations). Variable-length
    series are padded with null to the maximum length found in the file.

    Format reference: https://github.com/rakshitha123/TSForecasting/tree/master/utils
    """
    attributes: list[tuple[str, str]] = []
    data_started = False
    rows: list[dict[str, object]] = []

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            lower = line.lower()

            if lower.startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    attributes.append((parts[1], parts[2].strip().lower()))
                continue

            if lower.startswith("@data"):
                data_started = True
                continue

            if lower.startswith("@") or not data_started:
                continue

            # Each data row: attr1:attr2:...:v1,v2,v3,...
            parts = line.split(":")
            if len(parts) < len(attributes) + 1:
                logger.debug("Skipping malformed TSF row (too few fields): %s", line[:80])
                continue

            row: dict[str, object] = {}
            for i, (attr_name, attr_type) in enumerate(attributes):
                val = parts[i]
                if attr_type == "numeric":
                    try:
                        row[attr_name] = int(val) if "." not in val else float(val)
                    except ValueError:
                        row[attr_name] = None
                else:
                    row[attr_name] = val  # string or date — kept as string

            series_str = parts[len(attributes)]
            values: list[float | None] = []
            for token in series_str.split(","):
                tok = token.strip()
                if tok in ("?", ""):
                    values.append(None)
                else:
                    try:
                        values.append(float(tok))
                    except ValueError:
                        values.append(None)

            for j, v in enumerate(values):
                row[f"value_{j}"] = v

            rows.append(row)

    if not rows:
        schema = {name: pl.String for name, _ in attributes}
        return pl.DataFrame(schema=schema).lazy()

    max_len = max(sum(1 for k in r if k.startswith("value_")) for r in rows)
    for row in rows:
        for j in range(max_len):
            row.setdefault(f"value_{j}", None)

    return pl.DataFrame(rows).lazy()


def _read_first_line(path: Path) -> str | None:
    """Best-effort read of the raw header line, for delimiter sniffing. ``None`` if unreadable."""
    opener = gzip.open if path.name.endswith(".gz") else Path.open
    try:
        with opener(path, mode="rt", encoding="utf-8", errors="replace") as fh:
            return fh.readline()
    except OSError:
        return None


def _untried_delimiter(path: Path, configured_sep: str) -> str | None:
    """Return a delimiter the header shows evidence of, other than the one actually used."""
    header = _read_first_line(path)
    if not header:
        return None
    for candidate in _CANDIDATE_DELIMITERS:
        if candidate != configured_sep and candidate in header:
            return candidate
    return None


def _schema_has_non_string_column(path: Path, opts: dict[str, object]) -> bool:
    """Re-infer the schema over the whole file, to check whether a small sample is the cause.

    Only called for an all-string schema, which is otherwise accepted as
    genuine categorical data -- this distinguishes that from a numeric column
    whose configured or default ``infer_schema_length`` happened to sample
    only non-numeric-looking rows (blank leading rows are the common case).
    """
    full_opts = {**opts, "infer_schema_length": None}
    schema = pl.scan_csv(path, **full_opts).collect_schema()  # type: ignore[arg-type]
    return any(dtype != pl.String for dtype in schema.values())


def _assert_non_degenerate(lf: pl.LazyFrame, path: Path, fmt: str, opts: dict[str, object]) -> None:
    schema = lf.collect_schema()
    n_cols = len(schema)

    if n_cols == 0:
        raise SchemaError(f"'{path}' produced a frame with no columns.")

    if fmt not in ("csv", "tsv"):
        return  # no delimiter concept; a narrow or all-string frame here is simply the data

    n_string = sum(1 for dtype in schema.values() if dtype == pl.String)
    if n_cols > 1 and n_string < n_cols:
        return  # at least one column parsed as non-string: not a flat, undelimited dump

    configured_sep = str(opts.get("separator", opts.get("delimiter", ",")))
    missed = _untried_delimiter(path, configured_sep)
    if missed is not None:
        if n_cols == 1:
            raise SchemaError(
                f"'{path}' has only one column, and its header contains {missed!r} — this usually "
                "means the delimiter was not detected correctly. Check source.read_options['separator']."
            )
        raise SchemaError(
            f"All {n_cols} columns in '{path}' were inferred as String, and the header contains "
            f"{missed!r} — this usually means the delimiter was not detected correctly. "
            "Check source.read_options['separator']."
        )

    if n_cols > 1 and _schema_has_non_string_column(path, opts):
        raise SchemaError(
            f"All {n_cols} columns in '{path}' were inferred as String, but a full scan finds a "
            "numeric column among them -- the configured (or default) "
            "source.read_options['infer_schema_length'] sampled only non-numeric-looking rows "
            "(blank leading rows are a common cause). Set it larger, or to null for an unbounded scan."
        )
    # No evidence of a missed delimiter, and (for n_cols > 1) a full scan agrees every column is
    # genuinely String: a narrow or all-string CSV/TSV is valid data on its own.
