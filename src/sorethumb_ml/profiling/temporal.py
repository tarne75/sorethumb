"""Detect and apply date/time parsing for String (and epoch-integer) columns.

Before this module runs, a date or timestamp stored as text is just another
String column to the profiler -- frequency- or one-hot-encoded like any other
category, with no notion of hour, day-of-week or calendar order. Detection
turns such a column into a real ``Date`` / ``Datetime`` / ``Time`` column
*before* profiling ever sees it, so every existing temporal code path
(classification, ``derive_time``, period-window filtering) applies unchanged,
with no separate "string time" handling anywhere else.

Two ways a column becomes temporal:

1. **Auto-detected.** A String column where at least ``_MIN_PARSE_RATE`` of
   the non-null sampled values parse under exactly one of a fixed, ordered
   list of formats (ISO datetime, ISO date, time-of-day, then day-first or
   month-first numeric dates). Ambiguous day/month order (both directions
   parse acceptably) is never guessed -- the column is left as a string and a
   warning names ``columns.datetime_formats`` as the way to resolve it.
2. **Forced**, via ``columns.datetime_formats[col]``: a strptime-style format
   for a String column, or ``"epoch_s"`` / ``"epoch_ms"`` for an integer or
   float column. This is the only way an integer column is ever treated as
   time, and the only way to resolve an ambiguous date order.

Values that still fail to parse (forced or auto-detected) become null, same
as any other missing value -- not a reason to fail the run.

:func:`apply_temporal_parses` is idempotent: a column already at its target
dtype (the in-process pipeline casts once, early, before every later call)
is left alone, so it is safe to call again on already-parsed data, which is
exactly what a fresh ``score --from-run`` load needs.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl

from sorethumb_ml.config import ColumnsConfig
from sorethumb_ml.errors import AmbiguousDateFormatWarning, ConfigError
from sorethumb_ml.profiling.classify import _is_ignored, is_included

logger = logging.getLogger(__name__)

TemporalKind = Literal["date", "datetime", "time", "epoch_s", "epoch_ms"]

_MIN_PARSE_RATE = 0.99
_SAMPLE_SIZE = 2000

# Tried in order; the first to clear _MIN_PARSE_RATE wins. "%-separated" day/month
# candidates are generated per separator actually present in the sample (see
# _detect_ambiguous_numeric_date), not listed here.
_DATE_SEPARATORS: tuple[str, ...] = ("/", "-", ".")

# A leading 4-digit year unambiguously fixes field order (ISO 8601: year first).
# Polars' own format=None inference silently assumes day-first for every OTHER
# numeric date shape (01-02-2026, 01/02/2026, 01.02.2026 alike) -- exactly the
# guess this module refuses to make, so such a sample must skip _detect_iso
# entirely and go to _detect_ambiguous_numeric_date's explicit resolve-or-refuse.
_YEAR_FIRST_RE = re.compile(r"^\d{4}[-/.]")


@dataclass(frozen=True)
class TemporalParse:
    """How one column was turned into a real date/time value."""

    kind: TemporalKind
    format: str | None  # strptime format for kind in (date, datetime, time); None = ISO-flexible
    forced: bool  # True if this came from columns.datetime_formats, not auto-detection

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict form for ``FeaturePlan.to_json``."""
        return {"kind": self.kind, "format": self.format, "forced": self.forced}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> TemporalParse:
        """Restore from the dict produced by :meth:`to_dict`."""
        return cls(kind=d["kind"], format=d["format"], forced=d["forced"])  # type: ignore[arg-type]


def _parse_rate(parsed: pl.Series, original: pl.Series) -> float:
    """Fraction of *original*'s non-null values that *parsed* turned into a non-null value."""
    non_null = original.is_not_null()
    denom = int(non_null.sum())
    if denom == 0:
        return 0.0
    return float((parsed.is_not_null() & non_null).sum()) / denom


def _try_parse_rate(expr_fn: Callable[[], pl.Series], original: pl.Series) -> float:
    """``_parse_rate`` of a parsing attempt that may itself fail to resolve any format at all.

    Polars' auto format inference (``format=None``, or an explicit format too
    heterogeneous for the sample) can raise ``ComputeError`` rather than
    nulling every row when ``strict=False`` -- a column this candidate simply
    does not apply to, not a crash.
    """
    try:
        return _parse_rate(expr_fn(), original)
    except pl.exceptions.ComputeError:
        return 0.0


def _looks_year_first(series: pl.Series) -> bool:
    sample = series.drop_nulls()
    if sample.is_empty():
        return False
    return bool(_YEAR_FIRST_RE.match(sample[0]))


def _detect_iso(series: pl.Series) -> TemporalParse | None:
    if not _looks_year_first(series):
        return None  # ambiguous numeric date shape: _detect_ambiguous_numeric_date decides
    if _try_parse_rate(lambda: series.str.to_date(strict=False), series) >= _MIN_PARSE_RATE:
        return TemporalParse(kind="date", format=None, forced=False)
    if _try_parse_rate(lambda: series.str.to_datetime(strict=False), series) >= _MIN_PARSE_RATE:
        return TemporalParse(kind="datetime", format=None, forced=False)
    return None


def _to_time(series: pl.Series, fmt: str) -> pl.Series:
    return series.str.to_time(format=fmt, strict=False)


def _detect_time_of_day(series: pl.Series) -> TemporalParse | None:
    for fmt in ("%H:%M:%S%.f", "%H:%M:%S", "%H:%M"):
        rate = _try_parse_rate(lambda: _to_time(series, fmt), series)  # noqa: B023
        if rate >= _MIN_PARSE_RATE:
            return TemporalParse(kind="time", format=fmt, forced=False)
    return None


def _detect_ambiguous_numeric_date(series: pl.Series, column: str) -> TemporalParse | None:
    """Day-first vs month-first dates like ``31/01/2026``: accepted only when the sample settles it."""
    sample = series.drop_nulls()
    if sample.is_empty():
        return None
    first_value = sample[0]
    sep = next((s for s in _DATE_SEPARATORS if s in first_value), None)
    if sep is None:
        return None

    day_first = f"%d{sep}%m{sep}%Y"
    month_first = f"%m{sep}%d{sep}%Y"
    day_first_rate = _try_parse_rate(lambda: series.str.to_date(format=day_first, strict=False), series)
    month_first_rate = _try_parse_rate(lambda: series.str.to_date(format=month_first, strict=False), series)
    day_ok = day_first_rate >= _MIN_PARSE_RATE
    month_ok = month_first_rate >= _MIN_PARSE_RATE

    if day_ok and month_ok:
        # Never promoted by run.strict (see AmbiguousDateFormatWarning): the outcome --
        # the column stays text -- is exactly what every earlier version did.
        with warnings.catch_warnings():
            warnings.simplefilter("always", AmbiguousDateFormatWarning)
            warnings.warn(
                f"column {column!r}: values parse as a date under both day-first ({day_first!r}) "
                f"and month-first ({month_first!r}) order -- too ambiguous to guess, left as text. "
                f"Set columns.datetime_formats[{column!r}] to the correct one to parse it.",
                AmbiguousDateFormatWarning,
                stacklevel=2,
            )
        return None
    if day_ok:
        return TemporalParse(kind="date", format=day_first, forced=False)
    if month_ok:
        return TemporalParse(kind="date", format=month_first, forced=False)
    return None


def _detect_column(series: pl.Series, column: str) -> TemporalParse | None:
    sample = series if len(series) <= _SAMPLE_SIZE else series.sample(_SAMPLE_SIZE, seed=0)
    for detector in (_detect_iso, _detect_time_of_day):
        found = detector(sample)
        if found is not None:
            return found
    return _detect_ambiguous_numeric_date(sample, column)


_TIME_DIRECTIVES = ("%H", "%M", "%S", "%f")


def _forced_parse(column: str, spec: str, dtype: pl.DataType) -> TemporalParse:
    if spec in ("epoch_s", "epoch_ms"):
        if dtype not in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
            raise ConfigError(
                f"columns.datetime_formats[{column!r}] = {spec!r} requires an integer or float "
                f"column; {column!r} is {dtype}."
            )
        return TemporalParse(kind=spec, format=None, forced=True)  # type: ignore[arg-type]
    if dtype != pl.String:
        raise ConfigError(
            f"columns.datetime_formats[{column!r}] = {spec!r} is a strptime format, which requires "
            f"a String column; {column!r} is {dtype}."
        )
    # A format with no time directive is a pure date -- giving it kind="date" rather than
    # always "datetime" means it gets a real Date dtype (and skips the always-zero "hour"
    # derivative a Datetime stuck at midnight would otherwise emit).
    kind: TemporalKind = "datetime" if any(d in spec for d in _TIME_DIRECTIVES) else "date"
    return TemporalParse(kind=kind, format=spec, forced=True)


def detect_temporal_parses(df: pl.DataFrame, columns_config: ColumnsConfig) -> dict[str, TemporalParse]:
    """Return ``{column: TemporalParse}`` for every column that should become a real date/time.

    Raises:
        ConfigError: a ``columns.datetime_formats`` entry names a column not in *df*, or one
            whose dtype cannot take that override (an epoch spec on a non-numeric column, or a
            format string on a non-String column).

    """
    unknown = set(columns_config.datetime_formats) - set(df.columns)
    if unknown:
        raise ConfigError(f"columns.datetime_formats names column(s) not in the dataset: {sorted(unknown)}")

    parses: dict[str, TemporalParse] = {}
    for column in df.columns:
        dtype = df.schema[column]
        forced = columns_config.datetime_formats.get(column)
        if forced is not None:
            parses[column] = _forced_parse(column, forced, dtype)
        elif dtype == pl.String and not _excluded_from_auto_detection(column, dtype, columns_config):
            found = _detect_column(df[column], column)
            if found is not None:
                parses[column] = found
    return parses


def _excluded_from_auto_detection(column: str, dtype: pl.DataType, columns_config: ColumnsConfig) -> bool:
    """Columns auto-detection must leave as text (``datetime_formats`` can still force them).

    * Identity columns -- ``group_by``, ``id_column``, ``reference_column`` --
      unless the column is also ``time_column``: their values become group
      keys, row ids and labels, and recasting them changes those keys,
      orphaning every earlier run's history and models for those groups.
    * Columns ``include_only`` leaves out, and columns an ``ignore`` pattern
      matches against their *original* dtype:
      ``"type:String *"`` was written about the column as it appears in the
      source, and must not stop matching because detection recast it.
    """
    if column == columns_config.time_column:
        return False
    identity = {c for c in (columns_config.id_column, columns_config.reference_column) if c}
    identity |= set(columns_config.group_by)
    if column in identity:
        return True
    protected = identity | ({columns_config.time_column} if columns_config.time_column else set())
    if not is_included(column, columns_config, protected):
        return True
    return _is_ignored(column, str(dtype), columns_config.ignore, protected)


def parse_temporal_columns(
    df: pl.DataFrame, columns_config: ColumnsConfig
) -> tuple[pl.DataFrame, dict[str, TemporalParse]]:
    """Detect, cast and convert in one step: what every fresh (non-replay) plan build needs.

    Returns the cast frame and the detection result to record on the
    ``FeaturePlan`` (``build_feature_plan(..., parsed_temporal=...)``).
    """
    parsed = detect_temporal_parses(df, columns_config)
    df = apply_temporal_parses(df, parsed)
    return convert_durations_to_seconds(df), parsed


def _cast_expr(column: str, parse: TemporalParse) -> pl.Expr:
    col = pl.col(column)
    if parse.kind == "epoch_s":
        return pl.from_epoch(col, time_unit="s").alias(column)
    if parse.kind == "epoch_ms":
        return pl.from_epoch(col, time_unit="ms").alias(column)
    if parse.kind == "date":
        return col.str.to_date(format=parse.format, strict=False).alias(column)
    if parse.kind == "datetime":
        return col.str.to_datetime(format=parse.format, strict=False).alias(column)
    return col.str.to_time(format=parse.format, strict=False).alias(column)  # kind == "time"


def _still_in_source_dtype(dtype: pl.DataType, kind: TemporalKind) -> bool:
    """Whether *dtype* is what this parse reads from: text, or a number for epoch kinds.

    Anything else -- already cast by an earlier call, or a column whose dtype
    drifted since the plan was fitted -- is left alone: re-parsing is a no-op
    in the first case, and in the second ``apply_feature_plan``'s schema
    fingerprint check is what reports it. (An earlier prefix test,
    ``str(dtype).startswith("Date")``, also matched ``Datetime``.)
    """
    if kind in ("epoch_s", "epoch_ms"):
        return dtype.is_numeric()
    return dtype == pl.String


def apply_temporal_parses(df: pl.DataFrame, parsed: dict[str, TemporalParse]) -> pl.DataFrame:
    """Cast every column named in *parsed* to its target date/time dtype.

    Idempotent: only a column still in its source dtype (text, or a number for
    an epoch parse) is cast. Calling this twice on the same already-cast frame
    -- the in-process pipeline casts once, early, before ``apply_feature_plan``
    calls this again -- is therefore a no-op, and a fresh ``score --from-run``
    load (still raw text) gets cast for the first time here.
    """
    exprs = []
    for column, parse in parsed.items():
        if column not in df.columns:
            continue  # apply_feature_plan's own schema_fingerprint check reports a missing column
        if not _still_in_source_dtype(df.schema[column], parse.kind):
            continue  # already parsed (or drifted -- see _still_in_source_dtype)
        exprs.append(_cast_expr(column, parse))
    if not exprs:
        return df

    before_non_null = {
        column: int(df[column].is_not_null().sum()) for column in parsed if column in df.columns
    }
    df = df.with_columns(exprs)
    for column, parse in parsed.items():
        if column not in before_non_null:
            continue
        after_non_null = int(df[column].is_not_null().sum())
        lost = before_non_null[column] - after_non_null
        if lost > 0:
            logger.warning(
                "column %r: %d of %d non-null value(s) did not parse as %s (format=%r) and became null.",
                column,
                lost,
                before_non_null[column],
                parse.kind,
                parse.format,
            )
    return df


def convert_durations_to_seconds(df: pl.DataFrame) -> pl.DataFrame:
    """Cast every ``Duration`` column to its total seconds (a plain numeric ``Int64``).

    A duration has no calendar position (no hour-of-day, no day-of-week), so
    ``derive_time``'s per-dtype derivative set is empty for it -- left as
    ``Duration``, the column would be classified ``temporal`` and then silently
    produce zero output features. Converting it to a number first makes it an
    ordinary numeric column instead, carrying the one signal it actually has:
    magnitude. Idempotent: no ``Duration`` columns is a no-op.
    """
    exprs = [pl.col(c).dt.total_seconds().alias(c) for c, dtype in df.schema.items() if dtype == pl.Duration]
    return df.with_columns(exprs) if exprs else df
