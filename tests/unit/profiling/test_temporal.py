"""Unit tests for date/time-as-text detection and application (profiling/temporal.py)."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import polars as pl
import pytest

from sorethumb_ml.config import ColumnsConfig
from sorethumb_ml.errors import AmbiguousDateFormatWarning, ConfigError
from sorethumb_ml.profiling.temporal import (
    TemporalParse,
    apply_temporal_parses,
    convert_durations_to_seconds,
    detect_temporal_parses,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Auto-detection: one format family at a time
# ---------------------------------------------------------------------------


def test_detects_iso_date() -> None:
    df = pl.DataFrame({"when": ["2026-01-01", "2026-02-15", "2026-03-20"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed == {"when": TemporalParse(kind="date", format=None, forced=False)}
    out = apply_temporal_parses(df, parsed)
    assert out.schema["when"] == pl.Date


def test_detects_iso_datetime_with_mixed_separators_and_fractional_seconds() -> None:
    df = pl.DataFrame({"when": ["2026-01-01T10:30:00", "2026-02-15 11:00:00", "2026-03-20T12:00:00.5"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed["when"].kind == "datetime"
    out = apply_temporal_parses(df, parsed)
    assert str(out.schema["when"]).startswith("Datetime")


def test_detects_time_of_day() -> None:
    df = pl.DataFrame({"at": ["10:30:00", "23:59:59", "00:00:00"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed["at"] == TemporalParse(kind="time", format="%H:%M:%S%.f", forced=False)
    out = apply_temporal_parses(df, parsed)
    assert out.schema["at"] == pl.Time


def test_plain_categorical_text_is_not_detected() -> None:
    df = pl.DataFrame({"region": ["us", "eu", "apac", "us"]})
    assert detect_temporal_parses(df, ColumnsConfig()) == {}


def test_numeric_column_is_never_auto_detected_as_time() -> None:
    """Epoch integers look exactly like any other number; only a config override names one."""
    df = pl.DataFrame({"ts": [1700000000, 1700086400, 1700172800]})
    assert detect_temporal_parses(df, ColumnsConfig()) == {}


# ---------------------------------------------------------------------------
# Day-first / month-first resolution
# ---------------------------------------------------------------------------


def test_day_first_order_is_detected_when_the_sample_settles_it() -> None:
    """day=31 can only be day-first: month-first would be invalid for every row."""
    df = pl.DataFrame({"when": ["31/01/2026", "15/02/2026", "20/03/2026"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed["when"] == TemporalParse(kind="date", format="%d/%m/%Y", forced=False)


def test_month_first_order_is_detected_when_the_sample_settles_it() -> None:
    df = pl.DataFrame({"when": ["12/31/2026", "02/15/2026"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed["when"] == TemporalParse(kind="date", format="%m/%d/%Y", forced=False)


def test_ambiguous_day_month_order_is_left_as_text_and_warns() -> None:
    """Every value <= 12 in both positions: day-first and month-first both parse -- refuse to guess."""
    df = pl.DataFrame({"when": ["01/02/2026", "03/04/2026", "05/06/2026"]})
    with pytest.warns(AmbiguousDateFormatWarning, match="when.*day-first.*month-first"):
        parsed = detect_temporal_parses(df, ColumnsConfig())
    assert "when" not in parsed


def test_ambiguous_order_is_never_silently_resolved_via_polars_own_default() -> None:
    """Regression: polars' own flexible to_date(format=None) silently assumes day-first for
    ANY year-last numeric date (any separator) -- this must never be the path that decides."""
    df = pl.DataFrame({"when": ["01/02/2026", "03/04/2026"]})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        parsed = detect_temporal_parses(df, ColumnsConfig())
    assert "when" not in parsed
    out = apply_temporal_parses(df, parsed)
    assert out.schema["when"] == pl.String


@pytest.mark.parametrize("sep", ["/", "-", "."])
def test_day_first_resolution_works_for_every_separator(sep: str) -> None:
    df = pl.DataFrame({"when": [f"31{sep}01{sep}2026", f"15{sep}02{sep}2026"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed["when"].format == f"%d{sep}%m{sep}%Y"


# ---------------------------------------------------------------------------
# Forced parsing (columns.datetime_formats)
# ---------------------------------------------------------------------------


def test_forced_format_overrides_detection() -> None:
    """Forcing resolves what auto-detection alone would refuse as ambiguous."""
    df = pl.DataFrame({"when": ["01/02/2026", "03/04/2026"]})
    parsed = detect_temporal_parses(df, ColumnsConfig(datetime_formats={"when": "%m/%d/%Y"}))
    assert parsed["when"] == TemporalParse(kind="date", format="%m/%d/%Y", forced=True)


def test_forced_format_with_no_time_directive_yields_a_real_date() -> None:
    df = pl.DataFrame({"when": ["31/01/2026"]})
    parsed = detect_temporal_parses(df, ColumnsConfig(datetime_formats={"when": "%d/%m/%Y"}))
    assert parsed["when"].kind == "date"


def test_forced_format_with_a_time_directive_yields_a_datetime() -> None:
    df = pl.DataFrame({"when": ["31/01/2026 10:30"]})
    parsed = detect_temporal_parses(df, ColumnsConfig(datetime_formats={"when": "%d/%m/%Y %H:%M"}))
    assert parsed["when"].kind == "datetime"


def test_forced_epoch_seconds() -> None:
    df = pl.DataFrame({"ts": [1700000000, 1700086400]})
    parsed = detect_temporal_parses(df, ColumnsConfig(datetime_formats={"ts": "epoch_s"}))
    assert parsed == {"ts": TemporalParse(kind="epoch_s", format=None, forced=True)}
    out = apply_temporal_parses(df, parsed)
    assert str(out.schema["ts"]).startswith("Datetime")


def test_forced_epoch_milliseconds() -> None:
    df = pl.DataFrame({"ts": [1700000000000, 1700086400000]})
    parsed = detect_temporal_parses(df, ColumnsConfig(datetime_formats={"ts": "epoch_ms"}))
    out = apply_temporal_parses(df, parsed)
    assert out["ts"][0].year == 2023


def test_unknown_column_in_datetime_formats_is_a_config_error() -> None:
    df = pl.DataFrame({"amount": [1.0, 2.0]})
    with pytest.raises(ConfigError, match="nope"):
        detect_temporal_parses(df, ColumnsConfig(datetime_formats={"nope": "%Y-%m-%d"}))


def test_epoch_spec_on_a_string_column_is_a_config_error() -> None:
    df = pl.DataFrame({"when": ["2026-01-01"]})
    with pytest.raises(ConfigError, match="integer or float"):
        detect_temporal_parses(df, ColumnsConfig(datetime_formats={"when": "epoch_s"}))


def test_strptime_format_on_a_numeric_column_is_a_config_error() -> None:
    df = pl.DataFrame({"amount": [1.0, 2.0]})
    with pytest.raises(ConfigError, match="String column"):
        detect_temporal_parses(df, ColumnsConfig(datetime_formats={"amount": "%Y-%m-%d"}))


# ---------------------------------------------------------------------------
# Partial parse failure -> null, not a crash
# ---------------------------------------------------------------------------


def test_unparseable_values_become_null_and_log_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    df = pl.DataFrame({"when": ["2026-01-01", "not-a-date", "2026-01-03"]})
    parsed = {"when": TemporalParse(kind="date", format=None, forced=True)}
    with caplog.at_level("WARNING", logger="sorethumb_ml.profiling.temporal"):
        out = apply_temporal_parses(df, parsed)
    assert out["when"].to_list() == [date(2026, 1, 1), None, date(2026, 1, 3)]
    assert "1 of 3" in caplog.text


def test_preexisting_nulls_are_not_counted_as_parse_failures(caplog: pytest.LogCaptureFixture) -> None:
    df = pl.DataFrame({"when": ["2026-01-01", None, "2026-01-03"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    with caplog.at_level("WARNING", logger="sorethumb_ml.profiling.temporal"):
        apply_temporal_parses(df, parsed)
    assert "did not parse" not in caplog.text


# ---------------------------------------------------------------------------
# Idempotency: apply_temporal_parses must be safe to call on already-cast data
# ---------------------------------------------------------------------------


def test_apply_is_idempotent_on_already_cast_columns() -> None:
    df = pl.DataFrame({"when": ["2026-01-01", "2026-02-15"]})
    parsed = detect_temporal_parses(df, ColumnsConfig())
    once = apply_temporal_parses(df, parsed)
    twice = apply_temporal_parses(once, parsed)
    assert once.equals(twice)


def test_apply_skips_a_column_missing_from_the_frame() -> None:
    """apply_feature_plan's own schema_fingerprint check is what reports a missing column."""
    df = pl.DataFrame({"amount": [1.0]})
    parsed = {"when": TemporalParse(kind="date", format=None, forced=False)}
    out = apply_temporal_parses(df, parsed)
    assert out.columns == ["amount"]


def test_apply_with_no_parses_is_a_no_op_returning_the_same_frame() -> None:
    df = pl.DataFrame({"amount": [1.0]})
    assert apply_temporal_parses(df, {}) is df


# ---------------------------------------------------------------------------
# Duration -> total seconds
# ---------------------------------------------------------------------------


def test_duration_becomes_total_seconds() -> None:
    df = pl.DataFrame({"dur": [timedelta(seconds=90), timedelta(minutes=5), None]})
    out = convert_durations_to_seconds(df)
    assert out.schema["dur"] in (pl.Int64, pl.Int32)
    assert out["dur"].to_list() == [90, 300, None]


def test_convert_durations_is_a_no_op_with_none_present() -> None:
    df = pl.DataFrame({"amount": [1.0]})
    assert convert_durations_to_seconds(df) is df


# ---------------------------------------------------------------------------
# TemporalParse (de)serialisation, for FeaturePlan round-tripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parse",
    [
        TemporalParse(kind="date", format=None, forced=False),
        TemporalParse(kind="datetime", format="%d/%m/%Y %H:%M", forced=True),
        TemporalParse(kind="epoch_s", format=None, forced=True),
    ],
)
def test_temporal_parse_dict_roundtrip(parse: TemporalParse) -> None:
    assert TemporalParse.from_dict(parse.to_dict()) == parse


# ---------------------------------------------------------------------------
# Columns auto-detection must leave alone
# ---------------------------------------------------------------------------

_TEXT_DATES = ["2026-01-01", "2026-01-02", "2026-01-03"]


@pytest.mark.parametrize(
    "columns_kw",
    [{"group_by": ["day"]}, {"id_column": "day"}, {"reference_column": "day"}],
    ids=["group_by", "id_column", "reference_column"],
)
def test_identity_columns_are_never_auto_detected(columns_kw: dict[str, object]) -> None:
    """Recasting a group/id/reference column changes its keys and orphans earlier history."""
    df = pl.DataFrame({"day": _TEXT_DATES})
    assert detect_temporal_parses(df, ColumnsConfig(**columns_kw)) == {}  # type: ignore[arg-type]


def test_time_column_is_still_parsed_even_when_it_is_also_a_group_column() -> None:
    df = pl.DataFrame({"day": _TEXT_DATES})
    parsed = detect_temporal_parses(df, ColumnsConfig(group_by=["day"], time_column="day"))
    assert "day" in parsed


def test_a_forced_format_still_applies_to_an_identity_column() -> None:
    """Auto-detection stays out; an explicit datetime_formats entry is the user's own choice."""
    df = pl.DataFrame({"day": _TEXT_DATES})
    cfg = ColumnsConfig(group_by=["day"], datetime_formats={"day": "%Y-%m-%d"})
    assert detect_temporal_parses(df, cfg)["day"].forced is True


@pytest.mark.parametrize("pattern", ["type:String *", "type:String d*", "day"])
def test_columns_an_ignore_pattern_matches_on_their_original_dtype_are_not_detected(
    pattern: str,
) -> None:
    """Regression: 'type:String *' stopped matching a text date once it was recast to Date."""
    df = pl.DataFrame({"day": _TEXT_DATES})
    assert detect_temporal_parses(df, ColumnsConfig(ignore=[pattern])) == {}


def test_an_ignore_pattern_for_another_dtype_does_not_block_detection() -> None:
    df = pl.DataFrame({"day": _TEXT_DATES})
    assert "day" in detect_temporal_parses(df, ColumnsConfig(ignore=["type:Int64 *"]))


# ---------------------------------------------------------------------------
# run.strict and exact dtype checks
# ---------------------------------------------------------------------------


def test_ambiguous_date_warning_is_never_promoted_to_an_error() -> None:
    """Like ReportGenerationWarning: the outcome (column stays text) is what earlier versions
    did, so a strict run must not start failing over it."""
    from sorethumb_ml.errors import SorethumbWarning

    df = pl.DataFrame({"when": ["01/02/2026", "03/04/2026"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", SorethumbWarning)  # what run.strict installs
        parsed = detect_temporal_parses(df, ColumnsConfig())
    assert parsed == {}
    assert any(issubclass(w.category, AmbiguousDateFormatWarning) for w in caught)


def test_a_date_parse_is_not_skipped_for_a_column_that_is_already_a_datetime() -> None:
    """Regression: a str(dtype).startswith('Date') check treated Datetime as already a Date."""
    df = pl.DataFrame({"d": ["2026-01-01 10:00:00"]}).with_columns(pl.col("d").str.to_datetime())
    out = apply_temporal_parses(df, {"d": TemporalParse(kind="date", format=None, forced=False)})
    assert isinstance(out.schema["d"], pl.Datetime)  # left alone: not a String, nothing to parse
