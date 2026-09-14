"""Unit tests for sorethumb.history.periods: pure period-label/window math.

No Workspace, no SQLite -- see tests/integration/test_history.py for the
ledger/totals/rolling-window tests that need a real store.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sorethumb.history.periods import (
    period_bounds,
    period_range,
    resolve_period,
    step_back,
    step_forward,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# periods.py — resolve_period
# ---------------------------------------------------------------------------


class TestResolvePeriod:
    def test_day_label_equals_period_from(self):
        ref = datetime(2026, 9, 3, 14, 30, tzinfo=UTC)
        frm, to, label = resolve_period(ref, "day", False)
        assert label == frm == "2026-09-03"
        assert to == "2026-09-04"

    def test_day_roll_weekend_saturday(self):
        sat = datetime(2026, 9, 5, 10, 0, tzinfo=UTC)  # Saturday
        frm, _, label = resolve_period(sat, "day", True)
        assert label == "2026-09-04"  # Friday

    def test_day_roll_weekend_sunday(self):
        sun = datetime(2026, 9, 6, 10, 0, tzinfo=UTC)  # Sunday
        frm, _, label = resolve_period(sun, "day", True)
        assert label == "2026-09-04"  # Friday

    def test_day_no_roll_weekday(self):
        wed = datetime(2026, 9, 2, 0, 0, tzinfo=UTC)  # Wednesday
        frm, _, label = resolve_period(wed, "day", True)
        assert label == "2026-09-02"

    def test_day_window_half_open(self):
        ref = datetime(2026, 9, 3, tzinfo=UTC)
        frm, to, _ = resolve_period(ref, "day", False)
        assert frm == "2026-09-03"
        assert to == "2026-09-04"

    def test_week_label_is_monday(self):
        wed = datetime(2026, 9, 2, tzinfo=UTC)  # Wednesday
        frm, to, label = resolve_period(wed, "week", False)
        assert label == frm == "2026-08-31"  # Monday
        assert to == "2026-09-07"

    def test_month_label_is_first(self):
        ref = datetime(2026, 9, 15, tzinfo=UTC)
        frm, to, label = resolve_period(ref, "month", False)
        assert label == frm == "2026-09-01"
        assert to == "2026-10-01"

    def test_month_december_wraps(self):
        ref = datetime(2026, 12, 20, tzinfo=UTC)
        frm, to, label = resolve_period(ref, "month", False)
        assert label == "2026-12-01"
        assert to == "2027-01-01"

    def test_hour_label_and_window(self):
        ref = datetime(2026, 9, 3, 14, 45, tzinfo=UTC)
        frm, to, label = resolve_period(ref, "hour", False)
        assert label == frm == "2026-09-03T14"
        assert to == "2026-09-03T15"


class TestPeriodBounds:
    """period_bounds(label, g) must reproduce the window resolve_period built."""

    @pytest.mark.parametrize(
        ("ref", "granularity"),
        [
            (datetime(2026, 9, 3, 14, 30, tzinfo=UTC), "day"),
            (datetime(2026, 9, 2, tzinfo=UTC), "week"),
            (datetime(2026, 12, 20, tzinfo=UTC), "month"),
            (datetime(2026, 9, 3, 14, 45, tzinfo=UTC), "hour"),
        ],
    )
    def test_round_trips_resolve_period(self, ref, granularity):
        frm, to, label = resolve_period(ref, granularity, roll_non_business=False)
        assert period_bounds(label, granularity) == (frm, to)


# ---------------------------------------------------------------------------
# periods.py — step_back / step_forward / period_range
# ---------------------------------------------------------------------------


class TestStepNavigation:
    def test_step_back_day(self):
        assert step_back("2026-09-03", "day", 3) == "2026-08-31"

    def test_step_forward_day(self):
        assert step_forward("2026-09-03", "day", 1) == "2026-09-04"

    def test_step_back_week(self):
        assert step_back("2026-09-07", "week", 1) == "2026-08-31"

    def test_step_forward_week(self):
        assert step_forward("2026-08-31", "week", 1) == "2026-09-07"

    def test_step_back_month_crosses_year(self):
        assert step_back("2026-01-01", "month", 1) == "2025-12-01"

    def test_step_forward_month_crosses_year(self):
        assert step_forward("2025-12-01", "month", 1) == "2026-01-01"

    def test_step_back_hour(self):
        assert step_back("2026-09-03T14", "hour", 2) == "2026-09-03T12"

    def test_step_forward_hour(self):
        assert step_forward("2026-09-03T23", "hour", 1) == "2026-09-04T00"

    def test_period_range_inclusive(self):
        labels = period_range("2026-09-01", "2026-09-03", "day")
        assert labels == ["2026-09-01", "2026-09-02", "2026-09-03"]

    def test_period_range_single(self):
        assert period_range("2026-09-01", "2026-09-01", "day") == ["2026-09-01"]

    def test_period_range_empty_when_start_after_end(self):
        assert period_range("2026-09-04", "2026-09-01", "day") == []

    def test_period_range_month_four_periods(self):
        labels = period_range("2026-01-01", "2026-04-01", "month")
        assert labels == ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
