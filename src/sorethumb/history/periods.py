"""Period resolution: convert a reference datetime to a half-open period window.

The period_label always equals period_from. Deriving the label separately from
the window is precisely how off-by-one reporting bugs arise, so the two are
produced together in a single call and must never be computed independently.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

PeriodGranularity = Literal["hour", "day", "week", "month"]

_BUSINESS_WEEKDAYS = frozenset({0, 1, 2, 3, 4})  # Monday–Friday (Python weekday())


def _prev_business_day(d: date) -> date:
    """Return d, or the most recent Monday–Friday if d falls on a weekend."""
    while d.weekday() not in _BUSINESS_WEEKDAYS:
        d -= timedelta(days=1)
    return d


def resolve_period(
    reference: datetime,
    granularity: PeriodGranularity,
    roll_non_business: bool,
) -> tuple[str, str, str]:
    """Return (period_from, period_to_exclusive, period_label) for *reference*.

    period_label == period_from by construction. Rolling the label and window
    separately introduces off-by-one reporting bugs; this function produces both
    from the same truncated reference so they cannot disagree.

    When roll_non_business is True and granularity is 'day', a weekend reference
    is shifted back to the previous business day before truncation. The entire
    window moves with it, so a weekend run never silently merges three periods.
    """
    ref = reference.astimezone(UTC)

    if granularity == "hour":
        truncated = ref.replace(minute=0, second=0, microsecond=0)
        nxt = truncated + timedelta(hours=1)
        label = truncated.strftime("%Y-%m-%dT%H")
        return label, nxt.strftime("%Y-%m-%dT%H"), label

    if granularity == "day":
        ref_date = ref.date()
        if roll_non_business:
            ref_date = _prev_business_day(ref_date)
        label = ref_date.isoformat()
        nxt = (ref_date + timedelta(days=1)).isoformat()
        return label, nxt, label

    if granularity == "week":
        ref_date = ref.date()
        monday = ref_date - timedelta(days=ref_date.weekday())
        if roll_non_business:
            monday = _prev_business_day(monday)
        label = monday.isoformat()
        nxt = (monday + timedelta(weeks=1)).isoformat()
        return label, nxt, label

    if granularity == "month":
        ref_date = ref.date()
        first = ref_date.replace(day=1)
        label = first.isoformat()
        nxt = _add_months(first, 1).isoformat()
        return label, nxt, label

    msg = f"Unknown granularity: {granularity!r}"
    raise ValueError(msg)


def step_back(label: str, granularity: PeriodGranularity, n: int = 1) -> str:
    """Return the period label n periods before *label*."""
    if granularity == "hour":
        dt = _parse_hour_label(label)
        return (dt - timedelta(hours=n)).strftime("%Y-%m-%dT%H")
    if granularity == "day":
        return (date.fromisoformat(label) - timedelta(days=n)).isoformat()
    if granularity == "week":
        return (date.fromisoformat(label) - timedelta(weeks=n)).isoformat()
    if granularity == "month":
        return _add_months(date.fromisoformat(label), -n).isoformat()
    msg = f"Unknown granularity: {granularity!r}"
    raise ValueError(msg)


def step_forward(label: str, granularity: PeriodGranularity, n: int = 1) -> str:
    """Return the period label n periods after *label*."""
    if granularity == "hour":
        dt = _parse_hour_label(label)
        return (dt + timedelta(hours=n)).strftime("%Y-%m-%dT%H")
    if granularity == "day":
        return (date.fromisoformat(label) + timedelta(days=n)).isoformat()
    if granularity == "week":
        return (date.fromisoformat(label) + timedelta(weeks=n)).isoformat()
    if granularity == "month":
        return _add_months(date.fromisoformat(label), n).isoformat()
    msg = f"Unknown granularity: {granularity!r}"
    raise ValueError(msg)


def period_range(
    start_label: str,
    end_label_inclusive: str,
    granularity: PeriodGranularity,
) -> list[str]:
    """Return a sorted inclusive list of period labels from start to end."""
    labels: list[str] = []
    current = start_label
    while current <= end_label_inclusive:
        labels.append(current)
        current = step_forward(current, granularity)
    return labels


def _add_months(d: date, n: int) -> date:
    """Return d shifted by n calendar months, always landing on the 1st."""
    total = d.year * 12 + (d.month - 1) + n
    year, month_idx = divmod(total, 12)
    return date(year, month_idx + 1, 1)


def _parse_hour_label(label: str) -> datetime:
    """Parse a %Y-%m-%dT%H label back to a UTC datetime."""
    return datetime.strptime(label, "%Y-%m-%dT%H").replace(tzinfo=UTC)
