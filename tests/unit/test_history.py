"""Unit tests for M6: period resolution, ledger, totals, rolling windows."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from sorethumb.errors import PopulationMismatchWarning
from sorethumb.history.ledger import (
    clear_period,
    completed_groups,
    iter_pending_periods,
    last_complete_period,
    periods_missing_groups,
    resolve_backfill_range,
)
from sorethumb.history.periods import (
    period_range,
    resolve_period,
    step_back,
    step_forward,
)
from sorethumb.history.totals import compute_totals
from sorethumb.history.windows import WindowResult, compute_rolling_windows
from sorethumb.store.workspace import Workspace, make_group_key

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace.init(tmp_path / "ws")


def _seed_totals(
    ws: Workspace,
    dataset_fp: str,
    run_id: str,
    period_label: str,
    group_key: str,
    anomaly_count: int,
    population: int,
    rate: float | None = None,
) -> None:
    ws.store.upsert_dataset(dataset_fp, "uri", "s", "c", 100, 5)
    ws.store.insert_run(run_id, dataset_fp, "{}", 0)
    ws.store.upsert_total(
        dataset_fp,
        group_key,
        period_label,
        anomaly_count,
        population,
        rate if rate is not None else (anomaly_count / population if population > 0 else None),
        run_id,
    )


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


# ---------------------------------------------------------------------------
# ledger.py — backfill branches
# ---------------------------------------------------------------------------


class TestResolveBackfillRange:
    DS = "ds1"

    def test_cold_start_spans_bootstrap_periods(self, ws):
        with ws:
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=7,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        # Should span exactly 7 periods: 2026-08-27 → 2026-09-02
        assert len(labels) == 7
        assert labels[0] == step_back("2026-09-03", "day", 7)
        assert labels[-1] == step_back("2026-09-03", "day", 1)

    def test_warm_continuation_starts_after_last_complete(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "run1", "2026-08-30", "gk1", 5, 100)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        # Warm: from 2026-08-31 to 2026-09-02
        assert labels[0] == "2026-08-31"
        assert labels[-1] == "2026-09-02"
        assert len(labels) == 3

    def test_warm_capped_at_lookback_periods(self, ws):
        with ws:
            # last_complete is very old — more than lookback_periods back
            _seed_totals(ws, self.DS, "run1", "2020-01-01", "gk1", 0, 100)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=5,
                max_backfill_periods=30,
            )
        # Capped at 5 periods back from reference
        assert len(labels) == 5
        assert labels[0] == step_back("2026-09-03", "day", 5)

    def test_reference_already_complete_spans_lookback(self, ws):
        with ws:
            # Mark reference itself as complete
            _seed_totals(ws, self.DS, "run1", "2026-09-03", "gk1", 5, 100)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=7,
                max_backfill_periods=30,
            )
        # Branch 3: full lookback_periods scan (7 periods before reference)
        assert len(labels) == 7
        assert labels[0] == step_back("2026-09-03", "day", 7)
        assert labels[-1] == step_back("2026-09-03", "day", 1)

    def test_cold_start_clamped_by_max_backfill(self, ws):
        with ws:
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=28,
                max_backfill_periods=5,
            )
        assert len(labels) <= 5

    def test_empty_when_nothing_to_do(self, ws):
        with ws:
            # last_complete = reference - 1 → warm start → empty because no gap
            _seed_totals(ws, self.DS, "run1", "2026-09-02", "gk1", 0, 100)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        # Warm continuation from 2026-09-03 → 2026-09-02 end → empty
        assert labels == []


# ---------------------------------------------------------------------------
# ledger.py — iter_pending_periods / clear_period / periods_missing_groups
# ---------------------------------------------------------------------------


class TestLedgerHelpers:
    DS = "ds2"

    def test_iter_pending_excludes_completed(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100)
            pending = iter_pending_periods(
                ws.store,
                self.DS,
                ["2026-09-01", "2026-09-02", "2026-09-03"],
            )
        assert "2026-09-01" not in pending
        assert "2026-09-02" in pending
        assert "2026-09-03" in pending

    def test_iter_pending_forced_bypasses_ledger(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100)
            pending = iter_pending_periods(
                ws.store,
                self.DS,
                ["2026-09-01"],
                forced_periods=["2026-09-01"],
            )
        assert "2026-09-01" in pending

    def test_zero_anomaly_period_is_complete(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 0, 100)  # zero anomalies
            pending = iter_pending_periods(ws.store, self.DS, ["2026-09-01"])
        # zero anomalies is still complete — must be skipped
        assert pending == []

    def test_clear_period_removes_from_ledger(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100)
            assert completed_groups(ws.store, self.DS, "2026-09-01") != []
            clear_period(ws.store, self.DS, "2026-09-01")
            assert completed_groups(ws.store, self.DS, "2026-09-01") == []

    def test_periods_missing_groups_detects_re_widening(self, ws):
        with ws:
            ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
            ws.store.insert_run("r1", self.DS, "{}", 0)
            gk_a = make_group_key({"g": "A"})
            gk_b = make_group_key({"g": "B"})
            # Period 2026-09-01 completed with only gk_a (scope was narrow at that time)
            ws.store.upsert_total(self.DS, gk_a, "2026-09-01", 5, 100, 0.05, "r1")
            # gk_b was seen for a different period (confirms it's a real group, not a phantom)
            ws.store.upsert_total(self.DS, gk_b, "2026-08-01", 3, 100, 0.03, "r1")
            # Now re-widened: both gk_a and gk_b are requested
            missing = periods_missing_groups(
                ws.store,
                self.DS,
                [gk_a, gk_b],
                "day",
                28,
                "2026-09-03",
            )
        assert "2026-09-01" in missing

    def test_periods_missing_groups_bounded_to_seen(self, ws):
        with ws:
            ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
            ws.store.insert_run("r1", self.DS, "{}", 0)
            gk_never = make_group_key({"g": "NEVER"})
            # No totals for gk_never — it's unseen
            missing = periods_missing_groups(
                ws.store,
                self.DS,
                [gk_never],
                "day",
                28,
                "2026-09-03",
            )
        # Bounded: gk_never not in seen groups → empty result
        assert missing == []


# ---------------------------------------------------------------------------
# totals.py
# ---------------------------------------------------------------------------


class TestComputeTotals:
    DS = "ds3"

    def _run(self, ws: Workspace, run_id: str = "run1") -> None:
        ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 3)
        ws.store.insert_run(run_id, self.DS, "{}", 0)

    def test_aggregates_anomaly_count_by_group(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame(
                {
                    "country": ["US", "US", "AU", "AU"],
                    "anomaly_flag": [True, False, True, True],
                }
            )
            df = compute_totals(ws.store, results, None, ["country"], "2026-09-01", self.DS, "run1")
        counts = dict(zip(df["group_key"].to_list(), df["anomaly_count"].to_list(), strict=True))
        gk_us = make_group_key({"country": "US"})
        gk_au = make_group_key({"country": "AU"})
        assert counts[gk_us] == 1
        assert counts[gk_au] == 2

    def test_upsert_idempotent_single_row_per_natural_key(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame({"anomaly_flag": [True, False, True]})
            compute_totals(ws.store, results, None, [], "2026-09-01", self.DS, "run1")
            compute_totals(ws.store, results, None, [], "2026-09-01", self.DS, "run1")
            rows = ws.store._conn.execute(
                "SELECT COUNT(*) AS n FROM totals WHERE dataset_fp=? AND period_label=?",
                (self.DS, "2026-09-01"),
            ).fetchone()
        assert rows["n"] == 1

    def test_population_join_at_group_grain(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame(
                {
                    "country": ["US", "US", "AU"],
                    "anomaly_flag": [True, False, True],
                }
            )
            population = pl.DataFrame(
                {
                    "country": ["US", "AU"],
                    "population": [1000, 500],
                }
            )
            df = compute_totals(ws.store, results, population, ["country"], "2026-09-01", self.DS, "run1")
        gk_us = make_group_key({"country": "US"})
        row_us = df.filter(pl.col("group_key") == gk_us)
        assert row_us["population"][0] == 1000
        assert abs(row_us["rate"][0] - 1 / 1000) < 1e-9

    def test_population_missing_column_warns_unknown(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame(
                {
                    "country": ["US"],
                    "anomaly_flag": [True],
                }
            )
            pop_wrong = pl.DataFrame({"region": ["NA"], "population": [500]})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = compute_totals(ws.store, results, pop_wrong, ["country"], "2026-09-01", self.DS, "run1")
        assert any(issubclass(x.category, PopulationMismatchWarning) for x in w)
        assert df["population"][0] == -1
        assert df["rate"][0] is None

    def test_no_group_by_produces_global_group(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame({"anomaly_flag": [True, True, False]})
            df = compute_totals(ws.store, results, None, [], "2026-09-01", self.DS, "run1")
        assert len(df) == 1
        assert df["group_key"][0] == "__all__"
        assert df["anomaly_count"][0] == 2

    def test_rate_is_null_for_unknown_population(self, ws):
        with ws:
            self._run(ws)
            results = pl.DataFrame({"anomaly_flag": [True]})
            df = compute_totals(ws.store, results, None, [], "2026-09-01", self.DS, "run1")
        assert df["population"][0] == -1
        assert df["rate"][0] is None


# ---------------------------------------------------------------------------
# windows.py
# ---------------------------------------------------------------------------


class TestRollingWindows:
    DS = "ds4"

    def _setup(self, ws: Workspace, run_id: str = "run1") -> None:
        ws.store.upsert_dataset(self.DS, "uri", "s", "c", 1000, 5)
        ws.store.insert_run(run_id, self.DS, "{}", 0)

    def test_volume_weighted_rate(self, ws):
        """Rate = sum(anomalies) / sum(population), not mean of per-group rates."""
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            # Two periods with different group sizes
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 10, 1000, 0.01, "run1")
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 1, 100, 0.01, "run1")
            # Prior window: 2026-09-01 (no data → rate None)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [2],
                "day",
            )
        # W=2: current = [2026-09-02, 2026-09-03] → 11 anomalies / 1100 population
        r: WindowResult = results[0]
        assert r.current_anomaly_count == 11
        assert r.current_population == 1100
        assert abs(r.current_rate - 11 / 1100) < 1e-10  # type: ignore[operator]

    def test_unknown_population_excluded(self, ws):
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 1000, 0.005, "run1")
            ws.store.upsert_total(self.DS, make_group_key({"g": "B"}), "2026-09-03", 99, -1, None, "run1")
            results = compute_rolling_windows(ws.store, self.DS, "2026-09-03", [1], "day")
        r = results[0]
        # Unknown population row excluded → only 5 anomalies / 1000 population
        assert r.current_anomaly_count == 5
        assert r.current_population == 1000

    def test_low_volume_flag_on_shortest_window(self, ws):
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 1, 50, 0.02, "run1")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [1, 7],
                "day",
                low_volume_threshold=100,
            )
        by_w = {r.window_size: r for r in results}
        assert by_w[1].low_volume is True
        assert by_w[7].low_volume is False

    def test_calibration_break_detected(self, ws):
        with ws:
            ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
            # Two runs with different calibration modes
            ws.store.insert_run("r_self", self.DS, '{"calibration_mode": "self"}', 0)
            ws.store.insert_run("r_ref", self.DS, '{"calibration_mode": "reference"}', 0)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 5, 100, 0.05, "r_self")
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 100, 0.05, "r_ref")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [2],
                "day",
            )
        assert results[0].calibration_break is True

    def test_no_calibration_break_same_mode(self, ws):
        with ws:
            ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
            ws.store.insert_run("r1", self.DS, '{"calibration_mode": "self"}', 0)
            ws.store.insert_run("r2", self.DS, '{"calibration_mode": "self"}', 0)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 5, 100, 0.05, "r1")
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 100, 0.05, "r2")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [2],
                "day",
            )
        assert results[0].calibration_break is False

    def test_group_allow_list_limits_aggregation(self, ws):
        with ws:
            self._setup(ws)
            gk_a = make_group_key({"g": "A"})
            gk_b = make_group_key({"g": "B"})
            ws.store.upsert_total(self.DS, gk_a, "2026-09-03", 10, 1000, 0.01, "run1")
            ws.store.upsert_total(self.DS, gk_b, "2026-09-03", 50, 5000, 0.01, "run1")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [1],
                "day",
                group_keys=[gk_a],
            )
        r = results[0]
        assert r.current_anomaly_count == 10
        assert r.current_population == 1000

    def test_absolute_and_pct_change(self, ws):
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            # Prior: 2026-09-02 → 10/1000 = 0.01
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 10, 1000, 0.01, "run1")
            # Current: 2026-09-03 → 20/1000 = 0.02
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 20, 1000, 0.02, "run1")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [1],
                "day",
            )
        r = results[0]
        assert abs(r.absolute_change - 0.01) < 1e-10  # type: ignore[operator]
        assert abs(r.pct_change - 1.0) < 1e-10  # type: ignore[operator]  # 100% increase

    def test_empty_windows_list_returns_empty(self, ws):
        with ws:
            results = compute_rolling_windows(ws.store, self.DS, "2026-09-03", [], "day")
        assert results == []

    def test_results_sorted_by_window_size(self, ws):
        with ws:
            self._setup(ws)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                "2026-09-03",
                [14, 1, 7],
                "day",
            )
        sizes = [r.window_size for r in results]
        assert sizes == sorted(sizes)
