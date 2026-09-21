"""Integration tests for sorethumb.history: ledger, period completion, rolling
windows against a real Workspace/SQLite store.

See tests/unit/history/test_periods.py for the pure period-label/window math
(no store involved).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sorethumb import Config
from sorethumb._pipeline import run_detection
from sorethumb.config import DetectorConfig
from sorethumb.history.ledger import (
    clear_period,
    completed_groups,
    iter_pending_periods,
    last_complete_period,
    resolve_backfill_range,
)
from sorethumb.history.periods import step_back, step_forward
from sorethumb.history.windows import WindowResult, compute_rolling_windows
from sorethumb.store.workspace import Workspace, make_group_key
from tests.factories.configs import make_config
from tests.factories.frames import write_two_period_parquet

pytestmark = pytest.mark.integration


def _seed_totals(
    ws: Workspace,
    dataset_fp: str,
    run_id: str,
    period_label: str,
    group_key: str,
    anomaly_count: int,
    population: int,
    rate: float | None = None,
    config_hash: str = "cfg0",
) -> None:
    """Seed one group's totals row and mark the period complete under *config_hash*.

    Real production writes totals + completion atomically for every group in
    one call (``Store.record_period_completion``); this direct insert lets
    ledger/window tests build up history state one group at a time without
    running the full pipeline. Each call marks the period complete (failed
    groups are modeled directly via ``ws.store.record_period_completion`` in
    the tests that need a partial/incomplete period).
    """
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
        config_hash,
    )
    ws.store._conn.execute(
        """
        INSERT INTO period_execution
            (dataset_fp, period_label, config_hash, run_id, group_count, failed_count, complete)
        VALUES (?, ?, ?, ?, 1, 0, 1)
        ON CONFLICT(dataset_fp, period_label, config_hash) DO UPDATE SET
            group_count = group_count + 1,
            run_id      = excluded.run_id
        """,
        (dataset_fp, period_label, config_hash, run_id),
    )
    ws.store._conn.commit()


# ---------------------------------------------------------------------------
# ledger.py — backfill branches
# ---------------------------------------------------------------------------


class TestResolveBackfillRange:
    DS = "ds1"
    CFG = "cfg0"

    def test_cold_start_spans_bootstrap_periods(self, ws):
        with ws:
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
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
            _seed_totals(ws, self.DS, "run1", "2026-08-30", "gk1", 5, 100, config_hash=self.CFG)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
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
            _seed_totals(ws, self.DS, "run1", "2020-01-01", "gk1", 0, 100, config_hash=self.CFG)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
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
            _seed_totals(ws, self.DS, "run1", "2026-09-03", "gk1", 5, 100, config_hash=self.CFG)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
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
                self.CFG,
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
            _seed_totals(ws, self.DS, "run1", "2026-09-02", "gk1", 0, 100, config_hash=self.CFG)
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                "day",
                bootstrap_periods=28,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        # Warm continuation from 2026-09-03 → 2026-09-02 end → empty
        assert labels == []

    def test_last_complete_is_scoped_to_config_hash(self, ws):
        """A period complete under one config must not count as last_complete for another."""
        with ws:
            _seed_totals(ws, self.DS, "run1", "2026-09-02", "gk1", 5, 100, config_hash="other-cfg")
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                "day",
                bootstrap_periods=7,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        # self.CFG has never completed anything -- cold start, not warm continuation.
        assert len(labels) == 7

    def test_roll_non_business_drops_weekend_labels_from_cold_start(self, ws):
        """A bootstrap window wide enough to span a weekend must not include
        Sat/Sun labels when roll_non_business=True, even though only the
        *reference* is rolled by callers (resolve_period), never the labels
        resolve_backfill_range walks through internally."""
        with ws:
            # 2026-09-21 is a Monday; a 5-day bootstrap window from it spans
            # Wed(16)..Sun(20) with no filtering, so Sat(19)/Sun(20) would
            # otherwise leak in.
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-21",
                "day",
                bootstrap_periods=5,
                lookback_periods=28,
                max_backfill_periods=30,
                roll_non_business=True,
            )
        assert labels == ["2026-09-16", "2026-09-17", "2026-09-18"]

    def test_roll_non_business_false_keeps_weekend_labels(self, ws):
        """Default behaviour (roll_non_business=False) is unchanged: the
        weekend labels a business-only dataset would never have data for are
        still returned, same as before this was fixed."""
        with ws:
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-21",
                "day",
                bootstrap_periods=5,
                lookback_periods=28,
                max_backfill_periods=30,
            )
        assert labels == ["2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20"]

    def test_roll_non_business_drops_weekend_labels_from_warm_continuation(self, ws):
        """The warm-continuation branch's start (last_complete + 1) is not
        itself rolled -- if the last completed period was a Friday, "the day
        after" is a Saturday. This must still be filtered out, the same as
        the cold-start case above."""
        with ws:
            _seed_totals(ws, self.DS, "run1", "2026-09-18", "gk1", 5, 100, config_hash=self.CFG)  # Friday
            labels = resolve_backfill_range(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-21",  # Monday
                "day",
                bootstrap_periods=28,
                lookback_periods=28,
                max_backfill_periods=30,
                roll_non_business=True,
            )
        # Without filtering this would be [Sat 19, Sun 20]; both must be dropped.
        assert labels == []


# ---------------------------------------------------------------------------
# ledger.py — iter_pending_periods / clear_period
# ---------------------------------------------------------------------------


class TestLedgerHelpers:
    DS = "ds2"
    CFG = "cfg0"

    def test_iter_pending_excludes_completed(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100, config_hash=self.CFG)
            pending = iter_pending_periods(
                ws.store,
                self.DS,
                self.CFG,
                ["2026-09-01", "2026-09-02", "2026-09-03"],
            )
        assert "2026-09-01" not in pending
        assert "2026-09-02" in pending
        assert "2026-09-03" in pending

    def test_iter_pending_forced_bypasses_ledger(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100, config_hash=self.CFG)
            pending = iter_pending_periods(
                ws.store,
                self.DS,
                self.CFG,
                ["2026-09-01"],
                forced_periods=["2026-09-01"],
            )
        assert "2026-09-01" in pending

    def test_zero_anomaly_period_is_complete(self, ws):
        with ws:
            _seed_totals(
                ws, self.DS, "r1", "2026-09-01", "gk1", 0, 100, config_hash=self.CFG
            )  # zero anomalies
            pending = iter_pending_periods(ws.store, self.DS, self.CFG, ["2026-09-01"])
        # zero anomalies is still complete — must be skipped
        assert pending == []

    def test_clear_period_removes_from_ledger(self, ws):
        with ws:
            _seed_totals(ws, self.DS, "r1", "2026-09-01", "gk1", 5, 100, config_hash=self.CFG)
            assert completed_groups(ws.store, self.DS, "2026-09-01", self.CFG) != []
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is True
            clear_period(ws.store, self.DS, "2026-09-01", self.CFG)
            assert completed_groups(ws.store, self.DS, "2026-09-01", self.CFG) == []
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is False


# ---------------------------------------------------------------------------
# db.py — Store.record_period_completion / period_is_complete (P0-2)
# ---------------------------------------------------------------------------


class TestPeriodExecution:
    DS = "ds_pe"
    CFG = "cfgX"

    def _run(self, ws: Workspace, run_id: str = "run1") -> None:
        ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
        ws.store.insert_run(run_id, self.DS, "{}", 0)

    def test_complete_when_no_failed_groups(self, ws):
        with ws:
            self._run(ws)
            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[("gk1", 5, 100, 0.05)],
                failed_count=0,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is True

    def test_zero_anomaly_totals_still_complete(self, ws):
        """anomaly_count=0 across every group is a legitimate complete outcome,
        not a signal that nothing happened."""
        with ws:
            self._run(ws)
            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[("gk1", 0, 100, 0.0)],
                failed_count=0,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is True

    def test_not_complete_with_any_failed_group(self, ws):
        """One group failing among several must mark the period incomplete,
        even though the succeeding groups' totals are written."""
        with ws:
            self._run(ws)
            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[("gk1", 5, 100, 0.05)],  # only the succeeding group
                failed_count=1,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is False
            assert ws.store.completed_group_keys(self.DS, "2026-09-01", self.CFG) == ["gk1"]

    def test_not_complete_with_zero_groups_discovered(self, ws):
        with ws:
            self._run(ws)
            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[],
                failed_count=0,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is False

    def test_repeat_call_is_idempotent(self, ws):
        with ws:
            self._run(ws)
            for _ in range(2):
                ws.store.record_period_completion(
                    dataset_fp=self.DS,
                    period_label="2026-09-01",
                    period_from="2026-09-01",
                    period_to="2026-09-02",
                    config_hash=self.CFG,
                    run_id="run1",
                    totals=[("gk1", 5, 100, 0.05)],
                    failed_count=0,
                )
            totals = ws.store.totals_for_periods(self.DS, ["2026-09-01"], self.CFG)
            pe_count = ws.store._conn.execute(
                "SELECT COUNT(*) AS n FROM period_execution WHERE dataset_fp=? AND period_label=? AND config_hash=?",
                (self.DS, "2026-09-01", self.CFG),
            ).fetchone()["n"]
            assert len(totals) == 1
            assert pe_count == 1
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is True

    def test_recovers_a_previously_failed_period_once_retried_clean(self, ws):
        """A period first recorded incomplete (one failed group) must flip to
        complete once a later call for the same key has zero failures."""
        with ws:
            self._run(ws)
            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[("gk1", 5, 100, 0.05)],
                failed_count=1,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is False

            ws.store.record_period_completion(
                dataset_fp=self.DS,
                period_label="2026-09-01",
                period_from="2026-09-01",
                period_to="2026-09-02",
                config_hash=self.CFG,
                run_id="run1",
                totals=[("gk1", 5, 100, 0.05), ("gk2", 2, 50, 0.04)],
                failed_count=0,
            )
            assert ws.store.period_is_complete(self.DS, "2026-09-01", self.CFG) is True


# ---------------------------------------------------------------------------
# windows.py
# ---------------------------------------------------------------------------


class TestRollingWindows:
    DS = "ds4"
    CFG = "cfg0"

    def _setup(self, ws: Workspace, run_id: str = "run1") -> None:
        ws.store.upsert_dataset(self.DS, "uri", "s", "c", 1000, 5)
        ws.store.insert_run(run_id, self.DS, "{}", 0)

    def test_volume_weighted_rate(self, ws):
        """Rate = sum(anomalies) / sum(population), not mean of per-group rates."""
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            # Two periods with different group sizes
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 10, 1000, 0.01, "run1", self.CFG)
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 1, 100, 0.01, "run1", self.CFG)
            # Prior window: 2026-09-01 (no data → rate None)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
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
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 1000, 0.005, "run1", self.CFG)
            ws.store.upsert_total(
                self.DS, make_group_key({"g": "B"}), "2026-09-03", 99, -1, None, "run1", self.CFG
            )
            results = compute_rolling_windows(ws.store, self.DS, self.CFG, "2026-09-03", [1], "day")
        r = results[0]
        # Unknown population row excluded → only 5 anomalies / 1000 population
        assert r.current_anomaly_count == 5
        assert r.current_population == 1000

    def test_low_volume_flag_on_shortest_window(self, ws):
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 1, 50, 0.02, "run1", self.CFG)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                [1, 7],
                "day",
                low_volume_threshold=100,
            )
        by_w = {r.window_size: r for r in results}
        assert by_w[1].low_volume is True
        assert by_w[7].low_volume is False

    def test_calibration_break_detects_other_config_hash_in_window(self, ws):
        """calibration_break is real provenance now: it fires when this span
        also has totals recorded under a *different* config_hash -- those
        rows are excluded from the sums above, so the trend may be comparing
        an incomplete picture. This replaces a dead check that diffed a
        ``calibration_mode`` config field which no longer exists.
        """
        with ws:
            ws.store.upsert_dataset(self.DS, "uri", "s", "c", 100, 5)
            ws.store.insert_run("r_a", self.DS, "{}", 0)
            ws.store.insert_run("r_b", self.DS, "{}", 0)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 5, 100, 0.05, "r_a", self.CFG)
            # A different configuration also has a totals row in this span.
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 100, 0.05, "r_b", "other-cfg")
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                [2],
                "day",
            )
        assert results[0].calibration_break is True
        # Only self.CFG's own row (at 2026-09-02) is summed in; the other
        # config's row at 2026-09-03 is excluded, not added on top.
        assert results[0].current_anomaly_count == 5
        assert results[0].current_population == 100

    def test_no_calibration_break_single_config_hash(self, ws):
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 5, 100, 0.05, "run1", self.CFG)
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 5, 100, 0.05, "run1", self.CFG)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
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
            ws.store.upsert_total(self.DS, gk_a, "2026-09-03", 10, 1000, 0.01, "run1", self.CFG)
            ws.store.upsert_total(self.DS, gk_b, "2026-09-03", 50, 5000, 0.01, "run1", self.CFG)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
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
            ws.store.upsert_total(self.DS, gk, "2026-09-02", 10, 1000, 0.01, "run1", self.CFG)
            # Current: 2026-09-03 → 20/1000 = 0.02
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 20, 1000, 0.02, "run1", self.CFG)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                [1],
                "day",
            )
        r = results[0]
        assert abs(r.absolute_change - 0.01) < 1e-10  # type: ignore[operator]
        assert abs(r.pct_change - 1.0) < 1e-10  # type: ignore[operator]  # 100% increase

    def test_empty_windows_list_returns_empty(self, ws):
        with ws:
            results = compute_rolling_windows(ws.store, self.DS, self.CFG, "2026-09-03", [], "day")
        assert results == []

    def test_results_sorted_by_window_size(self, ws):
        with ws:
            self._setup(ws)
            results = compute_rolling_windows(
                ws.store,
                self.DS,
                self.CFG,
                "2026-09-03",
                [14, 1, 7],
                "day",
            )
        sizes = [r.window_size for r in results]
        assert sizes == sorted(sizes)

    def test_two_configs_totals_are_not_summed_together(self, ws):
        """The core P0-2 bug: totals recorded under a different config_hash
        for the same (dataset, group, period) must never be added into this
        config's aggregate."""
        with ws:
            self._setup(ws)
            gk = make_group_key({"g": "A"})
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 10, 1000, 0.01, "run1", self.CFG)
            ws.store.upsert_total(self.DS, gk, "2026-09-03", 999, 999000, 0.999, "run1", "other-cfg")
            results = compute_rolling_windows(ws.store, self.DS, self.CFG, "2026-09-03", [1], "day")
        r = results[0]
        assert r.current_anomaly_count == 10
        assert r.current_population == 1000


# ---------------------------------------------------------------------------
# Full pipeline: period-label overrides and history-ledger recording (P0-2)
# ---------------------------------------------------------------------------


def _period_config(parquet: Path, workdir: Path, *, detectors: list[DetectorConfig] | None = None) -> Config:
    return make_config(
        parquet,
        workdir,
        source_format="parquet",
        combination="composite",
        detectors=detectors,
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )


def _one_day_two_group_parquet(path: Path, *, per_group: int = 60, seed: int = 0) -> None:
    """One calendar day, two groups ("cat" A/B), each large enough to fit a detector."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = per_group * 2
    ts = [datetime(2024, 1, 15, 8, tzinfo=UTC) + timedelta(minutes=i) for i in range(n)]
    pl.DataFrame(
        {
            "id": list(range(n)),
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "cat": ["A"] * per_group + ["B"] * per_group,
            "num_a": rng.normal(0.0, 1.0, n).tolist(),
            "num_b": rng.normal(5.0, 2.0, n).tolist(),
        }
    ).write_parquet(str(path))


def _period_config_with_groups(parquet: Path, workdir: Path) -> Config:
    return make_config(
        parquet,
        workdir,
        source_format="parquet",
        combination="composite",
        time_column="ts",
        group_by=["cat"],
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )


@pytest.mark.parametrize("label", ["2024-01-15", "2024-01-16"])
def test_period_override_filters_to_that_window(tmp_path: Path, label: str) -> None:
    parquet = tmp_path / "two_periods.parquet"
    per_day = 100
    planted = write_two_period_parquet(parquet, per_day=per_day)
    cfg = _period_config(parquet, tmp_path / "ws")

    result = run_detection(cfg, period_label_override=label, no_report=True)

    assert result.period_label == label
    assert result.n_succeeded == 1
    group = result.groups[0]
    # Only that day's rows entered the pipeline — not the whole 200-row dataset.
    assert group.n_records == per_day

    flagged = set(pl.read_parquet(group.results_path)["row_id"].to_list())
    this_day = set(planted[label])
    other_day = set(planted["2024-01-16" if label == "2024-01-15" else "2024-01-15"])

    # This period's planted anomalies are caught; the other period's ids are
    # absent entirely (they were never in the filtered frame).
    assert this_day <= flagged, f"missed planted anomalies {this_day - flagged}"
    assert not (flagged & other_day)
    assert all(rid < per_day for rid in flagged) == (label == "2024-01-15")


def test_period_overrides_produce_distinct_runs(tmp_path: Path) -> None:
    parquet = tmp_path / "two_periods.parquet"
    write_two_period_parquet(parquet)
    cfg = _period_config(parquet, tmp_path / "ws")

    r15 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    r16 = run_detection(cfg, period_label_override="2024-01-16", no_report=True)

    assert r15.run_id != r16.run_id
    assert r15.n_anomalies > 0
    assert r16.n_anomalies > 0


def test_period_run_records_history_ledger(tmp_path: Path) -> None:
    """run_detection writes the period row + per-group totals so backfill is idempotent."""
    parquet = tmp_path / "two_periods.parquet"
    per_day = 100
    write_two_period_parquet(parquet, per_day=per_day)
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    result = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    dataset_fp = result.dataset_fp

    with Workspace.open(workdir) as ws:
        # period row carries the [from, to) window for the label
        prow = ws.store._conn.execute(
            "SELECT period_from, period_to FROM period WHERE dataset_fp=? AND period_label=?",
            (dataset_fp, "2024-01-15"),
        ).fetchone()
        assert prow is not None
        assert (prow["period_from"], prow["period_to"]) == ("2024-01-15", "2024-01-16")

        # one totals row per processed group, population == rows that entered the pipeline
        totals = ws.store.totals_for_periods(dataset_fp, ["2024-01-15"], result.config_hash)
        assert len(totals) == result.n_succeeded == 1
        row = totals[0]
        assert row["population"] == per_day
        assert row["anomaly_count"] == result.groups[0].n_anomalies
        assert row["run_id"] == result.run_id

        # the label is now complete: backfill would not re-queue it
        assert iter_pending_periods(
            ws.store, dataset_fp, result.config_hash, ["2024-01-15", "2024-01-16"]
        ) == ["2024-01-16"]


def test_repeat_period_run_is_idempotent_in_history_ledger(tmp_path: Path) -> None:
    parquet = tmp_path / "two_periods.parquet"
    write_two_period_parquet(parquet, per_day=100)
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r1.run_id == r2.run_id

    with Workspace.open(workdir) as ws:
        totals = ws.store.totals_for_periods(r1.dataset_fp, ["2024-01-15"], r1.config_hash)
        assert len(totals) == 1, "a repeat run must not duplicate the totals row"
        assert ws.store.period_is_complete(r1.dataset_fp, "2024-01-15", r1.config_hash) is True


def test_two_configs_on_one_period_do_not_double_count_or_block_each_other(tmp_path: Path) -> None:
    """Two configurations processing the same period_label must each get
    their own totals row (scoped by config_hash), and running one must not
    make the ledger think the other is already done.
    """
    parquet = tmp_path / "two_periods.parquet"
    write_two_period_parquet(parquet, per_day=100)
    workdir = tmp_path / "ws"
    cfg_a = _period_config(parquet, workdir, detectors=[DetectorConfig(name="isolation_forest")])
    cfg_b = _period_config(parquet, workdir, detectors=[DetectorConfig(name="kmeans_distance")])
    assert cfg_a.config_hash() != cfg_b.config_hash()

    ra = run_detection(cfg_a, period_label_override="2024-01-15", no_report=True)
    assert ra.n_succeeded == 1

    with Workspace.open(workdir) as ws:
        # Config B has never run this period -- must still be pending under
        # its own hash, regardless of config A having just completed it.
        pending_b = iter_pending_periods(ws.store, ra.dataset_fp, cfg_b.config_hash(), ["2024-01-15"])
        assert pending_b == ["2024-01-15"]

    rb = run_detection(cfg_b, period_label_override="2024-01-15", no_report=True)
    assert rb.n_succeeded == 1
    assert ra.run_id != rb.run_id

    with Workspace.open(workdir) as ws:
        totals_a = ws.store.totals_for_periods(ra.dataset_fp, ["2024-01-15"], ra.config_hash)
        totals_b = ws.store.totals_for_periods(rb.dataset_fp, ["2024-01-15"], rb.config_hash)
        assert len(totals_a) == 1
        assert len(totals_b) == 1
        # Each config's own read never includes the other's row.
        assert {t["config_hash"] for t in totals_a} == {ra.config_hash}
        assert {t["config_hash"] for t in totals_b} == {rb.config_hash}

        assert ws.store.period_is_complete(ra.dataset_fp, "2024-01-15", ra.config_hash) is True
        assert ws.store.period_is_complete(rb.dataset_fp, "2024-01-15", rb.config_hash) is True

        # Rolling-window aggregation for A must not pick up B's contribution.
        other = ws.store.other_config_hashes_for_periods(ra.dataset_fp, ["2024-01-15"], ra.config_hash)
        assert other == {rb.config_hash}


def test_period_with_one_failed_group_is_not_marked_complete(tmp_path: Path) -> None:
    """A period where one of several groups fails must not be recorded
    complete -- only the succeeding group's totals are written -- and a retry
    with the same inputs must re-execute the failed group and only then
    complete the period.
    """
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    parquet = tmp_path / "grouped.parquet"
    _one_day_two_group_parquet(parquet)
    workdir = tmp_path / "ws"
    cfg = _period_config_with_groups(parquet, workdir)

    calls = {"n": 0}
    orig_fit = IsolationForestDetector.fit

    def _flaky_fit(self: object, *a: object, **k: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("synthetic failure for the second group")
        return orig_fit(self, *a, **k)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IsolationForestDetector, "fit", _flaky_fit)
        r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    assert r1.n_succeeded == 1
    assert r1.n_failed == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r1.dataset_fp, "2024-01-15", r1.config_hash) is False
        totals = ws.store.totals_for_periods(r1.dataset_fp, ["2024-01-15"], r1.config_hash)
        assert len(totals) == 1, "only the succeeding group's totals are recorded"

    # Retry with the same inputs: the failed group must be re-executed (fit is
    # no longer patched), and only then does the period become complete.
    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r2.run_id == r1.run_id
    assert r2.n_succeeded == 1
    assert r2.n_skipped == 1

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r2.dataset_fp, "2024-01-15", r2.config_hash) is True
        totals = ws.store.totals_for_periods(r2.dataset_fp, ["2024-01-15"], r2.config_hash)
        assert len(totals) == 2


def test_interruption_between_groups_recovers_all_totals_on_retry(tmp_path: Path) -> None:
    """A crash after one group is marked complete in the run ledger but
    before the next group even starts -- so _record_period_history never
    runs at all -- must not permanently lose the finished group's totals: a
    retry must record every group, including the one that was only resumed
    (skipped) this time, not just the one it (re-)executes.
    """
    import sorethumb._pipeline as pipe

    parquet = tmp_path / "grouped.parquet"
    _one_day_two_group_parquet(parquet)
    workdir = tmp_path / "ws"
    cfg = _period_config_with_groups(parquet, workdir)

    real_execute_group = pipe._execute_group
    calls = {"n": 0}

    def _crash_on_second_group(*a: object, **k: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash between groups")
        return real_execute_group(*a, **k)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipe, "_execute_group", _crash_on_second_group)
        with pytest.raises(RuntimeError, match="simulated crash between groups"):
            run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    with Workspace.open(workdir) as ws:
        run_row = ws.store.list_runs(limit=1)[0]
        run_id = str(run_row["run_id"])
        dataset_fp = str(run_row["dataset_fp"])
        config_hash = cfg.config_hash()

        # The first group committed to the run ledger before the crash; the
        # second group was never attempted. Neither has a totals row yet, and
        # the period is not complete -- _record_period_history never ran.
        run_groups = ws.store.all_run_groups(run_id)
        assert len(run_groups) == 1
        assert run_groups[0]["status"] == "complete"
        assert ws.store.totals_for_periods(dataset_fp, ["2024-01-15"], config_hash) == []
        assert ws.store.period_is_complete(dataset_fp, "2024-01-15", config_hash) is False

    r2 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)
    assert r2.run_id == run_id
    assert r2.n_skipped == 1, "the first group must be resumed, not re-executed"
    assert r2.n_succeeded == 1, "the second group runs for the first time"

    with Workspace.open(workdir) as ws:
        assert ws.store.period_is_complete(r2.dataset_fp, "2024-01-15", r2.config_hash) is True
        totals = ws.store.totals_for_periods(r2.dataset_fp, ["2024-01-15"], r2.config_hash)
        assert len(totals) == 2, "the resumed (skipped) group's totals must be recorded too"


def _write_days_parquet(path: Path, *, days: list[str], per_day: int = 80, seed: int = 0) -> None:
    """Write ``per_day`` rows for each ISO day in *days*; row 3 of each day is a +999 anomaly."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    ts: list[datetime] = []
    num_a: list[float] = []
    for d, day in enumerate(days):
        base = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(hours=8)
        col = rng.normal(0.0, 1.0, per_day).tolist()
        col[3] = 999.0
        for r in range(per_day):
            ids.append(d * per_day + r)
            ts.append(base + timedelta(minutes=r))
            num_a.append(col[r])
    pl.DataFrame(
        {
            "id": ids,
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "num_a": num_a,
            "num_b": rng.normal(5.0, 2.0, len(ids)).tolist(),
        }
    ).write_parquet(str(path))


def test_appending_a_snapshot_keeps_dataset_identity_and_history(tmp_path: Path) -> None:
    """Appending a day of rows must not orphan the previous period's history."""
    parquet = tmp_path / "growing.parquet"
    workdir = tmp_path / "ws"
    cfg = _period_config(parquet, workdir)

    # Snapshot 1: only 2024-01-15 exists. Process that period.
    _write_days_parquet(parquet, days=["2024-01-15"])
    r1 = run_detection(cfg, period_label_override="2024-01-15", no_report=True)

    # Snapshot 2: 2024-01-16 appended. Process the new period.
    _write_days_parquet(parquet, days=["2024-01-15", "2024-01-16"])
    r2 = run_detection(cfg, period_label_override="2024-01-16", no_report=True)

    # Logical identity is unchanged; the snapshot fingerprint moved.
    assert r1.dataset_fp == r2.dataset_fp
    assert r1.snapshot_fp
    assert r2.snapshot_fp
    assert r1.snapshot_fp != r2.snapshot_fp
    assert r1.run_id != r2.run_id  # snapshot_fp is part of the run id

    with Workspace.open(workdir) as ws:
        dfp = r2.dataset_fp
        # Both periods' totals live under the one logical dataset — nothing orphaned.
        labels = {
            r["period_label"]
            for r in ws.store.totals_for_periods(dfp, ["2024-01-15", "2024-01-16"], r2.config_hash)
        }
        assert labels == {"2024-01-15", "2024-01-16"}
        assert iter_pending_periods(ws.store, dfp, r2.config_hash, ["2024-01-15", "2024-01-16"]) == []

        # Both snapshots are recorded against the same dataset_fp.
        snaps = {s["snapshot_fp"] for s in ws.store.dataset_snapshots(dfp)}
        assert snaps == {r1.snapshot_fp, r2.snapshot_fp}
