"""T0: the rolling-window trend says what its numbers mean and when they can mislead."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from sorethumb_ml.cli import _trend_notes, _trend_table
from sorethumb_ml.history.windows import (
    COMPARABILITY,
    compute_rolling_windows,
    default_reference_label,
    newcombe_interval,
    trend_basis,
    wilson_interval,
)
from sorethumb_ml.report.html import _windows_notes, _windows_table
from sorethumb_ml.store.workspace import Workspace, make_group_key

pytestmark = pytest.mark.integration

DS = "ds-trend"
CFG = "cfg-trend"
GA = make_group_key({"g": "A"})
GB = make_group_key({"g": "B"})


def _run(
    ws: Workspace, run_id: str, *, version: str = "0.1.0", policy: str | None = "intersection@1"
) -> None:
    ws.store.upsert_dataset(DS, "uri", "s", "c", 1000, 5)
    ws.store.insert_run(run_id, DS, "{}", 0, library_version=version, ensemble_policy=policy)


def _total(ws: Workspace, gk: str, label: str, count: int, pop: int, run_id: str = "r1") -> None:
    ws.store.upsert_total(DS, gk, label, count, pop, count / pop if pop > 0 else None, run_id, CFG)


def _windows(ws: Workspace, ref: str, sizes: list[int], **kw: object):
    return compute_rolling_windows(ws.store, DS, CFG, ref, sizes, "day", **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("combination", "contamination", "expected"),
    [
        ("composite", 0.05, "budget"),
        ("composite", "auto", "self_referenced"),
        ("intersection", 0.05, "self_referenced"),
        ("union", "auto", "self_referenced"),
    ],
)
def test_trend_basis(combination: str, contamination: str | float, expected: str) -> None:
    assert trend_basis(combination, contamination) == expected


def test_a_budget_trend_reports_counts_but_no_change(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-02", 5, 100)
        _total(ws, GA, "2026-09-03", 6, 120)  # same 5% budget, rounding apart
        (r,) = _windows(ws, "2026-09-03", [1], basis="budget")
    assert (r.current_anomaly_count, r.prior_anomaly_count) == (6, 5)
    assert r.current_rate == pytest.approx(0.05)
    assert r.absolute_change is None
    assert r.pct_change is None
    assert r.change_ci is None
    assert r.basis == "budget"


# ---------------------------------------------------------------------------
# Matched groups (Simpson's paradox)
# ---------------------------------------------------------------------------


def test_a_group_appearing_does_not_move_the_matched_rate(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-02", 10, 100)
        _total(ws, GA, "2026-09-03", 10, 100)
        _total(ws, GB, "2026-09-03", 90, 100)  # new, very anomalous group
        (matched,) = _windows(ws, "2026-09-03", [1])
        (unmatched,) = _windows(ws, "2026-09-03", [1], match_groups=False)
    assert matched.absolute_change == pytest.approx(0.0)
    assert (matched.groups_matched, matched.groups_current_only, matched.groups_prior_only) == (1, 1, 0)
    assert "group_mix_changed" in matched.comparability
    assert unmatched.absolute_change == pytest.approx(0.5 - 0.1)


def test_with_no_prior_groups_the_current_window_keeps_all_of_its_groups(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-03", 10, 100)
        _total(ws, GB, "2026-09-03", 30, 100)
        (r,) = _windows(ws, "2026-09-03", [1])
    assert r.current_anomaly_count == 40
    assert r.prior_rate is None
    assert r.absolute_change is None
    assert "group_mix_changed" not in r.comparability


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_counts_expected_periods(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        for label in ("2026-09-01", "2026-09-02", "2026-09-03"):
            _total(ws, GA, label, 1, 100)
        (r,) = _windows(ws, "2026-09-03", [7])
    assert (r.current_periods_covered, r.current_periods_expected) == (3, 7)
    assert (r.prior_periods_covered, r.prior_periods_expected) == (0, 7)
    assert not r.coverage_complete
    assert {"incomplete_current", "incomplete_prior"} <= set(r.comparability)


def test_weekends_are_not_expected_when_rolling_non_business_days(ws: Workspace) -> None:
    # 2026-09-07 is a Monday; the 7-day window back to Tuesday 09-01 holds 5 weekdays.
    with ws:
        _run(ws, "r1")
        for label in ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"):
            _total(ws, GA, label, 1, 100)
        (r,) = _windows(ws, "2026-09-07", [7], roll_non_business=True)
        (r_all,) = _windows(ws, "2026-09-07", [7])
    assert (r.current_periods_covered, r.current_periods_expected) == (5, 5)
    assert "incomplete_current" not in r.comparability
    assert (r_all.current_periods_covered, r_all.current_periods_expected) == (5, 7)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(10, 100)  # type: ignore[misc]
    assert lo == pytest.approx(0.0552, abs=1e-4)
    assert hi == pytest.approx(0.1744, abs=1e-4)
    assert wilson_interval(0, 0) is None
    lo0, _ = wilson_interval(0, 50)  # type: ignore[misc]
    assert lo0 == 0.0
    _, hi_all = wilson_interval(50, 50)  # type: ignore[misc]
    assert hi_all == 1.0


def test_newcombe_interval_matches_the_published_example() -> None:
    # Newcombe (1998), Statistics in Medicine 17:873, example (a): 56/70 vs 48/80, method 10.
    lo, hi = newcombe_interval(56, 70, 48, 80)  # type: ignore[misc]
    assert lo == pytest.approx(0.0524, abs=1e-4)
    assert hi == pytest.approx(0.3339, abs=1e-4)
    assert newcombe_interval(1, 0, 1, 10) is None


def test_change_interval_significance_and_withheld_relative_change(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-02", 5, 1000)  # prior count below min_prior_count
        _total(ws, GA, "2026-09-03", 60, 1000)
        (r,) = _windows(ws, "2026-09-03", [1])
    assert r.absolute_change == pytest.approx(0.055)
    assert r.change_ci is not None
    assert r.change_significant is True
    assert r.pct_change is None  # 5 prior anomalies: a relative change would be noise
    assert r.current_rate_ci is not None
    assert r.prior_rate_ci is not None


def test_low_volume_is_per_window(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-03", 1, 50)
        _total(ws, GA, "2026-08-27", 1, 500)
        by_w = {r.window_size: r for r in _windows(ws, "2026-09-03", [1, 7, 14], low_volume_threshold=100)}
    assert by_w[1].low_volume  # current 50
    assert by_w[7].low_volume  # current 50 + nothing
    assert not by_w[14].low_volume  # current 550, prior empty


# ---------------------------------------------------------------------------
# Comparability
# ---------------------------------------------------------------------------


def test_mixed_library_versions_and_policies_are_flagged(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1", version="0.1.0", policy="intersection@1")
        _run(ws, "r2", version="0.2.0", policy="composite-agreement@1")
        _total(ws, GA, "2026-09-02", 5, 100, run_id="r1")
        _total(ws, GA, "2026-09-03", 5, 100, run_id="r2")
        (r,) = _windows(ws, "2026-09-03", [1])
    assert {"mixed_library_versions", "mixed_ensemble_policies"} <= set(r.comparability)
    assert set(r.comparability) <= set(COMPARABILITY)


def test_other_configuration_in_span_keeps_calibration_break(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-03", 5, 100)
        ws.store.upsert_total(DS, GA, "2026-09-02", 5, 100, 0.05, "r1", "another-config")
        (r,) = _windows(ws, "2026-09-03", [2])
    assert r.calibration_break
    assert "other_config_in_span" in r.comparability


# ---------------------------------------------------------------------------
# Reference period
# ---------------------------------------------------------------------------


def test_default_reference_is_the_latest_elapsed_period(ws: Workspace) -> None:
    with ws:
        assert default_reference_label(ws.store, DS, CFG, "2026-09-05") == (None, False)
        _run(ws, "r1")
        _total(ws, GA, "2026-09-03", 1, 100)
        _total(ws, GA, "2026-09-05", 1, 100)  # the period in progress
        assert default_reference_label(ws.store, DS, CFG, "2026-09-05") == ("2026-09-03", False)
        assert default_reference_label(ws.store, DS, CFG, "2026-09-03") == ("2026-09-05", True)


def test_run_period_labels(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _run(ws, "r2")
        _total(ws, GA, "2026-09-03", 1, 100, run_id="r1")
        _total(ws, GB, "2026-09-03", 1, 100, run_id="r2")
        assert ws.store.run_period_labels("r1", GA) == ["2026-09-03"]
        assert ws.store.run_period_labels("r1", GB) == []
        assert ws.store.run_provenance([]) == {}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(table: object) -> str:
    console = Console(width=250, record=True, color_system=None)
    console.print(table)
    return console.export_text()


def test_cli_table_and_notes_explain_every_caveat(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-02", 10, 1000)
        _total(ws, GA, "2026-09-03", 60, 1000)
        _total(ws, GB, "2026-09-03", 5, 50)
        results = _windows(ws, "2026-09-03", [1, 2])
    text = _render(_trend_table(results, ref_label="2026-09-03", config_hash=CFG))
    assert "Rolling windows (ending 2026-09-03" in text
    assert "1 (+1/-0)" in text
    assert "+5.00 [" in text
    assert "*" in text
    notes = "\n".join(_trend_notes(results, basis="self_referenced", in_progress=True))
    assert "still in progress" in notes
    assert "group_mix_changed" in notes
    assert "incomplete_prior" in notes
    assert "fitted and calibrated on its own rows" in notes
    budget = "\n".join(_trend_notes(results, basis="budget", in_progress=False))
    assert "set by the review budget" in budget


def test_report_table_and_notes(ws: Workspace) -> None:
    with ws:
        _run(ws, "r1")
        _total(ws, GA, "2026-09-02", 10, 1000)
        _total(ws, GA, "2026-09-03", 60, 1000)
        results = _windows(ws, "2026-09-03", [1])
        budget = _windows(ws, "2026-09-03", [1], basis="budget")
    table = _windows_table(results)
    assert "Δ pp [95% CI]" in table
    assert "+5.00 [" in table
    notes = _windows_notes(results)
    assert "<code>sorethumb score --from-run</code>" in notes  # the shared note keeps its markup
    assert "Intervals cover sampling noise only" in notes
    assert "set by budget" in _windows_table(budget)
    assert "set by the review budget" in _windows_notes(budget)


def test_a_period_run_report_shows_its_trend(tmp_path: Path) -> None:
    """End to end: two periods through the pipeline, then the second run's report has a trend table."""
    from sorethumb_ml._pipeline import run_detection
    from tests.factories.configs import make_config
    from tests.factories.frames import write_two_period_parquet

    parquet = tmp_path / "two.parquet"
    labels = sorted(write_two_period_parquet(parquet))
    cfg = make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        combination="composite",
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
        report_kwargs={"formats": ["html"], "rolling_windows": [1]},
    )
    run_detection(cfg, period_label_override=labels[0])
    second = run_detection(cfg, period_label_override=labels[1])
    page = (tmp_path / "ws" / "reports" / second.run_id / "index.html").read_text(encoding="utf-8")
    # Once, in the records panel -- not once per nested div (the explanation detail has its own).
    assert page.count("Trend ending 2024-01-16") == 1
    assert "Explanation detail" in page
    for heading in ("1. Population", "2. Scores", "3. Threshold", "4. Counts"):
        assert heading in page
    assert "1/1 · 1/1" in page
    assert "Intervals cover sampling noise only" in page
    assert "set history.anchor_run" in page
