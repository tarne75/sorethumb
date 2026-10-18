"""T3: the trend as four signals with verdicts, in the CLI, --json and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml import Config
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import app
from sorethumb_ml.config import HistoryConfig
from sorethumb_ml.history.trend import SECTIONS, build_trend_report
from sorethumb_ml.store.workspace import Workspace, make_group_key
from tests.factories.configs import make_config
from tests.factories.frames import write_two_period_parquet

pytestmark = pytest.mark.integration

runner = CliRunner()
LABELS = ["2024-01-15", "2024-01-16"]


def _cfg(
    tmp_path: Path, *, anchor: str | None = None, formats: list[str] | None = None, **kw: object
) -> Config:
    parquet = tmp_path / "two.parquet"
    if not parquet.exists():
        write_two_period_parquet(parquet)
    return make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        combination=str(kw.pop("combination", "composite")),
        contamination=kw.pop("contamination", "auto"),  # type: ignore[arg-type]
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False, "anchor_run": anchor},
        report_kwargs={"formats": formats or ["html", "json"], "rolling_windows": [1]},
    )


def _anchored_history(tmp_path: Path) -> tuple[Config, str]:
    """Fit day 1, then run both days against it as the anchor."""
    fit = run_detection(_cfg(tmp_path), period_label_override=LABELS[0], no_report=True)
    anchored = _cfg(tmp_path, anchor=fit.run_id)
    run_detection(anchored, period_label_override=LABELS[0], no_report=True)
    last = run_detection(anchored, period_label_override=LABELS[1])
    return anchored, last.run_id


# ---------------------------------------------------------------------------
# Verdicts (seeded store)
# ---------------------------------------------------------------------------


def _seed(ws: Workspace, gk: str, label: str, n: int, own: int, anchor: int | None) -> None:
    ws.store.upsert_dataset("ds", "uri", "s", "c", 1000, 5)
    ws.store.insert_run("r1", "ds", "{}", 0)
    pct = [i / 100 for i in range(101)]
    stats: dict = {
        "stats_version": 2,
        "n_rows": n,
        "population": {"x": {"kind": "numeric", "n": n, "missing": 0, "percentiles": pct}},
        "scores": {},
        "composite": {"flagged": own},
    }
    if anchor is not None:
        stats["anchor"] = {
            "anchor_run_id": "run_a",
            "comparable": True,
            "n_rows": n,
            "flagged": anchor,
            "combined_percentiles": pct,
            "anchor_percentiles": pct,
            "tail_share": 0.05,
        }
    ws.store.record_period_completion(
        dataset_fp="ds",
        period_label=label,
        period_from=label,
        period_to=label,
        config_hash="cfg",
        run_id="r1",
        totals=[(gk, own, n, own / n)],
        failed_count=0,
        period_stats=[(gk, 2, json.dumps(stats))],
    )


def _config(anchor: str | None = None, contamination: str | float = "auto") -> Config:
    cfg = make_config(Path("x.csv"), Path("ws"), combination="composite", contamination=contamination)
    return cfg.model_copy(update={"history": HistoryConfig(anchor_run=anchor, roll_non_business=False)})


def test_verdicts_without_an_anchor(ws: Workspace) -> None:
    gk = make_group_key({"g": "a"})
    with ws:
        _seed(ws, gk, "2026-09-02", 1000, 50, None)
        _seed(ws, gk, "2026-09-03", 1000, 60, None)
        report = build_trend_report(
            ws.store,
            dataset_fp="ds",
            config_hash="cfg",
            config=_config(),
            reference_label="2026-09-03",
            windows=[1],
        )
    assert report.verdict("population", 1).status == "ok"
    assert report.verdict("scores", 1).reasons == ("no anchor run is set",)
    assert report.verdict("threshold", 1).status == "not_available"
    assert report.verdict("counts", 1).status == "ok"
    with pytest.raises(ValueError, match="unknown section"):
        report.verdict("vibes", 1)


def test_verdicts_with_an_anchor_and_a_budget(ws: Workspace) -> None:
    gk = make_group_key({"g": "a"})
    with ws:
        _seed(ws, gk, "2026-09-02", 1000, 50, 50)
        _seed(ws, gk, "2026-09-03", 1000, 50, 80)
        _seed(ws, make_group_key({"g": "b"}), "2026-09-03", 10, 1, None)  # unanchored cell, only current
        report = build_trend_report(
            ws.store,
            dataset_fp="ds",
            config_hash="cfg",
            config=_config(anchor="run_a", contamination=0.05),
            reference_label="2026-09-03",
            windows=[1, 2],
        )
    scores = report.verdict("scores", 1)
    assert scores.status == "caveats"
    assert "1 (group, period) cell(s) left out" in scores.reasons
    threshold = report.verdict("threshold", 1)
    assert any("mirrors score drift" in r for r in threshold.reasons)
    counts = report.verdict("counts", 1)
    assert counts.reasons[0] == "rate set by the review budget"
    assert "group_mix_changed" in counts.reasons
    assert report.verdict("population", 1).status == "caveats"
    # Window 2 has no prior anchored cells for any group.
    assert report.verdict("scores", 2).status == "not_available"


def test_a_configured_anchor_with_no_scored_periods(ws: Workspace) -> None:
    gk = make_group_key({"g": "a"})
    with ws:
        _seed(ws, gk, "2026-09-03", 1000, 50, None)
        report = build_trend_report(
            ws.store,
            dataset_fp="ds",
            config_hash="cfg",
            config=_config(anchor="run_a"),
            reference_label="2026-09-03",
            windows=[1],
        )
    assert report.verdict("scores", 1).reasons == (
        "no period in these windows was scored against the anchor",
    )
    assert report.verdict("population", 1).status == "not_available"


def test_to_dict_is_json_safe_and_complete(ws: Workspace) -> None:
    gk = make_group_key({"g": "a"})
    with ws:
        _seed(ws, gk, "2026-09-02", 1000, 50, 50)
        _seed(ws, gk, "2026-09-03", 1000, 60, 70)
        report = build_trend_report(
            ws.store,
            dataset_fp="ds",
            config_hash="cfg",
            config=_config(anchor="run_a"),
            reference_label="2026-09-03",
            windows=[1],
        )
    d = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert d["reference_period"] == "2026-09-03"
    assert d["anchor_run"] == "run_a"
    (w,) = d["windows"]
    assert w["current_periods"] == ["2026-09-03"]
    assert set(w["verdicts"]) == set(SECTIONS)
    assert w["anchor"]["volume"] + w["anchor"]["score_drift"] + w["anchor"][
        "threshold_drift"
    ] == pytest.approx(10)
    assert w["population"]["fields"]["x"]["band"] == "stable"
    assert "current_labels" not in w["counts"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_history_json_and_text(tmp_path: Path) -> None:
    _anchored_history(tmp_path)
    ws = str(tmp_path / "ws")
    res = runner.invoke(app, ["history", "--workdir", ws, "--json", "--period", LABELS[1], "--window", "1"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output)
    (w,) = d["windows"]
    assert w["verdicts"]["scores"]["status"] in ("ok", "caveats")
    assert w["anchor"]["anchor_run_id"] == d["anchor_run"]
    a = w["anchor"]
    assert a["volume"] + a["score_drift"] + a["threshold_drift"] == pytest.approx(a["count_change"])

    text = runner.invoke(
        app, ["history", "--workdir", ws, "--period", LABELS[1], "--window", "1"], env={"COLUMNS": "250"}
    )
    assert text.exit_code == 0, text.output
    out = text.output
    for heading in ("1. Population", "2. Scores", "3. Threshold", "4. Counts", "Trend ending 2024-01-16"):
        assert heading in out
    assert out.index("2. Scores: do") < out.index("Anchor rate cur") < out.index("3. Threshold: did")
    assert "Count change against anchor" in out
    assert "Window 1:" in out


def test_history_json_without_history(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("id,x\n" + "\n".join(f"{i},{i % 7}" for i in range(60)), encoding="utf-8")
    run_detection(make_config(csv, tmp_path / "ws"), no_report=True)  # no time column: no periods
    res = runner.invoke(app, ["history", "--workdir", str(tmp_path / "ws"), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["reference_period"] is None


def test_history_json_on_an_empty_workspace_is_a_json_error(tmp_path: Path) -> None:
    Workspace.init(tmp_path / "ws").close()
    res = runner.invoke(app, ["history", "--workdir", str(tmp_path / "ws"), "--json"])
    assert res.exit_code != 0
    assert "error" in json.loads(res.output)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_an_anchored_report_shows_all_four_sections(tmp_path: Path) -> None:
    _, run_id = _anchored_history(tmp_path)
    reports = tmp_path / "ws" / "reports" / run_id
    page = (reports / "index.html").read_text(encoding="utf-8")
    assert page.count("Trend ending 2024-01-16") == 1
    assert "Anchor rate cur % [95% CI]" in page
    assert "Threshold drift (rows)" in page
    assert "= volume" in page
    assert "set history.anchor_run" not in page
    payload = json.loads((reports / "index.json").read_text(encoding="utf-8"))
    trend = payload["groups"][0]["trend"]
    assert trend["anchor_run"] is not None
    assert trend["windows"][0]["anchor"]["cells_used"] == 2
