"""T1: per-period statistics recorded with each period's totals, and population drift in history."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

import sorethumb_ml._pipeline as pipeline_mod
from sorethumb_ml import Config, score_forward
from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import _drift_notes, _drift_table
from sorethumb_ml.history.windows import compute_population_drift
from sorethumb_ml.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_two_period_parquet

pytestmark = pytest.mark.integration


def _cfg(tmp_path: Path) -> tuple[Config, list[str]]:
    parquet = tmp_path / "two.parquet"
    labels = sorted(write_two_period_parquet(parquet))
    cfg = make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        combination="composite",
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False},
    )
    return cfg, labels


def _rows(cfg: Config, labels: list[str]) -> list[dict]:
    from sorethumb_ml.io.fingerprint import logical_dataset_id

    with Workspace.open(Path(cfg.run.workdir)) as ws:
        return ws.store.period_stats_for_periods(
            logical_dataset_id(cfg.source.dataset_id, cfg.source.uri), labels, cfg.config_hash()
        )


def test_each_period_records_statistics_with_its_totals(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    first = run_detection(cfg, period_label_override=labels[0], no_report=True)
    run_detection(cfg, period_label_override=labels[1], no_report=True)
    rows = _rows(cfg, labels)
    assert {r["period_label"] for r in rows} == set(labels)
    row = next(r for r in rows if r["period_label"] == labels[0])
    stats = row["stats"]
    group = first.groups[0]
    assert row["run_id"] == first.run_id
    assert stats["n_rows"] == group.n_records
    assert stats["composite"]["flagged"] == group.n_anomalies
    assert set(stats["population"]) >= {"num_a", "num_b"}
    assert "id" not in stats["population"]  # id and time columns change every period
    assert "ts" not in stats["population"]
    assert set(stats["scores"]) == {d.name for d in cfg.detectors if d.enabled}


def test_a_resumed_run_still_records_statistics(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    run_detection(cfg, period_label_override=labels[0], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        ws.store._conn.execute("DELETE FROM period_stats")  # as if the ledger write never happened
        ws.store._conn.commit()
    again = run_detection(cfg, period_label_override=labels[0], no_report=True)
    assert all(g.status == "skipped" for g in again.groups)
    assert len(_rows(cfg, labels)) == len(again.groups)


def test_a_statistics_failure_never_fails_the_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sorethumb_ml.history.period_stats as ps

    def _boom(**_kw: object) -> dict:
        raise RuntimeError("stats exploded")

    monkeypatch.setattr(ps, "compute_period_stats", _boom)
    cfg, labels = _cfg(tmp_path)
    result = run_detection(cfg, period_label_override=labels[0], no_report=True)
    assert all(g.status == "success" for g in result.groups)
    assert _rows(cfg, labels) == []


def test_an_unreadable_statistics_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = pipeline_mod._save_period_stats

    def _corrupt(ws: Workspace, run_id: str, group_key: str, **kw: object) -> None:
        real(ws, run_id, group_key, **kw)  # type: ignore[arg-type]
        (ws.results_dir(run_id, group_key) / "period_stats.json").write_text("{not json", encoding="utf-8")

    monkeypatch.setattr(pipeline_mod, "_save_period_stats", _corrupt)
    cfg, labels = _cfg(tmp_path)
    result = run_detection(cfg, period_label_override=labels[0], no_report=True)
    assert result.groups[0].status == "success"
    assert _rows(cfg, labels) == []


def test_score_forward_writes_no_statistics(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    src = run_detection(cfg, period_label_override=labels[0], no_report=True)
    fwd = score_forward(cfg, src.run_id, no_report=True, period_label_override=labels[1])
    ws_root = Path(cfg.run.workdir)
    assert not list((ws_root / "results" / fwd.run_id).rglob("period_stats.json"))


def test_history_shows_population_drift_between_periods(tmp_path: Path) -> None:
    from sorethumb_ml.io.fingerprint import logical_dataset_id

    cfg, labels = _cfg(tmp_path)
    run_detection(cfg, period_label_override=labels[0], no_report=True)
    run_detection(cfg, period_label_override=labels[1], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        (d,) = compute_population_drift(
            ws.store,
            logical_dataset_id(cfg.source.dataset_id, cfg.source.uri),
            cfg.config_hash(),
            labels[1],
            [1],
            "day",
        )
    assert (d.current_stats, d.prior_stats) == (1, 1)
    assert d.current_rows > 0
    assert d.prior_rows > 0
    assert d.volume_change is not None
    # The planted anomalies flip from num_a=+999 to -999 between the days; num_a moves the most.
    assert d.top_fields(1)[0][0] == "num_a"
    console = Console(width=200, record=True, color_system=None)
    console.print(_drift_table([d]))
    text = console.export_text()
    assert "Population drift" in text
    assert "num_a" in text
    notes = "\n".join(_drift_notes([d]))
    assert "Population Stability Index" in notes
    assert "without summaries" not in notes


def test_drift_without_statistics_says_so(tmp_path: Path) -> None:
    with Workspace.init(tmp_path / "ws") as ws:
        (d,) = compute_population_drift(ws.store, "ds", "cfg", "2026-09-03", [1], "day")
    assert (d.current_stats, d.prior_stats, d.fields) == (0, 0, {})
    assert d.max_band is None
    assert d.volume_change is None
    assert "without summaries" in "\n".join(_drift_notes([d]))
