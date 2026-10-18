"""T2: the anchor run and the exact split of a count change into volume, score drift and threshold drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from sorethumb_ml import Config, score_forward
from sorethumb_ml._pipeline import _make_run_id, run_detection
from sorethumb_ml.cli import _decomposition_notes, _decomposition_table
from sorethumb_ml.errors import ConfigError
from sorethumb_ml.history.anchor import RULE_FILENAME, load_rule
from sorethumb_ml.history.windows import compute_count_decomposition
from sorethumb_ml.io.fingerprint import logical_dataset_id
from sorethumb_ml.store.workspace import Workspace, make_group_key
from tests.factories.configs import make_config
from tests.factories.frames import write_two_period_parquet

pytestmark = pytest.mark.integration


def _cfg(tmp_path: Path, *, anchor: str | None = None, **kw: object) -> tuple[Config, list[str]]:
    parquet = tmp_path / "two.parquet"
    if not parquet.exists():
        write_two_period_parquet(parquet)
    labels = ["2024-01-15", "2024-01-16"]
    cfg = make_config(
        parquet,
        tmp_path / "ws",
        source_format="parquet",
        combination=str(kw.pop("combination", "composite")),
        contamination=kw.pop("contamination", "auto"),  # type: ignore[arg-type]
        time_column="ts",
        history_kwargs={"period_granularity": "day", "roll_non_business": False, "anchor_run": anchor},
    )
    return cfg, labels


def _ds(cfg: Config) -> str:
    return logical_dataset_id(cfg.source.dataset_id, cfg.source.uri)


def _stats(cfg: Config, label: str) -> list[dict]:
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        return ws.store.period_stats_for_periods(_ds(cfg), [label], cfg.config_hash())


# ---------------------------------------------------------------------------
# Config, run id and validation
# ---------------------------------------------------------------------------


def test_the_anchor_is_not_in_the_config_hash_but_is_in_the_run_id(tmp_path: Path) -> None:
    plain, _ = _cfg(tmp_path)
    anchored, _ = _cfg(tmp_path, anchor="run_x")
    assert plain.config_hash() == anchored.config_hash()
    assert _make_run_id("d", "c", "p") != _make_run_id("d", "c", "p", anchor_run="run_x")
    assert _make_run_id("d", "c", "p") == _make_run_id("d", "c", "p", anchor_run=None)


def test_an_unusable_anchor_is_refused(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    fwd = score_forward(cfg, fit.run_id, no_report=True, period_label_override=labels[1])
    for anchor, match in ((fwd.run_id, "score-forward"), ("run_nope", "not a run in this workspace")):
        bad, _ = _cfg(tmp_path, anchor=anchor)
        with pytest.raises(ConfigError, match=match):
            run_detection(bad, period_label_override=labels[1], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        ws.store._conn.execute("UPDATE run SET status='failed' WHERE run_id=?", [fit.run_id])
        ws.store._conn.commit()
    bad, _ = _cfg(tmp_path, anchor=fit.run_id)
    with pytest.raises(ConfigError, match="status 'failed'"):
        run_detection(bad, period_label_override=labels[1], no_report=True)


def test_an_anchor_from_another_dataset_is_refused(tmp_path: Path) -> None:
    other, labels = _cfg(tmp_path / "other")
    fit = run_detection(other, period_label_override=labels[0], no_report=True)
    cfg, _ = _cfg(tmp_path, anchor=fit.run_id)
    # Same workspace, different dataset: copy the run row's dataset by pointing at it.
    cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"workdir": other.run.workdir})})
    cfg = cfg.model_copy(
        update={"source": cfg.source.model_copy(update={"dataset_id": "a-different-dataset"})}
    )
    with pytest.raises(ConfigError, match="different dataset"):
        run_detection(cfg, period_label_override=labels[1], no_report=True)


# ---------------------------------------------------------------------------
# Rules and anchored periods
# ---------------------------------------------------------------------------


def test_every_fitting_run_saves_its_rule(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        rule = load_rule(ws, fit.run_id, fit.groups[0].group_key)
    assert rule is not None
    assert rule["combination"] == "composite"
    assert rule["threshold"] is not None
    assert set(rule["weights"]) == set(rule["detectors"])


@pytest.mark.parametrize(
    ("combination", "contamination"),
    [("composite", "auto"), ("composite", 0.05), ("intersection", "auto"), ("union", 0.05)],
)
def test_an_anchored_rerun_of_the_anchor_period_has_no_threshold_drift(
    tmp_path: Path, combination: str, contamination: str | float
) -> None:
    """Scoring the anchor's own period with its own models reproduces its own flags."""
    cfg, labels = _cfg(tmp_path, combination=combination, contamination=contamination)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    anchored, _ = _cfg(tmp_path, anchor=fit.run_id, combination=combination, contamination=contamination)
    again = run_detection(anchored, period_label_override=labels[0], no_report=True)
    assert again.run_id != fit.run_id
    (row,) = _stats(anchored, labels[0])
    anchor = row["stats"]["anchor"]
    assert anchor["comparable"] is True
    assert anchor["anchor_run_id"] == fit.run_id
    assert anchor["flagged"] == row["stats"]["composite"]["flagged"]
    assert anchor["tail_share"] == pytest.approx(0.05, abs=0.02)
    assert anchor["score_psi"] == pytest.approx(0.0, abs=1e-6)


def test_the_decomposition_adds_up_exactly(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    anchored, _ = _cfg(tmp_path, anchor=fit.run_id)
    run_detection(anchored, period_label_override=labels[0], no_report=True)
    run_detection(anchored, period_label_override=labels[1], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        (d,) = compute_count_decomposition(ws.store, _ds(cfg), cfg.config_hash(), labels[1], [1], "day")
    assert d.anchor_run_id == fit.run_id
    assert (d.groups_used, d.cells_used, d.cells_excluded) == (1, 2, 0)
    assert d.count_change == d.current_count - d.prior_count
    assert d.volume + d.score_drift + d.threshold_drift == pytest.approx(d.count_change, abs=1e-9)  # type: ignore[operator]
    assert d.anchor_change_ci is not None
    assert d.score_psi is not None
    assert d.tail_share is not None

    text = _render(_decomposition_table([d]))
    assert f"Count change against anchor {fit.run_id}" in text
    notes = "\n".join(_decomposition_notes([d], configured_anchor=fit.run_id))
    assert "volume" in notes
    assert "threshold drift" in notes


def test_a_period_the_anchor_cannot_score_is_not_comparable(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        (ws.root / "models" / fit.run_id / fit.groups[0].group_key / RULE_FILENAME).unlink()
    anchored, _ = _cfg(tmp_path, anchor=fit.run_id)
    run_detection(anchored, period_label_override=labels[1], no_report=True)
    (row,) = _stats(anchored, labels[1])
    assert row["stats"]["anchor"] == {
        "anchor_run_id": fit.run_id,
        "comparable": False,
        "reason": "the anchor run saved no fixed rule for this group",
    }


def test_a_broken_anchor_model_is_not_comparable_and_never_fails_the_group(tmp_path: Path) -> None:
    cfg, labels = _cfg(tmp_path)
    fit = run_detection(cfg, period_label_override=labels[0], no_report=True)
    gk = fit.groups[0].group_key
    with Workspace.open(Path(cfg.run.workdir)) as ws:
        for manifest in (ws.root / "models" / fit.run_id / gk).glob("*.joblib"):
            manifest.write_bytes(b"corrupt")
    anchored, _ = _cfg(tmp_path, anchor=fit.run_id)
    result = run_detection(anchored, period_label_override=labels[1], no_report=True)
    assert result.groups[0].status == "success"
    (row,) = _stats(anchored, labels[1])
    assert row["stats"]["anchor"]["comparable"] is False
    assert "ModelIntegrityError" in row["stats"]["anchor"]["reason"]


# ---------------------------------------------------------------------------
# Decomposition edge cases (seeded statistics)
# ---------------------------------------------------------------------------


def _seed(
    ws: Workspace, gk: str, label: str, n: int, own: int, anchor: int | None, anchor_id: str = "run_a"
) -> None:
    ws.store.upsert_dataset("ds", "uri", "s", "c", 1000, 5)
    ws.store.insert_run("r1", "ds", "{}", 0)
    stats = {"stats_version": 2, "n_rows": n, "population": {}, "scores": {}, "composite": {"flagged": own}}
    if anchor is not None:
        stats["anchor"] = {
            "anchor_run_id": anchor_id,
            "comparable": True,
            "n_rows": n,
            "flagged": anchor,
            "combined_percentiles": [i / 100 for i in range(101)],
            "anchor_percentiles": [i / 100 for i in range(101)],
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


def test_the_terms_isolate_each_cause(ws: Workspace) -> None:
    ga, gb = make_group_key({"g": "a"}), make_group_key({"g": "b"})
    with ws:
        # Group a: volume doubles, rates unchanged -> pure volume.
        _seed(ws, ga, "2026-09-02", 100, 10, 10)
        _seed(ws, ga, "2026-09-03", 200, 20, 20)
        # Group b: same volume; the anchor rate rises but the own rate does not -> the
        # self-recalibrated threshold absorbed a real score shift.
        _seed(ws, gb, "2026-09-02", 100, 5, 5)
        _seed(ws, gb, "2026-09-03", 100, 5, 15)
        (d,) = compute_count_decomposition(ws.store, "ds", "cfg", "2026-09-03", [1], "day")
    assert d.volume == pytest.approx(10.0)
    assert d.score_drift == pytest.approx(10.0)
    assert d.threshold_drift == pytest.approx(-10.0)
    assert d.count_change == 10
    assert d.groups_used == 2


def test_unanchored_and_mixed_cells(ws: Workspace) -> None:
    ga, gb = make_group_key({"g": "a"}), make_group_key({"g": "b"})
    with ws:
        _seed(ws, ga, "2026-09-02", 100, 10, None)  # run before an anchor was set
        _seed(ws, ga, "2026-09-03", 100, 10, 10)
        (none,) = compute_count_decomposition(ws.store, "ds", "cfg", "2026-09-03", [1], "day")
        _seed(ws, gb, "2026-09-02", 100, 10, 10, anchor_id="run_a")
        _seed(ws, gb, "2026-09-03", 100, 10, 10, anchor_id="run_b")
        (mixed,) = compute_count_decomposition(ws.store, "ds", "cfg", "2026-09-03", [1], "day", [gb])
    assert none.volume is None
    assert none.cells_excluded == 2
    assert mixed.anchor_run_id == "mixed"
    assert mixed.volume is None
    notes = "\n".join(_decomposition_notes([mixed], configured_anchor="run_b"))
    assert "More than one anchor run" in notes


def test_notes_without_an_anchor() -> None:
    from sorethumb_ml.history.windows import DecompositionResult

    empty = DecompositionResult(
        window_size=1,
        period_label="p",
        anchor_run_id=None,
        groups_used=0,
        cells_used=0,
        cells_excluded=0,
        current_rows=0,
        prior_rows=0,
        current_count=0,
        prior_count=0,
        current_anchor_count=0,
        prior_anchor_count=0,
        volume=None,
        score_drift=None,
        threshold_drift=None,
        count_change=None,
    )
    assert empty.current_anchor_rate is None
    unset = "\n".join(_decomposition_notes([empty], configured_anchor=None))
    assert "set history.anchor_run" in unset
    pending = "\n".join(_decomposition_notes([empty], configured_anchor="run_a"))
    assert "--force-period" in pending


def _render(table: object) -> str:
    console = Console(width=250, record=True, color_system=None)
    console.print(table)
    return console.export_text()
