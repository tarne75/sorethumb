"""What a first or zero-config run tells a reader about the default flag policy.

The default (``intersection`` at ``scoring.contamination = "auto"``) is kept, so its
shortlist size is set by the detectors' own boundaries, not by the user. A run must say
so prominently on a first or zero-config run, always show the top of the ranking
(flagged or not), and never present "no rows flagged" as "no anomalies".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb_ml._pipeline import run_detection
from sorethumb_ml.cli import app
from sorethumb_ml.config import DetectorConfig
from sorethumb_ml.store.results import RANKED_PREVIEW_ROWS, read_ranked_preview, read_results
from sorethumb_ml.store.workspace import Workspace
from tests.factories.configs import make_config
from tests.factories.frames import write_planted_csv

pytestmark = pytest.mark.integration

runner = CliRunner()


def _flat(text: str) -> str:
    """Collapse Rich's line wrapping so phrases can be matched."""
    return " ".join(text.split())


def _write_categorical_csv(path: Path) -> Path:
    # Two uniform categoricals: the detectors' own boundaries do not overlap, so the
    # default intersection flags nothing while every row is still ranked.
    rng = np.random.default_rng(1)
    pl.DataFrame(
        {"colour": rng.choice(["red", "green", "blue"], 400), "size": rng.choice(["s", "m", "l", "xl"], 400)}
    ).write_csv(path)
    return path


def _run_zero_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Path, *extra: str):
    monkeypatch.chdir(tmp_path)
    return runner.invoke(app, ["run", data.name, "--no-save-config", "--log-level", "WARNING", *extra])


# ── The ranked preview ────────────────────────────────────────────────────────


def test_the_ranked_preview_keeps_the_top_of_the_ranking_flagged_or_not(tmp_path: Path) -> None:
    csv = tmp_path / "d.csv"
    planted = write_planted_csv(csv, n_normal=280, n_anomaly=5)
    result = run_detection(make_config(csv, tmp_path / "ws"), no_report=True)
    (group,) = result.groups

    with Workspace.open(tmp_path / "ws") as ws:
        preview = read_ranked_preview(ws, result.run_id, group.group_key)
        flagged = read_results(ws, result.run_id, group.group_key, with_explanations=False)
    assert preview is not None
    assert preview.height == RANKED_PREVIEW_ROWS
    assert preview["score_rank"].to_list() == list(range(1, RANKED_PREVIEW_ROWS + 1))
    scores = preview["composite_score"].to_list()
    assert scores == sorted(scores, reverse=True)
    # The flag column agrees with anomalies.parquet, and the planted rows lead the ranking.
    assert flagged is not None
    flagged_ids = set(flagged["row_id"].to_list())
    assert all(
        (rid in flagged_ids) == f for rid, f in zip(preview["row_id"], preview["flagged"], strict=True)
    )
    assert set(preview["row_id"].head(len(planted)).to_list()) == set(planted)
    # Source values travel with the ranking.
    assert {"num_a", "num_b", "cat"} <= set(preview.columns)
    # GroupSummary.top_ranked carries the same rows for API callers.
    assert [r["row_id"] for r in group.top_ranked] == preview["row_id"].to_list()
    assert group.top_ranked[0]["values"]["num_a"] == 999.0


def test_a_group_with_no_flagged_rows_still_has_a_ranking(tmp_path: Path) -> None:
    csv = _write_categorical_csv(tmp_path / "cats.csv")
    # The default detectors (make_config's default is Isolation Forest alone).
    detectors = [DetectorConfig(name=n) for n in ("isolation_forest", "kmeans_distance", "one_class_svm")]
    result = run_detection(
        make_config(csv, tmp_path / "ws", id_column=None, detectors=detectors), no_report=True
    )
    (group,) = result.groups
    assert group.n_anomalies == 0, "test precondition: this data should flag nothing at the default"
    assert len(group.top_ranked) == RANKED_PREVIEW_ROWS
    assert not any(r["flagged"] for r in group.top_ranked)


# ── The run summary ───────────────────────────────────────────────────────────


def test_a_zero_config_run_warns_that_nobody_chose_the_shortlist_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_planted_csv(tmp_path / "data.csv")
    out = _flat(_run_zero_config(tmp_path, monkeypatch, tmp_path / "data.csv", "--no-report").output)

    assert "Shortlist size not chosen" in out
    assert "Nobody chose how many rows to flag" in out
    assert "only rows every detector flags are kept" in out
    assert "Treat the ranking as the result" in out
    assert "Top-ranked rows" in out
    assert "flagged=" in out
    assert "anomalies=" not in out


def test_a_configured_repeat_run_keeps_a_short_note_instead_of_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "data.csv"
    write_planted_csv(csv)
    _run_zero_config(tmp_path, monkeypatch, csv, "--no-report")  # first run of the workspace
    (tmp_path / "sorethumb.toml").write_text(
        f'[source]\nuri = {json.dumps(str(csv))}\n\n[run]\nworkdir = "sorethumb-workspace"\nseed = 7\n',
        encoding="utf-8",
    )
    out = _flat(runner.invoke(app, ["run", "--no-report", "--log-level", "WARNING"]).output)

    assert "Shortlist size not chosen" not in out
    assert 'Shortlist size: not chosen (scoring.contamination = "auto"' in out
    assert "Top-ranked rows" in out


def test_a_numeric_review_budget_gets_no_shortlist_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "data.csv"
    write_planted_csv(csv)
    (tmp_path / "sorethumb.toml").write_text(
        f"[source]\nuri = {json.dumps(str(csv))}\n\n[scoring]\ncontamination = 0.05\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    out = _flat(runner.invoke(app, ["run", "--no-report", "--log-level", "WARNING"]).output)

    assert "Shortlist size" not in out
    assert "Top-ranked rows" in out


def test_no_flagged_rows_is_never_presented_as_no_anomalies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = _write_categorical_csv(tmp_path / "cats.csv")
    result = _run_zero_config(tmp_path, monkeypatch, csv)
    out = _flat(result.output)

    assert result.exit_code == 0, out
    assert "Flagged for review: 0 of 400 rows" in out
    assert "not a finding that the data holds no anomalies" in out
    assert "Top-ranked rows" in out
    assert "Shortlist size not chosen" in out

    ranked = _flat(runner.invoke(app, ["anomalies"]).output)
    assert "No rows were flagged" in ranked
    assert "not a finding that the data holds no anomalies" in ranked
    assert "Top-ranked rows" in ranked
    assert "No anomaly rows found" not in ranked
    # The flagged-rows JSON contract is unchanged: still an empty list.
    assert json.loads(runner.invoke(app, ["anomalies", "--json"]).output) == []

    report = next((tmp_path / "sorethumb-workspace" / "reports").glob("*/index.html")).read_text(
        encoding="utf-8"
    )
    assert "No rows were flagged in this group" in report
    assert "not a finding that the group holds no anomalies" in report
    assert "Shortlist size not chosen" in report
    assert "Top-ranked rows" in report
    assert "No anomalies flagged" not in report


# ── JSON and `anomalies --ranked` ─────────────────────────────────────────────


def test_run_json_says_whether_the_shortlist_size_was_chosen_and_carries_the_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_planted_csv(tmp_path / "data.csv")
    payload = json.loads(
        _run_zero_config(tmp_path, monkeypatch, tmp_path / "data.csv", "--no-report", "--json").output
    )

    assert payload["shortlist_size_chosen"] is False
    (group,) = payload["groups"]
    assert len(group["top_ranked"]) == RANKED_PREVIEW_ROWS
    assert group["top_ranked"][0]["score_rank"] == 1
    assert set(group["top_ranked"][0]) == {"row_id", "score_rank", "composite_score", "flagged", "values"}


def test_anomalies_ranked_lists_the_top_of_the_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_planted_csv(tmp_path / "data.csv")
    _run_zero_config(tmp_path, monkeypatch, tmp_path / "data.csv", "--no-report")

    rows = json.loads(runner.invoke(app, ["anomalies", "--ranked", "--top", "7", "--json"]).output)
    assert len(rows) == 7
    assert [r["score_rank"] for r in rows] == list(range(1, 8))
    assert rows[0]["values"]["num_a"] == 999.0

    text = _flat(runner.invoke(app, ["anomalies", "--ranked", "--top", "3"]).output)
    assert "Top-ranked rows" in text
    assert "3 row(s) by combined score" in text

    flagged = _flat(runner.invoke(app, ["anomalies", "--top", "3"]).output)
    assert "Flagged rows" in flagged
    assert "flagged row(s)" in flagged
    assert "anomaly row(s)" not in flagged
