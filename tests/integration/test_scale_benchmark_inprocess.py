"""The scale benchmark's measuring path run in this process, so coverage sees it.

The published benchmark measures each run in a spawned process
(tests/benchmark/test_scale_benchmark_smoke.py runs it that way). Here the child step is
called directly instead, with every pipeline attribute it instruments restored afterwards,
so the instrumentation, the measurements and the result document are exercised on every PR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import sorethumb_ml._pipeline as pipeline
from sorethumb_ml.evaluate import scale_benchmark as sb
from sorethumb_ml.store import models

pytestmark = pytest.mark.integration


class _ListQueue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-set every attribute _instrument replaces, so monkeypatch restores the
    # originals at teardown, and keep the child's chdir from leaking.
    for attr, _stage in sb.TIMED_FUNCTIONS:
        monkeypatch.setattr(pipeline, attr, getattr(pipeline, attr))
    for attr in ("_fit_detector", "score_and_flag", "_timed_stage"):
        monkeypatch.setattr(pipeline, attr, getattr(pipeline, attr))
    monkeypatch.setattr(models, "score_and_flag", getattr(models, "score_and_flag"))  # noqa: B009
    monkeypatch.chdir(Path.cwd())

    def run_here(
        kind: str, data_path: Path, workdir: Path, cwd: Path, source_run_id: str | None
    ) -> dict[str, Any]:
        queue = _ListQueue()
        sb._measure_child(kind, str(data_path), str(workdir), str(cwd), source_run_id, queue)
        (out,) = queue.items
        assert out["ok"], out.get("error")
        return out

    monkeypatch.setattr(sb, "_run_in_child", run_here)


@pytest.mark.usefixtures("in_process")
def test_the_benchmark_document_carries_every_measurement(tmp_path: Path) -> None:
    doc = sb.run_scale_benchmark((1_500,), tmp_path / "scratch")

    assert doc["environment"]["packages"]["sorethumb-ml"]
    assert doc["environment"]["commit"]
    (case,) = doc["cases"]
    assert case["n_rows"] == 1_500
    assert case["n_input_columns"] == 14
    run, score = case["run"], case["score_forward"]
    assert run["config"]["detectors"] == ["isolation_forest", "kmeans_distance", "one_class_svm"]
    assert {f"fit:{d}" for d in run["config"]["detectors"]} <= set(run["stage_seconds"])
    assert {f"score:{d}" for d in run["config"]["detectors"]} <= set(score["stage_seconds"])
    assert "score-forward-setup" in score["stage_seconds"]
    assert run["feature_width"] > 0
    assert run["detectors"]["one_class_svm"] == {"train_rows": 1_500, "scored_rows": 1_500}
    assert 0 < run["n_explained"] <= run["n_flagged"]
    assert 0 <= run["flag_quality"]["flag_recall"] <= 1
    assert case["run_outputs_mb"]["total"] > 0
    assert case["workspace_after_score_mb"] >= case["run_outputs_mb"]["total"]
    assert "1,500 rows" in sb.render_markdown(doc)


@pytest.mark.usefixtures("in_process")
def test_a_failed_measurement_is_reported_not_swallowed(tmp_path: Path) -> None:
    queue = _ListQueue()
    (tmp_path / "cwd").mkdir()
    sb._measure_child(
        "run", str(tmp_path / "missing.csv"), str(tmp_path / "ws"), str(tmp_path / "cwd"), None, queue
    )
    (out,) = queue.items
    assert out["ok"] is False
    assert out["error"]
