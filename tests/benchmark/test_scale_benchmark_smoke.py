"""The scale benchmark end to end on a small size: every published field is produced.

Marked ``benchmark`` (spawned processes, real fitting); runs in the scheduled
full-nightly job. The published 100,000 and 1,000,000-row figures come from
``scripts/run_scale_benchmark.py``; this keeps the harness that produces them working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sorethumb_ml.evaluate.scale_benchmark import render_markdown, run_scale_case

pytestmark = pytest.mark.benchmark


def test_a_small_case_measures_every_published_field(tmp_path: Path) -> None:
    case = run_scale_case(3_000, tmp_path / "scratch")

    assert not (tmp_path / "scratch" / "n3000").exists()  # cleaned up
    for part in ("run", "score_forward"):
        measured = case[part]
        assert measured["total_seconds"] > 0
        assert measured["peak_mb"] >= measured["baseline_peak_mb"] > 0
        assert measured["flag_quality"]["flag_recall"] is not None
        stages = measured["stage_seconds"]
        assert {"load-dataset", "feature-apply", "explain"} <= set(stages)
        assert {f"score:{name}" for name in case["run"]["config"]["detectors"]} <= set(stages)
        # Exclusive stage times account for the measured total.
        assert sum(stages.values()) == pytest.approx(measured["total_seconds"], rel=0.05)
    run = case["run"]
    assert {"feature-fit", "report"} <= set(run["stage_seconds"])
    assert {f"fit:{name}" for name in run["config"]["detectors"]} <= set(run["stage_seconds"])
    assert run["feature_width"] > 0
    assert set(run["detectors"]) == set(run["config"]["detectors"])
    assert all(d["scored_rows"] == 3_000 for d in run["detectors"].values())
    assert run["n_explained"] > 0
    assert case["run_outputs_mb"]["total"] > 0

    text = render_markdown(
        {
            "schema_version": 1,
            "seed": 0,
            "environment": {
                "generated_at": "x",
                "commit": "x",
                "platform": "x",
                "cpu_count": 1,
                "python": "x",
                "packages": {
                    "sorethumb-ml": "x",
                    "numpy": "x",
                    "scikit-learn": "x",
                    "polars": "x",
                    "shap": "x",
                },
            },
            "cases": [case],
        }
    )
    assert "3,000 rows" in text
