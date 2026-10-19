"""The scale benchmark's own machinery: data, stage accounting, instrumentation, rendering.

The benchmark itself runs each measurement in a spawned process (see
tests/benchmark/test_scale_benchmark_smoke.py); these check the parts that decide what
it reports, in-process and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sorethumb_ml._pipeline as pipeline
import sorethumb_ml.store.models as models
from sorethumb_ml import cli
from sorethumb_ml.evaluate import scale_benchmark as sb

pytestmark = pytest.mark.unit


def test_every_instrumented_pipeline_function_still_exists() -> None:
    # A rename would otherwise drop a stage from the benchmark silently: its time
    # would fall into "other" and the published breakdown would be wrong.
    for attr, _stage in sb.TIMED_FUNCTIONS:
        assert callable(getattr(pipeline, attr, None)), f"_pipeline.{attr} is gone"
    # Wrapped directly in _instrument, for per-detector training and scoring.
    assert callable(pipeline._fit_detector)
    assert pipeline.score_and_flag is models.score_and_flag
    assert callable(pipeline._timed_stage)
    assert callable(cli._load_config)


def test_the_pipeline_still_names_the_stages_the_benchmark_keeps() -> None:
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    for name in sb.TIMED_STAGE_NAMES:
        assert f'_timed_stage("{name}"' in source


def test_stage_clock_records_exclusive_time_for_nested_stages() -> None:
    ticks = iter([0.0, 1.0, 4.0, 10.0])  # outer starts, inner 1..4, outer ends at 10
    clock = sb.StageClock(clock=lambda: next(ticks))
    with clock.stage("outer"), clock.stage("inner"):
        pass
    assert clock.seconds == {"inner": 3.0, "outer": 7.0}
    assert clock.calls == {"inner": 1, "outer": 1}


def test_stage_clock_attributes_peak_memory_growth_exclusively() -> None:
    peaks = iter([100.0, 100.0, 400.0, 450.0])
    clock = sb.StageClock(clock=lambda: 0.0, memory=lambda: next(peaks))
    with clock.stage("outer"), clock.stage("inner"):
        pass
    assert clock.peak_growth_mb == {"inner": 300.0, "outer": 50.0}


def test_stage_clock_wrap_times_each_call_and_returns_the_result() -> None:
    clock = sb.StageClock()
    double = clock.wrap(lambda x: 2 * x, "double")
    assert double(3) == 6
    assert double(4) == 8
    assert clock.calls["double"] == 2


def test_scale_frame_is_deterministic_and_plants_the_stated_fraction() -> None:
    frame, labels = sb.make_scale_frame(20_000, seed=3)
    again, again_labels = sb.make_scale_frame(20_000, seed=3)
    assert frame.equals(again)
    assert np.array_equal(labels, again_labels)
    assert frame.height == 20_000
    assert labels.sum() == round(20_000 * sb.ANOMALY_FRACTION)
    assert "label" not in frame.columns  # the labels never reach the pipeline
    assert frame["id"].to_list() == list(range(20_000))


def test_scale_frame_planted_anomalies_are_anomalous() -> None:
    frame, labels = sb.make_scale_frame(50_000, seed=0)
    planted = labels.astype(bool)
    # The unseen merchant appears only on planted rows, and only at 3 a.m.
    unseen = (frame["merchant"] == "m_unseen").to_numpy()
    assert unseen.any()
    assert not (unseen & ~planted).any()
    assert set(frame.filter(frame["merchant"] == "m_unseen")["event_time"].dt.hour().to_list()) == {3}
    # Normal rows are concentrated on the working day.
    hours = frame.filter(~planted)["event_time"].dt.hour().to_numpy()
    assert ((hours >= 7) & (hours <= 19)).mean() > 0.9
    # No ordinary merchant takes a large share of rows.
    assert frame.filter(~planted)["merchant"].value_counts()["count"].max() < 0.25 * 50_000


def _doc() -> dict:
    def measured(total: float) -> dict:
        return {
            "total_seconds": total,
            "stage_seconds": {
                "other": 0.5,
                "explain": 3.0,
                "score:isolation_forest": 0.25,
                "fit:isolation_forest": 2.0,
                "load-dataset": 1.0,
            },
            "stage_peak_growth_mb": {"fit:isolation_forest": 700.0},
            "peak_mb": 900.0,
            "baseline_peak_mb": 90.0,
            "feature_width": 27,
            "detectors": {"isolation_forest": {"train_rows": 100, "scored_rows": 100}},
            "n_flagged": 5,
            "n_explained": 5,
            "flag_quality": {"flag_precision": 0.4, "flag_recall": 0.5},
        }

    return {
        "schema_version": 1,
        "seed": 0,
        "environment": {
            "generated_at": "2026-10-19T00:00:00+00:00",
            "commit": "abc",
            "platform": "test",
            "cpu_count": 8,
            "python": "3.12",
            "packages": {
                "sorethumb-ml": "0.1.0",
                "numpy": "2",
                "scikit-learn": "1",
                "polars": "2",
                "shap": "0",
            },
        },
        "cases": [
            {
                "n_rows": 100,
                "input_csv_mb": 0.1,
                "run": measured(6.5),
                "run_outputs_mb": {"total": 1.0},
                "score_forward": measured(4.0),
            }
        ],
    }


def test_render_markdown_reports_every_measurement() -> None:
    text = sb.render_markdown(_doc())
    for expected in (
        "| Elapsed | 6.5 s |",
        "| Peak memory (RSS) | 900 MB |",
        "| of which: Python with sorethumb imported | 90 MB |",
        "| Feature width | 27 |",
        "| `isolation_forest` trained on | 100 of 100 |",
        "| Rows explained | 5 |",
        "| Workspace on disk | 1.0 MB |",
        "| Explanations | 3.0 s |",
        "| Train `isolation_forest` | 2.0 s · +700 MB |",
        "| Score every row: `isolation_forest` | 0.2 s · +0 MB |",
        "| Flags vs planted anomalies (precision / recall) | 0.40 / 0.50 |",
        "commit `abc`",
    ):
        assert expected in text, expected


def test_write_results_splices_between_markers_and_refuses_without_them(tmp_path: Path) -> None:
    md = tmp_path / "benchmarks.md"
    md.write_text(
        "intro\n<!-- scale-results-start -->\nold\n<!-- scale-results-end -->\noutro\n", encoding="utf-8"
    )
    sb.write_results(_doc(), tmp_path / "scale.json", md)
    text = md.read_text(encoding="utf-8")
    assert "old" not in text
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert json.loads((tmp_path / "scale.json").read_text(encoding="utf-8"))["cases"][0]["n_rows"] == 100

    bare = tmp_path / "bare.md"
    bare.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        sb.write_results(_doc(), tmp_path / "scale.json", bare)


def test_stages_render_in_pipeline_order() -> None:
    text = sb.render_markdown(_doc())
    order = [
        "Load and parse",
        "Train `isolation_forest`",
        "Score every row",
        "| Explanations | 3.0 s ·",
        "Other",
    ]
    positions = [text.index(label) for label in order]
    assert positions == sorted(positions)
