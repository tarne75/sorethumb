"""One behavioural test per config field that was documented but previously inert.

Fields covered: run.max_rows, run.reuse_models, run.strict, run.slow_stage_seconds,
explain.enabled, explain.kernel_shap, explain.permutation_importance, report.formats,
report.open_after, report.rolling_windows, source.cache.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb import _pipeline
from sorethumb._pipeline import _group_summary, run_detection
from sorethumb.cli import app
from sorethumb.config import (
    Config,
    DetectorConfig,
    ExplainConfig,
    ReportConfig,
    RunConfig,
    ScoringConfig,
    SourceConfig,
)
from sorethumb.errors import NonFiniteWarning, SampleTruncatedWarning, SlowStageWarning

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, n_rows: int = 120, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    num_a = rng.normal(0.0, 1.0, n_rows)
    num_a[:3] = 999.0  # planted anomalies so a report section has real rows
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "num_a": num_a.tolist(),
            "num_b": rng.normal(5.0, 2.0, n_rows).tolist(),
        }
    ).write_csv(str(path))
    return path


def _cfg(
    csv: Path,
    workdir: Path,
    *,
    detectors: list[DetectorConfig] | None = None,
    run_kwargs: dict | None = None,
    explain_kwargs: dict | None = None,
    report_kwargs: dict | None = None,
) -> Config:
    return Config(
        source=SourceConfig(uri=str(csv), format="csv"),
        run=RunConfig(workdir=str(workdir), seed=42, **(run_kwargs or {})),
        columns={"id_column": "id"},
        detectors=detectors or [DetectorConfig(name="isolation_forest")],
        scoring=ScoringConfig(contamination=0.05, combination="composite", min_records=5),
        explain=ExplainConfig(**(explain_kwargs or {})),
        report=ReportConfig(**(report_kwargs or {})),
    )


def _fake_clock(step: float):
    """A monotonic-ish clock that jumps `step` seconds every call."""
    t = [0.0]

    def _c() -> float:
        t[0] += step
        return t[0]

    return _c


# ---------------------------------------------------------------------------
# run.max_rows
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::sorethumb.errors.SampleTruncatedWarning")
def test_run_max_rows_truncates_the_input(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", n_rows=120)
    result = run_detection(_cfg(csv, tmp_path / "ws", run_kwargs={"max_rows": 40}), no_report=True)
    assert result.n_succeeded == 1
    assert result.groups[0].n_records == 40  # only the first 40 rows entered the pipeline


def test_run_max_rows_emits_sample_truncated_warning(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", n_rows=120)
    with pytest.warns(SampleTruncatedWarning, match="run.max_rows"):
        run_detection(_cfg(csv, tmp_path / "ws", run_kwargs={"max_rows": 40}), no_report=True)


# ---------------------------------------------------------------------------
# run.reuse_models
# ---------------------------------------------------------------------------


def test_run_reuse_models_skips_refitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    csv = _write_csv(tmp_path / "d.csv")
    workdir = tmp_path / "ws"

    # First run fits + persists the model.
    r1 = run_detection(_cfg(csv, workdir), no_report=True)
    assert r1.n_succeeded == 1

    # Now any fit attempt is a hard error.
    def _boom(*_a, **_k):
        raise AssertionError("detector was re-fitted despite run.reuse_models")

    monkeypatch.setattr(IsolationForestDetector, "fit", _boom)

    # force=True re-runs the (complete) group; reuse_models must load the model.
    r2 = run_detection(_cfg(csv, workdir, run_kwargs={"reuse_models": True}), force=True, no_report=True)
    assert r2.n_succeeded == 1
    assert r2.n_anomalies == r1.n_anomalies

    # Sanity: without reuse_models the same forced re-run hits the boom.
    r3 = run_detection(_cfg(csv, workdir), force=True, no_report=True)
    assert r3.n_failed == 1


# ---------------------------------------------------------------------------
# run.strict
# ---------------------------------------------------------------------------


def _warn_in_group(**kwargs):
    warnings.warn("synthetic non-finite in group", NonFiniteWarning, stacklevel=2)
    return _group_summary(
        kwargs["group_key"],
        kwargs["group_label"],
        kwargs["n_records"],
        status="success",
        n_anomalies=0,
        anomaly_rate=0.0,
    )


def test_run_strict_promotes_in_group_warnings_to_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_pipeline, "_run_group", _warn_in_group)
    csv = _write_csv(tmp_path / "d.csv")

    lenient = run_detection(_cfg(csv, tmp_path / "ws_lenient"), no_report=True)
    assert lenient.n_succeeded == 1
    assert any("non-finite" in w for w in lenient.warnings_issued)

    strict = run_detection(_cfg(csv, tmp_path / "ws_strict", run_kwargs={"strict": True}), no_report=True)
    assert strict.n_succeeded == 0
    assert strict.n_failed == 1
    assert "NonFiniteWarning" in (strict.groups[0].error or "")


# ---------------------------------------------------------------------------
# run.slow_stage_seconds
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("default::sorethumb.errors.SlowStageWarning")
def test_run_slow_stage_seconds_warns_on_slow_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_pipeline, "_clock", _fake_clock(step=999.0))
    csv = _write_csv(tmp_path / "d.csv")
    with pytest.warns(SlowStageWarning, match="run.slow_stage_seconds"):
        run_detection(_cfg(csv, tmp_path / "ws", run_kwargs={"slow_stage_seconds": 1}), no_report=True)


def test_run_slow_stage_seconds_silent_when_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Clock barely advances → no stage exceeds the (default 300s) threshold.
    monkeypatch.setattr(_pipeline, "_clock", _fake_clock(step=0.001))
    csv = _write_csv(tmp_path / "d.csv")
    result = run_detection(_cfg(csv, tmp_path / "ws"), no_report=True)  # raises if SlowStageWarning fires
    assert result.n_succeeded == 1


# ---------------------------------------------------------------------------
# explain.enabled
# ---------------------------------------------------------------------------


def test_explain_enabled_false_skips_attribution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    import sorethumb.explain.shap_tree as shap_tree

    real = shap_tree.tree_shap_attributions

    def _spy(*a, **k):
        calls.append("shap")
        return real(*a, **k)

    monkeypatch.setattr(shap_tree, "tree_shap_attributions", _spy)
    csv = _write_csv(tmp_path / "d.csv")

    run_detection(_cfg(csv, tmp_path / "ws_on", explain_kwargs={"enabled": True}), no_report=True)
    assert calls == ["shap"]

    calls.clear()
    run_detection(_cfg(csv, tmp_path / "ws_off", explain_kwargs={"enabled": False}), no_report=True)
    assert calls == []


# ---------------------------------------------------------------------------
# explain.kernel_shap
# ---------------------------------------------------------------------------


def test_explain_kernel_shap_routes_non_tree_detectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sorethumb.explain.gradient as gradient

    seen: list[str] = []

    def _stub(name: str):
        def _f(_det, x, **_k):
            seen.append(name)
            return np.zeros((x.shape[0], x.shape[1]), dtype=np.float64), "heuristic"

        return _f

    monkeypatch.setattr(gradient, "kernel_shap_attributions", _stub("kernel"))
    monkeypatch.setattr(gradient, "gradient_attributions", _stub("gradient"))

    csv = _write_csv(tmp_path / "d.csv")
    dets = [DetectorConfig(name="one_class_svm")]

    run_detection(
        _cfg(csv, tmp_path / "ws_grad", detectors=dets, explain_kwargs={"kernel_shap": False}),
        no_report=True,
    )
    run_detection(
        _cfg(csv, tmp_path / "ws_ks", detectors=dets, explain_kwargs={"kernel_shap": True}),
        no_report=True,
    )
    assert "gradient" in seen
    assert "kernel" in seen
    assert seen.index("gradient") < seen.index("kernel")


# ---------------------------------------------------------------------------
# explain.permutation_importance
# ---------------------------------------------------------------------------


def test_explain_permutation_importance_runs_the_crosscheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sorethumb.explain.project as project

    calls: list[int] = []

    def _spy(*_a, **_k):
        calls.append(1)
        return {"num_a": 1.0}

    monkeypatch.setattr(project, "permutation_importance", _spy)
    csv = _write_csv(tmp_path / "d.csv")

    run_detection(
        _cfg(csv, tmp_path / "ws_off", explain_kwargs={"permutation_importance": False}),
        no_report=True,
    )
    assert calls == []

    run_detection(
        _cfg(csv, tmp_path / "ws_on", explain_kwargs={"permutation_importance": True}),
        no_report=True,
    )
    assert calls == [1]


# ---------------------------------------------------------------------------
# report.formats
# ---------------------------------------------------------------------------


def test_report_formats_selects_which_artefacts_are_written(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv")

    r_json = run_detection(
        _cfg(csv, tmp_path / "ws_json", report_kwargs={"formats": ["json"]}), no_report=False
    )
    assert r_json.report_path is not None
    rdir = r_json.report_path.parent
    assert r_json.report_path.name == "index.json"
    assert not (rdir / "index.html").exists()
    assert list(rdir.glob("*.csv")) == []
    json.loads(r_json.report_path.read_text())  # valid JSON

    r_html = run_detection(
        _cfg(csv, tmp_path / "ws_html", report_kwargs={"formats": ["html"]}), no_report=False
    )
    assert r_html.report_path is not None
    assert r_html.report_path.name == "index.html"
    assert list(r_html.report_path.parent.glob("*.csv")) == []
    assert not (r_html.report_path.parent / "index.json").exists()


# ---------------------------------------------------------------------------
# report.open_after
# ---------------------------------------------------------------------------


def test_report_open_after_launches_browser_only_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(_pipeline.webbrowser, "open", opened.append)
    csv = _write_csv(tmp_path / "d.csv")

    run_detection(_cfg(csv, tmp_path / "ws_off", report_kwargs={"open_after": False}), no_report=False)
    assert opened == []

    run_detection(_cfg(csv, tmp_path / "ws_on", report_kwargs={"open_after": True}), no_report=False)
    assert len(opened) == 1
    assert opened[0].startswith("file://")


# ---------------------------------------------------------------------------
# report.rolling_windows
# ---------------------------------------------------------------------------


def test_history_defaults_to_report_rolling_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sorethumb.history.windows as hw

    seen: dict[str, list[int]] = {}

    def _fake(_store, _dataset_fp, _ref_label, windows, _granularity, **_kw):
        seen["windows"] = list(windows)
        return []

    monkeypatch.setattr(hw, "compute_rolling_windows", _fake)

    csv = _write_csv(tmp_path / "d.csv")
    workdir = tmp_path / "ws"
    toml = tmp_path / "sorethumb.toml"
    toml.write_text(
        "[source]\n"
        f'uri = "{csv}"\nformat = "csv"\n'
        "[run]\n"
        f'workdir = "{workdir}"\nseed = 0\n'
        '[columns]\nid_column = "id"\n'
        "[report]\nrolling_windows = [3, 9]\n",
        encoding="utf-8",
    )
    # --dry-run creates the workspace without a full run.
    runner.invoke(app, ["run", "--config", str(toml), "--dry-run"])
    result = runner.invoke(app, ["history", "--config", str(toml)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert seen["windows"] == [3, 9]


# ---------------------------------------------------------------------------
# source.cache
# ---------------------------------------------------------------------------


def test_source_cache_false_never_persists_a_fingerprint_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sorethumb.io import source as src

    n = {"i": 0}

    def _fake_download(_url: str, _headers: dict, dest: Path) -> None:
        n["i"] += 1
        dest.write_bytes(f"a,b\n{n['i']},{n['i']}\n".encode())

    monkeypatch.setattr(src, "_download_to", _fake_download)

    on_dir = tmp_path / "cache_on"
    p_on = src.resolve_source(SourceConfig(uri="https://x/data.csv", cache=True), on_dir)
    assert p_on.parent.parent == on_dir  # <fingerprint>/data.csv
    assert list(on_dir.glob("*/data.csv"))

    off_dir = tmp_path / "cache_off"
    cfg_off = SourceConfig(uri="https://x/data.csv", cache=False)
    p_off = src.resolve_source(cfg_off, off_dir)
    assert p_off.name.startswith("uncached_data")
    assert list(off_dir.glob("*/data.csv")) == []  # no fingerprint-keyed cache dir

    # cache=False is always-fresh: a second call re-downloads new content.
    first = p_off.read_text()
    p_off2 = src.resolve_source(cfg_off, off_dir)
    assert p_off2.read_text() != first
