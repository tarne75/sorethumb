"""Real-data backtests: pre-registration, the cross-dataset decision, publishing (no fitting)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sorethumb_ml.evaluate.real_backtest import (
    decide_real,
    inject_real_results,
    load_real_preregistration,
    render_real_results,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_PREREG_DIR = _REPO / "tests" / "benchmark" / "preregistrations" / "real"
_RESULTS_DIR = _REPO / "docs" / "benchmark-results" / "real-backtests"

_BODY = """
candidate = "composite-agreement@1"
hypothesis = "it flags better on real data"
datasets = ["kddcup99_sa", "covtype_2v4"]
require = "{require}"
"""


def _pre(tmp_path: Path, require: str = "all", extra: str = "") -> Any:
    path = tmp_path / "real-v1.toml"
    path.write_text(_BODY.format(require=require) + extra, encoding="utf-8")
    return load_real_preregistration(path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('\ncandidate = "nope@1"', "unknown candidate"),
        ('\nbaseline = "union@1"', "baseline must be the default"),
        ('\ndatasets = ["mnist"]', "datasets must be"),
        ('\ndatasets = ["covtype_2v4", "covtype_2v4"]', "datasets must be"),
        ('\nrequire = "most"', "require"),
        ("\nalpha = 2.0", "alpha"),
        ("\nn_resamples = 10", "n_resamples"),
        ('\ncontamination = "sometimes"', "contamination must be"),
    ],
)
def test_an_invalid_real_preregistration_is_rejected(tmp_path: Path, extra: str, message: str) -> None:
    lines = [
        ln
        for ln in _BODY.format(require="all").splitlines()
        if not ln.startswith(extra.split("=")[0].strip())
    ]
    path = tmp_path / "bad.toml"
    path.write_text("\n".join(lines) + extra + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_real_preregistration(path)


def _reps(f1: list[float], fp: float = 0.5, jac: float = 0.8) -> list[dict[str, float]]:
    return [{"flag_f1": v, "false_positive_budget_ratio": fp, "flag_jaccard_stability": jac} for v in f1]


_BASE = [0.30, 0.32, 0.31, 0.29, 0.30, 0.33, 0.28, 0.31]
_WIN = [b + 0.2 for b in _BASE]


def _per(
    kdd: list[dict[str, float]], cov: list[dict[str, float]]
) -> dict[str, dict[str, list[dict[str, float]]]]:
    return {
        "kddcup99_sa": {"intersection@1": _reps(_BASE), "composite-agreement@1": kdd},
        "covtype_2v4": {"intersection@1": _reps(_BASE), "composite-agreement@1": cov},
    }


def test_wins_on_every_dataset_outperform(tmp_path: Path) -> None:
    decision = decide_real(_pre(tmp_path), _per(_reps(_WIN), _reps(_WIN)))
    assert decision.outperforms
    assert all(c.improved for c in decision.comparisons)


def test_require_all_versus_any(tmp_path: Path) -> None:
    per = _per(_reps(_WIN), _reps(_BASE))
    assert not decide_real(_pre(tmp_path, "all"), per).outperforms
    assert decide_real(_pre(tmp_path, "any"), per).outperforms


def test_a_regression_on_any_dataset_blocks_a_win(tmp_path: Path) -> None:
    decision = decide_real(_pre(tmp_path, "any"), _per(_reps(_WIN), _reps(_BASE, fp=1.3)))
    assert not decision.outperforms
    assert any(r.startswith("covtype_2v4: spends more") for r in decision.reasons)
    assert decision.comparisons[1].regressions


def test_rendering_and_injection(tmp_path: Path) -> None:
    assert "No real-data" in render_real_results([], "x")
    decision = decide_real(_pre(tmp_path), _per(_reps(_WIN), _reps(_BASE))).as_dict()
    result = {
        "preregistration": {"id": "r-v1", "candidate": "c@1", "baseline": "b@1", "hypothesis": "h"},
        "decision": decision,
        "datasets": {"kddcup99_sa": {"slice": "rows A"}, "covtype_2v4": {"slice": "rows B"}},
    }
    doc = tmp_path / "d.md"
    doc.write_text(
        "x\n<!-- real-backtest-results-start -->\n<!-- real-backtest-results-end -->\ny\n", encoding="utf-8"
    )
    assert inject_real_results(doc, [result], "res") is True
    text = doc.read_text(encoding="utf-8")
    assert "**does not outperform**" in text
    assert "| covtype_2v4 | rows B |" in text
    assert inject_real_results(doc, [result], "res") is False
    doc.write_text("none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        inject_real_results(doc, [result], "res")


def test_every_repository_real_preregistration_is_valid() -> None:
    for path in sorted(_PREREG_DIR.glob("*.toml")):
        pre = load_real_preregistration(path)
        assert pre.expected_outcome, f"{path.name}: state the expected outcome before confirming"


def test_every_published_real_result_matches_its_preregistration_and_reproduces_its_verdict() -> None:
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        pre = load_real_preregistration(_REPO / result["preregistration"]["path"])
        assert pre.sha256 == result["preregistration"]["sha256"], f"{pre.id} edited after confirmation"
        per = {d: m["per_replicate"] for d, m in result["datasets"].items()}
        assert decide_real(pre, per).as_dict() == result["decision"], path.name
