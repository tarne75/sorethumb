"""The CLI's documented exit codes and machine-readable error shape (P1-6).

0 success | 1 runtime | 2 pre-flight | 3 not found | 4 partial success.
Every --json command emits the same failure document -- exactly
``{"error": str, "kind": str, "exit_code": int}`` -- for every failure class, and the
document's ``exit_code`` is the process exit code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import sorethumb_ml.cli as cli_mod
from sorethumb_ml import Workspace
from sorethumb_ml._pipeline import GroupSummary, RunResult
from sorethumb_ml.cli import ExitCode, app
from sorethumb_ml.errors import (
    ConfigError,
    DetectorError,
    NotFoundError,
    PlanError,
    SchemaError,
    SorethumbError,
    SourceError,
    StoreError,
)
from tests.contract.test_cli import _run_and_get_run_id, _write_toml
from tests.factories.frames import write_grouped_csv as _write_csv

pytestmark = pytest.mark.contract

runner = CliRunner()
_ERROR_KEYS = {"error", "kind", "exit_code"}


@pytest.fixture
def ws(tmp_path: Path) -> tuple[Path, Path]:
    """(toml_path, workdir) for a workspace that already holds one completed run."""
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=300)
    workdir = tmp_path / "ws"
    toml_path = _write_toml(tmp_path / "sorethumb.toml", csv_path, workdir)
    _run_and_get_run_id(toml_path, workdir)
    return toml_path, workdir


def _assert_error_doc(result: Any, code: ExitCode) -> dict[str, Any]:
    assert result.exit_code == code, result.output
    doc = json.loads(result.stdout)  # exactly one parseable document on stdout
    assert set(doc) == _ERROR_KEYS, doc
    assert isinstance(doc["error"], str)
    assert doc["error"]
    assert doc["kind"] == code.kind
    assert doc["exit_code"] == int(code)
    return doc


# ---------------------------------------------------------------------------
# The taxonomy itself
# ---------------------------------------------------------------------------


def test_exit_code_values_are_the_documented_stable_numbers() -> None:
    assert {c.name: int(c) for c in ExitCode} == {
        "OK": 0,
        "RUNTIME": 1,
        "PREFLIGHT": 2,
        "NOT_FOUND": 3,
        "PARTIAL": 4,
    }


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConfigError("x"), ExitCode.PREFLIGHT),
        (SchemaError("x"), ExitCode.PREFLIGHT),
        (SourceError("x"), ExitCode.PREFLIGHT),
        (PlanError("x"), ExitCode.PREFLIGHT),
        (NotFoundError("x"), ExitCode.NOT_FOUND),
        (StoreError("x"), ExitCode.RUNTIME),
        (DetectorError("x"), ExitCode.RUNTIME),
        (SorethumbError("x"), ExitCode.RUNTIME),
    ],
)
def test_errors_classify_by_their_failure_kind(exc: SorethumbError, expected: ExitCode) -> None:
    assert cli_mod._classify_error(exc) is expected


def test_not_found_is_still_a_store_error() -> None:
    assert issubclass(NotFoundError, StoreError)


def test_a_third_party_error_subclass_defaults_to_runtime() -> None:
    class PluginError(SorethumbError):
        pass

    assert cli_mod._classify_error(PluginError("x")) is ExitCode.RUNTIME


# ---------------------------------------------------------------------------
# Pre-flight (2): no work attempted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["score", "--from-run", "r"],
        ["runs"],
        ["show", "r"],
        ["anomalies"],
        ["explain-plan"],
        ["config", "check"],
        ["config", "show", "r"],
        ["workspace", "ls"],
    ],
)
def test_missing_config_is_preflight_for_every_json_command(tmp_path: Path, argv: list[str]) -> None:
    result = runner.invoke(app, [*argv, "--config", str(tmp_path / "nope.toml"), "--json"])
    doc = _assert_error_doc(result, ExitCode.PREFLIGHT)
    assert "Config file not found" in doc["error"]


def test_invalid_group_filter_regex_is_preflight(ws: tuple[Path, Path]) -> None:
    toml_path, _ = ws
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--group-filter", "(", "--json"])
    _assert_error_doc(result, ExitCode.PREFLIGHT)


def test_group_selector_matching_nothing_is_preflight(ws: tuple[Path, Path]) -> None:
    toml_path, _ = ws
    result = runner.invoke(
        app, ["run", "--config", str(toml_path), "--only-group", "no-such-group", "--force", "--json"]
    )
    assert result.exit_code == ExitCode.PREFLIGHT
    doc = json.loads(result.stdout)
    assert doc["outcome"] == "preflight"
    assert doc["exit_code"] == 2
    assert doc["group_selection_error"]


def test_source_rejected_before_any_work_is_preflight(ws: tuple[Path, Path]) -> None:
    toml_path, _ = ws
    (toml_path.parent / "data" / "test.csv").unlink()
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--force", "--json"])
    doc = _assert_error_doc(result, ExitCode.PREFLIGHT)
    assert doc["error"].startswith("run failed:")


# ---------------------------------------------------------------------------
# Not found (3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "run_nope"],
        ["anomalies", "run_nope"],
        ["explain-plan", "run_nope"],
        ["config", "show", "run_nope"],
        ["score", "--from-run", "run_nope"],
    ],
)
def test_unknown_run_is_not_found_for_every_json_command(ws: tuple[Path, Path], argv: list[str]) -> None:
    toml_path, _ = ws
    result = runner.invoke(app, [*argv, "--config", str(toml_path), "--json"])
    doc = _assert_error_doc(result, ExitCode.NOT_FOUND)
    assert "run_nope" in doc["error"]


@pytest.mark.parametrize(
    "argv",
    [
        ["runs"],
        ["show", "r"],
        ["anomalies"],
        ["explain-plan", "r"],
        ["config", "show", "r"],
        ["workspace", "ls"],
    ],
)
def test_missing_workspace_is_not_found(tmp_path: Path, argv: list[str]) -> None:
    csv_path = tmp_path / "d.csv"
    _write_csv(csv_path, n_rows=50)
    toml_path = _write_toml(tmp_path / "sorethumb.toml", csv_path, tmp_path / "never-created")
    result = runner.invoke(app, [*argv, "--config", str(toml_path), "--json"])
    _assert_error_doc(result, ExitCode.NOT_FOUND)


def test_run_without_a_persisted_plan_is_not_found(ws: tuple[Path, Path]) -> None:
    toml_path, workdir = ws
    with Workspace.open(workdir) as w:
        run_id = str(w.store.list_runs(limit=1)[0]["run_id"])
        (w.run_dir(run_id) / "plan.json").unlink()
    result = runner.invoke(app, ["explain-plan", run_id, "--config", str(toml_path), "--json"])
    doc = _assert_error_doc(result, ExitCode.NOT_FOUND)
    assert "No persisted FeaturePlan" in doc["error"]


def test_report_for_unknown_run_is_not_found(ws: tuple[Path, Path]) -> None:
    toml_path, _ = ws
    result = runner.invoke(app, ["report", "run_nope", "--config", str(toml_path)])
    assert result.exit_code == ExitCode.NOT_FOUND


# ---------------------------------------------------------------------------
# Runtime (1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["runs"],
        ["show", "r"],
        ["anomalies"],
        ["explain-plan", "r"],
        ["config", "show", "r"],
        ["workspace", "ls"],
    ],
)
def test_store_failure_is_runtime_for_every_workspace_command(
    ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    toml_path, _ = ws

    def _boom(_path: object) -> object:
        raise StoreError("database is locked")

    monkeypatch.setattr(cli_mod.Workspace, "open", staticmethod(_boom))
    result = runner.invoke(app, [*argv, "--config", str(toml_path), "--json"])
    doc = _assert_error_doc(result, ExitCode.RUNTIME)
    assert "database is locked" in doc["error"]


def test_run_that_raises_a_runtime_project_error_is_runtime(
    ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path, _ = ws

    def _boom(*_a: object, **_k: object) -> object:
        raise DetectorError("fit exploded")

    monkeypatch.setattr(cli_mod, "run_detection", _boom)
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--json"])
    doc = _assert_error_doc(result, ExitCode.RUNTIME)
    assert "fit exploded" in doc["error"]


def _result_with(statuses: list[str], *, report_status: str = "skipped") -> RunResult:
    groups = [
        GroupSummary(
            group_key=f"g{i}",
            group_label=f"g{i}",
            n_records=10,
            n_anomalies=0,
            anomaly_rate=None,
            results_path=None,
            status=status,
            error="injected" if status == "failed" else None,
            elapsed_seconds=0.1,
            drifted=False,
            refit_reason=None,
            warnings_issued=[],
        )
        for i, status in enumerate(statuses)
    ]
    return RunResult(
        run_id="fake",
        dataset_uri="u",
        dataset_fp="fp",
        config_hash="h",
        period_label=None,
        workspace_path=Path(),
        groups=groups,
        report_path=None,
        started_at="t0",
        finished_at="t1",
        report_status=report_status,
    )


def test_every_group_failing_is_runtime(ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path, _ = ws
    monkeypatch.setattr(cli_mod, "run_detection", lambda *_a, **_k: _result_with(["failed", "failed"]))
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--json"])
    assert result.exit_code == ExitCode.RUNTIME
    doc = json.loads(result.stdout)
    assert (doc["exit_code"], doc["outcome"]) == (1, "runtime")


# ---------------------------------------------------------------------------
# Partial success (4)
# ---------------------------------------------------------------------------


def test_some_groups_failing_is_partial_success(
    ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path, _ = ws
    monkeypatch.setattr(cli_mod, "run_detection", lambda *_a, **_k: _result_with(["success", "failed"]))
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--json"])
    assert result.exit_code == ExitCode.PARTIAL
    doc = json.loads(result.stdout)
    assert (doc["exit_code"], doc["outcome"]) == (4, "partial")
    assert doc["n_succeeded"] == 1
    assert doc["n_failed"] == 1


def test_report_failure_after_successful_detection_is_partial_success(
    ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path, _ = ws
    monkeypatch.setattr(
        cli_mod, "run_detection", lambda *_a, **_k: _result_with(["success"], report_status="failed")
    )
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--json"])
    assert result.exit_code == ExitCode.PARTIAL
    assert json.loads(result.stdout)["outcome"] == "partial"


def test_score_forward_partial_success_uses_the_same_code(
    ws: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path, _ = ws
    monkeypatch.setattr(cli_mod, "score_forward", lambda *_a, **_k: _result_with(["success", "failed"]))
    result = runner.invoke(app, ["score", "--from-run", "r", "--config", str(toml_path), "--json"])
    assert result.exit_code == ExitCode.PARTIAL
    assert json.loads(result.stdout)["exit_code"] == 4


def test_successful_run_document_reports_ok(ws: tuple[Path, Path]) -> None:
    toml_path, _ = ws
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--force", "--no-report", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert (doc["exit_code"], doc["outcome"]) == (0, "ok")


# ---------------------------------------------------------------------------
# Documentation is tied to the enum
# ---------------------------------------------------------------------------


def test_cli_reference_exit_code_table_matches_the_enum() -> None:
    text = (Path(__file__).resolve().parents[2] / "docs" / "cli_reference.md").read_text(encoding="utf-8")
    section = text.split("## Exit codes", 1)[1].split("\n## ", 1)[0]
    documented = {int(m.group(1)) for m in re.finditer(r"^\| `(\d+)` \|", section, flags=re.MULTILINE)}
    assert documented == {int(c) for c in ExitCode}
    for code in ExitCode:
        if code is ExitCode.OK:
            continue
        assert f"`{code.kind}`" in section, f"docs must name the JSON kind for exit code {int(code)}"
