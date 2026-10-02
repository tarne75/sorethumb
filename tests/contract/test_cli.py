"""CLI contract tests: exit codes, JSON/text output shape, and --help text --
for commands that don't need a full pipeline run, and for commands that do
but where the run is unavoidable setup, not the thing under test.

See tests/integration/test_cli.py for the workflow-correctness tests (does
`run`/`backfill`/`score --from-run` actually do the right thing).
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import sorethumb_ml
from sorethumb_ml.cli import app
from tests.factories.frames import write_grouped_csv as _write_csv
from tests.factories.golden import assert_matches_golden

pytestmark = pytest.mark.contract

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_toml(path: Path, csv_path: Path, workdir: Path, group_by: list[str] | None = None) -> Path:
    """Write a minimal sorethumb.toml to *path*."""
    group_by_line = f"group_by = {json.dumps(group_by or [])}" if group_by else "group_by = []"
    toml = f"""\
[source]
uri = {json.dumps(str(csv_path))}
format = "csv"

[run]
workdir = {json.dumps(str(workdir))}
seed = 0

[columns]
id_column = "id"
{group_by_line}

[profiling]
null_ratio_drop = 0.9

[features]
one_hot_max_cardinality = 20
scaler = "robust"
correlation_threshold = 0.95

[scoring]
contamination = 0.1
combination = "composite"
weighting = "equal"
min_records = 10

[[detectors]]
name = "isolation_forest"
enabled = true

[explain]
enabled = false
top_n = 3
max_rows = 100
"""
    path.write_text(toml, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path):
    """Return (csv_path, toml_path, workdir) for a fresh workspace."""
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=300)
    workdir = tmp_path / "ws"
    toml_path = tmp_path / "sorethumb.toml"
    _write_toml(toml_path, csv_path, workdir)
    return csv_path, toml_path, workdir


def _run_and_get_run_id(toml_path: Path, workdir: Path) -> str:
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    from sorethumb_ml import Workspace

    with Workspace.open(workdir) as ws:
        runs = ws.store.list_runs(limit=1)
    return runs[0]["run_id"]


# ---------------------------------------------------------------------------
# sorethumb --version
# ---------------------------------------------------------------------------


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # Compare with the single source of truth, not a literal that goes stale
    # at the next release: the CLI, the package attribute and the installed
    # distribution metadata must all agree.
    assert "sorethumb" in result.stdout
    assert sorethumb_ml.__version__ in result.stdout
    assert sorethumb_ml.__version__ == importlib.metadata.version("sorethumb-ml")


# ---------------------------------------------------------------------------
# sorethumb init
# ---------------------------------------------------------------------------


def test_init_creates_toml(tmp_path: Path):
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    toml_path = tmp_path / "sorethumb.toml"
    assert toml_path.exists()


def test_init_toml_is_valid_toml(tmp_path: Path):
    runner.invoke(app, ["init", str(tmp_path)])
    with (tmp_path / "sorethumb.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    assert "source" in raw
    assert "run" in raw


def test_init_workdir_matches_the_workspace_it_creates(tmp_path: Path):
    """Init pre-creates a workspace directory (sorethumb-workspace/,
    not the previous hidden .sorethumb_workspace/) -- the starter toml's
    run.workdir must actually point at it, not be left as the generic
    "required — no default" placeholder every other required field gets."""
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0

    with (tmp_path / "sorethumb.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    workdir = raw["run"]["workdir"]
    assert workdir == str(tmp_path / "sorethumb-workspace")
    assert (Path(workdir) / "sorethumb.db").is_file()


def test_init_does_not_overwrite_existing(tmp_path: Path):
    runner.invoke(app, ["init", str(tmp_path)])
    original = (tmp_path / "sorethumb.toml").read_text(encoding="utf-8")
    runner.invoke(app, ["init", str(tmp_path)])
    assert (tmp_path / "sorethumb.toml").read_text(encoding="utf-8") == original


def _flat(result) -> str:
    """Output with Rich's line wrapping collapsed, so phrases can be asserted on."""
    return " ".join(result.output.split())


def _assert_no_success_output(result) -> None:
    out = result.output
    for banner in ("Workspace created", "Config written", "Next steps"):
        assert banner not in out, out


def test_init_reports_a_failing_workspace_as_an_error_and_states_the_partial_result(tmp_path: Path):
    """A real failure, not a mock: a regular file sits where the workspace
    directory must go, so Workspace.init cannot create it (this fails the same way
    for root, unlike a chmod-based unwritable directory)."""
    (tmp_path / "sorethumb-workspace").write_text("in the way", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 1
    _assert_no_success_output(result)
    assert "Workspace initialisation failed" in _flat(result)
    # The config file was still written; the message says so, and names it.
    assert (tmp_path / "sorethumb.toml").is_file()
    assert "WAS written" in _flat(result)
    assert "sorethumb.toml" in _flat(result)
    assert "was not created" in _flat(result)


def test_init_workspace_failure_is_an_error_even_for_unexpected_exception_types(tmp_path: Path, monkeypatch):
    from sorethumb_ml import cli as cli_mod

    def _boom(_path):
        msg = "disk on fire"
        raise RuntimeError(msg)

    monkeypatch.setattr(cli_mod.Workspace, "init", staticmethod(_boom))
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 1
    _assert_no_success_output(result)
    assert "disk on fire" in _flat(result)
    assert "WAS written" in _flat(result)


def test_init_unusable_target_path_fails_before_writing_anything(tmp_path: Path):
    target = tmp_path / "not-a-dir"
    target.write_text("a file, not a directory", encoding="utf-8")
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 1
    _assert_no_success_output(result)
    assert "Nothing was written" in _flat(result)
    assert target.read_text(encoding="utf-8") == "a file, not a directory"


def test_init_unwritable_config_location_fails_without_a_success_banner(tmp_path: Path):
    # A read-only target directory makes the config write fail.
    import os

    if os.geteuid() == 0:  # pragma: no cover - root ignores directory permissions
        return
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result = runner.invoke(app, ["init", str(locked)])
    finally:
        locked.chmod(0o700)
    assert result.exit_code == 1
    _assert_no_success_output(result)
    assert "Nothing was written" in _flat(result)


def test_init_toml_lists_extra_params_commented(tmp_path: Path):
    """The starter file lists every sklearn extra_params key, commented out."""
    from sorethumb_ml.detectors.isolation_forest import IsolationForestDetector

    runner.invoke(app, ["init", str(tmp_path)])
    text = (tmp_path / "sorethumb.toml").read_text(encoding="utf-8")

    assert "# extra_params:" in text
    for key in IsolationForestDetector.available_extra_params():
        assert f"#   {key} " in text, f"{key} not documented in starter toml"
    # Still valid TOML — the extra_params keys are comments, params stays empty.
    raw = tomllib.loads(text)
    assert all(d.get("params", {}) == {} for d in raw["detectors"])


# ---------------------------------------------------------------------------
# TOML serialisation -- _write_minimal_toml must always produce
# syntactically valid TOML that round-trips into an equivalent Config,
# no matter what characters or nested structures the source Config holds.
# ---------------------------------------------------------------------------


def test_write_minimal_toml_escapes_windows_backslash_path(tmp_path: Path):
    from sorethumb_ml.cli import _write_minimal_toml
    from sorethumb_ml.config import Config, DetectorConfig, RunConfig, SourceConfig

    cfg = Config(
        source=SourceConfig(uri=r"C:\Users\alice\data\events.csv"),
        run=RunConfig(workdir=r"C:\Users\alice\workdir"),
        detectors=[DetectorConfig(name="isolation_forest")],
    )
    out_path = tmp_path / "sorethumb.toml"
    _write_minimal_toml(out_path, cfg)
    text = out_path.read_text(encoding="utf-8")

    with out_path.open("rb") as fh:
        raw = tomllib.load(fh)
    assert raw["source"]["uri"] == r"C:\Users\alice\data\events.csv"
    assert raw["run"]["workdir"] == r"C:\Users\alice\workdir"

    reloaded = Config.model_validate(raw)
    assert reloaded.source.uri == cfg.source.uri
    assert reloaded.run.workdir == cfg.run.workdir


def test_write_minimal_toml_escapes_quotes_and_control_characters(tmp_path: Path):
    from sorethumb_ml.cli import _write_minimal_toml
    from sorethumb_ml.config import Config, DetectorConfig, RunConfig, SourceConfig

    tricky = 'a "quoted" path\twith\ta tab\nand a newline'
    cfg = Config(
        source=SourceConfig(uri=f"/data/{tricky}.csv"),
        run=RunConfig(workdir="/tmp/sorethumb-workspace"),
        detectors=[DetectorConfig(name="isolation_forest")],
    )
    out_path = tmp_path / "sorethumb.toml"
    _write_minimal_toml(out_path, cfg)

    with out_path.open("rb") as fh:
        raw = tomllib.load(fh)
    assert raw["source"]["uri"] == cfg.source.uri
    reloaded = Config.model_validate(raw)
    assert reloaded.source.uri == cfg.source.uri


def test_write_minimal_toml_renders_every_builtin_detector(tmp_path: Path):
    """Every registered detector -- not just the three default-starter ones --
    must round-trip through the writer, since a live Config (which this
    function serves, unlike the static starter template) can contain any
    of them."""
    from sorethumb_ml.cli import _write_minimal_toml
    from sorethumb_ml.config import Config, DetectorConfig, RunConfig, SourceConfig
    from sorethumb_ml.detectors import registry

    cfg = Config(
        source=SourceConfig(uri="/data/events.csv"),
        run=RunConfig(workdir="/tmp/sorethumb-workspace"),
        detectors=[DetectorConfig(name=name) for name in registry],
    )
    out_path = tmp_path / "sorethumb.toml"
    _write_minimal_toml(out_path, cfg)

    with out_path.open("rb") as fh:
        raw = tomllib.load(fh)
    assert [d["name"] for d in raw["detectors"]] == list(registry)
    reloaded = Config.model_validate(raw)
    assert [d.name for d in reloaded.detectors] == list(registry)


def test_write_minimal_toml_renders_nested_extra_params(tmp_path: Path):
    """A non-empty nested params.extra_params dict (the exact shape that
    previously -- json.dumps previously emitted invalid TOML inline-table
    syntax for it) must round-trip correctly."""
    from sorethumb_ml.cli import _write_minimal_toml
    from sorethumb_ml.config import Config, DetectorConfig, RunConfig, SourceConfig

    cfg = Config(
        source=SourceConfig(uri="/data/events.csv"),
        run=RunConfig(workdir="/tmp/sorethumb-workspace"),
        detectors=[
            DetectorConfig(
                name="isolation_forest",
                params={"n_estimators": 50, "extra_params": {"n_jobs": 2, "bootstrap": True}},
            )
        ],
    )
    out_path = tmp_path / "sorethumb.toml"
    _write_minimal_toml(out_path, cfg)

    with out_path.open("rb") as fh:
        raw = tomllib.load(fh)
    params = raw["detectors"][0]["params"]
    assert params["n_estimators"] == 50
    assert params["extra_params"] == {"n_jobs": 2, "bootstrap": True}

    reloaded = Config.model_validate(raw)
    assert reloaded.detectors[0].params["extra_params"] == {"n_jobs": 2, "bootstrap": True}


# ---------------------------------------------------------------------------
# sorethumb config check / schema / show
# ---------------------------------------------------------------------------


def test_config_check_valid(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["config", "check", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()


def test_config_check_missing_file(tmp_path: Path):
    result = runner.invoke(app, ["config", "check", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2


def test_config_check_json_output(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["config", "check", "--json", "--config", str(toml_path)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "source" in data
    assert "detectors" in data


def test_config_check_json_redacts_uri_credentials(tmp_path: Path):
    """config check --json must never echo back a credential
    embedded in source.uri -- reuses the same redaction
    _pipeline._redacted_config_json already applies before a run's config
    is persisted to the database, rather than a second, separate rule."""
    toml_path = tmp_path / "sorethumb.toml"
    toml_path.write_text(
        f"""\
[source]
uri = "https://alice:s3cret-pw@example.com/data.csv"
format = "csv"

[run]
workdir = {json.dumps(str(tmp_path / "workdir"))}
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["config", "check", "--json", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "s3cret-pw" not in result.stdout
    data = json.loads(result.stdout)
    assert data["source"]["uri"] == "https://***@example.com/data.csv"


def test_redact_config_does_not_mutate_environment(monkeypatch: pytest.MonkeyPatch):
    """_redact_config previously mutated os.environ in place (setting
    SORETHUMB_TOKEN/SORETHUMB_PASSWORD to the literal string "REDACTED"),
    which would corrupt a real credential for the rest of the process. It
    must now be a pure function."""
    from sorethumb_ml.cli import _redact_config
    from sorethumb_ml.config import Config, RunConfig, SourceConfig

    monkeypatch.setenv("SORETHUMB_TOKEN", "do-not-touch-me")
    cfg = Config(
        source=SourceConfig(uri="https://x/data.csv", auth="bearer", auth_env_var="SORETHUMB_TOKEN"),
        run=RunConfig(workdir="/tmp/sorethumb-workspace"),
    )
    _redact_config(cfg)
    assert os.environ["SORETHUMB_TOKEN"] == "do-not-touch-me"


def test_redact_config_keeps_auth_env_var_name(monkeypatch: pytest.MonkeyPatch):
    """The env var *name* is not a secret (see
    test_auth_token_not_in_config_json) -- only its value is -- so it's
    kept, consistent with what's already persisted for a real run."""
    from sorethumb_ml.cli import _redact_config
    from sorethumb_ml.config import Config, RunConfig, SourceConfig

    monkeypatch.setenv("MY_TOKEN", "secret-value")
    cfg = Config(
        source=SourceConfig(uri="https://x/data.csv", auth="bearer", auth_env_var="MY_TOKEN"),
        run=RunConfig(workdir="/tmp/sorethumb-workspace"),
    )
    redacted = _redact_config(cfg)
    assert redacted["source"]["auth_env_var"] == "MY_TOKEN"
    assert "secret-value" not in json.dumps(redacted)


def test_config_schema_matches_golden(workspace):
    """The full JSON schema, not just a couple of top-level keys -- a renamed
    or dropped field anywhere in the config tree is a breaking change for
    anyone generating config files from this schema."""
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["config", "schema"])
    assert result.exit_code == 0
    assert_matches_golden(result.stdout, "cli_config_schema.json", is_json=True)


def test_config_show_prints_summary(workspace):
    _, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)
    result = runner.invoke(app, ["config", "show", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "isolation_forest" in result.stdout


def test_config_show_json_output(workspace):
    _, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)
    result = runner.invoke(app, ["config", "show", run_id, "--json", "--config", str(toml_path)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "source" in data
    assert "detectors" in data


def test_config_show_output_writes_toml(workspace, tmp_path):
    _, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)
    out_path = tmp_path / "recovered.toml"
    result = runner.invoke(
        app, ["config", "show", run_id, "--output", str(out_path), "--config", str(toml_path)]
    )
    assert result.exit_code == 0
    assert out_path.exists()
    with out_path.open("rb") as fh:
        raw = tomllib.load(fh)
    assert "source" in raw


def test_config_show_unknown_run_exits_not_found(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["config", "show", "nonexistent_run_id", "--config", str(toml_path)])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# sorethumb detectors
# ---------------------------------------------------------------------------


def test_detectors_lists_isolation_forest():
    result = runner.invoke(app, ["detectors"])
    assert result.exit_code == 0
    assert "isolation_forest" in result.stdout


def test_detectors_json_output_matches_golden():
    result = runner.invoke(app, ["detectors", "--json"])
    assert result.exit_code == 0
    assert_matches_golden(result.stdout, "cli_detectors.json", is_json=True)


# ---------------------------------------------------------------------------
# sorethumb inspect
# ---------------------------------------------------------------------------


def test_inspect_prints_feature_plan(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["inspect", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "feature plan" in result.stdout.lower() or "Feature plan" in result.stdout


# ---------------------------------------------------------------------------
# sorethumb run: output shape and input-validation exit codes
# ---------------------------------------------------------------------------


def test_run_invalid_group_filter_fails_immediately(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--group-filter", "[invalid("])
    assert result.exit_code == 2


def test_run_json_output(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "run_id" in data
    assert "groups" in data
    g = data["groups"][0]
    assert isinstance(g["n_flagged"], int)
    assert "n_anomalies" not in g  # the redundant alias was removed before the first release
    assert "n_anomalies" not in data
    assert data["n_flagged"] == sum(grp["n_flagged"] for grp in data["groups"])
    assert isinstance(g["detector_flag_rates"], dict)
    assert set(g["detector_flag_rates"]) == {"isolation_forest"}  # the workspace fixture's detector
    assert all(0.0 <= r <= 1.0 for r in g["detector_flag_rates"].values())
    assert g["dropped_detectors"] == []


def test_run_summary_frames_flagged_count_as_a_review_shortlist(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0
    out = result.stdout
    assert "Flagged for review" in out
    assert "not an estimate of true prevalence" in out
    assert "Realised detector flag rates" in out
    assert "Total anomalies" not in out


def test_run_summary_and_json_surface_warnings_issued():
    """A group-level warning (e.g. ZeroAnomalyWarning) must reach the
    user, both in the printed summary and the JSON payload -- previously
    RunResult.warnings_issued was populated but never displayed anywhere.

    Built directly from RunResult/GroupSummary rather than forcing a real
    warning through a full pipeline run (which would need engineering
    genuine ML-detector disagreement to be deterministic): this pins the
    formatting/serialisation contract these two functions own, the same way
    a to_dict() test would.
    """
    from sorethumb_ml._pipeline import GroupSummary, RunResult
    from sorethumb_ml.cli import _print_run_summary, _run_result_to_dict

    warning_msg = "Three-way intersection flagged zero rows: no row passed all 3 configured detectors."
    group = GroupSummary(
        group_key="gk1",
        group_label="US",
        n_records=10,
        n_anomalies=0,
        anomaly_rate=0.0,
        results_path=None,
        status="success",
        error=None,
        elapsed_seconds=0.1,
        drifted=False,
        refit_reason=None,
        warnings_issued=[warning_msg],
    )
    result = RunResult(
        run_id="run1",
        dataset_uri="file:///x.csv",
        dataset_fp="fp1",
        config_hash="ch1",
        period_label=None,
        workspace_path=Path("/tmp/ws"),
        groups=[group],
        report_path=None,
        started_at="t0",
        finished_at="t1",
        warnings_issued=[warning_msg],
    )

    import io

    from rich.console import Console

    import sorethumb_ml.cli as cli_mod

    buf = Console(file=io.StringIO(), width=200)
    original = cli_mod.console
    cli_mod.console = buf
    try:
        _print_run_summary(result)
    finally:
        cli_mod.console = original
    printed = buf.file.getvalue()
    assert warning_msg in printed

    payload = _run_result_to_dict(result)
    assert payload["warnings_issued"] == [warning_msg]
    assert payload["groups"][0]["warnings_issued"] == [warning_msg]


def test_run_json_flag_counts_use_one_name_and_no_alias():
    """Group and run documents expose ``n_flagged`` only; ``n_anomalies`` is gone."""
    from sorethumb_ml._pipeline import GroupSummary, RunResult
    from sorethumb_ml.cli import _run_result_to_dict

    def _group(key: str, n: int) -> GroupSummary:
        return GroupSummary(
            group_key=key,
            group_label=key,
            n_records=100,
            n_anomalies=n,
            anomaly_rate=n / 100,
            results_path=None,
            status="success",
            error=None,
            elapsed_seconds=0.1,
            drifted=False,
            refit_reason=None,
            warnings_issued=[],
        )

    result = RunResult(
        run_id="run1",
        dataset_uri="file:///x.csv",
        dataset_fp="fp1",
        config_hash="ch1",
        period_label=None,
        workspace_path=Path("/tmp/ws"),
        groups=[_group("a", 3), _group("b", 4)],
        report_path=None,
        started_at="t0",
        finished_at="t1",
    )
    payload = _run_result_to_dict(result)
    assert payload["n_flagged"] == 7
    assert [g["n_flagged"] for g in payload["groups"]] == [3, 4]
    assert "n_anomalies" not in payload
    assert all("n_anomalies" not in g for g in payload["groups"])


# ---------------------------------------------------------------------------
# sorethumb runs / show
# ---------------------------------------------------------------------------


def test_runs_lists_after_run(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["runs", "--config", str(toml_path)])
    assert result.exit_code == 0


def test_runs_json_output(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["runs", "--config", str(toml_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)


def test_show_prints_run_detail(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])

    from sorethumb_ml import Workspace

    with Workspace.open(workdir) as ws:
        runs = ws.store.list_runs(limit=1)
    run_id = runs[0]["run_id"]

    result = runner.invoke(app, ["show", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0
    assert run_id in result.stdout


def test_show_unknown_run_exits_not_found(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["show", "run_nonexistent", "--config", str(toml_path)])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# sorethumb workspace commands (exit-code contract only; see integration for
# workspace reset, which actually verifies the workspace is gone)
# ---------------------------------------------------------------------------


def test_workspace_ls(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["workspace", "ls", "--config", str(toml_path)])
    assert result.exit_code == 0


def test_workspace_du(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["workspace", "du", "--config", str(toml_path)])
    assert result.exit_code == 0


def test_workspace_prune_dry_run(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(
        app, ["workspace", "prune", "--config", str(toml_path), "--dry-run", "--days", "0"]
    )
    assert result.exit_code == 0


def test_workspace_prune_rejects_negative_days(workspace):
    """--days -1 must fail loudly with a clean exit code, not silently
    prune every artifact in the workspace."""
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["workspace", "prune", "--config", str(toml_path), "--days", "-1"])
    assert result.exit_code == 2
    assert "retention_days" in result.output


def test_workspace_vacuum(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["workspace", "vacuum", "--config", str(toml_path)])
    assert result.exit_code == 0


def test_workspace_migrate(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["workspace", "migrate", "--config", str(toml_path)])
    assert result.exit_code == 0


def test_workspace_migrate_dry_run(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["workspace", "migrate", "--config", str(toml_path), "--dry-run"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# sorethumb explain-plan
# ---------------------------------------------------------------------------


def test_explain_plan_prints_table(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["explain-plan", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "feature" in result.stdout.lower() or "plan" in result.stdout.lower()


def test_explain_plan_json_output(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["explain-plan", "--config", str(toml_path), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "decisions" in data or "output_features" in data


def _explain_plan_json(toml_path: Path, *args: str) -> dict:
    result = runner.invoke(app, ["explain-plan", *args, "--config", str(toml_path), "--json"])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_explain_plan_with_run_id_loads_persisted_plan_not_current_data(workspace):
    """A run ID selects the plan the run was fitted with; no run ID plans the
    current data. After the source data changes the two must differ."""
    import polars as pl

    csv_path, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)

    historical_before = _explain_plan_json(toml_path, run_id)
    current_before = _explain_plan_json(toml_path)
    assert "value_a" in historical_before["output_features"]
    assert current_before["output_features"] == historical_before["output_features"]

    # value_a becomes entirely null -> a fresh plan now drops it (null_ratio_drop = 0.9).
    df = pl.read_csv(csv_path).with_columns(pl.lit(None, dtype=pl.Float64).alias("value_a"))
    df.write_csv(str(csv_path))

    historical_after = _explain_plan_json(toml_path, run_id)
    current_after = _explain_plan_json(toml_path)

    assert historical_after == historical_before  # persisted plan is immutable
    assert "value_a" not in current_after["output_features"]
    assert current_after["output_features"] != historical_after["output_features"]


def test_explain_plan_with_run_id_does_not_read_source_data(workspace):
    csv_path, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)
    csv_path.unlink()
    result = runner.invoke(app, ["explain-plan", run_id, "--config", str(toml_path), "--json"])
    assert result.exit_code == 0, result.stdout
    assert "output_features" in json.loads(result.stdout)


def test_explain_plan_unknown_run_id_is_not_found(workspace):
    _, toml_path, workdir = workspace
    _run_and_get_run_id(toml_path, workdir)
    result = runner.invoke(app, ["explain-plan", "no-such-run", "--config", str(toml_path), "--json"])
    assert result.exit_code == 3
    assert "Run not found: no-such-run" in json.loads(result.stdout)["error"]

    human = runner.invoke(app, ["explain-plan", "no-such-run", "--config", str(toml_path)])
    assert human.exit_code == 3


def test_explain_plan_run_without_persisted_plan_is_clear_error(workspace):
    _, toml_path, workdir = workspace
    run_id = _run_and_get_run_id(toml_path, workdir)
    from sorethumb_ml import Workspace

    with Workspace.open(workdir) as ws:
        (ws.run_dir(run_id) / "plan.json").unlink()
    result = runner.invoke(app, ["explain-plan", run_id, "--config", str(toml_path), "--json"])
    assert result.exit_code == 3
    assert "No persisted FeaturePlan" in json.loads(result.stdout)["error"]


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_exit_0_on_success(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0


def test_exit_2_on_config_error():
    result = runner.invoke(app, ["run", "--config", "nonexistent_config.toml", "--no-report"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Import boundary: CLI must not import private library modules
# ---------------------------------------------------------------------------


def test_cli_only_imports_public_api():
    """Parse cli.py AST and assert no private sorethumb imports."""
    cli_path = Path(__file__).parent.parent.parent / "src" / "sorethumb_ml" / "cli.py"
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Collect all "from sorethumb_ml.X" and "import sorethumb_ml.X" statements
    private_imports: list[str] = []
    public_modules = {
        "sorethumb_ml",  # top-level package (allowed for __version__)
    }
    # Sub-imports inside the CLI body that are inside TYPE_CHECKING blocks
    # are allowed (they never execute at runtime).
    # We only flag module-level imports here.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("sorethumb_ml.") and module not in public_modules:
                # Allow only the explicit re-export through the public API
                private_imports.append(module)

    # The only allowed sorethumb_ml sub-import at the top level is sorethumb_ml
    # itself (via `import sorethumb_ml` for __version__). All library
    # sub-modules must be accessed through the public re-exports, OR inside
    # functions (PLC0415).
    # Filter out any that are inside function bodies (they're runtime-guarded).
    top_level_imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("sorethumb_ml.") and not module.startswith("sorethumb_ml.cli"):
                top_level_imports.append(module)

    # These sub-module imports are the public API surface (re-exported from __init__)
    allowed_sub_modules = {
        "sorethumb_ml",
    }
    forbidden = [m for m in top_level_imports if m not in allowed_sub_modules]
    assert forbidden == [], (
        f"CLI imports private library modules at top level: {forbidden}\n"
        "The CLI must import only from 'sorethumb' (the public API)."
    )


# ---------------------------------------------------------------------------
# score --from-run help must not imply sandboxing or safe loading of
# third-party workspaces
# ---------------------------------------------------------------------------


def test_score_help_documents_pickle_trust_boundary():
    result = runner.invoke(app, ["score", "--help"])
    assert result.exit_code == 0
    help_text = result.output.lower()
    assert "unpickle" in help_text or "pickle" in help_text
    assert "not sandboxed" in help_text or "no sandboxing" in help_text
    assert "trust" in help_text


def test_score_help_does_not_overstate_digest_safety():
    result = runner.invoke(app, ["score", "--help"])
    assert result.exit_code == 0
    help_text = result.output.lower()
    # A digest match must never be presented as proof a workspace is safe.
    forbidden_claims = [
        "digests ensure",
        "digests guarantee",
        "safe to load",
        "safely load",
        "sandboxed environment",
    ]
    for claim in forbidden_claims:
        assert claim not in help_text, f"score --help overstates safety with: {claim!r}"


# ---------------------------------------------------------------------------
# JSON mode: no prompts, no stray notices, exactly one JSON document on both
# success and error paths, and a closed stdin never hangs.
#
# These spawn a real child process with stdin closed (subprocess.DEVNULL),
# not CliRunner: CliRunner's fake stdin is an in-memory, in-process stream,
# and these are specifically stdin-closed / whole-process-output contracts —
# the outcome a real automation pipeline piping ``sorethumb ... --json``
# actually depends on, against the real entry point, not an in-process call.
# ---------------------------------------------------------------------------


def _run_json_subprocess(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run the CLI as a real child process with stdin closed.

    ``python -c "from sorethumb_ml.cli import app; app()"`` stands in for the
    installed ``sorethumb`` console script (identical callable, no PATH
    dependency), invoked as a genuine subprocess so a real, closed stdin file
    descriptor is exercised -- not an in-memory stream. ``timeout`` makes a
    regression to "prompts and blocks forever" fail the test instead of
    hanging the suite.
    """
    return subprocess.run(
        [sys.executable, "-c", "from sorethumb_ml.cli import app; app()", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def test_run_json_first_run_does_not_prompt_with_closed_stdin(tmp_path: Path):
    """The exact scenario that used to hang: --json, no config file yet, a
    bare data-file argument (which triggers the "save settings?" prompt),
    and closed stdin. Must run to completion with exactly one JSON document
    on stdout, never asking to save a config."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, n_rows=200)
    workdir = tmp_path / "ws"

    proc = _run_json_subprocess(["run", str(csv_path), "--workdir", str(workdir), "--no-report", "--json"])

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)  # fails outright if anything but pure JSON is on stdout
    assert "run_id" in data
    assert not (tmp_path / "sorethumb.toml").exists()  # never asked, never wrote one
    assert "Save settings" not in proc.stdout
    assert "Save settings" not in proc.stderr


def test_run_json_group_filter_error_emits_single_json_document(workspace):
    _, toml_path, _ = workspace
    proc = _run_json_subprocess(["run", "--config", str(toml_path), "--group-filter", "[invalid(", "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert "error" in data
    assert "group-filter" in data["error"].lower()


def test_config_check_json_missing_file_emits_single_json_document(tmp_path: Path):
    missing = tmp_path / "does_not_exist.toml"
    proc = _run_json_subprocess(["config", "check", "--config", str(missing), "--json"])
    assert proc.returncode == 2
    data = json.loads(proc.stdout)
    assert "error" in data
    assert str(missing) in data["error"]


def test_runs_json_missing_workspace_emits_single_json_document(tmp_path: Path):
    """Previously an unhandled StoreError crashed with a traceback instead of
    the JSON caller's expected single document."""
    toml_path = tmp_path / "sorethumb.toml"
    _write_toml(toml_path, tmp_path / "unused.csv", tmp_path / "never-created-ws")
    proc = _run_json_subprocess(["runs", "--config", str(toml_path), "--json"])
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert "error" in data


def test_show_json_run_not_found_emits_single_json_document(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    proc = _run_json_subprocess(["show", "does-not-exist", "--config", str(toml_path), "--json"])
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert "error" in data
    assert "does-not-exist" in data["error"]


def test_anomalies_json_zero_anomalies_emits_empty_json_array(workspace):
    """A run with no flagged rows must still emit valid JSON in --json mode
    -- previously this printed a plain sentence instead, so a --json
    caller had to special-case "nothing found" as a JSON parse failure.

    Emptying the persisted results directly (rather than engineering genuine
    zero-anomaly detector agreement, which isn't reliably deterministic --
    see test_run_summary_and_json_surface_warnings_issued's docstring for
    the same reasoning) isolates this command's own output-shape contract.
    """
    import polars as pl

    from sorethumb_ml import Workspace

    _, toml_path, workdir = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0

    with Workspace.open(workdir) as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        for g in ws.store.all_run_groups(run_id):
            parquet = ws.results_dir(run_id, g["group_key"]) / "anomalies.parquet"
            pl.DataFrame({"rank": []}, schema={"rank": pl.Int64}).write_parquet(parquet)

    proc = _run_json_subprocess(["anomalies", run_id, "--config", str(toml_path), "--json"])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_anomalies_json_run_not_found_emits_single_json_document(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    proc = _run_json_subprocess(["anomalies", "does-not-exist", "--config", str(toml_path), "--json"])
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert "error" in data


def test_config_show_json_run_not_found_emits_single_json_document(workspace):
    _, toml_path, _ = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    proc = _run_json_subprocess(["config", "show", "does-not-exist", "--config", str(toml_path), "--json"])
    assert proc.returncode == 3
    data = json.loads(proc.stdout)
    assert "error" in data
