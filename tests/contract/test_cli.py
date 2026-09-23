"""CLI contract tests: exit codes, JSON/text output shape, and --help text --
for commands that don't need a full pipeline run, and for commands that do
but where the run is unavoidable setup, not the thing under test.

See tests/integration/test_cli.py for the workflow-correctness tests (does
`run`/`backfill`/`score --from-run` actually do the right thing).
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb.cli import app
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
    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        runs = ws.store.list_runs(limit=1)
    return runs[0]["run_id"]


# ---------------------------------------------------------------------------
# sorethumb --version
# ---------------------------------------------------------------------------


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "sorethumb" in result.stdout
    assert "0.1" in result.stdout


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
    """P3-5: init pre-creates a workspace directory (sorethumb-workspace/,
    not the pre-P3-5 hidden .sorethumb_workspace/) -- the starter toml's
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


def test_init_toml_lists_extra_params_commented(tmp_path: Path):
    """The starter file lists every sklearn extra_params key, commented out."""
    from sorethumb.detectors.isolation_forest import IsolationForestDetector

    runner.invoke(app, ["init", str(tmp_path)])
    text = (tmp_path / "sorethumb.toml").read_text(encoding="utf-8")

    assert "# extra_params:" in text
    for key in IsolationForestDetector.available_extra_params():
        assert f"#   {key} " in text, f"{key} not documented in starter toml"
    # Still valid TOML — the extra_params keys are comments, params stays empty.
    raw = tomllib.loads(text)
    assert all(d.get("params", {}) == {} for d in raw["detectors"])


# ---------------------------------------------------------------------------
# TOML serialisation (P0-4) -- _write_minimal_toml must always produce
# syntactically valid TOML that round-trips into an equivalent Config,
# no matter what characters or nested structures the source Config holds.
# ---------------------------------------------------------------------------


def test_write_minimal_toml_escapes_windows_backslash_path(tmp_path: Path):
    from sorethumb.cli import _write_minimal_toml
    from sorethumb.config import Config, DetectorConfig, RunConfig, SourceConfig

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
    from sorethumb.cli import _write_minimal_toml
    from sorethumb.config import Config, DetectorConfig, RunConfig, SourceConfig

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
    from sorethumb.cli import _write_minimal_toml
    from sorethumb.config import Config, DetectorConfig, RunConfig, SourceConfig
    from sorethumb.detectors import registry

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
    """A non-empty nested params.extra_params dict (the exact shape P0-4
    called out -- json.dumps previously emitted invalid TOML inline-table
    syntax for it) must round-trip correctly."""
    from sorethumb.cli import _write_minimal_toml
    from sorethumb.config import Config, DetectorConfig, RunConfig, SourceConfig

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


def test_config_show_unknown_run_exits_1(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["config", "show", "nonexistent_run_id", "--config", str(toml_path)])
    assert result.exit_code == 1


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
    assert g["n_flagged"] == g["n_anomalies"]  # review-shortlist size, renamed for honesty
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
    """P2-4: a group-level warning (e.g. ZeroAnomalyWarning) must reach the
    user, both in the printed summary and the JSON payload -- previously
    RunResult.warnings_issued was populated but never displayed anywhere.

    Built directly from RunResult/GroupSummary rather than forcing a real
    warning through a full pipeline run (which would need engineering
    genuine ML-detector disagreement to be deterministic): this pins the
    formatting/serialisation contract these two functions own, the same way
    a to_dict() test would.
    """
    from sorethumb._pipeline import GroupSummary, RunResult
    from sorethumb.cli import _print_run_summary, _run_result_to_dict

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

    import sorethumb.cli as cli_mod

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

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        runs = ws.store.list_runs(limit=1)
    run_id = runs[0]["run_id"]

    result = runner.invoke(app, ["show", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0
    assert run_id in result.stdout


def test_show_unknown_run_exits_1(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["show", "run_nonexistent", "--config", str(toml_path)])
    assert result.exit_code == 1


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
    """P2-6: --days -1 must fail loudly with a clean exit code, not silently
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
    cli_path = Path(__file__).parent.parent.parent / "src" / "sorethumb" / "cli.py"
    source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Collect all "from sorethumb.X" and "import sorethumb.X" statements
    private_imports: list[str] = []
    public_modules = {
        "sorethumb",  # top-level package (allowed for __version__)
    }
    # Sub-imports inside the CLI body that are inside TYPE_CHECKING blocks
    # are allowed (they never execute at runtime).
    # We only flag module-level imports here.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("sorethumb.") and module not in public_modules:
                # Allow only the explicit re-export through the public API
                private_imports.append(module)

    # The only allowed sorethumb sub-import at the top level is sorethumb itself
    # (via `import sorethumb` for __version__). All library sub-modules must
    # be accessed through the public re-exports, OR inside functions (PLC0415).
    # Filter out any that are inside function bodies (they're runtime-guarded).
    top_level_imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("sorethumb.") and not module.startswith("sorethumb.cli"):
                top_level_imports.append(module)

    # These sub-module imports are the public API surface (re-exported from __init__)
    allowed_sub_modules = {
        "sorethumb",
    }
    forbidden = [m for m in top_level_imports if m not in allowed_sub_modules]
    assert forbidden == [], (
        f"CLI imports private library modules at top level: {forbidden}\n"
        "The CLI must import only from 'sorethumb' (the public API)."
    )


# ---------------------------------------------------------------------------
# score --from-run help must not imply sandboxing or safe loading of
# third-party workspaces (P0-8)
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
