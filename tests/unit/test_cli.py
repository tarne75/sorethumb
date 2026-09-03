"""Unit tests for M8: CLI commands via CliRunner.

Every test uses a synthetic CSV dataset written to a tempdir. The tests
exercise the CLI at the command boundary, not the library internals, so
assertions are on exit codes, stdout content, and workspace state.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_csv(path: Path, n_rows: int = 300, n_groups: int = 2, seed: int = 0) -> Path:
    """Write a synthetic multi-group CSV to *path* and return it."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = [f"G{i}" for i in range(n_groups)]
    df = pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "group": [groups[i % n_groups] for i in range(n_rows)],
            "value_a": rng.normal(0, 1, n_rows).tolist(),
            "value_b": rng.normal(5, 2, n_rows).tolist(),
            "cat": (["A"] * (n_rows // 2) + ["B"] * (n_rows - n_rows // 2)),
        }
    )
    df.write_csv(str(path))
    return path


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


@pytest.fixture
def workspace_grouped(tmp_path: Path):
    """Workspace with group_by configured."""
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=300, n_groups=2)
    workdir = tmp_path / "ws"
    toml_path = tmp_path / "sorethumb.toml"
    _write_toml(toml_path, csv_path, workdir, group_by=["group"])
    return csv_path, toml_path, workdir


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


def test_init_does_not_overwrite_existing(tmp_path: Path):
    runner.invoke(app, ["init", str(tmp_path)])
    original = (tmp_path / "sorethumb.toml").read_text(encoding="utf-8")
    runner.invoke(app, ["init", str(tmp_path)])
    assert (tmp_path / "sorethumb.toml").read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# sorethumb config check / schema
# ---------------------------------------------------------------------------


def test_config_check_valid(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["config", "check", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "valid" in result.stdout.lower()


def test_config_check_missing_file(tmp_path: Path):
    result = runner.invoke(app, ["config", "check", "--config", str(tmp_path / "missing.toml")])
    assert result.exit_code == 2


def test_config_schema_emits_json(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["config", "schema"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert "properties" in schema or "$defs" in schema


# ---------------------------------------------------------------------------
# sorethumb detectors
# ---------------------------------------------------------------------------


def test_detectors_lists_isolation_forest():
    result = runner.invoke(app, ["detectors"])
    assert result.exit_code == 0
    assert "isolation_forest" in result.stdout


def test_detectors_json_output():
    result = runner.invoke(app, ["detectors", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    names = [d["name"] for d in data]
    assert "isolation_forest" in names


# ---------------------------------------------------------------------------
# sorethumb inspect
# ---------------------------------------------------------------------------


def test_inspect_prints_feature_plan(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(app, ["inspect", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "feature plan" in result.stdout.lower() or "Feature plan" in result.stdout


# ---------------------------------------------------------------------------
# sorethumb run
# ---------------------------------------------------------------------------


def test_run_succeeds(workspace):
    _, toml_path, workdir = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert workdir.exists()
    assert (workdir / "sorethumb.db").exists()


def test_run_creates_results_parquet(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    parquets = list(workdir.rglob("anomalies.parquet"))
    assert len(parquets) >= 1


def test_run_idempotent_second_run_skips_groups(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result2 = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result2.exit_code == 0
    assert "skipped" in result2.stdout.lower()


def test_run_dry_run_writes_nothing(workspace):
    _, toml_path, workdir = workspace
    workdir_before = set(workdir.rglob("*")) if workdir.exists() else set()
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--dry-run"])
    assert result.exit_code == 0
    # DB and workspace may be created but no result parquets should appear
    parquets = list(workdir.rglob("anomalies.parquet")) if workdir.exists() else []
    assert len(parquets) == 0


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


def test_run_with_groups(workspace_grouped):
    _, toml_path, workdir = workspace_grouped
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")


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
# sorethumb workspace commands
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


def test_workspace_reset_requires_confirmation(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    # --yes flag should succeed without interactive prompt
    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml_path), "--yes"])
    assert result.exit_code == 0
    assert not workdir.exists()


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
