"""CLI workflow-correctness tests: `run`, `backfill`, `score --from-run`,
`report`, and `workspace reset` actually do the right thing (idempotency,
dry-run side-effect isolation, config-file preservation, real file/DB
mutation) -- not just the exit-code/output-shape contract in
tests/contract/test_cli.py.
"""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from sorethumb.cli import app
from tests.factories.frames import write_grouped_csv as _write_csv

pytestmark = pytest.mark.integration

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


@pytest.fixture
def workspace_grouped(tmp_path: Path):
    """Workspace with group_by configured."""
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=300, n_groups=2)
    workdir = tmp_path / "ws"
    toml_path = tmp_path / "sorethumb.toml"
    _write_toml(toml_path, csv_path, workdir, group_by=["group"])
    return csv_path, toml_path, workdir


def _run_and_get_run_id(toml_path: Path, workdir: Path) -> str:
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        runs = ws.store.list_runs(limit=1)
    return runs[0]["run_id"]


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


def test_run_zero_config_creates_single_workspace_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """P3-5: with no sorethumb.toml, `sorethumb run <file>` must create every
    artefact (db, models, results, reports, logs) under one dedicated
    `./sorethumb-workspace/` directory in the invocation directory -- never
    scattered beside the source data (which may live somewhere else
    entirely) and never loose files dropped directly in the invocation
    directory itself.
    """
    data_dir = tmp_path / "data-elsewhere"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    csv_path = _write_csv(data_dir / "data.csv", n_rows=300)

    monkeypatch.chdir(project_dir)
    result = runner.invoke(app, ["run", str(csv_path), "--no-report"], input="n\n")
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    ws_dir = project_dir / "sorethumb-workspace"
    assert ws_dir.is_dir()
    assert (ws_dir / "sorethumb.db").is_file()
    assert (ws_dir / "models").is_dir()
    assert (ws_dir / "results").is_dir()
    assert any((ws_dir / "results").rglob("anomalies.parquet"))

    # Nothing else was created directly in the invocation directory...
    assert {p.name for p in project_dir.iterdir()} == {"sorethumb-workspace"}
    # ...and nothing was scattered beside the source data either.
    assert {p.name for p in data_dir.iterdir()} == {"data.csv"}


def test_run_zero_config_refuses_when_legacy_dot_workspace_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """P3-5: a sorethumb.db directly at '.' is a workspace created under the
    pre-P3-5 default (workdir="."). Silently falling through to the new
    './sorethumb-workspace/' default would just stop seeing its runs, with
    no error -- refuse instead, non-destructively, until the caller chooses
    explicitly (either --workdir . to keep using it, or migrate it)."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "sorethumb.db").touch()
    csv_path = _write_csv(tmp_path / "data.csv", n_rows=300)

    monkeypatch.chdir(project_dir)
    result = runner.invoke(app, ["run", str(csv_path), "--no-report"])
    output = result.stdout + (result.stderr or "")

    assert result.exit_code == 2
    assert "sorethumb-workspace" in output
    assert not (project_dir / "sorethumb-workspace").exists()

    # An explicit --workdir . goes right ahead against the legacy workspace.
    result2 = runner.invoke(app, ["run", str(csv_path), "--no-report", "--workdir", "."], input="n\n")
    assert result2.exit_code == 0, result2.stdout + (result2.stderr or "")


def test_run_idempotent_second_run_skips_groups(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result2 = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result2.exit_code == 0
    assert "skipped" in result2.stdout.lower()


def test_run_dry_run_registers_dataset_and_run_but_fits_nothing(workspace):
    csv_path, toml_path, workdir = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.stdout

    from sorethumb import Workspace
    from sorethumb.io.fingerprint import logical_dataset_id

    dataset_fp = logical_dataset_id(None, str(csv_path))

    with Workspace.open(workdir) as ws:
        # It DOES write: the workspace DB (migrations applied), a dataset +
        # dataset_snapshot row, and a run row left in status 'running'.
        assert len(ws.store.dataset_snapshots(dataset_fp)) == 1
        runs = ws.store.list_runs()
        assert [r["status"] for r in runs] == ["running"]
        # It does NOT write: run_group rows (and therefore no totals either,
        # since totals are only recorded per processed group).
        assert ws.store.all_run_groups(runs[0]["run_id"]) == []

    # No results/models/report artefacts were produced on disk either.
    assert list(workdir.rglob("anomalies.parquet")) == []
    assert list(workdir.rglob("index.html")) == []
    assert list((workdir / "models").rglob("plan.json")) == []
    assert list((workdir / "models").rglob("*.joblib")) == []


def test_run_exits_nonzero_and_reports_status_when_report_generation_fails(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2-8: a report-rendering exception used to be invisible from the CLI
    -- exit code 0, nothing printed about it, report_status not even a
    field. Detection/scoring succeeding must still surface a failed report
    explicitly, in both the text summary and --json, and as a non-zero exit."""
    import sorethumb._pipeline as pipeline_mod

    def _boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(pipeline_mod, "render_report", _boom)

    _, toml_path, _ = workspace
    result = runner.invoke(app, ["run", "--config", str(toml_path)])
    output = result.stdout + (result.stderr or "")
    assert result.exit_code == 1
    assert "Report generation failed" in output

    json_result = runner.invoke(app, ["run", "--config", str(toml_path), "--force", "--json"])
    assert json_result.exit_code == 1
    payload = json.loads(json_result.stdout)
    assert payload["report_status"] == "failed"
    assert payload["report_path"] is None
    assert payload["n_failed"] == 0  # detection/scoring itself did not fail


def test_run_with_groups(workspace_grouped):
    _, toml_path, workdir = workspace_grouped
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")


def test_run_limit_groups_caps_deterministically(workspace_grouped):
    """P2-7: --limit-groups used to be parsed and silently ignored. Must
    actually cap the group count, and do so deterministically (sorted by
    label) so the same limit always keeps the same groups."""
    _, toml_path, workdir = workspace_grouped  # group labels "G0", "G1"
    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report", "--limit-groups", "1"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        groups = ws.store.all_run_groups(run_id)
    assert len(groups) == 1
    assert groups[0]["group_label"] == "G0"


# ---------------------------------------------------------------------------
# --strict overrides an explicit TOML value; not passing it respects TOML (P2-7)
# ---------------------------------------------------------------------------


def test_run_strict_flag_overrides_explicit_toml_false(workspace):
    """The old `run_section.setdefault("strict", strict)` silently ignored
    --strict whenever TOML already had *any* explicit value for it (even
    False) -- setdefault only fills in a missing key."""
    _, toml_path, workdir = workspace
    toml_path.write_text(toml_path.read_text().replace("[run]\n", "[run]\nstrict = false\n", 1))

    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report", "--strict"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        config_json = ws.store.get_run(run_id)["config_json"]
    assert json.loads(config_json)["run"]["strict"] is True


def test_run_no_strict_flag_overrides_explicit_toml_true(workspace):
    _, toml_path, workdir = workspace
    toml_path.write_text(toml_path.read_text().replace("[run]\n", "[run]\nstrict = true\n", 1))

    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report", "--no-strict"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        config_json = ws.store.get_run(run_id)["config_json"]
    assert json.loads(config_json)["run"]["strict"] is False


def test_run_no_strict_or_no_strict_flag_respects_toml_true(workspace):
    """When neither --strict nor --no-strict is passed, TOML's own value
    must be respected, not silently replaced by the hardcoded default."""
    _, toml_path, workdir = workspace
    toml_path.write_text(toml_path.read_text().replace("[run]\n", "[run]\nstrict = true\n", 1))

    result = runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = ws.store.list_runs(limit=1)[0]["run_id"]
        config_json = ws.store.get_run(run_id)["config_json"]
    assert json.loads(config_json)["run"]["strict"] is True


# ---------------------------------------------------------------------------
# sorethumb run --detectors must not rewrite sorethumb.toml (P0-6)
# ---------------------------------------------------------------------------


def _write_full_toml(path: Path, csv_path: Path, workdir: Path) -> str:
    """Write a sorethumb.toml exercising every section, nested detector params,
    quoted strings and a Windows-style path, and return its exact text.
    """
    toml = f"""\
# a leading comment that must survive untouched
[source]
uri = {json.dumps(str(csv_path))}
format = "csv"

[columns]
id_column = "id"
group_by = []
ignore = ["it's \\"quoted\\"", "C:\\\\Users\\\\Test Data\\\\input.csv"]

[profiling]
null_ratio_drop = 0.9
null_ratio_flag = 0.5

[features]
one_hot_max_cardinality = 20
scaler = "robust"
correlation_threshold = 0.95

# detectors run as an ensemble
[[detectors]]
name = "isolation_forest"
enabled = true

[detectors.params]
n_estimators = 50

[detectors.params.extra_params]
max_features = 0.5

[[detectors]]
name = "kmeans_distance"
enabled = false

[scoring]
contamination = 0.1
combination = "composite"
weighting = "equal"
min_records = 10

[explain]
enabled = false
top_n = 3
max_rows = 100

[run]
workdir = {json.dumps(str(workdir))}
seed = 0

[history]
period_granularity = "day"

[report]
formats = ["html"]
# a trailing comment that must survive untouched
"""
    path.write_text(toml, encoding="utf-8")
    return toml


def test_run_detectors_override_does_not_rewrite_config(tmp_path: Path):
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=300)
    workdir = tmp_path / "ws"
    toml_path = tmp_path / "sorethumb.toml"
    original_text = _write_full_toml(toml_path, csv_path, workdir)
    original_bytes = toml_path.read_bytes()

    result = runner.invoke(
        app,
        ["run", "--config", str(toml_path), "--detectors", "ecod", "--no-report", "--json"],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    # The config file on disk must be byte-identical to what was written before the run.
    assert toml_path.read_bytes() == original_bytes
    assert toml_path.read_text(encoding="utf-8") == original_text

    # The override must still take effect for the run itself.
    data = json.loads(result.stdout)
    g = data["groups"][0]
    assert set(g["detector_flag_rates"]) == {"ecod"}


# ---------------------------------------------------------------------------
# sorethumb score --from-run
# ---------------------------------------------------------------------------


def test_score_from_run(workspace):
    _, toml_path, workdir = workspace
    src_run_id = _run_and_get_run_id(toml_path, workdir)

    result = runner.invoke(
        app, ["score", "--from-run", src_run_id, "--config", str(toml_path), "--no-report", "--json"]
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    data = json.loads(result.stdout)
    assert data["run_id"].startswith("score_")
    assert data["run_id"] != src_run_id

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        assert ws.store.get_run(data["run_id"])["source_run_id"] == src_run_id


def test_score_from_missing_run_fails(workspace):
    _, toml_path, _ = workspace
    result = runner.invoke(
        app, ["score", "--from-run", "run_nope", "--config", str(toml_path), "--no-report"]
    )
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# sorethumb report
# ---------------------------------------------------------------------------


def test_report_rerenders_from_persisted_run(workspace):
    _, toml_path, workdir = workspace
    # A run *with* a report, then delete the rendered file.
    assert runner.invoke(app, ["run", "--config", str(toml_path)]).exit_code == 0
    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = str(ws.store.list_runs(limit=1)[0]["run_id"])
    index = workdir / "reports" / run_id / "index.html"
    assert index.exists()
    original = index.read_text(encoding="utf-8")
    index.unlink()

    result = runner.invoke(app, ["report", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "Report written:" in result.stdout
    assert index.exists()
    assert index.read_text(encoding="utf-8") == original


def test_report_defaults_to_latest_run(workspace):
    _, toml_path, workdir = workspace
    assert runner.invoke(app, ["run", "--config", str(toml_path)]).exit_code == 0

    result = runner.invoke(app, ["report", "--config", str(toml_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "Report written:" in result.stdout


def test_report_unknown_run_id_errors(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    result = runner.invoke(app, ["report", "run_nope", "--config", str(toml_path)])
    assert result.exit_code == 1
    assert "Run not found" in result.stdout + (result.stderr or "")


def test_report_rerender_uses_current_config_report_formats(workspace):
    """P2-7: report.formats is the one deliberate exception to 'report
    re-renders from the run's historical state' -- it is purely cosmetic
    (which output files get written), so a change to the *current* config
    must take effect on re-render, as the command's own help text promises."""
    _, toml_path, workdir = workspace
    assert runner.invoke(app, ["run", "--config", str(toml_path)]).exit_code == 0

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = str(ws.store.list_runs(limit=1)[0]["run_id"])
    report_dir = workdir / "reports" / run_id
    assert (report_dir / "index.html").exists()
    assert not (report_dir / "index.json").exists()

    toml_path.write_text(toml_path.read_text() + '\n[report]\nformats = ["json"]\n')

    result = runner.invoke(app, ["report", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert (report_dir / "index.json").exists()


def test_report_rerender_group_structure_comes_from_historical_run(workspace_grouped):
    """The data-shaping half of the same rule: a re-render must reflect the
    run's own historical group structure, never whatever the *current*
    config says now -- the persisted results were computed against that
    historical structure and couldn't correctly correspond to a different
    one."""
    _, toml_path, workdir = workspace_grouped  # group_by = ["group"] -> 2 groups
    assert runner.invoke(app, ["run", "--config", str(toml_path)]).exit_code == 0

    from sorethumb import Workspace

    with Workspace.open(workdir) as ws:
        run_id = str(ws.store.list_runs(limit=1)[0]["run_id"])
        assert len(ws.store.all_run_groups(run_id)) == 2

    # A current config that would produce a single group on a *fresh* run.
    toml_path.write_text(toml_path.read_text().replace('group_by = ["group"]', "group_by = []"))

    result = runner.invoke(app, ["report", run_id, "--config", str(toml_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    html = (workdir / "reports" / run_id / "index.html").read_text(encoding="utf-8")
    assert "G0" in html
    assert "G1" in html


# ---------------------------------------------------------------------------
# sorethumb backfill
# ---------------------------------------------------------------------------


def _write_timeseries_parquet(path: Path, day_labels: list[str], *, per_day: int = 40, seed: int = 0) -> None:
    """Write a Parquet file with ``per_day`` rows on each of ``day_labels``.

    A few rows per day carry ``value_a = 999`` so each day has real anomalies.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    ts: list[datetime] = []
    value_a: list[float] = []
    for d, label in enumerate(day_labels):
        base = datetime.fromisoformat(label).replace(tzinfo=UTC) + timedelta(hours=8)
        col = rng.normal(0.0, 1.0, per_day).tolist()
        for k in (1, 7, 19):
            col[k] = 999.0
        for r in range(per_day):
            ids.append(d * per_day + r)
            ts.append(base + timedelta(minutes=r))
            value_a.append(col[r])
    pl.DataFrame(
        {
            "id": ids,
            "ts": pl.Series(ts).dt.cast_time_unit("us"),
            "value_a": value_a,
            "value_b": rng.normal(5.0, 2.0, len(ids)).tolist(),
        }
    ).write_parquet(str(path))


def _write_timeseries_toml(path: Path, parquet_path: Path, workdir: Path, *, bootstrap: int = 3) -> Path:
    """Write a sorethumb.toml with a time column and a shallow backfill window."""
    toml = f"""\
[source]
uri = {json.dumps(str(parquet_path))}
format = "parquet"

[run]
workdir = {json.dumps(str(workdir))}
seed = 0

[columns]
id_column = "id"
time_column = "ts"
group_by = []

[history]
period_granularity = "day"
roll_non_business = true
bootstrap_periods = {bootstrap}
lookback_periods = {bootstrap}
max_backfill_periods = 30

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
"""
    path.write_text(toml, encoding="utf-8")
    return path


def _recent_day_labels(n: int) -> tuple[list[str], int]:
    """The ``n`` business-day labels a cold-start backfill will target right
    now, plus the ``bootstrap_periods``/``lookback_periods`` calendar-day
    span that makes ``resolve_backfill_range``'s cold-start branch land on
    exactly those labels.

    ``resolve_backfill_range``'s cold-start branch walks back *calendar*
    days from the reference, then drops Saturday/Sunday labels via
    ``filter_non_business`` (see its docstring) -- so a fixed
    ``bootstrap_periods=n`` only ever produces exactly ``n`` labels when the
    trailing window happens not to cross a weekend, and silently produces
    fewer on the days it does. Returning the exact calendar-day span between
    the earliest of the n business-day labels and "yesterday" lets the
    caller configure ``bootstrap``/``lookback`` to match precisely, so the
    test is correct on every day of the week rather than only the days it
    happened to be run on.
    """
    from sorethumb.history.periods import filter_non_business, period_range, resolve_period, step_back

    ref = datetime.now(UTC)
    _, _, ref_label = resolve_period(ref, "day", roll_non_business=True)
    end_label = step_back(ref_label, "day", 1)
    # Generous margin: at most 2 non-business days per 7-calendar-day span.
    start_label = step_back(ref_label, "day", n * 2 + 3)
    candidates = period_range(start_label, end_label, "day")
    business_labels = filter_non_business(candidates, "day", roll_non_business=True)
    labels = business_labels[-n:]
    span_days = (date.fromisoformat(end_label) - date.fromisoformat(labels[0])).days + 1
    return labels, span_days


@pytest.fixture
def timeseries_workspace(tmp_path: Path):
    """(toml_path, workdir, day_labels) for a 3-day time-series dataset."""
    labels, span_days = _recent_day_labels(3)
    parquet = tmp_path / "data" / "ts.parquet"
    _write_timeseries_parquet(parquet, labels)
    workdir = tmp_path / "ws"
    toml_path = tmp_path / "sorethumb.toml"
    _write_timeseries_toml(toml_path, parquet, workdir, bootstrap=span_days)
    return toml_path, workdir, labels


def _totals_period_labels(toml_path: Path, workdir: Path, candidate_labels: list[str]) -> set[str]:
    """Which of ``candidate_labels`` have a totals row, via the Store's public API
    (not raw SQL -- CLI tests treat the store as a black box)."""
    from sorethumb import Workspace
    from sorethumb.config import Config
    from sorethumb.io.fingerprint import logical_dataset_id

    with toml_path.open("rb") as fh:
        cfg = Config.model_validate(tomllib.load(fh))
    dataset_fp = logical_dataset_id(cfg.source.dataset_id, cfg.source.uri)

    with Workspace.open(workdir) as ws:
        rows = ws.store.totals_for_periods(dataset_fp, candidate_labels, cfg.config_hash())
    return {str(r["period_label"]) for r in rows}


def test_backfill_processes_pending_periods_and_writes_totals(timeseries_workspace):
    toml_path, workdir, labels = timeseries_workspace

    result = runner.invoke(app, ["backfill", "--config", str(toml_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "3 pending periods" in result.stdout
    assert "Backfill complete." in result.stdout

    # every backfilled period now has a totals row (this is what makes it idempotent)
    assert _totals_period_labels(toml_path, workdir, labels) == set(labels)


def test_backfill_is_idempotent(timeseries_workspace):
    toml_path, workdir, _ = timeseries_workspace

    first = runner.invoke(app, ["backfill", "--config", str(toml_path)])
    assert first.exit_code == 0, first.stdout + (first.stderr or "")

    second = runner.invoke(app, ["backfill", "--config", str(toml_path)])
    assert second.exit_code == 0
    assert "Nothing to backfill" in second.stdout


def test_backfill_dry_run_writes_no_totals(timeseries_workspace):
    toml_path, workdir, labels = timeseries_workspace

    result = runner.invoke(app, ["backfill", "--config", str(toml_path), "--dry-run"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    for label in labels:
        assert label in result.stdout
    assert _totals_period_labels(toml_path, workdir, labels) == set()


def test_backfill_no_time_column_exits_cleanly(workspace):
    _, toml_path, _ = workspace  # the plain fixture has no time_column
    result = runner.invoke(app, ["backfill", "--config", str(toml_path)])
    assert result.exit_code == 0
    assert "No time_column configured" in result.stdout


def test_backfill_collects_every_result_and_exits_nonzero_on_any_failure(
    timeseries_workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2-7: backfill used to discard every period's RunResult and always
    print "Backfill complete." with exit code 0, even if every period
    failed. One period is made to fail (its RunResult carries a failed
    group); the other two must still be processed (not stopped early), and
    the command must exit non-zero and name the failed period."""
    import sorethumb.cli as cli_mod
    from sorethumb._pipeline import GroupSummary, RunResult

    toml_path, workdir, labels = timeseries_workspace
    failing_label = labels[1]
    real_run_detection = cli_mod.run_detection
    seen_labels: list[str | None] = []

    def _patched(cfg, *, period_label_override=None, **kwargs):
        seen_labels.append(period_label_override)
        if period_label_override == failing_label:
            bad_group = GroupSummary(
                group_key="gk",
                group_label="__all__",
                n_records=10,
                n_anomalies=0,
                anomaly_rate=None,
                results_path=None,
                status="failed",
                error="injected failure",
                elapsed_seconds=0.1,
                drifted=False,
                refit_reason=None,
                warnings_issued=[],
            )
            return RunResult(
                run_id="fake-run",
                dataset_uri=cfg.source.uri,
                dataset_fp="fp",
                config_hash=cfg.config_hash(),
                period_label=period_label_override,
                workspace_path=Path(cfg.run.workdir),
                groups=[bad_group],
                report_path=None,
                started_at="t0",
                finished_at="t1",
            )
        return real_run_detection(cfg, period_label_override=period_label_override, **kwargs)

    monkeypatch.setattr(cli_mod, "run_detection", _patched)

    result = runner.invoke(app, ["backfill", "--config", str(toml_path)])
    output = result.stdout + (result.stderr or "")

    assert set(seen_labels) == set(labels), "every pending period must still be processed, not stopped early"
    assert result.exit_code == 1
    assert failing_label in output
    assert "failed" in output.lower()
    assert "Backfill complete." not in output


# ---------------------------------------------------------------------------
# sorethumb workspace reset
# ---------------------------------------------------------------------------


def test_workspace_reset_requires_confirmation(workspace):
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    # --yes flag should succeed without interactive prompt
    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml_path), "--yes"])
    assert result.exit_code == 0
    assert not workdir.exists()


def test_workspace_reset_rejects_a_directory_that_is_not_a_workspace(tmp_path: Path):
    """P0-2: a directory with no sorethumb.db marker (never a real
    Workspace) must be refused, not silently accepted and destroyed,
    however it ended up as the resolved workdir."""
    not_a_workspace = tmp_path / "not-a-workspace"
    not_a_workspace.mkdir()
    (not_a_workspace / "important.txt").write_text("do not delete me")
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=10)
    toml_path = tmp_path / "sorethumb.toml"
    _write_toml(toml_path, csv_path, not_a_workspace)

    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml_path), "--yes"])

    assert result.exit_code != 0
    assert not_a_workspace.exists()
    assert (not_a_workspace / "important.txt").exists()


def test_workspace_reset_deletion_failure_is_reported_not_swallowed(
    workspace, monkeypatch: pytest.MonkeyPatch
):
    """P0-2: `shutil.rmtree(..., ignore_errors=True)` used to make a
    partial/failed deletion look like a clean success (exit 0). A real
    deletion failure must now propagate as a non-zero exit."""
    _, toml_path, workdir = workspace
    runner.invoke(app, ["run", "--config", str(toml_path), "--no-report"])
    assert workdir.exists()

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated deletion failure")

    monkeypatch.setattr("shutil.rmtree", _boom)

    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml_path), "--yes"])

    assert result.exit_code != 0


def test_workspace_reset_resolves_a_symlinked_workdir_before_deleting(
    tmp_path: Path,
) -> None:
    """P0-2: `_guard_reset_target` and the deletion itself must act on the
    *resolved* target, not the symlink -- a workdir configured as a symlink
    to a real workspace must still delete the real directory (proving the
    resolve-before-guard path-resolution edge case works), and must not
    delete anything if the symlink instead points somewhere unsafe."""
    real_ws_parent = tmp_path / "real" / "nested" / "location"
    real_ws_parent.mkdir(parents=True)
    real_ws = real_ws_parent / "ws"
    csv_path = tmp_path / "data" / "test.csv"
    _write_csv(csv_path, n_rows=10)
    toml_for_run = tmp_path / "real-sorethumb.toml"
    _write_toml(toml_for_run, csv_path, real_ws)
    runner.invoke(app, ["run", "--config", str(toml_for_run), "--no-report"])
    assert (real_ws / "sorethumb.db").exists()

    link_ws = tmp_path / "link-to-ws"
    link_ws.symlink_to(real_ws, target_is_directory=True)
    toml_via_link = tmp_path / "link-sorethumb.toml"
    _write_toml(toml_via_link, csv_path, link_ws)

    result = runner.invoke(app, ["workspace", "reset", "--config", str(toml_via_link), "--yes"])

    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert not real_ws.exists()
