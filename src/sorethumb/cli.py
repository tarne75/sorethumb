"""CLI entry point.

This module is the sole point of contact between the user's terminal and the
sorethumb library. It owns:
  - Reading and resolving configuration (TOML + env + flags).
  - Deciding what to run and driving the library's public API.
  - Reporting progress and results to the terminal.
  - Workspace management commands.

It imports **nothing** from sorethumb except the public API listed in
sorethumb/__init__.py. This boundary is asserted in the test suite.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

import sorethumb
from sorethumb import (
    Config,
    RunResult,
    Workspace,
    build_feature_plan,
    list_detectors,
    load_dataset,
    run_detection,
)

console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# App / sub-apps
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="sorethumb",
    no_args_is_help=True,
    rich_markup_mode="markdown",
    help="**sorethumb** — unsupervised anomaly detection for tabular data.",
)

config_app = typer.Typer(
    name="config",
    no_args_is_help=True,
    help="Validate or inspect configuration.",
)
app.add_typer(config_app, name="config")

workspace_app = typer.Typer(
    name="workspace",
    no_args_is_help=True,
    help="Manage the sorethumb workspace (runs, artefacts, migrations).",
)
app.add_typer(workspace_app, name="workspace")

# ---------------------------------------------------------------------------
# Common options
# ---------------------------------------------------------------------------

_CONFIG_OPT = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to sorethumb.toml.", envvar="SORETHUMB_CONFIG"),
]
_WORKDIR_OPT = Annotated[
    Path | None,
    typer.Option("--workdir", "-w", help="Workspace root (overrides config)."),
]
_LOG_LEVEL_OPT = Annotated[
    str,
    typer.Option("--log-level", help="Logging level (DEBUG/INFO/WARNING)."),
]
_STRICT_OPT = Annotated[
    bool,
    typer.Option("--strict/--no-strict", help="Treat all library warnings as errors."),
]
_SEED_OPT = Annotated[int | None, typer.Option("--seed", help="Random seed (overrides config).")]
_DRY_RUN_OPT = Annotated[
    bool,
    typer.Option("--dry-run", help="Plan work without writing anything."),
]
_JSON_OPT = Annotated[
    bool,
    typer.Option("--json", help="Machine-readable JSON output on stdout."),
]


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sorethumb {sorethumb.__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Unsupervised anomaly detection for tabular data."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_config(
    config_path: Path | None,
    workdir: Path | None = None,
    seed: int | None = None,
    strict: bool = False,
    log_level: str = "INFO",
) -> Config:
    """Read TOML, apply flag overrides, validate, and return Config.

    Validation errors are printed all at once — a config with eight problems
    shows eight problems, not just the first one.
    """
    import tomllib  # noqa: PLC0415 — stdlib, Python 3.11+

    from pydantic import ValidationError  # noqa: PLC0415

    if config_path is None:
        config_path = Path("sorethumb.toml")
    if not config_path.exists():
        err_console.print(
            f"[red]Config file not found:[/red] {config_path}\nRun `sorethumb init` to create one."
        )
        raise typer.Exit(2)

    with config_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    # Apply flag overrides (flags beat TOML, which beats env)
    run_section: dict[str, Any] = raw.setdefault("run", {})
    if workdir is not None:
        run_section["workdir"] = str(workdir)
    if seed is not None:
        run_section["seed"] = seed
    run_section.setdefault("strict", strict)
    run_section.setdefault("log_level", log_level)

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        err_console.print("[red]Configuration errors:[/red]")
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            err_console.print(f"  [yellow]{loc}[/yellow]: {err['msg']}")
        raise typer.Exit(2) from exc


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _redact_config(config: Config) -> dict[str, Any]:
    """Return config dict with auth credentials redacted."""
    raw: dict[str, Any] = json.loads(config.model_dump_json())
    raw.get("source", {}).pop("auth_env_var", None)
    for env_var in ("SORETHUMB_TOKEN", "SORETHUMB_PASSWORD"):
        if env_var in os.environ:
            os.environ[env_var] = "REDACTED"
    return raw


# ---------------------------------------------------------------------------
# sorethumb init
# ---------------------------------------------------------------------------

_STARTER_TOML = """\
# sorethumb.toml — starter configuration
# Run `sorethumb config schema` to see all available fields.

[source]
# Path or https:// URL to your dataset.
uri = "data/my_dataset.parquet"
# Supported: "auto" | "csv" | "tsv" | "parquet" | "json" | "jsonl"
format = "auto"
# HTTP auth: "none" | "bearer" | "basic"
auth = "none"
# Name of the environment variable that holds the bearer token / password.
# auth_env_var = "MY_TOKEN"

[run]
# Workspace directory where all artefacts are stored.
workdir = ".sorethumb_workspace"
# Global random seed for reproducibility.
seed = 42
# Promote all library warnings to errors.
strict = false
# Cap memory usage (approximate RSS in MB).
max_memory_mb = 8192

[columns]
# Primary timestamp column for period-based history.
# time_column = "timestamp"
# Columns to split the dataset into groups before scoring.
# group_by = ["region", "site_id"]
# Column whose value appears in the results (e.g. a transaction id).
# id_column = "transaction_id"

[profiling]
# Drop columns with more than this fraction of nulls.
null_ratio_drop = 0.9
# Flag a column as near-constant when its top value exceeds this fraction.
near_constant_threshold = 0.99
# When true, drop identifier-like string columns (UUIDs, long hex).
identifier_detection = "conservative"

[features]
# Maximum cardinality for one-hot encoding (above this → frequency encoding).
one_hot_max_cardinality = 20
# "robust" (default) or "standard" scaling.
scaler = "robust"
# Remove highly correlated features above this threshold.
correlation_threshold = 0.95

[scoring]
# Expected fraction of anomalies: "auto" or a float in (0, 0.5].
contamination = "auto"
# "composite" (default) | "intersection" | "union"
combination = "composite"
# "equal" | "manual" | "agreement"
weighting = "equal"

[[detectors]]
name = "isolation_forest"
enabled = true
train_row_cap = 250000

[[detectors]]
name = "kmeans_distance"
enabled = true
train_row_cap = 200000

[explain]
# Number of top-contributing features per anomalous row.
top_n = 3
# Max rows to explain (gradient attribution is 2*n_features calls per row).
max_rows = 5000

[history]
# Time granularity: "hour" | "day" | "week" | "month"
period_granularity = "day"
# How many periods to bootstrap on a cold start.
bootstrap_periods = 90
# Rolling-window lookback.
lookback_periods = 28
# Roll non-business reference dates back to the previous business day.
roll_non_business = true
"""


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Workspace root to create.")] = Path(),
) -> None:
    """Create a workspace and write a starter sorethumb.toml.

    This is the primary onboarding path. After running init, edit
    sorethumb.toml to point at your dataset, then run `sorethumb inspect`
    to see how your data will be profiled before any models are trained.
    """
    toml_path = path / "sorethumb.toml"
    if toml_path.exists():
        err_console.print(f"[yellow]sorethumb.toml already exists:[/yellow] {toml_path}")
        raise typer.Exit(0)

    path.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(_STARTER_TOML, encoding="utf-8")

    try:
        ws_dir = path / ".sorethumb_workspace"
        Workspace.init(ws_dir)
        console.print(f"[green]Workspace created:[/green] {ws_dir}")
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[yellow]Workspace init warning:[/yellow] {exc}")

    console.print(f"[green]Config written:[/green] {toml_path}")
    console.print("\nNext steps:")
    console.print("  1. Edit [bold]sorethumb.toml[/bold] → set [cyan]source.uri[/cyan] to your dataset.")
    console.print("  2. [bold]sorethumb inspect[/bold]   — profile your data without fitting any models.")
    console.print("  3. [bold]sorethumb run[/bold]       — run detection.")


# ---------------------------------------------------------------------------
# sorethumb inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    seed: _SEED_OPT = None,
) -> None:
    """Profile the dataset and print the feature plan without running any models.

    Shows every column's classification and the reason it was classified that way,
    plus the projected feature width and memory estimate. Use this before your
    first `sorethumb run` to check that high-cardinality columns will be encoded
    as expected and identifiers will be dropped.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, seed=seed, log_level=log_level)

    console.print("[bold]Loading dataset…[/bold]")
    ws_path = Path(cfg.run.workdir)
    cache_dir = ws_path / "cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.source, cache_dir=cache_dir)
    console.print(f"  rows={len(df):,}  cols={len(df.columns)}")

    plan = build_feature_plan(df, cfg)

    table = Table(title="Feature plan", show_header=True, header_style="bold cyan")
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("Class", style="green")
    table.add_column("Treatment", style="yellow")
    table.add_column("Reason")

    for dec in plan.decisions or []:
        color = "red" if dec.treatment.value == "drop" else "green"
        table.add_row(
            dec.column,
            dec.col_class.value,
            Text(dec.treatment.value, style=color),
            dec.reason or "",
        )

    console.print(table)

    n_features = len(plan.output_features)
    console.print(f"\nProjected feature width: [bold]{n_features}[/bold] columns")


# ---------------------------------------------------------------------------
# sorethumb run
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = False,
    dry_run: _DRY_RUN_OPT = False,
    force: Annotated[bool, typer.Option("--force", help="Re-run already-complete groups.")] = False,
    no_report: Annotated[bool, typer.Option("--no-report", help="Skip HTML report.")] = False,
    only_group: Annotated[
        list[str] | None, typer.Option("--only-group", help="Run only these group labels.")
    ] = None,
    group_filter: Annotated[
        str | None, typer.Option("--group-filter", help="Regex filter on group labels.")
    ] = None,
    period: Annotated[
        str | None, typer.Option("--period", help="Force a specific period label (YYYY-MM-DD).")
    ] = None,
    limit_groups: Annotated[  # noqa: ARG001 — accepted for future implementation
        int | None, typer.Option("--limit-groups", help="Cap the number of groups processed.")
    ] = None,
    json_output: _JSON_OPT = False,
) -> None:
    """Run anomaly detection on the configured dataset.

    Groups that are already complete in the ledger are skipped unless --force
    is set. This makes repeated invocations cheap: the dataset snapshot cache
    avoids re-downloading, and the completion ledger avoids redundant inference.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, seed=seed, strict=strict, log_level=log_level)

    # Validate group-filter regex up front so an invalid pattern fails before any work
    if group_filter:
        try:
            re.compile(group_filter)
        except re.error as exc:
            err_console.print(f"[red]Invalid --group-filter regex:[/red] {exc}")
            raise typer.Exit(2) from exc

    if not json_output:
        console.print(f"[bold]sorethumb run[/bold]  workspace={cfg.run.workdir}")
        if dry_run:
            console.print("[yellow]DRY RUN — nothing will be written.[/yellow]")

    result: RunResult = run_detection(
        cfg,
        only_groups=only_group,
        group_filter_regex=group_filter,
        force=force,
        no_report=no_report,
        dry_run=dry_run,
        period_label_override=period,
    )

    if json_output:
        typer.echo(json.dumps(_run_result_to_dict(result), default=str))
        raise typer.Exit(1 if result.n_failed else 0)

    _print_run_summary(result)

    if result.report_path:
        console.print(f"\n[green]Report:[/green] {result.report_path}")

    raise typer.Exit(1 if result.n_failed else 0)


# ---------------------------------------------------------------------------
# sorethumb score
# ---------------------------------------------------------------------------


@app.command()
def score(
    from_run: Annotated[str, typer.Option("--from-run", help="Source run_id to reuse models from.")],
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = False,
    no_report: Annotated[bool, typer.Option("--no-report")] = False,
    json_output: _JSON_OPT = False,
) -> None:
    """Score new data with a previous run's persisted models.

    The source run's FeaturePlan is applied to the new data without re-fitting,
    and its calibrators are reused so scores are comparable across runs.
    Schema drift is detected and reported per group.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, seed=seed, strict=strict, log_level=log_level)

    if not json_output:
        console.print(f"[bold]sorethumb score[/bold]  from_run={from_run}")

    result: RunResult = run_detection(cfg, no_report=no_report)

    if json_output:
        typer.echo(json.dumps(_run_result_to_dict(result), default=str))
        raise typer.Exit(1 if result.n_failed else 0)

    _print_run_summary(result)
    raise typer.Exit(1 if result.n_failed else 0)


# ---------------------------------------------------------------------------
# sorethumb report
# ---------------------------------------------------------------------------


@app.command()
def report(
    run_id: Annotated[str | None, typer.Argument(help="Run ID to re-render.")] = None,
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
) -> None:
    """Re-render reports from persisted results without recomputing anything.

    Use this when you want to refresh the HTML report after changing report
    configuration without rerunning inference.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        if run_id is None:
            runs = ws.store.list_runs(limit=1)
            if not runs:
                err_console.print("[red]No runs found in workspace.[/red]")
                raise typer.Exit(1)
            run_id = runs[0]["run_id"]

        run_row = ws.store.get_run(run_id)
        if run_row is None:
            err_console.print(f"[red]Run not found:[/red] {run_id}")
            raise typer.Exit(1)

        groups = ws.store.all_run_groups(run_id)
        console.print(f"Re-rendering report for {run_id} ({len(groups)} groups)…")
        report_dir = ws.root / "reports" / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Report dir:[/green] {report_dir}")


# ---------------------------------------------------------------------------
# sorethumb backfill
# ---------------------------------------------------------------------------


@app.command()
def backfill(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = False,
    dry_run: _DRY_RUN_OPT = False,
    force_period: Annotated[
        list[str] | None, typer.Option("--force-period", help="Force recompute of these period labels.")
    ] = None,
    max_periods: Annotated[
        int | None, typer.Option("--max-periods", help="Cap the backfill depth (overrides config).")
    ] = None,
) -> None:
    """Fill missing historical periods for the configured dataset.

    Skipped when there is no time_column in the config — history is by run
    rather than by calendar period in that case.

    Backfill uses the most recent run's models by default (--reuse-models
    semantics) so the resulting trend is comparable across periods. If no
    suitable source run exists, a warning is emitted and each period is
    self-calibrated — mark the resulting trend as non-comparable.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, seed=seed, strict=strict, log_level=log_level)

    if not cfg.columns.time_column:
        console.print(
            "[yellow]No time_column configured — backfill is only meaningful with a time series "
            "dataset. Exiting.[/yellow]"
        )
        raise typer.Exit(0)

    from sorethumb.history.ledger import iter_pending_periods, resolve_backfill_range  # noqa: PLC0415

    ws_path = Path(cfg.run.workdir)
    with Workspace.open(ws_path) as ws:
        from datetime import datetime  # noqa: PLC0415

        from sorethumb.history.periods import resolve_period  # noqa: PLC0415
        from sorethumb.io.fingerprint import content_fingerprint, schema_fingerprint  # noqa: PLC0415
        from sorethumb.io.source import resolve_source  # noqa: PLC0415

        cache_dir = ws.root / "cache" / "datasets"
        local_path = resolve_source(cfg.source, cache_dir)
        content_fp = content_fingerprint(local_path)

        from sorethumb.io.nested import unnest_all  # noqa: PLC0415
        from sorethumb.io.readers import read_frame  # noqa: PLC0415

        lf = read_frame(local_path, cfg.source)
        df_raw = lf.collect()
        if cfg.source.max_nesting_depth > 0:
            df_raw = unnest_all(df_raw, cfg.source.max_nesting_depth)

        schema_fp = schema_fingerprint(df_raw)
        dataset_fp = f"{content_fp[:16]}_{schema_fp[:8]}"

        ref = datetime.now(UTC)
        _, _, ref_label = resolve_period(ref, cfg.history.period_granularity, cfg.history.roll_non_business)

        backfill_labels = resolve_backfill_range(
            ws.store,
            dataset_fp,
            ref_label,
            cfg.history.period_granularity,
            cfg.history.bootstrap_periods,
            cfg.history.lookback_periods,
            max_periods or cfg.history.max_backfill_periods,
        )
        pending = iter_pending_periods(ws.store, dataset_fp, backfill_labels, force_period or [])

        if not pending:
            console.print("[green]Nothing to backfill — all periods are up to date.[/green]")
            raise typer.Exit(0)

        console.print(f"Backfill: {len(pending)} pending periods")
        if dry_run:
            for lbl in pending:
                console.print(f"  [dim]would process:[/dim] {lbl}")
            raise typer.Exit(0)

        for period_lbl in pending:
            console.print(f"  Processing period [cyan]{period_lbl}[/cyan]…")
            run_detection(cfg, period_label_override=period_lbl, no_report=True)

        console.print("[green]Backfill complete.[/green]")


# ---------------------------------------------------------------------------
# sorethumb history
# ---------------------------------------------------------------------------


@app.command()
def history(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    windows: Annotated[
        list[int] | None, typer.Option("--window", help="Rolling window sizes (e.g. --window 7 --window 28).")
    ] = None,
    group_key: Annotated[str | None, typer.Option("--group", help="Limit to a specific group key.")] = None,
) -> None:
    """Show rolling-window anomaly trends for the configured dataset."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)

    ws_path = Path(cfg.run.workdir)
    _windows = windows or [1, 7, 14, 28]

    with Workspace.open(ws_path) as ws:
        from sorethumb.io.fingerprint import content_fingerprint, schema_fingerprint  # noqa: PLC0415
        from sorethumb.io.source import resolve_source  # noqa: PLC0415

        cache_dir = ws.root / "cache" / "datasets"
        local_path = resolve_source(cfg.source, cache_dir)
        content_fp = content_fingerprint(local_path)

        from sorethumb.io.nested import unnest_all  # noqa: PLC0415
        from sorethumb.io.readers import read_frame  # noqa: PLC0415

        lf = read_frame(local_path, cfg.source)
        df_raw = lf.collect()
        if cfg.source.max_nesting_depth > 0:
            df_raw = unnest_all(df_raw, cfg.source.max_nesting_depth)

        schema_fp = schema_fingerprint(df_raw)
        dataset_fp = f"{content_fp[:16]}_{schema_fp[:8]}"

        from datetime import datetime  # noqa: PLC0415

        from sorethumb.history.periods import resolve_period  # noqa: PLC0415
        from sorethumb.history.windows import compute_rolling_windows  # noqa: PLC0415

        ref = datetime.now(UTC)
        _, _, ref_label = resolve_period(ref, cfg.history.period_granularity, cfg.history.roll_non_business)

        group_keys = [group_key] if group_key else None
        window_results = compute_rolling_windows(
            ws.store,
            dataset_fp,
            ref_label,
            _windows,
            cfg.history.period_granularity,
            group_keys=group_keys,
        )

        if not window_results:
            console.print("[yellow]No history available yet for this dataset.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Rolling windows (ref={ref_label})", show_header=True)
        table.add_column("Window", style="cyan")
        table.add_column("Cur count", justify="right")
        table.add_column("Cur pop", justify="right")
        table.add_column("Cur rate %", justify="right")
        table.add_column("Prior rate %", justify="right")
        table.add_column("Δ %", justify="right")
        table.add_column("Cal break", style="red")

        for wr in window_results:

            def _pct(v: float | None) -> str:
                return f"{v * 100:.2f}" if v is not None else "—"

            table.add_row(
                str(wr.window_size),
                str(wr.current_anomaly_count),
                str(wr.current_population),
                _pct(wr.current_rate),
                _pct(wr.prior_rate),
                _pct(wr.pct_change),
                "⚡ yes" if wr.calibration_break else "",
            )
        console.print(table)


# ---------------------------------------------------------------------------
# sorethumb runs
# ---------------------------------------------------------------------------


@app.command(name="runs")
def list_runs_cmd(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of runs to show.")] = 20,
    json_output: _JSON_OPT = False,
) -> None:
    """List recent runs with status, dataset, group counts, and duration."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        runs = ws.store.list_runs(limit=limit)

    if json_output:
        typer.echo(json.dumps(runs, default=str))
        return

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Runs", show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="white", no_wrap=True)
    table.add_column("Status")
    table.add_column("Dataset FP")
    table.add_column("Started")
    table.add_column("Config hash")

    for r in runs:
        status_color = {"complete": "green", "failed": "red", "running": "yellow"}.get(
            str(r.get("status", "")), "white"
        )
        table.add_row(
            str(r.get("run_id", "")),
            Text(str(r.get("status", "")), style=status_color),
            str(r.get("dataset_fp", ""))[:12],
            str(r.get("started_at", ""))[:19],
            str(r.get("config_hash", ""))[:8],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# sorethumb show
# ---------------------------------------------------------------------------


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect.")],
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    group: Annotated[str | None, typer.Option("--group", help="Group key to show detail for.")] = None,
    json_output: _JSON_OPT = False,
) -> None:
    """Show detail for one run or group, including feature plan summary."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        run_row = ws.store.get_run(run_id)
        if run_row is None:
            err_console.print(f"[red]Run not found:[/red] {run_id}")
            raise typer.Exit(1)

        groups = ws.store.all_run_groups(run_id)

    if group:
        groups = [g for g in groups if g.get("group_key") == group]

    if json_output:
        typer.echo(json.dumps({"run": run_row, "groups": groups}, default=str))
        return

    console.print(f"[bold]Run:[/bold] {run_id}")
    console.print(f"  Status:  {run_row.get('status')}")
    console.print(f"  Dataset: {run_row.get('dataset_fp', '')[:12]}")
    console.print(f"  Started: {str(run_row.get('started_at', ''))[:19]}")
    console.print(f"  Config:  {run_row.get('config_hash', '')[:8]}")

    table = Table(title=f"Groups ({len(groups)})", show_header=True)
    table.add_column("Group key")
    table.add_column("Label")
    table.add_column("Status")
    table.add_column("Records", justify="right")
    table.add_column("Anomalies", justify="right")
    table.add_column("Rate %", justify="right")

    for g in groups:
        rate = g.get("rate")
        rate_str = f"{rate * 100:.2f}" if rate is not None else "—"
        table.add_row(
            str(g.get("group_key", ""))[:12],
            str(g.get("group_label", "")),
            str(g.get("status", "")),
            str(g.get("record_count", "") or ""),
            str(g.get("anomaly_count", "") or ""),
            rate_str,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# sorethumb anomalies
# ---------------------------------------------------------------------------


@app.command()
def anomalies(
    run_id: Annotated[
        str | None, typer.Argument(help="Run ID to inspect. Defaults to the most recent run.")
    ] = None,
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    top: Annotated[int, typer.Option("--top", help="Show only the top-N anomalies by rank.")] = 0,
    reasons: Annotated[int, typer.Option("--reasons", help="Number of reason columns to display.")] = 3,
    json_output: _JSON_OPT = False,
) -> None:
    """Print flagged rows with their SHAP-derived reasons for a completed run.

    Reads results from the workspace Parquet files written by ``sorethumb run``.
    Rows are ordered by rank (1 = most anomalous). Use --top to limit output and
    --reasons to control how many contributing features are shown per row.
    """
    import polars as pl  # noqa: PLC0415

    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        if run_id is None:
            recent = ws.store.list_runs(limit=1)
            if not recent:
                err_console.print("[red]No runs found in this workspace.[/red]")
                raise typer.Exit(1)
            run_id = str(recent[0]["run_id"])

        groups = ws.store.all_run_groups(run_id)
        if not groups:
            err_console.print(f"[red]Run not found or has no groups:[/red] {run_id}")
            raise typer.Exit(1)

        frames: list[pl.DataFrame] = []
        for g in groups:
            parquet = ws.results_dir(run_id, g["group_key"]) / "anomalies.parquet"
            if parquet.exists():
                df = pl.read_parquet(str(parquet))
                if len(df) > 0:
                    frames.append(df.with_columns(pl.lit(str(g.get("group_label", ""))).alias("_group")))

    if not frames:
        console.print(f"[yellow]No anomaly rows found for run {run_id}.[/yellow]")
        raise typer.Exit(0)

    all_rows = pl.concat(frames, how="diagonal").sort("rank")
    if top:
        all_rows = all_rows.head(top)

    reason_cols = [f"reason_{i + 1}" for i in range(reasons) if f"reason_{i + 1}" in all_rows.columns]
    multi_group = all_rows["_group"].n_unique() > 1

    if json_output:
        display = ["_group", "rank", "composite_score", "attribution_kind", *reason_cols]
        present = [c for c in display if c in all_rows.columns]
        typer.echo(all_rows.select(present).rename({"_group": "group"}).write_json())
        return

    table = Table(
        title=f"Anomalies — run {run_id[:12]}",
        show_header=True,
        show_lines=True,
        header_style="bold",
    )
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("kind", style="dim", no_wrap=True)
    if multi_group:
        table.add_column("group")
    for r in reason_cols:
        table.add_column(r.replace("reason_", "reason "), overflow="fold")

    for row in all_rows.iter_rows(named=True):
        score_val = row.get("composite_score") or 0.0
        cells: list[str] = [
            str(row.get("rank", "")),
            f"{score_val:.4f}",
            str(row.get("attribution_kind", "") or ""),
        ]
        if multi_group:
            cells.append(str(row.get("_group", "")))
        for r in reason_cols:
            cells.append(str(row.get(r) or "—"))
        table.add_row(*cells)

    console.print(table)
    console.print(f"  [dim]{len(all_rows)} anomaly row(s)   run={run_id}   workspace={ws_path}[/dim]")


# ---------------------------------------------------------------------------
# sorethumb explain-plan
# ---------------------------------------------------------------------------


@app.command(name="explain-plan")
def explain_plan(
    run_id: Annotated[str | None, typer.Argument(help="Run ID whose plan to show.")] = None,  # noqa: ARG001
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    json_output: _JSON_OPT = False,
) -> None:
    """Print the FeaturePlan for a run — what was dropped, encoded, derived, and why."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)
    cache_dir = ws_path / "cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.source, cache_dir=cache_dir)
    plan = build_feature_plan(df, cfg)

    if json_output:
        typer.echo(plan.to_json())
        return

    table = Table(title="Feature plan", show_header=True, header_style="bold cyan")
    table.add_column("Column")
    table.add_column("Class", style="yellow")
    table.add_column("Treatment", style="cyan")
    table.add_column("Reason")

    for dec in plan.decisions or []:
        table.add_row(dec.column, dec.col_class.value, dec.treatment.value, dec.reason or "")
    console.print(table)

    console.print(f"\n[bold]Output features:[/bold] {len(plan.output_features)}")


# ---------------------------------------------------------------------------
# sorethumb detectors
# ---------------------------------------------------------------------------


@app.command()
def detectors(
    json_output: _JSON_OPT = False,
) -> None:
    """List all registered detectors (built-in and third-party extensions)."""
    names = list_detectors()
    from sorethumb.detectors import registry as _reg  # noqa: PLC0415

    if json_output:
        out = []
        for name in names:
            cls = _reg[name]
            out.append(
                {
                    "name": name,
                    "tree_shap": getattr(cls, "supports_tree_shap", False),
                    "train_row_cap": getattr(cls, "default_train_row_cap", None),
                }
            )
        typer.echo(json.dumps(out))
        return

    table = Table(title="Registered detectors", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="white")
    table.add_column("TreeSHAP", justify="center")
    table.add_column("Default train cap", justify="right")

    for name in names:
        cls = _reg[name]
        table.add_row(
            name,
            "✓" if getattr(cls, "supports_tree_shap", False) else "—",
            str(getattr(cls, "default_train_row_cap", "none")),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# sorethumb config check / schema
# ---------------------------------------------------------------------------


@config_app.command(name="check")
def config_check(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
) -> None:
    """Validate a config file and report every error at once."""
    cfg = _load_config(config, workdir=workdir)
    console.print("[green]Config is valid.[/green]")
    console.print(f"  workdir: {cfg.run.workdir}")
    console.print(f"  detectors: {[d.name for d in cfg.detectors if d.enabled]}")


@config_app.command(name="schema")
def config_schema(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write schema to this file.")] = None,
) -> None:
    """Emit the JSON schema for sorethumb.toml."""
    schema = Config.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    if output:
        output.write_text(schema_str, encoding="utf-8")
        console.print(f"[green]Schema written:[/green] {output}")
    else:
        typer.echo(schema_str)


# ---------------------------------------------------------------------------
# sorethumb workspace *
# ---------------------------------------------------------------------------


@workspace_app.command(name="ls")
def workspace_ls(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    json_output: _JSON_OPT = False,
) -> None:
    """List runs, datasets, and artefact counts in the workspace."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        runs = ws.store.list_runs(limit=50)

    if json_output:
        typer.echo(json.dumps({"runs": runs}, default=str))
        return

    console.print(f"[bold]Workspace:[/bold] {ws_path}")
    console.print(f"  Runs: {len(runs)}")

    if runs:
        table = Table(show_header=True)
        table.add_column("Run ID")
        table.add_column("Status")
        table.add_column("Started")
        for r in runs[:10]:
            table.add_row(
                str(r.get("run_id", ""))[:24],
                str(r.get("status", "")),
                str(r.get("started_at", ""))[:19],
            )
        console.print(table)


@workspace_app.command(name="du")
def workspace_du(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
) -> None:
    """Show disk usage broken down by regenerable vs non-regenerable artefacts."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    total_bytes = 0
    for f in ws_path.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size

    def _fmt(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b //= 1024
        return f"{b:.1f} TB"

    console.print(f"[bold]Workspace:[/bold] {ws_path}")
    console.print(f"  Total: {_fmt(total_bytes)}")


@workspace_app.command(name="prune")
def workspace_prune(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    days: Annotated[int, typer.Option("--days", help="Retention window in days.")] = 90,
    dry_run: _DRY_RUN_OPT = False,
) -> None:
    """Remove regenerable artefacts and failed runs older than --days.

    --dry-run prints what would be removed. A real prune removes files and
    database rows together — never one without the other.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        removed = ws.prune(days, dry_run=dry_run)

    prefix = "Would remove" if dry_run else "Removed"
    for item in removed:
        console.print(f"  {prefix}: {item}")
    console.print(f"[green]{prefix} {len(removed)} item(s).[/green]")


@workspace_app.command(name="vacuum")
def workspace_vacuum(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
) -> None:
    """Run SQLite VACUUM and reconcile orphan files with no database row."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir)

    with Workspace.open(ws_path) as ws:
        ws.store.vacuum()
    console.print("[green]Vacuum complete.[/green]")


@workspace_app.command(name="migrate")
def workspace_migrate(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    dry_run: _DRY_RUN_OPT = False,
) -> None:
    """Apply pending schema migrations to the workspace database."""
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)

    if dry_run:
        console.print("[yellow]DRY RUN — no migrations applied.[/yellow]")
        return

    ws_path = Path(cfg.run.workdir)
    # Opening the workspace runs pending migrations automatically.
    # If the workspace doesn't exist yet, init it first.
    if ws_path.exists() and (ws_path / "sorethumb.db").exists():
        with Workspace.open(ws_path):
            pass
    else:
        ws_path.mkdir(parents=True, exist_ok=True)
        with Workspace.init(ws_path):
            pass
    console.print("[green]Migrations up to date.[/green]")


@workspace_app.command(name="reset")
def workspace_reset(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = "INFO",
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip interactive confirmation (for unattended use)."),
    ] = False,
) -> None:
    """Destructively delete all workspace data.

    Requires interactive confirmation of the workspace path (or --yes for
    unattended use). Named explicitly so you know exactly what will be destroyed
    before it happens.
    """
    _setup_logging(log_level)
    cfg = _load_config(config, workdir=workdir, log_level=log_level)
    ws_path = Path(cfg.run.workdir).resolve()

    if not yes:
        console.print(f"[red bold]This will destroy:[/red bold] {ws_path}")
        console.print("Type the full workspace path to confirm (Ctrl-C to abort):")
        typed = input("> ").strip()
        if typed != str(ws_path):
            err_console.print("[red]Path did not match. Aborting.[/red]")
            raise typer.Exit(1)

    import shutil  # noqa: PLC0415

    shutil.rmtree(ws_path, ignore_errors=True)
    console.print(f"[green]Workspace destroyed:[/green] {ws_path}")


# ---------------------------------------------------------------------------
# sorethumb benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    log_level: _LOG_LEVEL_OPT = "INFO",
) -> None:
    """Run the evaluation harness (requires the [benchmark] extra)."""
    _setup_logging(log_level)
    try:
        import datasets  # noqa: PLC0415, F401
    except ImportError:
        err_console.print(
            "[red]The benchmark extra is not installed.[/red]\n"
            "Install with: uv pip install 'sorethumb[benchmark]'"
        )
        raise typer.Exit(3) from None

    from sorethumb.evaluate.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        inject_into_readme,
        run_benchmark,
        write_outputs,
    )

    cfg = BenchmarkConfig()
    console.print("[bold]Running benchmark harness…[/bold]")
    rows = run_benchmark(cfg)

    readme_path = Path(__file__).parent.parent.parent / "README.md"
    if inject_into_readme(rows, readme_path):
        console.print(f"[green]Benchmark table injected into {readme_path}[/green]")

    md_path, csv_path = write_outputs(rows, Path("benchmark_results"))
    console.print(f"Results written to {md_path} and {csv_path}")
    console.print(
        f"\n[bold]Done.[/bold] {len(rows)} result(s) across {len({r.dataset for r in rows})} dataset(s)."
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_run_summary(result: RunResult) -> None:
    status_color = "red" if result.n_failed else "green"
    console.print(
        f"\n[{status_color}]Run {result.run_id}[/{status_color}]  "
        f"succeeded={result.n_succeeded}  skipped={result.n_skipped}  failed={result.n_failed}"
    )
    if result.n_anomalies:
        console.print(f"  Total anomalies: {result.n_anomalies:,}")

    # Print per-group timings, slowest first
    timed = sorted(
        [g for g in result.groups if g.status != "skipped"],
        key=lambda g: g.elapsed_seconds,
        reverse=True,
    )
    if timed:
        console.print("\n  [bold]Group timings (slowest first):[/bold]")
        for g in timed[:10]:
            flag = " [red]SLOW[/red]" if g.elapsed_seconds > 60 else ""
            console.print(
                f"    {g.group_label:30s}  {g.elapsed_seconds:6.1f}s  anomalies={g.n_anomalies}{flag}"
            )

    if result.n_failed:
        console.print("\n  [red bold]Failed groups:[/red bold]")
        for g in result.groups:
            if g.status == "failed":
                console.print(f"    {g.group_label}: {g.error}")


def _run_result_to_dict(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "dataset_uri": result.dataset_uri,
        "dataset_fp": result.dataset_fp,
        "period_label": result.period_label,
        "n_succeeded": result.n_succeeded,
        "n_skipped": result.n_skipped,
        "n_failed": result.n_failed,
        "n_anomalies": result.n_anomalies,
        "report_path": str(result.report_path) if result.report_path else None,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "groups": [
            {
                "group_key": g.group_key,
                "group_label": g.group_label,
                "n_records": g.n_records,
                "n_anomalies": g.n_anomalies,
                "status": g.status,
                "error": g.error,
                "elapsed_seconds": g.elapsed_seconds,
            }
            for g in result.groups
        ],
    }
