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
import logging.handlers
import re
import sys
from collections.abc import Callable
from datetime import UTC
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

import sorethumb_ml
from sorethumb_ml import (
    Config,
    RunResult,
    SorethumbError,
    Workspace,
    build_feature_plan,
    list_detectors,
    load_dataset,
    run_detection,
    score_forward,
)

console = Console()
err_console = Console(stderr=True)


# Rich parses "[...]" in any printed string as markup, so a column named
# "[/x]" crashed the CLI with MarkupError and "amt [usd]" or a hint like
# "pip install 'sorethumb-ml[explain]'" lost its brackets. Everything not
# written by us -- column names, group labels, category values, reasons,
# paths, URIs, exception and warning messages -- goes through one of these.
def _e(value: object) -> str:
    """Escape *value* for interpolation into a Rich markup string."""
    return escape(str(value))


def _add_row(table: Table, *cells: object) -> None:
    """Add a table row whose cells are shown literally (``Text`` cells keep their style)."""
    table.add_row(*(cell if isinstance(cell, Text) else Text(str(cell)) for cell in cells))


# ---------------------------------------------------------------------------
# App / sub-apps
# ---------------------------------------------------------------------------

# Every Typer app sets these explicitly rather than trusting the installed
# Typer's defaults. pretty_exceptions_show_locals defaults to True on our
# declared floor (typer 0.16.0): an unexpected exception would then print every
# frame's local variables -- during an HTTP download, the headers dict holding
# the Authorization credential from source.auth_env_var -- into the terminal
# and CI logs. pretty_exceptions_short keeps an unexpected error to the frames
# that matter instead of a page of library internals.
_TYPER_EXCEPTION_SETTINGS: dict[str, Any] = {
    "pretty_exceptions_show_locals": False,
    "pretty_exceptions_short": True,
}


def _configure_output_streams(streams: tuple[Any, ...] | None = None) -> None:
    """Make stdout/stderr unable to abort a command over an unencodable character.

    A console already gets UTF-8 on every platform (Windows has used the
    console's Unicode API since Python 3.6). A pipe or file does not: on Windows
    it gets the ANSI code page (usually cp1252), so a column name like "温度" or
    an arrow in our own text raised UnicodeEncodeError mid-command. So:

    - a stream that is not a TTY switches to UTF-8, unless PYTHONIOENCODING
      says otherwise or it already is UTF-8 -- the encoding Python itself makes
      the default from 3.15 (PEP 686), and what a script reading our output
      should expect;
    - stdout's error handler becomes "replace" (stderr already uses
      "backslashreplace"), so an explicitly chosen narrow encoding degrades to
      "?" instead of crashing.

    --json output is unaffected either way: it is pure ASCII.
    """
    import os  # noqa: PLC0415

    targets = streams if streams is not None else (sys.stdout, sys.stderr)
    for stream in targets:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        changes: dict[str, str] = {}
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("_", "-")
        try:
            is_tty = bool(stream.isatty())
        except (OSError, ValueError):
            is_tty = False
        errors = getattr(stream, "errors", None) or "strict"
        if not is_tty and encoding not in ("utf-8", "utf8") and not os.environ.get("PYTHONIOENCODING"):
            changes["encoding"] = "utf-8"
        # Always pass errors alongside a new encoding: reconfigure(encoding=...)
        # on its own resets the handler to "strict".
        if errors == "strict" or changes:
            changes["errors"] = "replace" if errors == "strict" else errors
        if changes:
            try:
                reconfigure(**changes)
            except (OSError, ValueError):  # a stream that can't be reconfigured now: leave it
                continue


class _SorethumbTyper(typer.Typer):
    """Typer app that configures the output streams before running a command.

    Done in ``__call__`` (the console script and ``python -c "...; app()"``),
    not at import, so importing this module -- or invoking it in-process with
    Click's CliRunner -- never touches the caller's streams.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _configure_output_streams()
        return super().__call__(*args, **kwargs)


app = _SorethumbTyper(
    name="sorethumb",
    no_args_is_help=True,
    **_TYPER_EXCEPTION_SETTINGS,
    rich_markup_mode="markdown",
    help="**sorethumb** — unsupervised anomaly detection for tabular data.",
)

config_app = typer.Typer(
    name="config",
    no_args_is_help=True,
    **_TYPER_EXCEPTION_SETTINGS,
    help="Validate or inspect configuration.",
)
app.add_typer(config_app, name="config")

workspace_app = typer.Typer(
    name="workspace",
    no_args_is_help=True,
    **_TYPER_EXCEPTION_SETTINGS,
    help="Manage the sorethumb workspace (runs, artefacts, migrations).",
)
app.add_typer(workspace_app, name="workspace")

# ---------------------------------------------------------------------------
# Common options
# ---------------------------------------------------------------------------

# Default workspace root: used only when neither --workdir nor a config file's
# run.workdir is given -- with *or without* a config file. It is a path relative
# to the CURRENT DIRECTORY (not to the config file or the data file). A dedicated
# directory, not ".", so a first run never scatters sorethumb.db/models/results/
# reports/logs beside the source data or other files already in the current
# directory. docs and README state this value; tests/repo_check/test_workspace_docs.py
# fails if they drift from it.
_DEFAULT_WORKDIR = "sorethumb-workspace"

_CONFIG_OPT = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to sorethumb.toml.", envvar="SORETHUMB_CONFIG"),
]
_WORKDIR_OPT = Annotated[
    Path | None,
    typer.Option(
        "--workdir",
        "-w",
        help=f"Workspace root (overrides run.workdir; default ./{_DEFAULT_WORKDIR}/ in the current directory).",
    ),
]
_LOG_LEVEL_OPT = Annotated[
    str | None,
    typer.Option("--log-level", help="Logging level (DEBUG/INFO/WARNING).", show_default="INFO"),
]
_STRICT_OPT = Annotated[
    bool | None,
    typer.Option("--strict/--no-strict", help="Treat all library warnings as errors.", show_default="False"),
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

# The marker file Workspace.init creates (store/workspace.py's _MARKER_DB) --
# duplicated here, not imported, since it is that module's private constant.
_WORKSPACE_MARKER_FILENAME = "sorethumb.db"


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sorethumb {sorethumb_ml.__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(  # noqa: ARG001 -- handled by the eager _version_callback
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Unsupervised anomaly detection for tabular data."""
    # Whatever log file a command opens is closed when it finishes, so nothing
    # outlives the invocation (Windows can't delete or rotate an open file).
    ctx.call_on_close(_detach_file_handlers)


# ---------------------------------------------------------------------------
# Detector alias resolution
# ---------------------------------------------------------------------------

_DETECTOR_ALIASES: dict[str, str] = {
    # Short aliases
    "if": "isolation_forest",
    "km": "kmeans_distance",
    "oc": "one_class_svm",
    "ocsvm": "one_class_svm",
    # Full names also accepted
    "isolation_forest": "isolation_forest",
    "kmeans_distance": "kmeans_distance",
    "one_class_svm": "one_class_svm",
    "ecod": "ecod",
    "lof": "lof",
    "hbos": "hbos",
}


def _parse_detectors_flag(value: str) -> list[dict[str, Any]]:
    """Parse a comma-separated detector alias string into a raw detectors list.

    Looks up each alias in _DETECTOR_ALIASES, resolves the canonical name, then
    reads default_train_row_cap from the live registry so the cap is always
    consistent with the detector implementation.
    """
    from sorethumb_ml.detectors import registry  # noqa: PLC0415

    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        err_console.print("[red]--detectors: empty list — provide at least one detector alias.[/red]")
        raise typer.Exit(int(ExitCode.PREFLIGHT))

    result: list[dict[str, Any]] = []
    for alias in parts:
        name = _DETECTOR_ALIASES.get(alias.lower())
        if name is None:
            valid = ", ".join(sorted(_DETECTOR_ALIASES))
            err_console.print(
                f"[red]Unknown detector alias:[/red] {_e(repr(alias))}\nValid aliases: {_e(valid)}"
            )
            raise typer.Exit(int(ExitCode.PREFLIGHT))
        det_cls = registry.get(name)
        cap: int | None = det_cls.default_train_row_cap if det_cls else None
        result.append({"name": name, "train_row_cap": cap})

    return result


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


class ExitCode(IntEnum):
    """The CLI's documented, stable exit codes (see docs/cli_reference.md#exit-codes).

    0 success; 1 runtime failure (work was attempted and failed); 2 pre-flight
    failure (usage, configuration, schema or source rejected before any work);
    3 not found (a requested run, workspace or persisted file does not exist);
    4 partial success (results were produced, but some groups, periods or the
    requested report failed).
    """

    OK = 0
    RUNTIME = 1
    PREFLIGHT = 2
    NOT_FOUND = 3
    PARTIAL = 4

    @property
    def kind(self) -> str:
        """Lower-case name used as the ``kind`` field of JSON error documents."""
        return self.name.lower()


_FAILURE_KIND_TO_EXIT: dict[str, ExitCode] = {
    "runtime": ExitCode.RUNTIME,
    "preflight": ExitCode.PREFLIGHT,
    "not_found": ExitCode.NOT_FOUND,
}


def _classify_error(exc: SorethumbError) -> ExitCode:
    """Map a project error to its exit code via the exception's ``failure_kind``.

    Unknown kinds (a third-party subclass that sets something odd) fall back to
    the runtime code rather than inventing a new one.
    """
    return _FAILURE_KIND_TO_EXIT.get(getattr(exc, "failure_kind", "runtime"), ExitCode.RUNTIME)


def _fail(json_output: bool, message: str, code: ExitCode) -> NoReturn:
    """Report *message* as the command's failure and exit with *code*.

    JSON mode: a single document on stdout with the same shape for every
    machine-readable command and every failure class --
    ``{"error": <message>, "kind": <"runtime"|"preflight"|"not_found">,
    "exit_code": <int>}`` -- never Rich text on stderr a --json caller has no
    reason to read. Human mode: Rich-formatted text on stderr.
    """
    if json_output:
        typer.echo(json.dumps({"error": message, "kind": code.kind, "exit_code": int(code)}))
    else:
        err_console.print(f"[red]{_e(message)}[/red]")
    raise typer.Exit(int(code))


def _guard_legacy_dot_workspace(*, json_output: bool = False) -> None:
    """Refuse to silently switch a pre-existing "." workspace to the new default.

    Previously, the zero-config default workdir was ".". A directory that
    already has a `sorethumb.db` marker at "." is a workspace created under
    that old default; falling through to the new `_DEFAULT_WORKDIR` here
    would not touch or delete anything (non-destructive by construction --
    this function only ever reads), but it would silently make `sorethumb
    history`/`runs`/etc. stop seeing that workspace's existing runs, which is
    exactly the kind of surprise an explicit choice is supposed to prevent.
    Only fires when neither --workdir nor config's run.workdir was given --
    an explicit choice, in either direction, always wins outright.
    """
    if Path(_WORKSPACE_MARKER_FILENAME).exists():
        _fail(
            json_output,
            f"Found an existing workspace at '.' ({_WORKSPACE_MARKER_FILENAME}), but no "
            f"workdir is configured. sorethumb's zero-config default workspace changed "
            f"from '.' to './{_DEFAULT_WORKDIR}/' -- continuing would look for runs in "
            f"the new location and never see this one. Choose explicitly: keep using "
            f'this workspace by passing --workdir . (or setting run.workdir = "." in '
            f"sorethumb.toml), or migrate to the new default by moving its contents "
            f"into ./{_DEFAULT_WORKDIR}/ yourself and re-running without --workdir.",
            ExitCode.PREFLIGHT,
        )


def _load_config(
    config_path: Path | None,
    workdir: Path | None = None,
    seed: int | None = None,
    strict: bool | None = None,
    log_level: str | None = None,
    uri_override: str | None = None,
    detectors_override: list[dict[str, Any]] | None = None,
    *,
    json_output: bool = False,
) -> Config:
    """Read TOML, apply flag overrides, validate, and return Config.

    Validation errors are printed all at once — a config with eight problems
    shows eight problems, not just the first one.

    When *uri_override* is provided and no config file exists, an empty raw dict
    is used so the caller can proceed with defaults (workdir defaults to
    ``_DEFAULT_WORKDIR``, "./sorethumb-workspace/"). If a legacy workspace
    marker is found at "." in that case, refuses instead of silently
    resolving against the new default — see ``_guard_legacy_dot_workspace``.
    When *detectors_override* is provided it replaces the detectors list entirely.

    *strict* and *log_level* are ``None`` when the caller's CLI flag was not
    explicitly given (see ``_STRICT_OPT``/``_LOG_LEVEL_OPT``) — that is the
    only way to tell "not specified" apart from "explicitly set to the same
    value as the default", which a plain ``bool``/``str`` parameter can't.
    An explicit value always overrides TOML; when not given, TOML's own
    value (or the hardcoded default) applies untouched.

    *json_output*: callers from a JSON-capable command pass their own
    --json flag through here so a config error becomes a single JSON
    document on stdout instead of Rich text on stderr, matching every other
    error path in that command. Callers without a JSON mode leave it False.
    """
    import tomllib  # noqa: PLC0415 — stdlib, Python 3.11+

    from pydantic import ValidationError  # noqa: PLC0415

    if config_path is None:
        config_path = Path("sorethumb.toml")

    raw: dict[str, Any]
    if not config_path.exists():
        if uri_override is None:
            _fail(
                json_output,
                f"Config file not found: {config_path}. Run `sorethumb init` to create one.",
                ExitCode.PREFLIGHT,
            )
        raw = {}
    else:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)

    # URI override always wins (positional argument or explicit flag)
    if uri_override is not None:
        raw.setdefault("source", {})["uri"] = uri_override

    # Detectors override replaces the entire detectors list
    if detectors_override is not None:
        raw["detectors"] = detectors_override

    # Apply flag overrides (flags beat TOML, which beats env)
    run_section: dict[str, Any] = raw.setdefault("run", {})
    if workdir is not None:
        run_section["workdir"] = str(workdir)
    elif "workdir" not in run_section:
        _guard_legacy_dot_workspace(json_output=json_output)
        run_section["workdir"] = _DEFAULT_WORKDIR
    if seed is not None:
        run_section["seed"] = seed
    # An explicitly-passed flag overrides TOML outright; not passing one
    # (None) leaves whatever TOML already has, falling back to the
    # documented default only when TOML is silent too.
    if strict is not None:
        run_section["strict"] = strict
    else:
        run_section.setdefault("strict", False)
    if log_level is not None:
        run_section["log_level"] = log_level
    else:
        run_section.setdefault("log_level", "INFO")

    try:
        cfg = Config.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{' → '.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()]
        _fail(json_output, "Configuration errors: " + "; ".join(lines), ExitCode.PREFLIGHT)

    _add_file_handler(Path(cfg.run.workdir), cfg.run.log_level)
    return cfg


def _resolve_workdir(
    config_path: Path | None,
    workdir: Path | None,
    log_level: str | None,
    *,
    json_output: bool = False,
) -> tuple[Path, Config | None]:
    """Locate the workspace for a command that only reads it (or maintains it).

    These commands (``runs``, ``show``, ``anomalies``, ``report``, ``history``,
    ``explain-plan RUN_ID`` and ``workspace *``) never read the data source, so
    they don't need a ``sorethumb.toml``. Resolution:

    - ``--config`` given (or ``SORETHUMB_CONFIG`` set), or ``./sorethumb.toml``
      exists: load it exactly as every other command does, including exit 2
      when an explicitly named file is missing or invalid.
    - Otherwise: ``--workdir``, falling back to ``./sorethumb-workspace/`` --
      the same default ``sorethumb run DATA_FILE`` uses with no config, so the
      follow-up commands it suggests find the workspace it just wrote. The
      legacy ``.`` workspace guard still applies when no workdir is given.

    Returns ``(workdir, config)``; *config* is ``None`` in the second case, and
    callers use the run's own persisted settings or the library defaults.
    """
    if config_path is not None or Path("sorethumb.toml").exists():
        cfg = _load_config(config_path, workdir=workdir, log_level=log_level, json_output=json_output)
        return Path(cfg.run.workdir), cfg
    if workdir is None:
        _guard_legacy_dot_workspace(json_output=json_output)
        workdir = Path(_DEFAULT_WORKDIR)
    # Log to the workspace only when it already exists: a read-only command
    # pointed at a path with no workspace must not create one there.
    if workdir.is_dir():
        _add_file_handler(workdir, log_level or "INFO")
    return workdir, None


def _write_minimal_toml(path: Path, cfg: Config) -> None:
    """Write a full starter sorethumb.toml with required fields and detectors filled in."""
    from sorethumb_ml.io.toml_write import render_toml_key, render_toml_value  # noqa: PLC0415

    content = _generate_starter_toml()

    # Stamp the file as coming from `sorethumb run`, not `sorethumb init`
    content = content.replace(
        "# sorethumb.toml — generated by `sorethumb init`",
        "# sorethumb.toml — created by `sorethumb run`",
    )
    # Fill in the two required fields that have no default
    content = content.replace(
        "# uri =  # required — no default",
        f"uri = {render_toml_value(cfg.source.uri)}",
    )
    content = content.replace(
        "# workdir =  # required — no default",
        f"workdir = {render_toml_value(str(cfg.run.workdir))}",
    )
    # Replace the entire [[detectors]] section with the actual configured detectors
    marker = "# Detectors run as an ensemble; add or remove [[detectors]] blocks freely."
    preamble, _, _ = content.partition(marker)
    det_lines = [marker]
    for det in cfg.detectors:
        det_lines.append("")
        det_lines.append("[[detectors]]")
        det_lines.append(f"name = {render_toml_value(det.name)}")
        if not det.enabled:
            det_lines.append("enabled = false")
        if det.params:
            det_lines.append("[detectors.params]")
            for k, v in det.params.items():
                det_lines.append(f"{render_toml_key(k)} = {render_toml_value(v)}")
        if det.train_row_cap is not None:
            det_lines.append(f"train_row_cap = {det.train_row_cap}")
    det_lines.append("")
    content = preamble + "\n".join(det_lines)

    path.write_text(content, encoding="utf-8")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


class _WorkspaceLogHandler(logging.handlers.RotatingFileHandler):
    """The per-command file handler writing {workdir}/logs/sorethumb.log.

    Opened lazily (``delay=True``) and closed when the command finishes (see
    ``_add_file_handler``): Windows can't delete or rotate a file another
    handle still holds, so a handler left open blocks ``workspace reset`` and
    pins a workspace's log for the rest of an in-process session.

    Rotation renames the open log file, which fails on Windows while another
    sorethumb process has the same log open. Losing that race must not cost
    any log records, so a failed rollover keeps appending to the current file
    and is retried on a later record.
    """

    def doRollover(self) -> None:  # noqa: N802 -- logging's API name
        try:
            super().doRollover()
        except OSError:
            # Leave rotation for a later record. The stream may have been closed
            # by the base implementation before the rename failed; reopen it.
            if self.stream is None or self.stream.closed:
                self.stream = self._open()

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802 -- logging's API name
        try:
            return bool(super().shouldRollover(record))
        except OSError:
            return False


def _detach_file_handlers() -> None:
    """Close and remove every workspace log handler from the sorethumb_ml logger."""
    sorethumb_logger = logging.getLogger("sorethumb_ml")
    for handler in list(sorethumb_logger.handlers):
        if isinstance(handler, _WorkspaceLogHandler):
            sorethumb_logger.removeHandler(handler)
            handler.close()


def _add_file_handler(workdir: Path, level: str) -> None:
    """Log this command to {workdir}/logs/sorethumb.log, rotating at 10 MB, keeping 5 backups.

    One handler per command: the root callback (``main``) registers
    ``_detach_file_handlers`` to run when the command's context closes (normal
    exit, typer.Exit, or an exception), and a handler left over
    for a *different* workspace -- an earlier in-process invocation -- is
    replaced rather than reused, so records always land in the workspace the
    current command is using.
    """
    log_path = (workdir / "logs" / "sorethumb.log").resolve()
    sorethumb_logger = logging.getLogger("sorethumb_ml")
    for handler in sorethumb_logger.handlers:
        if isinstance(handler, _WorkspaceLogHandler) and Path(handler.baseFilename) == log_path:
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            return
    _detach_file_handlers()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = _WorkspaceLogHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    sorethumb_logger.addHandler(handler)


def _redact_config(config: Config) -> dict[str, Any]:
    """Return a config dict safe to print or echo back to the user.

    Config never holds a credential *value* -- ``SourceConfig.auth_env_var``
    is only the *name* of an environment variable read fresh at request
    time (see ``io/source.py``'s ``_build_auth_headers``), so there is
    nothing to strip there (see ``test_auth_token_not_in_config_json``).
    ``source.uri`` can itself carry embedded userinfo (``user:pass@host``)
    or a signed-download token in its query string, though -- redact that
    the same way run persistence already does
    (``_pipeline._redacted_config_json``), rather than invent a second,
    possibly-inconsistent redaction rule here.
    """
    from sorethumb_ml._pipeline import _redacted_config_json  # noqa: PLC0415

    raw: dict[str, Any] = json.loads(_redacted_config_json(config))
    return raw


# ---------------------------------------------------------------------------
# sorethumb init
# ---------------------------------------------------------------------------


def _render_toml_scalar(value: object) -> str | None:
    """Render a Python scalar as a TOML literal, or None if it has no literal form."""
    from sorethumb_ml.io.toml_write import render_toml_value  # noqa: PLC0415

    try:
        return render_toml_value(value)
    except TypeError:
        return None


def _detector_params_block(det_name: str, description: str) -> list[str]:
    """Render the detector's `params` line plus a commented `extra_params` catalogue."""
    import textwrap  # noqa: PLC0415

    from sorethumb_ml.detectors import registry  # noqa: PLC0415

    lines: list[str] = []
    desc = description.strip()
    if desc:
        lines.append(textwrap.fill(desc, 76, initial_indent="# ", subsequent_indent="# "))
    lines.append("params = {}")

    getter = getattr(registry.get(det_name), "available_extra_params", None)
    extras: dict[str, Any] = getter() if callable(getter) else {}
    if not extras:
        return lines

    first_key = next(iter(extras))
    lines += [
        "#",
        "# extra_params: forwarded verbatim to this detector's underlying scikit-learn",
        f"# estimator. Nest inside params, e.g. params = {{ extra_params = {{ {first_key} = ... }} }}",
        "# Every accepted key is listed below (commented) with its scikit-learn default:",
    ]
    for key, default in extras.items():
        rendered = _render_toml_scalar(default)
        lines.append(
            f"#   {key} = {rendered}" if rendered is not None else f"#   {key} =   # default: {default!r}"
        )
    return lines


def _starter_detectors_section(field_block: Callable[..., list[str]]) -> list[str]:
    """Render the `[[detectors]]` blocks for the three default detectors.

    ``train_row_cap`` is shown at each detector's built-in default (from the
    registry) so the starter file reflects the effective cap without a second
    hard-coded copy of the numbers.
    """
    from sorethumb_ml.config import DetectorConfig  # noqa: PLC0415
    from sorethumb_ml.detectors import registry  # noqa: PLC0415

    out = ["# Detectors run as an ensemble; add or remove [[detectors]] blocks freely."]
    for det in (
        DetectorConfig(name=name, train_row_cap=registry[name].default_train_row_cap)
        for name in ("isolation_forest", "kmeans_distance", "one_class_svm")
    ):
        out += ["", "[[detectors]]"]
        for i, (fname, fi) in enumerate(DetectorConfig.model_fields.items()):
            if i:
                out.append("")
            if fname == "params":
                out.extend(_detector_params_block(det.name, fi.description or ""))
            else:
                out.extend(field_block(fname, fi, override=getattr(det, fname)))
    out.append("")
    return out


def _generate_starter_toml() -> str:
    """Build a complete sorethumb.toml from the live Pydantic models.

    Every field is shown with its default value (or a commented placeholder
    for required/complex fields), plus its description as a TOML comment.
    """
    import textwrap  # noqa: PLC0415

    from pydantic.fields import FieldInfo  # noqa: PLC0415
    from pydantic_core import PydanticUndefined  # noqa: PLC0415

    from sorethumb_ml.config import (  # noqa: PLC0415
        ColumnsConfig,
        ExplainConfig,
        FeaturesConfig,
        HistoryConfig,
        ProfilingConfig,
        ReportConfig,
        RunConfig,
        ScoringConfig,
        SourceConfig,
    )
    from sorethumb_ml.io.toml_write import render_toml_value  # noqa: PLC0415

    _MISSING = object()

    def _scalar(v: object) -> str | None:
        try:
            return render_toml_value(v)
        except TypeError:
            return None

    def _field_block(name: str, fi: FieldInfo, override: object = _MISSING) -> list[str]:
        lines: list[str] = []
        desc = (fi.description or "").strip()
        if desc:
            wrapped = textwrap.fill(desc, 76, initial_indent="# ", subsequent_indent="# ")
            lines.append(wrapped)
        if override is not _MISSING:
            default = override
        elif fi.default is not PydanticUndefined:
            default = fi.default
        elif fi.default_factory is not None:
            try:
                default = fi.default_factory({})  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                default = _MISSING
        else:
            default = _MISSING
        if default is _MISSING:
            lines.append(f"# {name} =  # required — no default")
        elif default is None:
            lines.append(f"# {name} =  # optional, unset by default")
        else:
            toml_v = _scalar(default)
            if toml_v is not None:
                lines.append(f"{name} = {toml_v}")
            else:
                lines.append(f"# {name} =  # complex type, see docs")
        return lines

    def _section(header: str, model_cls: type[Any]) -> list[str]:
        lines = [f"[{header}]"]
        first = True
        for fname, fi in model_cls.model_fields.items():
            if not first:
                lines.append("")
            first = False
            lines.extend(_field_block(fname, fi))
        return lines

    out: list[str] = [
        "# sorethumb.toml — generated by `sorethumb init`",
        "# Every field is shown with its default value.",
        "# Fields marked '# optional' are unset by default; uncomment to override.",
        "# Fields marked '# required' must be set before `sorethumb run` will work.",
        "# Run `sorethumb config schema` for the full JSON schema.",
        "",
    ]
    for header, cls in [
        ("source", SourceConfig),
        ("columns", ColumnsConfig),
        ("profiling", ProfilingConfig),
        ("features", FeaturesConfig),
        ("scoring", ScoringConfig),
        ("explain", ExplainConfig),
        ("run", RunConfig),
        ("history", HistoryConfig),
        ("report", ReportConfig),
    ]:
        out.extend(_section(header, cls))
        out.append("")
    out.extend(_starter_detectors_section(_field_block))
    return "\n".join(out)


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
        err_console.print(f"[yellow]sorethumb.toml already exists:[/yellow] {_e(toml_path)}")
        raise typer.Exit(0)

    from sorethumb_ml.io.toml_write import render_toml_value  # noqa: PLC0415

    ws_dir = path / _DEFAULT_WORKDIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(False, f"Cannot create {path}: {exc}. Nothing was written.", ExitCode.RUNTIME)
    # Fill in the one field _generate_starter_toml() leaves as "required — no
    # default" that init already has a real answer for, so the written file
    # matches the workspace just created below rather than needing a manual
    # edit before it can be used.
    content = _generate_starter_toml().replace(
        "# workdir =  # required — no default",
        f"workdir = {render_toml_value(str(ws_dir))}",
    )
    try:
        toml_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        _fail(False, f"Cannot write {toml_path}: {exc}. Nothing was written.", ExitCode.RUNTIME)

    try:
        with Workspace.init(ws_dir):
            pass
    except Exception as exc:  # noqa: BLE001
        # Partial result, stated explicitly: the config file exists but the
        # workspace it points at does not. No success banner, non-zero exit.
        _fail(
            False,
            f"Workspace initialisation failed: {exc}. {toml_path} WAS written, but the workspace "
            f"{ws_dir} was not created. Fix the problem and remove {toml_path} before re-running "
            "`sorethumb init` (init leaves an existing sorethumb.toml untouched).",
            ExitCode.RUNTIME,
        )

    console.print(f"[green]Workspace created:[/green] {_e(ws_dir)}")
    console.print(f"[green]Config written:[/green] {_e(toml_path)}")
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
    log_level: _LOG_LEVEL_OPT = None,
    seed: _SEED_OPT = None,
) -> None:
    """Profile the dataset and print the feature plan without running any models.

    Shows every column's classification and the reason it was classified that way,
    plus the projected feature width and memory estimate. Use this before your
    first `sorethumb run` to check that high-cardinality columns will be encoded
    as expected and identifiers will be dropped.
    """
    _setup_logging(log_level or "INFO")
    cfg = _load_config(config, workdir=workdir, seed=seed, log_level=log_level)

    console.print("[bold]Loading dataset…[/bold]")
    ws_path = Path(cfg.run.workdir)
    cache_dir = ws_path / "cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.source, cache_dir=cache_dir)
    console.print(f"  rows={len(df):,}  cols={_e(len(df.columns))}")

    plan = build_feature_plan(df, cfg)

    table = Table(title="Feature plan", show_header=True, header_style="bold cyan")
    table.add_column("Column", style="white", no_wrap=True)
    table.add_column("Class", style="green")
    table.add_column("Treatment", style="yellow")
    table.add_column("Reason")

    for dec in plan.decisions or []:
        color = "red" if dec.treatment.value == "drop" else "green"
        _add_row(
            table,
            dec.column,
            dec.col_class.value,
            Text(dec.treatment.value, style=color),
            dec.reason or "",
        )

    console.print(table)

    n_features = len(plan.output_features)
    console.print(f"\nProjected feature width: [bold]{_e(n_features)}[/bold] columns")


# ---------------------------------------------------------------------------
# sorethumb run
# ---------------------------------------------------------------------------


@app.command()
def run(
    data_file: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Path to a data file. When supplied, overrides source.uri in the config. "
                "If no sorethumb.toml exists, all settings default and workdir defaults to "
                "'./sorethumb-workspace/' — you will be prompted to save a config file."
            )
        ),
    ] = None,
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Resolve the plan and register the run, but fit no models. "
                "Still writes: the workspace + schema migrations, the dataset and "
                "dataset_snapshot rows, and the run row (left in status 'running'). "
                "Skips: the feature plan, detector models, per-group results, "
                "history rows, and the report."
            ),
        ),
    ] = False,
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
    limit_groups: Annotated[
        int | None,
        typer.Option(
            "--limit-groups",
            help=(
                "Cap the number of groups processed, applied after --only-group/"
                "--group-filter. Groups are sorted by label first, so the same "
                "limit always keeps the same groups. Must be >= 1 when given."
            ),
        ),
    ] = None,
    detectors: Annotated[
        str | None,
        typer.Option(
            "--detectors",
            "-d",
            help=(
                "Comma-separated detector aliases to use, replacing the config list "
                "for this invocation only — the config file is never modified. "
                "Aliases: if=isolation_forest  km=kmeans_distance  oc=one_class_svm  "
                "ecod  lof  hbos. "
                "Full names are also accepted."
            ),
        ),
    ] = None,
    save_config: Annotated[
        bool | None,
        typer.Option(
            "--save-config/--no-save-config",
            help=(
                "With DATA_FILE and no sorethumb.toml: save (or don't save) this run's "
                "settings to sorethumb.toml without asking. Default: ask on an "
                "interactive terminal, don't save otherwise."
            ),
            show_default=False,
        ),
    ] = None,
    json_output: _JSON_OPT = False,
) -> None:
    """Run anomaly detection on the configured dataset.

    Groups that are already complete in the ledger are skipped unless --force
    is set. This makes repeated invocations cheap: the dataset snapshot cache
    avoids re-downloading, and the completion ledger avoids redundant inference.
    """
    _setup_logging(log_level or "INFO")

    config_path = config or Path("sorethumb.toml")
    config_existed = config_path.exists()

    detectors_override = _parse_detectors_flag(detectors) if detectors else None

    cfg = _load_config(
        config,
        workdir=workdir,
        seed=seed,
        strict=strict,
        log_level=log_level,
        uri_override=data_file,
        detectors_override=detectors_override,
        json_output=json_output,
    )

    if not config_existed and data_file is not None:
        _maybe_save_zero_config(config_path, cfg, save_config=save_config, json_output=json_output)

    # Validate group-filter regex up front so an invalid pattern fails before any work
    if group_filter:
        try:
            re.compile(group_filter)
        except re.error as exc:
            _fail(json_output, f"Invalid --group-filter regex: {exc}", ExitCode.PREFLIGHT)

    if not json_output:
        console.print(f"[bold]sorethumb run[/bold]  workspace={_e(cfg.run.workdir)}")
        if dry_run:
            console.print(
                "[yellow]DRY RUN[/yellow] — resolving the plan and registering the run; "
                "no models are fitted.\n"
                "  writes: workspace + schema migrations, the [cyan]dataset[/cyan] / "
                "[cyan]dataset_snapshot[/cyan] rows, and the [cyan]run[/cyan] row "
                "(status stays 'running').\n"
                "  skips:  feature plan, detector models, per-group results, history rows, report."
            )

    try:
        result: RunResult = run_detection(
            cfg,
            only_groups=only_group,
            group_filter_regex=group_filter,
            limit_groups=limit_groups,
            force=force,
            no_report=no_report,
            dry_run=dry_run,
            period_label_override=period,
        )
    except SorethumbError as exc:
        _fail(json_output, f"run failed: {exc}", _classify_error(exc))

    if json_output:
        typer.echo(json.dumps(_run_result_to_dict(result), default=str))
        raise typer.Exit(int(_exit_code_for(result)))

    _print_run_summary(result)

    raise typer.Exit(int(_exit_code_for(result)))


def _maybe_save_zero_config(
    config_path: Path, cfg: Config, *, save_config: bool | None, json_output: bool
) -> None:
    """Offer to save a zero-config run's settings, without ever blocking a script.

    ``--save-config``/``--no-save-config`` decide outright. Otherwise the user
    is asked only when stdin is an interactive terminal; under cron, CI or a
    pipe there is nobody to answer, so the answer is "no" and a note on stderr
    says how to save a config. ``--json`` never prompts and never prints to
    stdout: its caller gets exactly the run's JSON result.
    """
    if not json_output:
        console.print(
            f"[dim]No sorethumb.toml found — running with defaults, workdir={_e(repr(cfg.run.workdir))}.[/dim]"
        )
    if save_config is None:
        if json_output:
            return
        if not sys.stdin.isatty():
            err_console.print(
                "Not saving settings (stdin is not a terminal). Run `sorethumb init` to create "
                "sorethumb.toml, or pass --save-config.",
                markup=False,
                highlight=False,
            )
            return
        save_config = typer.confirm("Save settings to sorethumb.toml for future runs?", default=False)
    if save_config:
        _write_minimal_toml(config_path, cfg)
        if not json_output:
            console.print(f"[green]Saved {_e(config_path)}[/green]")


# ---------------------------------------------------------------------------
# sorethumb score
# ---------------------------------------------------------------------------


@app.command()
def score(
    from_run: Annotated[str, typer.Option("--from-run", help="Source run_id to reuse models from.")],
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = None,
    no_report: Annotated[bool, typer.Option("--no-report")] = False,
    json_output: _JSON_OPT = False,
) -> None:
    """Score new data with a previous run's persisted plan and models.

    The source run's fitted FeaturePlan and per-detector models + calibrators are
    loaded and applied to the new data without re-fitting. The calibrators map
    scores onto the source run's reference distribution, so the numbers are
    comparable across runs. Schema and library-version drift are detected per
    group (``--strict`` makes them errors). A new, distinct run is written that
    records the source run.

    Loading a run unpickles its persisted estimator and calibrator files
    (joblib), which is code execution, not sandboxed data loading. File
    digests only catch corruption or a swapped file, not a deliberately
    malicious one. Only use --from-run against a workspace you created
    yourself or fully trust — see SECURITY.md.
    """
    _setup_logging(log_level or "INFO")
    cfg = _load_config(
        config, workdir=workdir, seed=seed, strict=strict, log_level=log_level, json_output=json_output
    )

    if not json_output:
        console.print(f"[bold]sorethumb score[/bold]  from_run={_e(from_run)}")

    try:
        result: RunResult = score_forward(cfg, from_run, strict=cfg.run.strict, no_report=no_report)
    except SorethumbError as exc:
        _fail(json_output, f"score --from-run failed: {exc}", _classify_error(exc))

    if json_output:
        typer.echo(json.dumps(_run_result_to_dict(result), default=str))
        raise typer.Exit(int(_exit_code_for(result)))

    _print_run_summary(result)
    if any(g.drifted for g in result.groups):
        console.print("[yellow]note:[/yellow] one or more groups showed schema/version drift.")
    raise typer.Exit(int(_exit_code_for(result)))


# ---------------------------------------------------------------------------
# sorethumb report
# ---------------------------------------------------------------------------


@app.command()
def report(
    run_id: Annotated[str | None, typer.Argument(help="Run ID to re-render (default: latest run).")] = None,
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
) -> None:
    """Re-render a run's HTML report from persisted results — no recompute.

    Reads the run's stored config, FeaturePlan and per-group results Parquet
    -- never today's ``--config``/``sorethumb.toml`` -- and rewrites
    ``{workdir}/reports/{run_id}/index.html``. The persisted results were
    computed against that historical plan and group structure, so anything
    that could change what they *mean* (columns, detectors, scoring, ...)
    always comes from the run's own history, never from the current config;
    otherwise a re-render could silently stop corresponding to the data it's
    rendering. ``report.formats`` is the one deliberate exception -- purely
    cosmetic (which output files get written), so it *is* taken from the
    current config: change it and re-run this command to pick up the new
    formats, or to rebuild a report that was deleted. ``--config``/
    ``--workdir`` otherwise only locate the workspace the run lives in.
    """
    _setup_logging(log_level or "INFO")
    from sorethumb_ml.config import ReportConfig  # noqa: PLC0415

    ws_path, cfg = _resolve_workdir(config, workdir, log_level)
    formats = cfg.report.formats if cfg is not None else ReportConfig().formats

    try:
        ws_cm = Workspace.open(ws_path)
    except SorethumbError as exc:
        _fail(False, str(exc), _classify_error(exc))
    with ws_cm as ws:
        if run_id is None:
            runs = ws.store.list_runs(limit=1)
            if not runs:
                _fail(False, "No runs found in workspace.", ExitCode.NOT_FOUND)
            run_id = str(runs[0]["run_id"])

        if ws.store.get_run(run_id) is None:
            _fail(False, f"Run not found: {run_id}", ExitCode.NOT_FOUND)

        n_groups = len(ws.store.all_run_groups(run_id))
        console.print(f"Re-rendering report for [cyan]{_e(run_id)}[/cyan] ({_e(n_groups)} groups)…")
        from sorethumb_ml._pipeline import _render_report_or_reason  # noqa: PLC0415

        path, reason = _render_report_or_reason(ws, run_id, formats=formats)
        if path is None:
            _fail(
                False,
                f"Could not render report for {run_id}: {reason}. See the log for details.",
                ExitCode.RUNTIME,
            )
        console.print(f"[green]Report written:[/green] {_e(path)}")


# ---------------------------------------------------------------------------
# sorethumb backfill
# ---------------------------------------------------------------------------


@app.command()
def backfill(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    seed: _SEED_OPT = None,
    strict: _STRICT_OPT = None,
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

    Each pending period is fitted and scored independently (a full ``sorethumb
    run`` over that period's window) and self-calibrated. The ``sorethumb
    history`` trend that results shows relative period-to-period movement, not an
    absolute anomaly level on a shared scale — for that, score every period
    against one fixed run with ``sorethumb score --from-run``.
    """
    _setup_logging(log_level or "INFO")
    cfg = _load_config(config, workdir=workdir, seed=seed, strict=strict, log_level=log_level)

    if not cfg.columns.time_column:
        console.print(
            "[yellow]No time_column configured — backfill is only meaningful with a time series "
            "dataset. Exiting.[/yellow]"
        )
        raise typer.Exit(0)

    from sorethumb_ml.history.ledger import iter_pending_periods, resolve_backfill_range  # noqa: PLC0415

    ws_path = Path(cfg.run.workdir)
    # Match `run`: open an existing workspace, otherwise create it so `backfill`
    # works as a first command on a fresh workdir.
    if ws_path.exists() and (ws_path / "sorethumb.db").exists():
        ws_cm = Workspace.open(ws_path)
    else:
        ws_cm = Workspace.init(ws_path)
    with ws_cm as ws:
        from datetime import datetime  # noqa: PLC0415

        from sorethumb_ml.history.periods import resolve_period  # noqa: PLC0415
        from sorethumb_ml.io.fingerprint import logical_dataset_id  # noqa: PLC0415

        # History is keyed on the stable logical dataset id -- the same value
        # run_detection will register -- so backfill's pending-period math does
        # not depend on the current snapshot's content.
        dataset_fp = logical_dataset_id(cfg.source.dataset_id, cfg.source.uri)
        config_hash = cfg.config_hash()

        ref = datetime.now(UTC)
        _, _, ref_label = resolve_period(ref, cfg.history.period_granularity, cfg.history.roll_non_business)

        backfill_labels = resolve_backfill_range(
            ws.store,
            dataset_fp,
            config_hash,
            ref_label,
            cfg.history.period_granularity,
            cfg.history.bootstrap_periods,
            cfg.history.lookback_periods,
            max_periods or cfg.history.max_backfill_periods,
            roll_non_business=cfg.history.roll_non_business,
        )
        pending = iter_pending_periods(ws.store, dataset_fp, config_hash, backfill_labels, force_period or [])

        if not pending:
            console.print(
                f"[green]Nothing to backfill for config {_e(config_hash[:8])} — all periods are up to date.[/green]"
            )
            raise typer.Exit(0)

        console.print(f"Backfill: {_e(len(pending))} pending periods (config {_e(config_hash[:8])})")
        if dry_run:
            for lbl in pending:
                console.print(f"  [dim]would process:[/dim] {_e(lbl)}")
            raise typer.Exit(0)

        # Each period runs independently, so one period failing must not stop the
        # rest, and the command must still exit non-zero and name every period that
        # failed. Two kinds of failure are kept apart: a period whose run finished
        # with failed *groups* (a RunResult with n_failed), and a period whose run
        # *raised* a project error before producing a result (source unreadable,
        # store error, ...), which has no groups to list.
        failed_groups: list[tuple[str, RunResult]] = []
        raised: list[tuple[str, str]] = []
        n_ok = 0
        for period_lbl in pending:
            console.print(f"  Processing period [cyan]{_e(period_lbl)}[/cyan]…")
            try:
                result = run_detection(cfg, period_label_override=period_lbl, no_report=True)
            except SorethumbError as exc:
                raised.append((period_lbl, f"{type(exc).__name__}: {exc}"))
                err_console.print(f"  [red]Period {_e(period_lbl)} raised an error:[/red] {_e(exc)}")
                continue
            if result.n_failed:
                failed_groups.append((period_lbl, result))
            else:
                n_ok += 1

        if failed_groups or raised:
            err_console.print(
                f"\n[red bold]Backfill finished with {_e(len(failed_groups) + len(raised))} failed period(s) "
                f"of {_e(len(pending))} ({_e(n_ok)} succeeded):[/red bold]"
            )
            for period_lbl, result in failed_groups:
                names = [g.group_label for g in result.groups if g.status == "failed"]
                err_console.print(f"  {_e(period_lbl)}: failed group(s): {_e(', '.join(names))}")
            for period_lbl, message in raised:
                err_console.print(f"  {_e(period_lbl)}: raised {_e(message)}")
            produced_results = n_ok > 0 or any(r.n_succeeded for _, r in failed_groups)
            raise typer.Exit(int(ExitCode.PARTIAL if produced_results else ExitCode.RUNTIME))

        console.print("[green]Backfill complete.[/green]")


# ---------------------------------------------------------------------------
# sorethumb history
# ---------------------------------------------------------------------------


@app.command()
def history(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    windows: Annotated[
        list[int] | None, typer.Option("--window", help="Rolling window sizes (e.g. --window 7 --window 28).")
    ] = None,
    group_key: Annotated[str | None, typer.Option("--group", help="Limit to a specific group key.")] = None,
) -> None:
    """Show rolling-window anomaly trends for the configured dataset.

    With no sorethumb.toml (and no --config), shows the dataset and
    configuration of the workspace's most recent run, using that run's own
    history and report settings.
    """
    _setup_logging(log_level or "INFO")
    ws_path, loaded = _resolve_workdir(config, workdir, log_level)

    try:
        ws_cm = Workspace.open(ws_path)
    except SorethumbError as exc:
        _fail(False, str(exc), _classify_error(exc))
    with ws_cm as ws:
        from sorethumb_ml.io.fingerprint import logical_dataset_id  # noqa: PLC0415

        # Trends are read straight from the ledger, keyed on the stable logical
        # dataset id -- no need to touch the source file. Aggregation is scoped
        # to this config's hash -- a different configuration's totals for the
        # same periods are separate rows and are never silently folded in.
        if loaded is not None:
            cfg = loaded
            dataset_fp = logical_dataset_id(cfg.source.dataset_id, cfg.source.uri)
            config_hash = cfg.config_hash()
        else:
            latest = ws.store.list_runs(limit=1)
            if not latest:
                _fail(False, "No runs found in workspace.", ExitCode.NOT_FOUND)
            run_row = latest[0]
            try:
                cfg = Config.model_validate_json(run_row["config_json"])
            except Exception as exc:  # noqa: BLE001 -- any unreadable stored config is the same failure
                _fail(
                    False,
                    f"The latest run's stored config could not be read ({exc}); pass --config.",
                    ExitCode.RUNTIME,
                )
            dataset_fp = str(run_row["dataset_fp"])
            config_hash = str(run_row["config_hash"])
        _windows = windows or cfg.report.rolling_windows

        from datetime import datetime  # noqa: PLC0415

        from sorethumb_ml.history.periods import resolve_period  # noqa: PLC0415
        from sorethumb_ml.history.windows import compute_rolling_windows  # noqa: PLC0415

        ref = datetime.now(UTC)
        _, _, ref_label = resolve_period(ref, cfg.history.period_granularity, cfg.history.roll_non_business)

        group_keys = [group_key] if group_key else None
        window_results = compute_rolling_windows(
            ws.store,
            dataset_fp,
            config_hash,
            ref_label,
            _windows,
            cfg.history.period_granularity,
            group_keys=group_keys,
        )

        if not window_results:
            console.print(
                f"[yellow]No history available yet for this dataset under config {_e(config_hash[:8])}.[/yellow]"
            )
            raise typer.Exit(0)

        table = Table(
            title=f"Rolling windows (ref={_e(ref_label)}, config={_e(config_hash[:8])})", show_header=True
        )
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

            _add_row(
                table,
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
    log_level: _LOG_LEVEL_OPT = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of runs to show.")] = 20,
    json_output: _JSON_OPT = False,
) -> None:
    """List recent runs with status, dataset, group counts, and duration."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level, json_output=json_output)

    try:
        with Workspace.open(ws_path) as ws:
            runs = ws.store.list_runs(limit=limit)
    except SorethumbError as exc:
        _fail(json_output, str(exc), _classify_error(exc))

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
        _add_row(
            table,
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
    log_level: _LOG_LEVEL_OPT = None,
    group: Annotated[str | None, typer.Option("--group", help="Group key to show detail for.")] = None,
    json_output: _JSON_OPT = False,
) -> None:
    """Show detail for one run or group, including feature plan summary."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level, json_output=json_output)

    try:
        with Workspace.open(ws_path) as ws:
            run_row = ws.store.get_run(run_id)
            if run_row is None:
                _fail(json_output, f"Run not found: {run_id}", ExitCode.NOT_FOUND)

            groups = ws.store.all_run_groups(run_id)
    except SorethumbError as exc:
        _fail(json_output, str(exc), _classify_error(exc))

    if group:
        groups = [g for g in groups if g.get("group_key") == group]

    if json_output:
        typer.echo(json.dumps({"run": run_row, "groups": groups}, default=str))
        return

    console.print(f"[bold]Run:[/bold] {_e(run_id)}")
    console.print(f"  Status:  {_e(run_row.get('status'))}")
    console.print(f"  Dataset: {_e(run_row.get('dataset_fp', '')[:12])}")
    console.print(f"  Started: {_e(str(run_row.get('started_at', ''))[:19])}")
    console.print(f"  Config:  {_e(run_row.get('config_hash', '')[:8])}")

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
        _add_row(
            table,
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
    log_level: _LOG_LEVEL_OPT = None,
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

    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level, json_output=json_output)

    try:
        with Workspace.open(ws_path) as ws:
            if run_id is None:
                recent = ws.store.list_runs(limit=1)
                if not recent:
                    _fail(json_output, "No runs found in this workspace.", ExitCode.NOT_FOUND)
                run_id = str(recent[0]["run_id"])

            groups = ws.store.all_run_groups(run_id)
            if not groups:
                _fail(json_output, f"Run not found or has no groups: {run_id}", ExitCode.NOT_FOUND)

            frames: list[pl.DataFrame] = []
            for g in groups:
                parquet = ws.results_dir(run_id, g["group_key"]) / "anomalies.parquet"
                if parquet.exists():
                    df = pl.read_parquet(str(parquet))
                    if len(df) > 0:
                        frames.append(df.with_columns(pl.lit(str(g.get("group_label", ""))).alias("_group")))
    except SorethumbError as exc:
        _fail(json_output, str(exc), _classify_error(exc))

    if not frames:
        if json_output:
            typer.echo("[]")
            raise typer.Exit(0)
        console.print(f"[yellow]No anomaly rows found for run {_e(run_id)}.[/yellow]")
        raise typer.Exit(0)

    all_rows = pl.concat(frames, how="diagonal").sort("rank")
    if top:
        all_rows = all_rows.head(top)

    reason_cols = [f"reason_{i + 1}" for i in range(reasons) if f"reason_{i + 1}" in all_rows.columns]
    multi_group = all_rows["_group"].n_unique() > 1

    if json_output:
        display = ["_group", "rank", "composite_score", "attribution_kind", *reason_cols]
        present = [c for c in display if c in all_rows.columns]
        # Re-encoded through json.dumps (ensure_ascii) like every other --json
        # output: polars writes non-ASCII as raw characters, which a narrow
        # output encoding (PYTHONIOENCODING=cp1252) would turn into "?".
        typer.echo(json.dumps(json.loads(all_rows.select(present).rename({"_group": "group"}).write_json())))
        return

    table = Table(
        title=f"Anomalies — run {_e(run_id[:12])}",
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
        _add_row(table, *cells)

    console.print(table)
    console.print(
        f"  [dim]{_e(len(all_rows))} anomaly row(s)   run={_e(run_id)}   workspace={_e(ws_path)}[/dim]"
    )


# ---------------------------------------------------------------------------
# sorethumb explain-plan
# ---------------------------------------------------------------------------


@app.command(name="explain-plan")
def explain_plan(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run ID whose persisted plan to show (default: plan the current source data)."),
    ] = None,
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    json_output: _JSON_OPT = False,
) -> None:
    """Print the FeaturePlan -- what was dropped, encoded, derived, and why.

    With a RUN_ID, shows the plan that run was actually fitted with (loaded
    from the selected workspace). Without one, plans the current source data
    under the current config, which may differ from any historical run.
    """
    from sorethumb_ml.store.models import load_plan  # noqa: PLC0415

    _setup_logging(log_level or "INFO")
    if run_id is not None:
        ws_path, cfg_or_none = _resolve_workdir(config, workdir, log_level, json_output=json_output)
    else:
        # Planning the current source data needs a data source, so a config.
        cfg_or_none = _load_config(config, workdir=workdir, log_level=log_level, json_output=json_output)
        ws_path = Path(cfg_or_none.run.workdir)

    if run_id is not None:
        try:
            with Workspace.open(ws_path) as ws:
                if ws.store.get_run(run_id) is None:
                    _fail(json_output, f"Run not found: {run_id}", ExitCode.NOT_FOUND)
                plan = load_plan(ws, run_id)
        except SorethumbError as exc:
            _fail(json_output, str(exc), _classify_error(exc))
    else:
        cfg = cfg_or_none
        assert cfg is not None  # loaded above whenever run_id is None
        cache_dir = ws_path / "cache" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            df = load_dataset(cfg.source, cache_dir=cache_dir)
            plan = build_feature_plan(df, cfg)
        except SorethumbError as exc:
            _fail(json_output, str(exc), _classify_error(exc))

    if json_output:
        typer.echo(json.dumps(json.loads(plan.to_json())))  # ASCII-escaped, like every --json output
        return

    table = Table(title="Feature plan", show_header=True, header_style="bold cyan")
    table.add_column("Column")
    table.add_column("Class", style="yellow")
    table.add_column("Treatment", style="cyan")
    table.add_column("Reason")

    for dec in plan.decisions or []:
        _add_row(table, dec.column, dec.col_class.value, dec.treatment.value, dec.reason or "")
    console.print(table)

    console.print(f"\n[bold]Output features:[/bold] {_e(len(plan.output_features))}")


# ---------------------------------------------------------------------------
# sorethumb detectors
# ---------------------------------------------------------------------------


@app.command()
def detectors(
    json_output: _JSON_OPT = False,
) -> None:
    """List all registered detectors (built-in and third-party extensions)."""
    names = list_detectors()
    from sorethumb_ml.detectors import registry as _reg  # noqa: PLC0415

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
        _add_row(
            table,
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
    json_output: _JSON_OPT = False,
) -> None:
    """Validate a config file and report every error at once.

    Use --json to print the fully-resolved config (defaults, env vars, and
    CLI overrides all applied) as JSON, with credentials redacted -- useful
    to inspect what a run would actually use before there's a run to
    `config show` from.
    """
    cfg = _load_config(config, workdir=workdir, json_output=json_output)
    if json_output:
        typer.echo(json.dumps(_redact_config(cfg), indent=2))
        return
    console.print("[green]Config is valid.[/green]")
    console.print(f"  workdir: {_e(cfg.run.workdir)}")
    console.print(f"  detectors: {_e([d.name for d in cfg.detectors if d.enabled])}")


@config_app.command(name="schema")
def config_schema(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write schema to this file.")] = None,
) -> None:
    """Emit the JSON schema for sorethumb.toml."""
    schema = Config.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    if output:
        output.write_text(schema_str, encoding="utf-8")
        console.print(f"[green]Schema written:[/green] {_e(output)}")
    else:
        typer.echo(schema_str)


@config_app.command(name="show")
def config_show(
    run_id: Annotated[str, typer.Argument(help="Run ID whose config to display.")],
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    json_output: _JSON_OPT = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write config as a re-usable sorethumb.toml to this path."),
    ] = None,
) -> None:
    """Display the exact config used for a past run.

    By default prints a human-readable summary. Use --json for the raw JSON
    or --output <path> to reconstruct a sorethumb.toml you can edit and re-run.
    """
    cfg = _load_config(config, workdir=workdir, json_output=json_output)
    ws_path = Path(cfg.run.workdir)

    try:
        with Workspace.open(ws_path) as ws:
            run_row = ws.store.get_run(run_id)
    except SorethumbError as exc:
        _fail(json_output, str(exc), _classify_error(exc))

    if run_row is None:
        _fail(json_output, f"Run not found: {run_id}", ExitCode.NOT_FOUND)

    config_json: str = run_row.get("config_json") or "{}"

    if json_output:
        typer.echo(json.dumps(json.loads(config_json), indent=2))
        return

    run_cfg = Config.model_validate_json(config_json)

    if output is not None:
        _write_minimal_toml(output, run_cfg)
        console.print(f"[green]Config written:[/green] {_e(output)}")
        return

    config_hash = run_row.get("config_hash", "")
    console.print(f"[bold]Config for run:[/bold] {_e(run_id)}  [dim](hash: {_e(config_hash[:8])})[/dim]")
    console.print(f"  Source URI: {_e(run_cfg.source.uri)}")
    console.print(f"  Workdir:    {_e(run_cfg.run.workdir)}")
    console.print(f"  Seed:       {_e(run_cfg.run.seed)}")

    det_table = Table(title="Detectors", show_header=True, header_style="bold cyan")
    det_table.add_column("Name", style="white")
    det_table.add_column("Enabled", justify="center")
    det_table.add_column("Train row cap", justify="right")
    det_table.add_column("Params")

    for det in run_cfg.detectors:
        cap = str(det.train_row_cap) if det.train_row_cap is not None else "default"
        params_str = ", ".join(f"{k}={v}" for k, v in (det.params or {}).items()) or "—"
        _add_row(
            det_table,
            det.name,
            "yes" if det.enabled else "no",
            cap,
            params_str,
        )
    console.print(det_table)

    scoring = run_cfg.scoring
    console.print("[bold]Scoring:[/bold]")
    console.print(f"  combination   = {_e(scoring.combination)}")
    console.print(f"  contamination = {_e(scoring.contamination)}")

    console.print("\n[dim]Use --json for the full config or --output <path> to save as sorethumb.toml[/dim]")


# ---------------------------------------------------------------------------
# sorethumb workspace *
# ---------------------------------------------------------------------------


@workspace_app.command(name="ls")
def workspace_ls(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    json_output: _JSON_OPT = False,
) -> None:
    """List runs, datasets, and artefact counts in the workspace."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level, json_output=json_output)

    try:
        with Workspace.open(ws_path) as ws:
            runs = ws.store.list_runs(limit=50)
    except SorethumbError as exc:
        _fail(json_output, str(exc), _classify_error(exc))

    if json_output:
        typer.echo(json.dumps({"runs": runs}, default=str))
        return

    console.print(f"[bold]Workspace:[/bold] {_e(ws_path)}")
    console.print(f"  Runs: {_e(len(runs))}")

    if runs:
        table = Table(show_header=True)
        table.add_column("Run ID")
        table.add_column("Status")
        table.add_column("Started")
        for r in runs[:10]:
            _add_row(
                table,
                str(r.get("run_id", ""))[:24],
                str(r.get("status", "")),
                str(r.get("started_at", ""))[:19],
            )
        console.print(table)


@workspace_app.command(name="du")
def workspace_du(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
) -> None:
    """Show disk usage broken down by regenerable vs non-regenerable artefacts."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level)

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

    console.print(f"[bold]Workspace:[/bold] {_e(ws_path)}")
    console.print(f"  Total: {_e(_fmt(total_bytes))}")


@workspace_app.command(name="prune")
def workspace_prune(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    days: Annotated[int, typer.Option("--days", help="Retention window in days.")] = 90,
    dry_run: _DRY_RUN_OPT = False,
) -> None:
    """Remove regenerable artefacts and failed runs older than --days.

    --dry-run prints what would be removed. A real prune removes files and
    database rows together — never one without the other.
    """
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level)

    try:
        with Workspace.open(ws_path) as ws:
            removed = ws.prune(days, dry_run=dry_run)
    except SorethumbError as exc:
        _fail(False, f"workspace prune failed: {exc}", _classify_error(exc))

    prefix = "Would remove" if dry_run else "Removed"
    for item in removed:
        console.print(f"  {_e(prefix)}: {_e(item)}")
    console.print(f"[green]{_e(prefix)} {_e(len(removed))} item(s).[/green]")


@workspace_app.command(name="vacuum")
def workspace_vacuum(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
) -> None:
    """Run SQLite VACUUM and reconcile orphan files with no database row."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level)

    with Workspace.open(ws_path) as ws:
        ws.store.vacuum()
    console.print("[green]Vacuum complete.[/green]")


@workspace_app.command(name="migrate")
def workspace_migrate(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    dry_run: _DRY_RUN_OPT = False,
) -> None:
    """Apply pending schema migrations to the workspace database."""
    _setup_logging(log_level or "INFO")
    ws_path, _ = _resolve_workdir(config, workdir, log_level)

    if dry_run:
        console.print("[yellow]DRY RUN — no migrations applied.[/yellow]")
        return

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


# Minimum path segments after the filesystem/drive anchor a reset target
# must have. Below this, a path is too shallow to plausibly be a dedicated,
# disposable workspace directory rather than an important top-level location
# reached by a typo or a misconfigured workdir (e.g. "/Users", "/home").
_MIN_RESET_DEPTH = 3


def _guard_reset_target(ws_path: Path) -> None:
    """Refuse `workspace reset` against a path that is almost certainly not a disposable workspace, regardless of --yes.

    Applied *in addition to* -- never instead of -- verifying the target
    actually opens as a Workspace: a marker file existing inside one of
    these locations, however unlikely, would still not make deleting it
    safe.
    """
    reasons: list[str] = []
    if ws_path == Path(ws_path.anchor):
        reasons.append("it is a filesystem/drive root")
    if ws_path == Path.home().resolve():
        reasons.append("it is your home directory")
    if ws_path == Path.cwd().resolve():
        reasons.append("it is the current working directory")
    if (ws_path / ".git").exists():
        reasons.append("it looks like a git repository root (contains .git)")
    depth = len(ws_path.parts) - 1  # segments after the anchor
    if depth < _MIN_RESET_DEPTH:
        reasons.append(
            f"it is only {depth} path segment(s) below the root (minimum "
            f"{_MIN_RESET_DEPTH}) -- too shallow to plausibly be a dedicated "
            "workspace directory"
        )
    if reasons:
        err_console.print(f"[red]Refusing to reset {_e(ws_path)}:[/red]")
        for reason in reasons:
            err_console.print(f"  - {_e(reason)}")
        raise typer.Exit(int(ExitCode.PREFLIGHT))


@workspace_app.command(name="reset")
def workspace_reset(
    config: _CONFIG_OPT = None,
    workdir: _WORKDIR_OPT = None,
    log_level: _LOG_LEVEL_OPT = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip interactive confirmation (for unattended use)."),
    ] = False,
) -> None:
    """Destructively delete everything sorethumb stored in the workspace.

    Deletes only what sorethumb creates there: sorethumb.db (and its
    -wal/-shm/-journal files) and the cache/, logs/, models/, reports/,
    results/ and tmp/ directories. Anything else in the directory -- your own
    data, notebooks, notes -- is left alone, and the directory itself is
    removed only if nothing else remains. A symlink among those entries is
    removed, never followed.

    Requires interactive confirmation of the workspace path (or --yes for
    unattended use). Named explicitly so you know exactly what will be destroyed
    before it happens.

    Refuses outright -- regardless of --yes -- unless the target actually
    opens as a sorethumb workspace, and rejects a filesystem/drive root, your
    home directory, the current working directory, a git repository root, or
    a suspiciously shallow path (see _guard_reset_target).
    """
    _setup_logging(log_level or "INFO")
    ws_root, _ = _resolve_workdir(config, workdir, log_level)
    ws_path = ws_root.resolve()

    _guard_reset_target(ws_path)

    try:
        with Workspace.open(ws_path):
            pass
    except SorethumbError as exc:
        _fail(False, f"Refusing to reset {ws_path}: not a sorethumb workspace ({exc})", _classify_error(exc))

    from sorethumb_ml.store.workspace import owned_entry_names  # noqa: PLC0415

    owned = [
        ws_path / name
        for name in owned_entry_names()
        if (ws_path / name).is_symlink() or (ws_path / name).exists()
    ]
    console.print(f"[red bold]This will destroy sorethumb's data in:[/red bold] {_e(ws_path)}")
    for entry in owned:
        console.print(f"  - {_e(entry.name)}{'/' if entry.is_dir() and not entry.is_symlink() else ''}")
    if not yes:
        console.print("Type the full workspace path to confirm (Ctrl-C to abort):")
        try:
            typed = input("> ").strip()
        except EOFError:
            # Closed stdin must terminate cleanly, not crash with a
            # traceback -- same outcome as typing the wrong path: abort.
            err_console.print("[red]No input available (stdin closed). Aborting.[/red]")
            raise typer.Exit(int(ExitCode.RUNTIME)) from None
        if typed != str(ws_path):
            err_console.print("[red]Path did not match. Aborting.[/red]")
            raise typer.Exit(int(ExitCode.RUNTIME))

    import shutil  # noqa: PLC0415

    from sorethumb_ml._atomic import unlink_with_retry  # noqa: PLC0415

    # This command's own log handler holds logs/sorethumb.log open; Windows
    # refuses to delete an open file, so close it before deleting anything.
    _detach_file_handlers()
    try:
        for entry in owned:
            # A symlink is removed itself, never followed; rmtree never follows
            # symlinks inside the directories it deletes either.
            if entry.is_symlink() or not entry.is_dir():
                unlink_with_retry(entry)
            else:
                shutil.rmtree(entry)
        remaining = sorted(ws_path.iterdir())
        if not remaining:
            ws_path.rmdir()
    except OSError as exc:
        err_console.print(f"[red]Failed to fully delete {_e(ws_path)}:[/red] {_e(exc)}")
        raise typer.Exit(int(ExitCode.RUNTIME)) from exc

    if remaining:
        console.print(
            f"[green]Workspace data deleted;[/green] kept {_e(ws_path)} because it holds other files:"
        )
        for entry in remaining:
            console.print(f"  - {_e(entry.name)}")
    else:
        console.print(f"[green]Workspace destroyed:[/green] {_e(ws_path)}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _exit_code_for(result: RunResult) -> ExitCode:
    """Classify a finished run into one of the documented exit codes.

    - ``PREFLIGHT`` (2): a group selector (--only-group/--group-filter) matched
      nothing, so zero groups were processed -- almost certainly a typo, and
      distinct from "nothing to do, all fine" (an empty groups list makes
      ``n_failed`` 0 too).
    - ``PARTIAL`` (4): results exist but something is missing: some groups failed
      while at least one succeeded, or every group succeeded and the explicitly
      requested report failed to render (the detection results are still right,
      but a success-shaped exit would hide the missing report).
    - ``RUNTIME`` (1): groups were attempted and none succeeded.
    - ``OK`` (0) otherwise.
    """
    if result.group_selection_error:
        return ExitCode.PREFLIGHT
    if result.n_failed:
        return ExitCode.PARTIAL if result.n_succeeded else ExitCode.RUNTIME
    if result.report_status == "failed":
        return ExitCode.PARTIAL
    return ExitCode.OK


def _print_run_summary(result: RunResult) -> None:
    status_color = "red" if (result.n_failed or result.group_selection_error) else "green"
    console.print(
        f"\n[{status_color}]Run {_e(result.run_id)}[/{status_color}]  "
        f"succeeded={_e(result.n_succeeded)}  skipped={_e(result.n_skipped)}  failed={_e(result.n_failed)}"
    )
    if result.group_selection_error:
        err_console.print(f"  [red]{_e(result.group_selection_error)}[/red]")
    if result.source_run_id:
        console.print(f"  [dim]Scored forward from run {_e(result.source_run_id)} (no re-fitting).[/dim]")

    total_rows = sum(g.n_records for g in result.groups if g.status in ("success", "skipped"))
    if result.n_anomalies or total_rows:
        pct = (
            f" ({100.0 * result.n_anomalies / total_rows:.2f}% of {total_rows:,} rows)" if total_rows else ""
        )
        console.print(
            f"  Flagged for review: {result.n_anomalies:,}{_e(pct)} — "
            "the review shortlist at the chosen budget, [dim]not an estimate of true prevalence[/dim]"
        )

    # Realised per-detector rates: what each detector's own heuristic boundary
    # flagged. contamination='auto' is the median of these — surfaced so it is
    # not read as "how many anomalies you have".
    rate_groups = [g for g in result.groups if g.detector_flag_rates]
    if rate_groups:
        console.print("\n  [bold]Realised detector flag rates[/bold] (own natural boundary):")
        for g in rate_groups[:8]:
            parts = ", ".join(f"{d}={r * 100:.2f}%" for d, r in g.detector_flag_rates.items())
            dropped = (
                f"  [yellow]dropped: {_e(', '.join(g.dropped_detectors))}[/yellow]"
                if g.dropped_detectors
                else ""
            )
            label = "" if g.group_label == "__all__" else f"[{g.group_label}] "
            console.print(f"    {_e(label)}{_e(parts)}{dropped}")
        if len(rate_groups) > 8:
            console.print(f"    [dim]… and {_e(len(rate_groups) - 8)} more groups[/dim]")

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
                f"    {_e(g.group_label):30s}  {g.elapsed_seconds:6.1f}s  anomalies={_e(g.n_anomalies)}{flag}"
            )

    if result.warnings_issued:
        console.print("\n  [yellow bold]Warnings:[/yellow bold]")
        for msg in result.warnings_issued:
            console.print(f"    [yellow]{_e(msg)}[/yellow]")

    if result.n_failed:
        console.print("\n  [red bold]Failed groups:[/red bold]")
        for g in result.groups:
            if g.status == "failed":
                console.print(f"    {_e(g.group_label)}: {_e(g.error)}")

    if result.report_status == "success":
        console.print(f"\n[green]Report:[/green] {_e(result.report_path)}")
    elif result.report_status == "failed":
        err_console.print(
            "\n[red bold]Report generation failed[/red bold] -- detection and scoring "
            "completed successfully; see the log for the underlying exception."
        )


def _run_result_to_dict(result: RunResult) -> dict[str, Any]:
    code = _exit_code_for(result)
    return {
        "exit_code": int(code),
        "outcome": code.kind,
        "run_id": result.run_id,
        "dataset_uri": result.dataset_uri,
        "dataset_fp": result.dataset_fp,
        "snapshot_fp": result.snapshot_fp,
        "source_run_id": result.source_run_id,
        "id_identity_scope": result.id_identity_scope,
        "group_selection_error": result.group_selection_error,
        "period_label": result.period_label,
        "n_succeeded": result.n_succeeded,
        "n_skipped": result.n_skipped,
        "n_failed": result.n_failed,
        "n_flagged": result.n_anomalies,  # review shortlist size, not a prevalence estimate
        "report_path": str(result.report_path) if result.report_path else None,
        "report_status": result.report_status,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "warnings_issued": result.warnings_issued,
        "groups": [
            {
                "group_key": g.group_key,
                "group_label": g.group_label,
                "n_records": g.n_records,
                "n_flagged": g.n_anomalies,  # review shortlist size, not a prevalence estimate
                "detector_flag_rates": g.detector_flag_rates,
                "dropped_detectors": g.dropped_detectors,
                "status": g.status,
                "error": g.error,
                "elapsed_seconds": g.elapsed_seconds,
                "warnings_issued": g.warnings_issued,
            }
            for g in result.groups
        ],
    }
