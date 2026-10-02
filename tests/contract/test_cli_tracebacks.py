"""An unexpected CLI crash never prints local variables, so it can't leak credentials.

Typer's pretty tracebacks show every frame's locals when
``pretty_exceptions_show_locals`` is on, and our declared floor (typer 0.16.0)
turns it on by default. During an HTTP download the locals include the request
headers, which carry the ``Authorization`` credential read from
``source.auth_env_var``. These tests pin the setting on every Typer app and
check, end to end in a subprocess, that a crash mid-download never prints the
token. Both are ``contract`` tests so the lowest-direct CI lane, which installs
exactly typer 0.16.0 and runs ``-m "unit or contract"``, exercises them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from sorethumb_ml import cli

pytestmark = pytest.mark.contract

_SENTINEL = "sentinel-token-7f3c9a1e5b"


def _all_typer_apps() -> list[typer.Typer]:
    apps = [cli.app]
    apps.extend(group.typer_instance for group in cli.app.registered_groups if group.typer_instance)
    return apps


def test_every_typer_app_hides_locals_in_tracebacks() -> None:
    apps = _all_typer_apps()
    assert {cli.config_app, cli.workspace_app} <= set(apps)
    for app in apps:
        assert app.pretty_exceptions_show_locals is False, app.info.name
        assert app.pretty_exceptions_short is True, app.info.name


# Runs the real CLI in a fresh interpreter with the download transport replaced by
# one that raises an unexpected RuntimeError, and the SSRF host check (which would
# need DNS) stubbed out. Nothing touches the network.
_CRASHING_DOWNLOAD = """
import sys
import httpx
import sorethumb_ml.io.source as source

def _boom(request):
    raise RuntimeError("simulated transport failure")

source._assert_host_is_safe = lambda url: None
source.httpx.HTTPTransport = lambda **kwargs: httpx.MockTransport(_boom)

from sorethumb_ml.cli import app
sys.argv = ["sorethumb", *sys.argv[1:]]
app()
"""


@pytest.mark.parametrize("auth", ["bearer", "basic"])
def test_crash_during_download_does_not_print_the_credential(tmp_path: Path, auth: str) -> None:
    secret = _SENTINEL if auth == "bearer" else f"user:{_SENTINEL}"
    (tmp_path / "sorethumb.toml").write_text(
        f"""\
[source]
uri = "https://data.example.com/table.csv"
auth = "{auth}"
auth_env_var = "SORETHUMB_TEST_TOKEN"
cache = false

[run]
workdir = "ws"
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "SORETHUMB_TEST_TOKEN": secret,
        "COLUMNS": "200",
        # Force Typer's pretty (Rich) traceback path -- the one that can print locals.
        "TERM": "xterm-256color",
    }
    env.pop("_TYPER_STANDARD_TRACEBACK", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CRASHING_DOWNLOAD, "inspect"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "simulated transport failure" in output, output  # the crash really happened mid-download
    assert _SENTINEL not in output
    assert "Authorization" not in output
