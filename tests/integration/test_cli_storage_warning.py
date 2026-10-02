"""The CLI warns once, on stderr, when a workspace is on risky storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml import cli
from sorethumb_ml.cli import app
from sorethumb_ml.store import storage
from tests.factories.frames import write_grouped_csv

pytestmark = pytest.mark.integration


def _project(root: Path) -> Path:
    write_grouped_csv(root / "data.csv", n_rows=200)
    toml = root / "sorethumb.toml"
    toml.write_text(
        f"[source]\nuri = {json.dumps(str(root / 'data.csv'))}\n\n[run]\nworkdir = {json.dumps(str(root / 'ws'))}\n",
        encoding="utf-8",
    )
    return toml


def test_risky_storage_is_warned_about_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_STORAGE_WARNED", set())
    monkeypatch.setattr(storage, "detect_risky_storage", lambda _path: "a OneDrive-synced folder")
    result = CliRunner().invoke(app, ["run", "--config", str(_project(tmp_path)), "--no-report"])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert flat.count("is on a OneDrive-synced folder") == 1
    assert "local, non-synced drive" in flat


def test_local_storage_prints_no_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_STORAGE_WARNED", set())
    monkeypatch.setattr(storage, "detect_risky_storage", lambda _path: None)
    result = CliRunner().invoke(app, ["run", "--config", str(_project(tmp_path)), "--no-report"])
    assert result.exit_code == 0, result.output
    assert "non-synced drive" not in result.output
