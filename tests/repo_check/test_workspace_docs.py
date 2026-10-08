"""The documented workspace rules match the CLI constant and behaviour.

The default workspace directory is a single constant (``cli._DEFAULT_WORKDIR``) but is
quoted in the README, the CLI reference, the config reference and SECURITY.md. A past
copy in SECURITY.md named a directory (``.sorethumb/``) the CLI never used. These tests
tie every documented mention to the constant, and check the documented resolution rules
(flag > config > default; default relative to the current directory; ``init`` layout)
against the real code.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sorethumb_ml.cli import _DEFAULT_WORKDIR, _detach_file_handlers, _load_config, app

pytestmark = pytest.mark.repo_check


@pytest.fixture(autouse=True)
def _close_logs_opened_outside_a_command() -> Iterator[None]:
    # Some tests call _load_config directly, outside a CLI command, so no
    # command context is there to close the log handler it attaches.
    yield
    _detach_file_handlers()


_ROOT = Path(__file__).resolve().parents[2]
_DOCS = [
    "README.md",
    "SECURITY.md",
    "docs/cli_reference.md",
    "docs/configuration.md",
    "docs/generate_config_docs.py",
]
# Any directory-looking name that the docs could be using for the workspace default.
_WORKSPACE_NAME = re.compile(r"(?<![\w-])(\.?sorethumb[-_.]?(?:workspace|work|ws)?)/")


def _text(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def test_the_constant_is_the_documented_value() -> None:
    assert _DEFAULT_WORKDIR == "sorethumb-workspace"


@pytest.mark.parametrize(
    "rel", ["README.md", "SECURITY.md", "docs/cli_reference.md", "docs/configuration.md"]
)
def test_every_user_facing_doc_states_the_default(rel: str) -> None:
    assert f"./{_DEFAULT_WORKDIR}/" in _text(rel) or f'"{_DEFAULT_WORKDIR}"' in _text(rel), rel


@pytest.mark.parametrize("rel", _DOCS)
def test_no_doc_names_a_different_default_workspace(rel: str) -> None:
    """Catches a stale or invented default (``.sorethumb/`` and friends)."""
    names = {m.group(1) for m in _WORKSPACE_NAME.finditer(_text(rel))}
    # 'sorethumb/' is the repo/URL path segment (github.com/tarne75/sorethumb/...), not a workspace.
    names.discard("sorethumb")
    assert names <= {_DEFAULT_WORKDIR}, (
        f"{rel} names workspace directories other than the default: {sorted(names)}"
    )


@pytest.mark.parametrize("rel", ["README.md", "SECURITY.md", "docs/cli_reference.md"])
def test_docs_say_the_workspace_is_executable_and_name_the_model_files(rel: str) -> None:
    text = _text(rel).lower()
    assert "executable" in text, rel
    assert "joblib" in text or "pickle" in text, rel


def test_cli_reference_documents_the_resolution_order() -> None:
    text = _text("docs/cli_reference.md")
    section = text.split("## Where the workspace lives", 1)[1].split("\n## ", 1)[0]
    # Precedence is spelled out in order, and the "relative to the current directory" rule is stated.
    assert (
        section.index("`--workdir PATH`")
        < section.index("`run.workdir`")
        < section.index(f"`./{_DEFAULT_WORKDIR}/`")
    )
    assert "relative to the current directory" in section.replace("\n", " ")


def test_default_applies_without_a_config_file_and_relative_to_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _load_config(None, uri_override=str(tmp_path / "d.csv"))
    assert cfg.run.workdir == _DEFAULT_WORKDIR
    assert not Path(cfg.run.workdir).is_absolute()  # resolved against the cwd, not beside the data


def test_default_also_applies_to_a_config_file_that_omits_workdir_and_is_not_config_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    toml = cfg_dir / "sorethumb.toml"
    toml.write_text('[source]\nuri = "d.csv"\n', encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = _load_config(toml)
    assert cfg.run.workdir == _DEFAULT_WORKDIR
    assert (elsewhere / cfg.run.workdir).parent == elsewhere  # i.e. ./sorethumb-workspace/ under the cwd


def test_flag_beats_config_beats_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    toml = tmp_path / "sorethumb.toml"
    toml.write_text('[source]\nuri = "d.csv"\n[run]\nworkdir = "from-config"\n', encoding="utf-8")
    assert _load_config(toml).run.workdir == "from-config"
    assert _load_config(toml, workdir=Path("from-flag")).run.workdir == "from-flag"


def test_init_creates_the_documented_layout(tmp_path: Path) -> None:
    target = tmp_path / "my-analysis"
    result = CliRunner().invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "sorethumb.toml").is_file()
    workspace = target / _DEFAULT_WORKDIR
    assert (workspace / "sorethumb.db").is_file()
    written = tomllib.loads((target / "sorethumb.toml").read_text(encoding="utf-8"))
    assert written["run"]["workdir"] == str(workspace)  # parsed: a Windows path is escaped in the file


def test_init_does_nothing_when_a_config_already_exists(tmp_path: Path) -> None:
    (tmp_path / "sorethumb.toml").write_text("# mine\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "sorethumb.toml").read_text(encoding="utf-8") == "# mine\n"
    assert not (tmp_path / _DEFAULT_WORKDIR).exists()


def test_workdir_option_help_names_the_default() -> None:
    import typer.main

    command = typer.main.get_command(app).commands["runs"]  # type: ignore[attr-defined]
    help_text = next(p.help for p in command.params if p.name == "workdir")
    assert f"./{_DEFAULT_WORKDIR}/" in help_text
