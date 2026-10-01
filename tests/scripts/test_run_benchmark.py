"""P0-6: benchmark generation is a maintainer script with explicit paths, not a public command.

The old ``sorethumb benchmark`` command edited the README found by walking up
from the installed package location and wrote ``benchmark_results/`` into the
current directory. These tests pin the replacement's contract: nothing is
written anywhere that wasn't named on the command line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import run_benchmark
from typer.testing import CliRunner

import sorethumb_ml
from sorethumb_ml.cli import app
from sorethumb_ml.evaluate import pipeline_benchmark

pytestmark = pytest.mark.unit

_START = "<!-- pipeline-benchmark-results-start -->"
_END = "<!-- pipeline-benchmark-results-end -->"


def _readme(path: Path) -> Path:
    path.write_text(f"# Title\n\n{_START}\nOLD TABLE\n{_END}\n\nfooter\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Replace the (slow, subprocess-per-cell) pipeline suite with an instant, complete, empty one."""
    monkeypatch.setattr(pipeline_benchmark, "run_pipeline_benchmark", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_benchmark, "assert_complete_and_error_free", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# The public command is gone
# ---------------------------------------------------------------------------


def test_benchmark_is_not_a_sorethumb_command():
    """Checked by inspecting the command tree: *invoking* it would (on a regression) start a real benchmark."""
    import typer.main

    registered = set(typer.main.get_command(app).commands)  # type: ignore[attr-defined]
    assert "benchmark" not in registered
    assert "benchmark" not in CliRunner().invoke(app, ["--help"]).output


# ---------------------------------------------------------------------------
# Explicit paths only
# ---------------------------------------------------------------------------


def test_output_dir_is_required(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        run_benchmark.main(["--no-legacy"])
    assert exc.value.code == 2
    assert "--output-dir" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_pipeline")
def test_without_readme_flag_no_readme_is_touched_and_nothing_lands_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    bystander = _readme(cwd / "README.md")  # a README right where the old command would look
    before = bystander.read_bytes()
    out = tmp_path / "out"

    assert run_benchmark.main(["--no-legacy", "--output-dir", str(out)]) == 0

    assert bystander.read_bytes() == before
    assert (out / "pipeline_benchmark_results.md").is_file()
    assert (out / "pipeline_benchmark_results.csv").is_file()
    assert sorted(p.name for p in cwd.iterdir()) == ["README.md"]  # no stray benchmark_results/


@pytest.mark.usefixtures("fake_pipeline")
def test_readme_flag_injects_into_exactly_that_file(tmp_path: Path):
    target = _readme(tmp_path / "TARGET.md")
    other = _readme(tmp_path / "OTHER.md")
    other_before = other.read_bytes()

    code = run_benchmark.main(["--no-legacy", "--output-dir", str(tmp_path / "out"), "--readme", str(target)])

    assert code == 0
    text = target.read_text(encoding="utf-8")
    assert "OLD TABLE" not in text
    assert "_No pipeline-benchmark results._" in text
    assert "python scripts/run_benchmark.py" in text  # the regenerate hint points at the script
    assert "sorethumb benchmark" not in text
    assert other.read_bytes() == other_before


@pytest.mark.usefixtures("fake_pipeline")
def test_missing_readme_path_is_an_argument_error(tmp_path: Path, capsys):
    code = run_benchmark.main(
        ["--no-legacy", "--output-dir", str(tmp_path / "out"), "--readme", str(tmp_path / "nope.md")]
    )
    assert code == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.usefixtures("fake_pipeline")
def test_readme_inside_the_package_directory_is_refused(tmp_path: Path, capsys):
    package_file = Path(sorethumb_ml.__file__).resolve().parent / "py.typed"
    before = package_file.read_bytes()
    code = run_benchmark.main(
        ["--no-legacy", "--output-dir", str(tmp_path / "out"), "--readme", str(package_file)]
    )
    assert code == 2
    assert "package" in capsys.readouterr().err
    assert package_file.read_bytes() == before
    assert not (tmp_path / "out").exists()


def test_incomplete_matrix_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """The real completeness check (not the fixture's no-op) must still stop a publish."""
    monkeypatch.setattr(pipeline_benchmark, "run_pipeline_benchmark", lambda *_a, **_k: [])
    target = _readme(tmp_path / "README.md")
    before = target.read_bytes()
    out = tmp_path / "out"

    code = run_benchmark.main(["--no-legacy", "--output-dir", str(out), "--readme", str(target)])

    assert code == 1
    assert target.read_bytes() == before
    assert not out.exists()
    assert capsys.readouterr().err.startswith("error:")
