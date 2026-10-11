"""The installed-distribution checker flags every way an install can disagree with its build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import check_installed_dist as cid

pytestmark = pytest.mark.unit

WHEEL = "sorethumb_ml-0.1.0-py3-none-any.whl"
SDIST = "sorethumb_ml-0.1.0.tar.gz"


@pytest.mark.parametrize(
    ("filename", "version"),
    [
        (WHEEL, "0.1.0"),
        (SDIST, "0.1.0"),
        ("sorethumb_ml-1.2.3rc1.tar.gz", "1.2.3rc1"),
        ("sorethumb_ml-2.0.0.dev1-py3-none-any.whl", "2.0.0.dev1"),
        ("other-0.1.0.tar.gz", None),
        ("sorethumb_ml-0.1.0.zip", None),
    ],
)
def test_artifact_version_is_read_from_the_file_name(filename: str, version: str | None) -> None:
    assert cid.artifact_version(filename) == version


def test_agreeing_versions_have_no_problems() -> None:
    assert cid.version_problems({"a": "0.1.0", "b": "0.1.0", "c": "0.1.0"}) == []


@pytest.mark.parametrize("odd_one", ["a", "b", "c", "d"])
def test_any_single_disagreeing_version_is_reported(odd_one: str) -> None:
    versions: dict[str, str | None] = {"a": "0.1.0", "b": "0.1.0", "c": "0.1.0", "d": "0.1.0"}
    versions[odd_one] = "0.1.1"
    problems = cid.version_problems(versions)
    assert len(problems) == 1
    assert "disagree" in problems[0]
    assert "0.1.1" in problems[0]


def test_a_missing_version_is_reported_by_source() -> None:
    problems = cid.version_problems({"pyproject.toml": "0.1.0", "artifact file name": None})
    assert problems == ["no version found from: ['artifact file name']"]


def _direct_url(filename: str, base: str = "file:///D:/a/sorethumb/dist") -> str:
    return json.dumps({"url": f"{base}/{filename}", "archive_info": {}})


@pytest.mark.parametrize(("artifact", "filename"), [("wheel", WHEEL), ("sdist", SDIST)])
def test_install_from_the_named_artifact_passes(artifact: str, filename: str) -> None:
    assert cid.direct_url_problems(_direct_url(filename), artifact, filename) == []


def test_percent_encoded_url_still_matches() -> None:
    url = json.dumps({"url": "file:///C:/Users/runner%20admin/dist/" + SDIST})
    assert cid.direct_url_problems(url, "sdist", SDIST) == []


def test_sdist_job_that_installed_a_wheel_is_reported() -> None:
    problems = cid.direct_url_problems(_direct_url(WHEEL), "sdist", SDIST)
    assert problems == [f"installed from {WHEEL!r}, expected {SDIST!r}"]


def test_wrong_kind_of_file_for_the_artifact_is_reported() -> None:
    problems = cid.direct_url_problems(_direct_url(SDIST), "wheel", SDIST)
    assert len(problems) == 1
    assert "needs a .whl" in problems[0]


def test_install_from_an_index_has_no_direct_url() -> None:
    problems = cid.direct_url_problems(None, "sdist", SDIST)
    assert len(problems) == 1
    assert "direct_url.json" in problems[0]


@pytest.mark.parametrize("text", ["not json", "[]", '{"nope": 1}'])
def test_unreadable_direct_url_is_reported(text: str) -> None:
    problems = cid.direct_url_problems(text, "sdist", SDIST)
    assert len(problems) == 1
    assert "unreadable" in problems[0]


def test_console_script_version_must_match_exactly() -> None:
    assert cid.console_script_problems("sorethumb 0.1.0\n", 0, "0.1.0") == []
    assert "expected 'sorethumb 0.1.0'" in cid.console_script_problems("sorethumb 0.1.1\n", 0, "0.1.0")[0]
    assert "exited 2" in cid.console_script_problems("boom", 2, "0.1.0")[0]


def test_package_outside_the_venv_is_reported(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    inside = venv / "lib" / "site-packages" / "sorethumb_ml" / "__init__.py"
    inside.parent.mkdir(parents=True)
    inside.touch()
    checkout = tmp_path / "checkout" / "src" / "sorethumb_ml" / "__init__.py"
    checkout.parent.mkdir(parents=True)
    checkout.touch()
    assert cid.location_problems(str(inside), str(venv)) == []
    assert "outside the venv" in cid.location_problems(str(checkout), str(venv))[0]


def test_pyproject_for_another_project_is_a_problem(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "other"\nversion = "0.1.0"\n')
    version, problems = cid.pyproject_version(pyproject)
    assert version is None
    assert problems == [f"{pyproject}: [project] name is 'other', expected 'sorethumb-ml'"]


def test_unreadable_pyproject_is_a_problem(tmp_path: Path) -> None:
    version, problems = cid.pyproject_version(tmp_path / "missing.toml")
    assert version is None
    assert len(problems) == 1
    assert problems[0].startswith("cannot read")


_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _ROOT / "pyproject.toml"


def _current_wheel_name() -> str:
    version, _ = cid.pyproject_version(_PYPROJECT)
    return f"sorethumb_ml-{version}-py3-none-any.whl"


def _run_main(capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str]:
    code = cid.main(
        [
            "--artifact",
            "wheel",
            "--file",
            f"dist/{_current_wheel_name()}",
            "--pyproject",
            str(_PYPROJECT),
            *extra,
        ]
    )
    return code, capsys.readouterr().err


def test_main_rejects_a_development_install_for_the_right_reasons(capsys: pytest.CaptureFixture[str]) -> None:
    """The test environment is an editable install: its versions agree, but it is not the artifact."""
    code, err = _run_main(capsys)
    assert code == 1
    assert "versions disagree" not in err
    assert "no version found" not in err
    assert f"expected {_current_wheel_name()!r}" in err  # direct_url.json names the checkout, not a wheel
    assert "outside the venv" in err  # imported from src/, not site-packages


def test_main_reports_a_missing_distribution_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_installed(name: str) -> None:
        raise cid.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cid.importlib.metadata, "distribution", not_installed)
    code, err = _run_main(capsys)
    assert code == 1
    assert "sorethumb-ml is not installed" in err
    assert "no version found from: ['installed distribution metadata']" in err


def test_main_reports_a_failing_import_alongside_the_other_problems(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = cid.importlib.import_module

    def broken(name: str) -> object:
        if name == "sorethumb_ml":
            raise ImportError("simulated")
        return real_import(name)

    monkeypatch.setattr(cid.importlib, "import_module", broken)
    code, err = _run_main(capsys)
    assert code == 1
    assert "import sorethumb_ml failed: ImportError: simulated" in err
    assert "no version found from: ['runtime sorethumb_ml.__version__']" in err
    assert f"expected {_current_wheel_name()!r}" in err  # the direct_url check still ran


def test_main_reports_a_console_script_that_cannot_run(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code, err = _run_main(capsys, "--console-script", str(tmp_path / "no-such-sorethumb"))
    assert code == 1
    assert "could not run" in err
