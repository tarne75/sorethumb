"""P0-9: the release-state checker rejects a tag whose repository state contradicts it.

Every case builds a small synthetic repository in ``tmp_path`` in the state a
correct release-state commit leaves behind, then breaks exactly one thing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from scripts import check_release_state as crs

pytestmark = pytest.mark.unit

TODAY = date(2026, 10, 15)
TAG_DATE = TODAY

_CHANGELOG = """\
# Changelog

## [Unreleased]

## [0.2.0] - 2026-10-15

Highlights of this release.

### Fixed

- Something real.

## [0.1.0] - 2026-09-21

First release.

[Unreleased]: https://github.com/example/repo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/example/repo/releases/tag/v0.2.0
[0.1.0]: https://github.com/example/repo/releases/tag/v0.1.0
"""

_SECURITY = """\
# Security Policy

## Supported Versions

`sorethumb` is pre-1.0 (currently `0.x`, alpha). The latest `0.x` release on
PyPI is supported.
"""

_README = "# Project\n\n> **Status: alpha.** Install with `pip install sorethumb-ml`.\n"


def _repo(
    tmp_path: Path,
    *,
    pyproject_version: str = "0.2.0",
    changelog: str = _CHANGELOG,
    security: str = _SECURITY,
    readme: str = _README,
) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{pyproject_version}"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    (tmp_path / "SECURITY.md").write_text(security, encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def _check(root: Path, version: str = "0.2.0", **kwargs) -> list[str]:
    kwargs.setdefault("tag_date", TAG_DATE)
    kwargs.setdefault("today", TODAY)
    return crs.check_release_state(root, version, **kwargs)


def _one(problems: list[str], needle: str) -> None:
    assert len(problems) == 1, problems
    assert needle in problems[0], problems


def test_a_correct_release_state_commit_passes(tmp_path):
    assert _check(_repo(tmp_path)) == []


# --- entries left under [Unreleased] -------------------------------------------------


def test_entries_left_under_unreleased_are_rejected(tmp_path):
    changelog = _CHANGELOG.replace(
        "## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n\n- Forgotten fix.\n", 1
    )
    _one(_check(_repo(tmp_path, changelog=changelog)), "under [Unreleased]")


def test_unreleased_with_only_empty_subheadings_passes(tmp_path):
    changelog = _CHANGELOG.replace("## [Unreleased]\n", "## [Unreleased]\n\n### Fixed\n\n", 1)
    assert _check(_repo(tmp_path, changelog=changelog)) == []


def test_unreleased_entries_are_reported_alongside_a_wrong_newest_section(tmp_path):
    changelog = _CHANGELOG.replace("## [Unreleased]\n", "## [Unreleased]\n\n- Pending.\n", 1)
    problems = _check(_repo(tmp_path, changelog=changelog, pyproject_version="0.3.0"), version="0.3.0")
    assert any("newest release section" in p for p in problems), problems
    assert any("under [Unreleased]" in p for p in problems), problems


def test_a_changelog_with_no_unreleased_section_passes(tmp_path):
    changelog = _CHANGELOG.replace("## [Unreleased]\n\n", "", 1).replace(
        "[Unreleased]: https://github.com/example/repo/compare/v0.2.0...HEAD\n", ""
    )
    assert _check(_repo(tmp_path, changelog=changelog)) == []


# --- changelog date -------------------------------------------------------------------


def test_future_changelog_date_is_rejected_even_without_a_tag_date(tmp_path):
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2026-10-16")
    problems = _check(_repo(tmp_path, changelog=changelog), tag_date=None)
    _one(problems, "in the future")


def test_changelog_date_must_be_the_tag_date(tmp_path):
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2026-10-14")
    problems = _check(_repo(tmp_path, changelog=changelog))
    _one(problems, "tag date is 2026-10-15")


def test_past_changelog_date_is_fine_when_no_tag_date_is_given(tmp_path):
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2026-10-01")
    assert _check(_repo(tmp_path, changelog=changelog), tag_date=None) == []


@pytest.mark.parametrize(
    "heading", ["## [0.2.0] - YYYY-MM-DD", "## [0.2.0]", "## [0.2.0] - TBD", "## [0.2.0] - 2026-13-45"]
)
def test_placeholder_or_invalid_date_is_rejected(tmp_path, heading):
    changelog = _CHANGELOG.replace("## [0.2.0] - 2026-10-15", heading)
    problems = _check(_repo(tmp_path, changelog=changelog))
    assert len(problems) == 1, problems
    assert "date" in problems[0], problems


# --- version agreement ------------------------------------------------------------------


def test_pyproject_version_mismatch_is_rejected(tmp_path):
    _one(_check(_repo(tmp_path, pyproject_version="0.1.9")), "pyproject.toml says version = '0.1.9'")


def test_newest_changelog_section_must_be_the_version_being_tagged(tmp_path):
    changelog = _CHANGELOG.replace(
        "## [Unreleased]\n\n## [0.2.0] - 2026-10-15", "## [Unreleased]\n\n## [0.3.0] - 2026-10-15"
    )
    _one(_check(_repo(tmp_path, changelog=changelog)), "newest release section is [0.3.0], not [0.2.0]")


def test_only_unreleased_heading_means_no_release_section(tmp_path):
    problems = _check(_repo(tmp_path, changelog="# Changelog\n\n## [Unreleased]\n\n- stuff\n"))
    assert any("no released" in p for p in problems), problems
    assert any("under [Unreleased]" in p for p in problems), problems


def test_empty_release_section_is_rejected(tmp_path):
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-10-15\n\n## [0.1.0] - 2026-09-21\n\nx\n\n"
        "[Unreleased]: https://x/compare/v0.2.0...HEAD\n[0.2.0]: https://x/releases/tag/v0.2.0\n"
    )
    _one(_check(_repo(tmp_path, changelog=changelog)), "[0.2.0] section is empty")


def test_unreleased_footer_link_must_name_the_released_version(tmp_path):
    changelog = _CHANGELOG.replace("compare/v0.2.0...HEAD", "compare/v0.1.0...HEAD")
    _one(_check(_repo(tmp_path, changelog=changelog)), "[Unreleased] link")


def test_version_footer_link_must_exist_and_name_the_tag(tmp_path):
    missing = _CHANGELOG.replace("[0.2.0]: https://github.com/example/repo/releases/tag/v0.2.0\n", "")
    _one(_check(_repo(tmp_path, changelog=missing)), "no '[0.2.0]: <url>' link")
    wrong = _CHANGELOG.replace("releases/tag/v0.2.0", "releases/tag/v0.1.0")
    _one(_check(_repo(tmp_path, changelog=wrong)), "[0.2.0] link")


@pytest.mark.parametrize("bad", ["v0.2.0", "0.2", "0.2.0rc1", "0.2.0.dev1", ""])
def test_version_argument_must_be_a_final_x_y_z(tmp_path, bad):
    problems = _check(_repo(tmp_path), version=bad)
    _one(problems, "final X.Y.Z")


# --- pre-release wording ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "text", "needle"),
    [
        (
            "SECURITY.md",
            "`sorethumb` is pre-1.0 and has not yet had its first\ntagged release. Only latest.\n",
            "no tagged release",
        ),
        ("SECURITY.md", "It has NOT YET HAD ITS FIRST TAGGED RELEASE.\n", "no tagged release"),
        ("README.md", "Not yet on PyPI. Install from a clone:\n", "not yet on PyPI"),
        (
            "README.md",
            "> the package is not yet published\n> to PyPI -- install from source\n",
            "not yet published/released",
        ),
        ("README.md", "> **Status: pre-release (0.2.0).** API may change.\n", "'Status: pre-release'"),
    ],
)
def test_pre_release_wording_is_rejected_across_line_wraps_and_case(tmp_path, filename, text, needle):
    root = _repo(tmp_path)
    (root / filename).write_text(text, encoding="utf-8")
    problems = _check(root)
    assert any(needle in p and p.startswith(filename) for p in problems), problems


def test_pre_1_0_wording_is_only_a_contradiction_from_1_0_on(tmp_path):
    assert _check(_repo(tmp_path)) == []  # 0.2.0: "pre-1.0" is accurate
    changelog = _CHANGELOG.replace("0.2.0", "1.0.0")
    root = _repo(tmp_path, pyproject_version="1.0.0", changelog=changelog)
    problems = _check(root, version="1.0.0")
    _one(problems, "pre-1.0")


def test_every_problem_is_reported_at_once(tmp_path):
    root = _repo(
        tmp_path,
        pyproject_version="0.1.0",
        security="has not yet had its first tagged release\n",
        readme="Not yet on PyPI\n",
    )
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2026-10-20")
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    assert len(_check(root)) >= 5


# --- CLI ------------------------------------------------------------------------------------


def test_cli_success_exit_code(tmp_path, capsys):
    # A past date, so the CLI's real "today" accepts it without a --tag-date.
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2026-09-21")
    root = _repo(tmp_path, changelog=changelog)
    assert crs.main(["--version", "0.2.0", "--root", str(root)]) == 0
    assert "ok:" in capsys.readouterr().out


def test_cli_failure_lists_every_problem_and_exits_1(tmp_path, capsys):
    changelog = _CHANGELOG.replace("[0.2.0] - 2026-10-15", "[0.2.0] - 2999-01-01")
    root = _repo(tmp_path, changelog=changelog, readme="Not yet on PyPI\n")
    assert crs.main(["--version", "0.2.0", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "in the future" in err
    assert "not yet on PyPI" in err
    assert "2 problem(s): not ready to tag v0.2.0" in err


def test_the_checker_runs_against_the_real_repository_without_error():
    """Whatever state the repo is in, the checker must parse the real files (and say something coherent)."""
    import tomllib

    with (crs.REPO_ROOT / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]
    problems = crs.check_release_state(crs.REPO_ROOT, version)
    assert all(isinstance(p, str) and p for p in problems)
