"""Validate the built sdist/wheel in ``dist/``.

Run after ``uv build`` -- locally, or in CI's ``release-validation.yml``
``build`` job. By default every test here skips when ``dist/`` hasn't been
built, so a plain ``pytest -m repo_check`` (no prior build step) never fails
on this; it exists to be run deliberately, after a build, not as part of the
ambient repo-check suite.

Setting ``SORETHUMB_REQUIRE_DIST=1`` turns every "artifact not built" skip
into a failure. The ``build`` job sets it, so a build step that silently
produced nothing (or the wrong version) cannot pass validation by skipping --
the same pattern as ``SORETHUMB_REQUIRE_NETWORK`` for the real-dataset smoke.

Guards against exactly the two failure modes the wheel-only pipeline missed:
a source distribution that is either broken (missing something the wheel
needs) or over-inclusive (shipping repository-only material, dev-tool
caches, or local settings that were never meant to leave this machine --
``.claude/`` and ``.hypothesis/`` both did, silently, before
``[tool.hatch.build.targets.sdist]`` was given an explicit allowlist). The
sdist's expected top level is *derived* from that allowlist, so adding or
removing an entry there needs no matching edit here.
"""

from __future__ import annotations

import email
import os
import re
import tarfile
import tomllib
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.repo_check

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIST_DIR = _REPO_ROOT / "dist"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

REQUIRE_ENV = "SORETHUMB_REQUIRE_DIST"

_REQUIRED_WHEEL_PATHS = frozenset(
    {
        "sorethumb_ml/py.typed",
        "sorethumb_ml/store/migrations/001_initial.sql",
    }
)
_REQUIRED_SDIST_PATHS = frozenset(
    {
        "src/sorethumb_ml/py.typed",
        "src/sorethumb_ml/store/migrations/001_initial.sql",
        "LICENSE",
        "README.md",
    }
)

# Top-level entries the build backend (hatchling) adds to every sdist
# regardless of ``include``: the generated core metadata, and the VCS
# exclusion file it force-includes so a later build from the sdist applies the
# same ignore rules. Anything else at the top level must come from the
# allowlist.
_BACKEND_ADDED_SDIST_ENTRIES = frozenset({"PKG-INFO", ".gitignore"})

# An allowlist entry is an anchored literal path ("/src", "/docs/x.md"); a glob
# or unanchored pattern cannot be turned into an expected-entry set reliably.
_LITERAL_INCLUDE = re.compile(r"/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*")

# Anything under these prefixes, or whose name contains one of these
# substrings, has no business in a source distribution: it is either
# repository-only material (dev-tool config, CI workflows, planning notes),
# a local cache/settings directory, or a build/test artefact.
_FORBIDDEN_SDIST_PREFIXES = (
    ".claude/",
    ".hypothesis/",
    ".github/",
    ".git/",
    "prompts/",
    "scripts/",
    "benchmark_results/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    "htmlcov/",
)
_FORBIDDEN_NAME_SUBSTRINGS = ("__pycache__", ".pyc", ".DS_Store", ".coverage", ".smbdelete")

_METADATA_FIELDS_MUST_MATCH = (
    "Name",
    "Version",
    "Summary",
    "License-Expression",
    "Requires-Python",
    "Author-email",
)


def _require_dist() -> bool:
    return os.environ.get(REQUIRE_ENV) == "1"


def _unavailable(reason: str) -> None:
    """Skip, or -- when ``SORETHUMB_REQUIRE_DIST=1`` -- fail: a required artifact is missing."""
    if _require_dist():
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1 makes a missing artifact fatal)", pytrace=False)
    pytest.skip(reason)


def _pyproject() -> dict[str, Any]:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _project_version() -> str:
    return str(_pyproject()["project"]["version"])


def _project_name() -> str:
    return str(_pyproject()["project"]["name"])


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _select_artifact(pattern: str, description: str) -> Path:
    """The one artifact in ``dist/`` matching ``pattern`` for the version in pyproject.toml.

    Artifacts of other versions (a stale ``dist/``) are ignored, never validated
    in place of the current build. More than one match for the current version
    is ambiguous and always an error.
    """
    matches = sorted(_DIST_DIR.glob(pattern))
    if not matches:
        _unavailable(f"dist/{pattern} ({description}) not built -- run `uv build` first")
    if len(matches) > 1:
        pytest.fail(f"expected one dist/{pattern}, found {[m.name for m in matches]}", pytrace=False)
    return matches[0]


def _wheel_path() -> Path:
    return _select_artifact(f"sorethumb_ml-{_project_version()}-*.whl", "wheel")


def _sdist_path() -> Path:
    return _select_artifact(f"sorethumb_ml-{_project_version()}.tar.gz", "sdist")


def _wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def _sdist_split(path: Path) -> tuple[set[str], list[str]]:
    """Return ``(wrapping directory names, member paths with the wrapper stripped)``."""
    with tarfile.open(path) as tf:
        names = tf.getnames()
    roots: set[str] = set()
    relative: list[str] = []
    for name in names:
        root, _sep, rest = name.partition("/")
        roots.add(root)
        if rest:
            relative.append(rest)
    return roots, relative


def _sdist_names(path: Path) -> list[str]:
    """Return sdist member paths with the wrapping "<name>-<version>/" dir stripped."""
    return _sdist_split(path)[1]


# hatch options that add files outside ``include``; the derivation below does not model them.
_UNMODELLED_SDIST_OPTIONS = ("only-include", "force-include")


def _sdist_include(sdist_config: dict[str, Any] | None = None) -> list[str]:
    """The sdist ``include`` allowlist, refusing hatch options that would add files outside it."""
    if sdist_config is None:
        sdist_config = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]
    unmodelled = [option for option in _UNMODELLED_SDIST_OPTIONS if option in sdist_config]
    if unmodelled:
        pytest.fail(
            f"[tool.hatch.build.targets.sdist] uses {unmodelled}; the expected sdist contents are "
            f"derived from `include` alone -- extend _expected_sdist_top_level to cover them",
            pytrace=False,
        )
    return list(sdist_config["include"])


def _expected_sdist_top_level(include: list[str]) -> set[str]:
    """The sdist's complete expected top-level entries, derived from the hatch allowlist."""
    bad = [
        entry
        for entry in include
        if not _LITERAL_INCLUDE.fullmatch(entry) or {".", ".."} & set(entry.split("/"))
    ]
    if bad:
        pytest.fail(
            f"[tool.hatch.build.targets.sdist].include entries must be anchored literal paths "
            f"(like '/src'); cannot derive the expected sdist contents from {bad}",
            pytrace=False,
        )
    return {entry.lstrip("/").split("/")[0] for entry in include} | _BACKEND_ADDED_SDIST_ENTRIES


def _sdist_layout_problems(members: list[str], include: list[str]) -> list[str]:
    """Differences between an sdist's members and what its allowlist promises."""
    problems = []
    top_level = {member.split("/")[0] for member in members}
    expected = _expected_sdist_top_level(include)
    if missing_top := sorted(expected - top_level):
        problems.append(f"top-level entries missing from the sdist: {missing_top}")
    if extra_top := sorted(top_level - expected):
        problems.append(f"unexpected top-level entries in the sdist: {extra_top}")
    present = set(members)
    for entry in include:
        path = entry.lstrip("/")
        if path not in present and not any(member.startswith(f"{path}/") for member in members):
            problems.append(f"allowlisted path {entry!r} is absent from the sdist")
    return problems


def test_wheel_contains_required_paths() -> None:
    names = set(_wheel_names(_wheel_path()))
    missing = _REQUIRED_WHEEL_PATHS - names
    assert not missing, f"wheel is missing required path(s): {sorted(missing)}"


def test_sdist_contains_required_paths() -> None:
    names = set(_sdist_names(_sdist_path()))
    missing = _REQUIRED_SDIST_PATHS - names
    assert not missing, f"sdist is missing required path(s): {sorted(missing)}"


def _repo_license_bytes() -> bytes:
    path = _DIST_DIR.parent / "LICENSE"
    if not path.is_file():
        _unavailable("LICENSE is not present")
    return path.read_bytes()


def test_wheel_ships_the_repository_license() -> None:
    """The wheel's dist-info/licenses/LICENSE must be the file in the repository, byte for byte."""
    expected = _repo_license_bytes()
    with zipfile.ZipFile(_wheel_path()) as zf:
        members = [n for n in zf.namelist() if re.fullmatch(r"[^/]+\.dist-info/licenses/LICENSE", n)]
        assert len(members) == 1, f"expected one dist-info/licenses/LICENSE in the wheel, found {members}"
        assert zf.read(members[0]) == expected, "wheel LICENSE differs from the repository LICENSE"


def test_sdist_top_level_matches_the_hatch_allowlist() -> None:
    """The sdist holds exactly the allowlisted top-level entries (plus what hatchling always adds).

    Expected contents are derived from ``[tool.hatch.build.targets.sdist].include``:
    every allowlisted path must be present (a directory with at least one file),
    and nothing else may sit at the top level.
    """
    roots, members = _sdist_split(_sdist_path())
    expected_root = f"sorethumb_ml-{_project_version()}"
    assert roots == {expected_root}, (
        f"sdist must have the single wrapper dir {expected_root!r}, found {roots}"
    )
    problems = _sdist_layout_problems(members, _sdist_include())
    assert not problems, "; ".join(problems)


def test_sdist_ships_the_repository_license() -> None:
    """The sdist's top-level LICENSE must be the file in the repository, byte for byte."""
    expected = _repo_license_bytes()
    with tarfile.open(_sdist_path()) as tf:
        members = [m for m in tf.getmembers() if m.name.count("/") == 1 and m.name.endswith("/LICENSE")]
        assert len(members) == 1, (
            f"expected one top-level LICENSE in the sdist, found {[m.name for m in members]}"
        )
        extracted = tf.extractfile(members[0])
        assert extracted is not None, "sdist LICENSE member has no extractable content"
        assert extracted.read() == expected, "sdist LICENSE differs from the repository LICENSE"


def test_sdist_excludes_repository_only_material() -> None:
    names = _sdist_names(_sdist_path())
    offenders = [
        n
        for n in names
        if n.startswith(_FORBIDDEN_SDIST_PREFIXES) or any(sub in n for sub in _FORBIDDEN_NAME_SUBSTRINGS)
    ]
    assert not offenders, f"sdist contains unintended file(s): {offenders}"


_REVIEW_DOCUMENT = re.compile(r"review[^/]*\.(md|docx|pdf)$", re.IGNORECASE)


def test_neither_artifact_contains_a_review_document() -> None:
    """Review documents are working material about the repository (see .gitignore); an
    allowlisted sdist should never pick one up, and a wheel never would. This keeps it so."""
    offenders = [
        n for n in (*_sdist_names(_sdist_path()), *_wheel_names(_wheel_path())) if _REVIEW_DOCUMENT.search(n)
    ]
    assert not offenders, f"review document(s) in a distribution artifact: {offenders}"


def _read_wheel_metadata(path: Path) -> Message:
    with zipfile.ZipFile(path) as zf:
        metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(zf.read(metadata_name))


def _read_sdist_metadata(path: Path) -> Message:
    with tarfile.open(path) as tf:
        pkg_info = next(m for m in tf.getmembers() if m.name.endswith("/PKG-INFO") or m.name == "PKG-INFO")
        extracted = tf.extractfile(pkg_info)
        assert extracted is not None, "PKG-INFO member has no extractable content"
        return email.message_from_bytes(extracted.read())


def test_wheel_and_sdist_report_the_same_version_and_metadata() -> None:
    """The two artifacts a release publishes must describe the same package.

    A build-backend bug, a stale cached artifact, or a race between building
    the two could otherwise ship a wheel and sdist that silently disagree.
    """
    wheel_metadata = _read_wheel_metadata(_wheel_path())
    sdist_metadata = _read_sdist_metadata(_sdist_path())

    mismatches = {
        field: (wheel_metadata.get(field), sdist_metadata.get(field))
        for field in _METADATA_FIELDS_MUST_MATCH
        if wheel_metadata.get(field) != sdist_metadata.get(field)
    }
    assert not mismatches, f"wheel/sdist metadata mismatch: {mismatches}"


def _normalised_name_and_version(metadata: Message) -> tuple[str, str]:
    return _normalise_name(metadata.get("Name", "")), str(metadata.get("Version"))


def test_both_artifacts_match_pyproject_name_and_version() -> None:
    """Each artifact's own metadata must name the project and version pyproject.toml declares."""
    expected = (_normalise_name(_project_name()), _project_version())
    actual = {
        "wheel": _normalised_name_and_version(_read_wheel_metadata(_wheel_path())),
        "sdist": _normalised_name_and_version(_read_sdist_metadata(_sdist_path())),
    }
    wrong = {kind: found for kind, found in actual.items() if found != expected}
    assert not wrong, f"artifact name/version differ from pyproject.toml {expected}: {wrong}"


# --- the checks above, exercised without a build ------------------------------------------------


@pytest.fixture
def empty_dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at an empty ``dist/`` (and a repo root with no LICENSE)."""
    dist = tmp_path / "dist"
    dist.mkdir()
    monkeypatch.setattr(f"{__name__}._DIST_DIR", dist)
    return dist


@pytest.mark.usefixtures("empty_dist")
@pytest.mark.parametrize("finder", [_wheel_path, _sdist_path, _repo_license_bytes])
def test_missing_artifact_skips_without_the_require_variable(
    monkeypatch: pytest.MonkeyPatch, finder: Any
) -> None:
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        finder()


@pytest.mark.usefixtures("empty_dist")
@pytest.mark.parametrize("finder", [_wheel_path, _sdist_path, _repo_license_bytes])
def test_missing_artifact_fails_when_dist_is_required(monkeypatch: pytest.MonkeyPatch, finder: Any) -> None:
    monkeypatch.setenv(REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match=REQUIRE_ENV):
        finder()


@pytest.mark.usefixtures("empty_dist")
@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_the_literal_one_requires_dist(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(REQUIRE_ENV, value)
    with pytest.raises(pytest.skip.Exception):
        _sdist_path()


def test_artifacts_of_another_version_are_not_validated_in_place_of_the_current_one(
    empty_dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (empty_dist / "sorethumb_ml-0.0.1.tar.gz").write_bytes(b"")
    (empty_dist / "sorethumb_ml-0.0.1-py3-none-any.whl").write_bytes(b"")
    monkeypatch.setenv(REQUIRE_ENV, "1")
    for finder in (_wheel_path, _sdist_path):
        with pytest.raises(pytest.fail.Exception, match="not built"):
            finder()


def test_two_artifacts_for_the_current_version_are_always_an_error(
    empty_dist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    version = _project_version()
    (empty_dist / f"sorethumb_ml-{version}-py3-none-any.whl").write_bytes(b"")
    (empty_dist / f"sorethumb_ml-{version}-py3-none-macosx.whl").write_bytes(b"")
    with pytest.raises(pytest.fail.Exception, match="expected one"):
        _wheel_path()


_SAMPLE_INCLUDE = ["/src", "/tests", "/README.md", "/pyproject.toml"]
_SAMPLE_MEMBERS = [
    "src/pkg/__init__.py",
    "tests/test_x.py",
    "README.md",
    "pyproject.toml",
    "PKG-INFO",
    ".gitignore",
]


def test_expected_sdist_top_level_is_derived_from_the_allowlist() -> None:
    assert _expected_sdist_top_level(_SAMPLE_INCLUDE) == {
        "src",
        "tests",
        "README.md",
        "pyproject.toml",
        "PKG-INFO",
        ".gitignore",
    }
    assert _expected_sdist_top_level(["/docs/guide.md"]) == {"docs", "PKG-INFO", ".gitignore"}


@pytest.mark.parametrize("entry", ["src", "/src/**", "/src/*.py", "/", "", "/src/../x"])
def test_non_literal_allowlist_entries_are_rejected(entry: str) -> None:
    with pytest.raises(pytest.fail.Exception, match="anchored literal"):
        _expected_sdist_top_level([entry])


def test_a_complete_sdist_has_no_layout_problems() -> None:
    assert _sdist_layout_problems(_SAMPLE_MEMBERS, _SAMPLE_INCLUDE) == []


def test_an_allowlisted_path_missing_from_the_sdist_is_reported() -> None:
    members = [m for m in _SAMPLE_MEMBERS if m != "README.md"]
    problems = _sdist_layout_problems(members, _SAMPLE_INCLUDE)
    assert any("README.md" in p and "absent" in p for p in problems)


def test_an_allowlisted_directory_with_no_files_is_reported() -> None:
    members = [m for m in _SAMPLE_MEMBERS if not m.startswith("tests/")]
    problems = _sdist_layout_problems(members, _SAMPLE_INCLUDE)
    assert any("'/tests'" in p and "absent" in p for p in problems)


def test_an_unlisted_top_level_entry_is_reported() -> None:
    problems = _sdist_layout_problems([*_SAMPLE_MEMBERS, "stray.txt", "extra/dir/file"], _SAMPLE_INCLUDE)
    assert any("unexpected top-level" in p and "stray.txt" in p and "extra" in p for p in problems)


def test_a_missing_backend_added_entry_is_reported() -> None:
    members = [m for m in _SAMPLE_MEMBERS if m != "PKG-INFO"]
    problems = _sdist_layout_problems(members, _SAMPLE_INCLUDE)
    assert any("missing from the sdist" in p and "PKG-INFO" in p for p in problems)


@pytest.mark.parametrize("option", _UNMODELLED_SDIST_OPTIONS)
def test_sdist_options_outside_include_are_named_not_misreported(option: str) -> None:
    with pytest.raises(pytest.fail.Exception, match=option):
        _sdist_include({"include": ["/src"], option: {}})


def test_the_real_allowlist_is_derivable() -> None:
    """The repository's own allowlist must stay literal, or the real-sdist check cannot run."""
    expected = _expected_sdist_top_level(_sdist_include())
    assert {"src", "tests", "LICENSE", "pyproject.toml", "PKG-INFO"} <= expected
