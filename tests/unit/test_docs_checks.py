"""Guards for the doc-consistency tooling in docs/.

`docs/generate_config_docs.py --check` and `docs/check_readme_snippets.py` run in
CI; these tests keep their detection logic honest and confirm the committed docs
are currently clean.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / "docs" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gcd = _load("generate_config_docs")
crs = _load("check_readme_snippets")


# ---------------------------------------------------------------------------
# generate_config_docs: prose-drift check
# ---------------------------------------------------------------------------


def test_prose_docs_have_no_schema_drift() -> None:
    assert gcd.check_doc_drift(_ROOT) == []


def test_configuration_md_is_up_to_date() -> None:
    committed = (_ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    assert committed == gcd.generate(), "run: python docs/generate_config_docs.py"


def test_no_broken_relative_markdown_links() -> None:
    assert gcd.check_broken_links(_ROOT) == []


def test_broken_link_check_flags_a_missing_target(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "a.md").write_text("see [gone](docs/nope.md) and [ok](a.md)\n", encoding="utf-8")
    problems = gcd.check_broken_links(tmp_path)
    assert any("nope.md" in p for p in problems)
    assert not any("[ok]" in p or "a.md: link -> a.md" in p for p in problems)


def test_drift_check_flags_a_wrong_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text(
        "Raise `profiling.null_ratio_flag` (default 0.30) to reduce indicators.\n"
        "`scoring.combination` defaults to `composite`.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gcd, "_PROSE_DOCS", ["bad.md"])
    problems = gcd.check_doc_drift(tmp_path)
    assert any("null_ratio_flag" in p and "0.0" in p for p in problems)
    assert any("combination" in p and "intersection" in p for p in problems)


def test_drift_check_flags_an_unknown_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text("Tune `profiling.null_ratio_flagg` for your data.\n", encoding="utf-8")
    monkeypatch.setattr(gcd, "_PROSE_DOCS", ["bad.md"])
    problems = gcd.check_doc_drift(tmp_path)
    assert any("null_ratio_flagg" in p and "not a field" in p for p in problems)


# ---------------------------------------------------------------------------
# check_readme_snippets: static (non-executing) checks
# ---------------------------------------------------------------------------


def test_readme_toml_fragments_validate_against_the_schema() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    toml_blocks = [body for _ln, lang, body in crs._blocks(readme) if lang == "toml"]
    assert toml_blocks  # README has toml examples
    for body in toml_blocks:
        assert crs._check_toml(body) is None, body


def test_readme_bash_commands_and_flags_exist() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    for _ln, lang, body in crs._blocks(readme):
        if lang == "bash":
            assert crs._check_bash(body) == [], body


def test_snippet_bash_check_catches_a_bogus_command_and_flag() -> None:
    problems = crs._check_bash("sorethumb frobnicate\nsorethumb run --nope")
    assert any("frobnicate" in p for p in problems)
    assert any("--nope" in p for p in problems)


def test_snippet_toml_check_catches_an_invalid_config_value() -> None:
    assert crs._check_toml('[scoring]\ncombination = "banana"\n') is not None
