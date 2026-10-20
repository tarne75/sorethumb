"""The README is the entry page: the PyPI long description and the first thing a reader sees.

It leads with how to install, a core-only example and the short list of limitations, then
the decision guidance (is it the right tool, which setup), a short command-line start,
the platform table and a map of the documentation. Everything else lives behind the
documentation index: the full limitations in ``docs/limitations.md``, the walkthrough in
``docs/getting-started.md``, the benchmark summary and matrices in ``docs/benchmarks.md``.
These checks keep that shape, keep the README short, and keep the benchmark summary
honest: every figure it quotes has to appear in the generated results.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_BENCH_DOC = _ROOT / "docs" / "benchmarks.md"
_LIMITATIONS = _ROOT / "docs" / "limitations.md"
# blob/main before the first release; the released tag after it (the release-state commit).
_DOCS = re.search(r"https://github\.com/tarne75/sorethumb/blob/[^/\s]+/docs/", _README).group(0)
# An entry page, not the manual: room for what a newcomer needs and no more.
_MAX_README_LINES = 300

_MARKER_PAIRS = (
    ("<!-- pipeline-benchmark-results-start -->", "<!-- pipeline-benchmark-results-end -->"),
    ("<!-- benchmark-results-start -->", "<!-- benchmark-results-end -->"),
)


def _h2_titles(text: str) -> list[str]:
    in_fence = False
    titles: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("## "):
            titles.append(line[3:].strip())
    return titles


def _section(text: str, title: str) -> str:
    return text.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_readme_leads_with_install_a_core_only_example_and_the_limitations_summary() -> None:
    assert _h2_titles(_README)[:3] == ["Installation", "60-second quickstart", "What it does not do"]


def test_the_decision_guidance_follows_the_limitations() -> None:
    titles = _h2_titles(_README)
    assert titles[3] == "Is it the right tool?"
    assert "### Which setup?" in _section(_README, "Is it the right tool?")


def test_the_readme_stays_an_entry_page() -> None:
    lines = len(_README.splitlines())
    assert lines <= _MAX_README_LINES, (
        f"README is {lines} lines; keep it an entry page and move detail behind docs/index.md"
    )
    assert f"{_DOCS}index.md" in _section(_README, "Documentation")


def test_the_core_only_example_needs_no_extras_and_no_network() -> None:
    section = _section(_README, "60-second quickstart")
    assert "core dependencies" in section
    assert "No network access" in section
    assert "pip install" not in section  # nothing beyond the core install


def test_the_limitations_summary_links_to_the_full_list() -> None:
    summary = _section(_README, "What it does not do")
    assert f"{_DOCS}limitations.md" in summary
    full = _LIMITATIONS.read_text(encoding="utf-8")
    # The full list expands the summary's points.
    for topic in ("max_memory_mb", "review budget", "executable", "exact", "anchor_run", "Correlation"):
        assert topic in full, f"docs/limitations.md lost its {topic!r} point"


def test_the_full_benchmark_matrices_are_not_in_the_readme() -> None:
    for start, end in _MARKER_PAIRS:
        assert start not in _README
        assert end not in _README
    wide = [line for line in _README.splitlines() if line.startswith("|") and line.count("|") >= 11]
    assert not wide, f"a wide results matrix is back in the README: {wide[0][:80]}"


def test_the_full_matrices_live_in_the_benchmark_page_with_their_injection_markers() -> None:
    text = _BENCH_DOC.read_text(encoding="utf-8")
    for start, end in _MARKER_PAIRS:
        assert text.count(start) == 1
        assert text.count(end) == 1
        assert text.index(start) < text.index(end)
    assert "--readme docs/benchmarks.md" in text  # the regeneration command targets this file


def test_every_figure_in_the_headline_summary_appears_in_the_generated_results() -> None:
    text = _BENCH_DOC.read_text(encoding="utf-8")
    summary = _section(text, "Headline results")
    figures = re.findall(r"\d\.\d{4} ± \d\.\d{4}", summary)
    assert len(figures) >= 10, "the headline summary lost its figures"
    generated = "".join(text[text.index(start) : text.index(end)] for start, end in _MARKER_PAIRS)
    missing = sorted({f for f in figures if f not in generated})
    assert not missing, (
        f"quoted in the headline summary but absent from the generated results (stale?): {missing}"
    )


def test_the_readme_links_to_the_full_results() -> None:
    assert "docs/benchmarks.md" in _README


def test_readme_links_survive_being_rendered_on_pypi() -> None:
    """PyPI shows the README without the repository around it, so a relative link is dead there.
    Only absolute URLs and in-page anchors are allowed."""
    in_fence = False
    bad: list[str] = []
    for line in _README.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for target in re.findall(r"\]\(([^)\s]+)", line) + re.findall(r'src="([^"]+)"', line):
            if not target.startswith(("http://", "https://", "#", "mailto:")):
                bad.append(target)
    assert not bad, f"relative links in the README: {bad}"
