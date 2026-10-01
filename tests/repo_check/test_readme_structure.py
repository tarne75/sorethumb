"""P2-6: the README is the PyPI long description and the first thing a reader sees.

It leads with how to install, a core-only example and the short list of limitations;
the full benchmark matrices live in ``docs/benchmarks.md``, with only a compact summary
(and a link) in the README. These checks keep that shape, and keep the summary honest:
every figure quoted in the README has to appear in the full results page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_check

_ROOT = Path(__file__).resolve().parents[2]
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_BENCH_DOC = _ROOT / "docs" / "benchmarks.md"

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


def test_readme_leads_with_install_a_core_only_example_and_the_limitations_summary() -> None:
    assert _h2_titles(_README)[:3] == ["Installation", "60-second quickstart", "What it does not do"]


def test_the_core_only_example_needs_no_extras_and_no_network() -> None:
    section = _README.split("## 60-second quickstart", 1)[1].split("\n## ", 1)[0]
    assert "core dependencies" in section
    assert "No network access" in section
    assert "pip install" not in section  # nothing beyond the core install


def test_the_limitations_summary_links_to_the_full_section() -> None:
    summary = _README.split("## What it does not do", 1)[1].split("\n## ", 1)[0]
    assert "(#honest-limitations)" in summary
    assert "## Honest limitations" in _README  # and that section still exists


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


def test_every_figure_in_the_readme_summary_appears_in_the_full_results() -> None:
    summary = _README.split("## Benchmark results (summary)", 1)[1].split("\n## ", 1)[0]
    figures = re.findall(r"\d\.\d{4} ± \d\.\d{4}", summary)
    assert len(figures) >= 10, "the summary table lost its figures"
    full = _BENCH_DOC.read_text(encoding="utf-8")
    missing = sorted({f for f in figures if f not in full})
    assert not missing, f"quoted in the README but absent from docs/benchmarks.md (stale?): {missing}"


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
