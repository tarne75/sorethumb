#!/usr/bin/env python3
"""Summarise pytest JUnit XML failures, grouped by cause.

Groups every failed or errored test case by (exception type, first frame inside
this repository), so a run with hundreds of failures that share one root cause
reads as one line with a count rather than hundreds of tracebacks.

Two output modes:

* plain text (default) -- for a terminal or a pasted log;
* ``--github-annotations`` -- one GitHub Actions ``::error`` workflow command
  per failure group, so the grouped summary is readable through the checks API
  (``/repos/{owner}/{repo}/check-runs/{id}/annotations``) even where a job's raw
  logs and uploaded artifacts can't be downloaded. GitHub keeps at most 10
  error annotations per step, so groups beyond ``--max-annotations`` are folded
  into the last one.

Stdlib only, so CI can run it without installing the project.

Usage:
    python3 scripts/summarise_junit.py results.xml [more.xml ...]
    python3 scripts/summarise_junit.py --github-annotations --label windows-fast results.xml

Exit code: 0 always (this reports; the test step itself is what fails a job),
2 on bad usage or an unreadable file.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# A traceback frame line pytest writes in --tb=short/long output, e.g.
#   src/sorethumb_ml/_atomic.py:95: in _fsync_path
#   D:\a\sorethumb\sorethumb\src\sorethumb_ml\_atomic.py:95: OSError
_FRAME_RE = re.compile(
    r"^\s*(?P<path>(?:[A-Za-z]:)?[^\s:]*(?:src|tests|scripts|docs)[\\/][^:\s]+\.py):(?P<line>\d+)"
)
_EXC_RE = re.compile(
    r"^E\s+(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Failure|failed)\b)"
)
# The "path:line: ExceptionType" line pytest ends each frame section with.
_TAIL_EXC_RE = re.compile(r"\.py:\d+: (?P<exc>[A-Za-z_][\w.]*)\s*$")
_MAX_DETAIL_CHARS = 3500


@dataclass
class Group:
    """Failures sharing one exception type and first in-repo frame."""

    key: tuple[str, str]
    tests: list[str] = field(default_factory=list)
    sample: str = ""


def _repo_relative(path: str) -> str:
    norm = path.replace("\\", "/")
    for anchor in ("/src/", "/tests/", "/scripts/", "/docs/"):
        idx = norm.find(anchor)
        if idx != -1:
            return norm[idx + 1 :]
    return norm


def _classify(text: str, message: str) -> tuple[str, str]:
    """Return (exception type, most specific in-repo frame) for one failure.

    The frame is the deepest one under ``src/`` (where the error was raised in
    library code), falling back to the deepest in-repo frame of any kind.
    """
    frames: list[str] = []
    exc = ""
    for line in text.splitlines():
        m = _FRAME_RE.match(line)
        if m:
            frames.append(f"{_repo_relative(m.group('path'))}:{m.group('line')}")
            tail = _TAIL_EXC_RE.search(line)
            if tail:
                exc = tail.group("exc")
        if not exc:
            e = _EXC_RE.match(line)
            if e:
                exc = e.group("exc")
    if not exc:
        exc = "AssertionError" if message.lstrip().startswith("assert") else message.split(":", 1)[0].strip()
        exc = (exc or "unknown")[:80]
    src_frames = [f for f in frames if f.startswith("src/")]
    frame = (src_frames or frames or ["(no in-repo frame)"])[-1]
    return exc, frame


def collect(paths: list[Path]) -> dict[tuple[str, str], Group]:
    """Parse every JUnit file and group failing/erroring test cases."""
    groups: dict[tuple[str, str], Group] = {}
    for path in paths:
        root = ET.parse(path).getroot()  # noqa: S314 -- our own CI output, not untrusted input
        for case in root.iter("testcase"):
            for kind in ("failure", "error"):
                node = case.find(kind)
                if node is None:
                    continue
                text = node.text or ""
                message = node.get("message", "")
                key = _classify(text, message)
                group = groups.setdefault(key, Group(key=key))
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                group.tests.append(name if kind == "failure" else f"{name} [{kind}]")
                if not group.sample:
                    group.sample = (message + "\n" + text).strip()
    return groups


def _counts(paths: list[Path]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for path in paths:
        root = ET.parse(path).getroot()  # noqa: S314
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for suite in suites:
            for attr in ("tests", "failures", "errors", "skipped"):
                totals[attr] += int(suite.get(attr, "0") or 0)
    return dict(totals)


def _render_group(group: Group, max_tests: int = 15) -> str:
    exc, frame = group.key
    lines = [f"{len(group.tests)} x {exc} at {frame}"]
    lines += [f"  - {t}" for t in group.tests[:max_tests]]
    if len(group.tests) > max_tests:
        lines.append(f"  ... and {len(group.tests) - max_tests} more")
    sample = group.sample
    if len(sample) > _MAX_DETAIL_CHARS:
        sample = sample[:1500] + "\n[...]\n" + sample[-(_MAX_DETAIL_CHARS - 1500) :]
    lines.append("  first failure detail:")
    lines += [f"    {s}" for s in sample.splitlines()]
    return "\n".join(lines)


def _escape_annotation(text: str) -> str:
    # GitHub workflow-command data escaping.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--github-annotations", action="store_true")
    parser.add_argument("--label", default="tests")
    parser.add_argument("--max-annotations", type=int, default=9)
    args = parser.parse_args(argv)

    existing = [p for p in args.files if p.is_file()]
    missing = [p for p in args.files if not p.is_file()]
    try:
        groups = collect(existing)
        counts = _counts(existing)
    except (OSError, ET.ParseError) as exc:
        print(f"cannot read JUnit XML: {exc}", file=sys.stderr)
        return 2

    ordered = sorted(groups.values(), key=lambda g: (-len(g.tests), g.key))
    header = (
        f"[{args.label}] tests={counts.get('tests', 0)} failures={counts.get('failures', 0)} "
        f"errors={counts.get('errors', 0)} skipped={counts.get('skipped', 0)} groups={len(ordered)}"
    )
    if missing:
        header += f" missing-files={','.join(str(p) for p in missing)}"

    if not args.github_annotations:
        print(header)
        for group in ordered:
            print(_render_group(group))
        return 0

    # The header always goes out as a notice, so a fully green run still reports its counts.
    print(f"::notice title={args.label} summary::{_escape_annotation(header)}")
    head = ordered[: args.max_annotations]
    rest = ordered[args.max_annotations :]
    for idx, group in enumerate(head, 1):
        print(
            f"::error title={args.label} group {idx}/{len(ordered)}::{_escape_annotation(_render_group(group))}"
        )
    if rest:
        folded = "\n\n".join(_render_group(g, max_tests=5) for g in rest)
        print(f"::error title={args.label} remaining {len(rest)} groups::{_escape_annotation(folded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
