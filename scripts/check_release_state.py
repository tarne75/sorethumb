#!/usr/bin/env python3
"""Reject a release tag whose repository state contradicts the release.

Run for the version about to be tagged, against the exact commit being tagged.
It exists because a release is more than a version bump: the changelog date,
SECURITY.md and README all make claims that are true *before* the tag ("not yet
released", "Unreleased", a placeholder date) and false *after* it, and nothing
else forces those edits. They belong in one release-state commit, made on the
tag day; this check is what stops a tag from being cut without it.

Checks (stdlib only, so CI can run it without installing the project):

* the version is a final ``X.Y.Z`` (the only shape ``publish.yml`` tags);
* ``pyproject.toml``'s version equals it;
* ``CHANGELOG.md``'s newest release section is for that version, has content,
  and a real ``YYYY-MM-DD`` date that is not in the future and -- when
  ``--tag-date`` is given -- is exactly the tag date; the ``[Unreleased]`` and
  ``[X.Y.Z]`` footer links name ``vX.Y.Z``; and ``[Unreleased]`` holds no entries
  (anything left there would be missing from the release's own notes);
* ``SECURITY.md`` and ``README.md`` carry no pre-release wording ("has not yet
  had its first tagged release", "not yet on PyPI", "Status: pre-release", and
  -- from 1.0 on -- "pre-1.0") that the version being tagged contradicts.

Dates are UTC: the tag date is the UTC date the tag is pushed.

Usage:
    python3 scripts/check_release_state.py --version 0.1.0
    python3 scripts/check_release_state.py --version 0.1.0 --tag-date "$(date -u +%F)"

Exit codes: 0 consistent; 1 one or more problems (all are listed); 2 bad usage.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_HEADING_RE = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>\S+))?\s*$", re.MULTILINE)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# (regex, what it claims). Searched case-insensitively in whitespace-collapsed text
# (line wraps and blockquote markers removed), so a sentence split over lines still matches.
_PRE_RELEASE_WORDING: list[tuple[str, str]] = [
    (r"has not yet had its first tagged release", "says there has been no tagged release"),
    (r"not yet (?:been )?(?:published|released|tagged)", "says the package is not yet published/released"),
    (r"not yet on pypi", "says the package is not yet on PyPI"),
    (r"status:\s*pre-?release", "labels the project 'Status: pre-release'"),
]
_PRE_1_0_WORDING = (r"pre-1\.0", "calls the project pre-1.0")
_WORDING_FILES = ("SECURITY.md", "README.md")


def _collapse(text: str) -> str:
    """Join wrapped lines (and drop leading blockquote markers) so phrases match across line breaks."""
    return re.sub(r"\s*\n\s*(?:>\s*)?", " ", text)


def _unreleased_problems(text: str) -> list[str]:
    """Flag entries still sitting under ``## [Unreleased]`` when a release is being tagged.

    Everything that ships belongs in the release's own section; whatever is left
    under [Unreleased] is, by definition, not part of the release notes the tag
    publishes. A trailing link-reference block is part of the file, not of the section.
    """
    heading = re.search(r"^## \[Unreleased\]\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if heading is None:
        return []
    following = _HEADING_RE.search(text, heading.end())
    body = text[heading.end() : following.start() if following else len(text)]
    body = re.sub(r"^\[[^\]]+\]:\s*\S+\s*$", "", body, flags=re.MULTILINE)
    entries = [line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not entries:
        return []
    return [
        f"CHANGELOG.md still has {len(entries)} line(s) of entries under [Unreleased]: move them into the "
        "release section, so the release notes the tag publishes include them, and leave [Unreleased] empty."
    ]


def _changelog_problems(text: str, version: str, tag_date: date | None, today: date) -> list[str]:
    return _release_section_problems(text, version, tag_date, today) + _unreleased_problems(text)


def _release_section_problems(text: str, version: str, tag_date: date | None, today: date) -> list[str]:
    problems: list[str] = []
    headings = [m for m in _HEADING_RE.finditer(text) if m.group("name").lower() != "unreleased"]
    if not headings:
        return ["CHANGELOG.md has no released '## [X.Y.Z] - YYYY-MM-DD' section."]

    top = headings[0]
    if top.group("name") != version:
        problems.append(
            f"CHANGELOG.md's newest release section is [{top.group('name')}], not [{version}]: "
            f"rename [Unreleased] to '## [{version}] - <tag date>' (with real content under it)."
        )
        return problems

    raw_date = top.group("date")
    if raw_date is None or not _ISO_DATE_RE.match(raw_date):
        problems.append(
            f"CHANGELOG.md's [{version}] heading has no valid YYYY-MM-DD date (found {raw_date!r}); "
            "set it to the actual tag date."
        )
    else:
        try:
            changelog_date = date.fromisoformat(raw_date)
        except ValueError:
            problems.append(f"CHANGELOG.md's [{version}] date {raw_date!r} is not a real calendar date.")
        else:
            if changelog_date > today:
                problems.append(
                    f"CHANGELOG.md's [{version}] date {raw_date} is in the future (today is {today} UTC)."
                )
            if tag_date is not None and changelog_date != tag_date:
                problems.append(
                    f"CHANGELOG.md's [{version}] date is {raw_date} but the tag date is {tag_date} (UTC): "
                    "set the changelog date to the actual tag date, in the release-state commit."
                )

    next_heading = _HEADING_RE.search(text, top.end())
    body = text[top.end() : next_heading.start() if next_heading else len(text)]
    # A trailing link-reference block belongs to the file, not to the section.
    body = re.sub(r"^\[[^\]]+\]:\s*\S+\s*$", "", body, flags=re.MULTILINE)
    if not body.strip():
        problems.append(f"CHANGELOG.md's [{version}] section is empty.")

    for label, link_re in (
        ("[Unreleased]", r"^\[Unreleased\]:\s*(\S+)\s*$"),
        (f"[{version}]", rf"^\[{re.escape(version)}\]:\s*(\S+)\s*$"),
    ):
        link = re.search(link_re, text, flags=re.MULTILINE)
        if link is None:
            if label != "[Unreleased]":
                problems.append(
                    f"CHANGELOG.md has no '{label}: <url>' link reference at the foot of the file."
                )
        elif f"v{version}" not in link.group(1):
            problems.append(f"CHANGELOG.md's {label} link ({link.group(1)}) does not refer to v{version}.")
    return problems


def _wording_problems(root: Path, major: int) -> list[str]:
    patterns = list(_PRE_RELEASE_WORDING)
    if major >= 1:
        patterns.append(_PRE_1_0_WORDING)
    problems: list[str] = []
    for name in _WORDING_FILES:
        path = root / name
        if not path.is_file():
            continue
        text = _collapse(path.read_text(encoding="utf-8"))
        for pattern, claim in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                problems.append(
                    f"{name} {claim} ('...{match.group(0)}...'), which the version being tagged contradicts: "
                    "update it in the release-state commit."
                )
    return problems


def check_release_state(
    root: Path, version: str, *, tag_date: date | None = None, today: date | None = None
) -> list[str]:
    """Return every problem found (empty list = consistent) for tagging *version* at *root*."""
    today = today or datetime.now(UTC).date()
    match = _VERSION_RE.match(version)
    if match is None:
        return [f"version must be a final X.Y.Z (the only shape publish.yml tags), got {version!r}."]

    problems: list[str] = []
    pyproject_path = root / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        pyproject_version = tomllib.load(fh)["project"]["version"]
    if pyproject_version != version:
        problems.append(f"pyproject.toml says version = {pyproject_version!r}, not {version!r}.")

    changelog_path = root / "CHANGELOG.md"
    if changelog_path.is_file():
        problems += _changelog_problems(changelog_path.read_text(encoding="utf-8"), version, tag_date, today)
    else:
        problems.append("CHANGELOG.md is missing.")

    problems += _wording_problems(root, int(match.group(1)))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", required=True, help="The X.Y.Z version about to be tagged (no leading 'v')."
    )
    parser.add_argument(
        "--tag-date",
        type=date.fromisoformat,
        default=None,
        help="The UTC date the tag is being created, YYYY-MM-DD; the changelog date must equal it.",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="Repository root (default: this checkout)."
    )
    args = parser.parse_args(argv)

    problems = check_release_state(args.root, args.version, tag_date=args.tag_date)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s): not ready to tag v{args.version}.", file=sys.stderr)
        return 1
    print(f"ok: repository state is consistent with tagging v{args.version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
