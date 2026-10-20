#!/usr/bin/env python3
"""Check (and optionally apply) a TestPyPI rehearsal version.

A rehearsal publishes a pre-release of the version in ``pyproject.toml``, e.g.
``0.1.0rc1`` for ``0.1.0``. TestPyPI files are immutable, so each rehearsal needs a
version not used before; a pre-release can never take the real release's version.

Usage:
    python scripts/rehearsal_version.py 0.1.0rc1           # check only
    python scripts/rehearsal_version.py 0.1.0rc1 --apply   # also write it to pyproject.toml

Stdlib only: runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
_PRE_RELEASE = re.compile(r"(?P<base>\d+\.\d+\.\d+)(?:a|b|rc)\d+")
_VERSION_LINE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)


def problem(version: str, base: str) -> str | None:
    """Return why *version* is not a valid rehearsal of *base*, or None."""
    match = _PRE_RELEASE.fullmatch(version)
    if match is None:
        return f"{version!r} is not a pre-release (X.Y.ZaN, X.Y.ZbN or X.Y.ZrcN), e.g. {base}rc1"
    if match["base"] != base:
        return f"{version!r} is not a pre-release of {base} (pyproject.toml's version)"
    return None


def apply(version: str, path: Path = PYPROJECT) -> None:
    """Replace ``[project] version`` in *path* with *version*."""
    text = path.read_text(encoding="utf-8")
    new, count = _VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if count != 1:
        msg = f"no 'version = \"...\"' line in {path}"
        raise SystemExit(msg)
    path.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version")
    parser.add_argument("--apply", action="store_true", help="write the version to pyproject.toml")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    args = parser.parse_args(argv)

    base = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))["project"]["version"]
    reason = problem(args.version, base)
    if reason:
        print(f"error: {reason}", file=sys.stderr)
        return 2
    if args.apply:
        apply(args.version, args.pyproject)
        print(f"pyproject.toml version: {base} -> {args.version}")
    else:
        print(f"ok: {args.version} is a rehearsal of {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
