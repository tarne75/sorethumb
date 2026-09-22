"""Identity inputs for staleness detection: code revision and dependency versions.

Each function here does light, cheap I/O (a subprocess call, reading
installed package metadata) but no dataset loading or model fitting — kept
separate from schema.py only because schema.py must stay import-cheap and
dependency-free beyond polars.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def code_revision(repo_root: Path) -> str:
    """Return a short revision string identifying the current code state.

    A short git SHA, suffixed with "+dirty" when the working tree has
    uncommitted changes -- an uncommitted edit is a different "code
    revision" for resume purposes even though HEAD hasn't moved. Falls back
    to "unknown" outside a git checkout (e.g. an extracted sdist) rather than
    raising, since this is only used to invalidate cached results, not to
    gate correctness.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"

    try:
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
            != ""
        )
    except (subprocess.SubprocessError, OSError):
        dirty = False

    return f"{sha}+dirty" if dirty else sha


# Packages whose version can change the numbers this script produces.
# "sorethumb-ml" is this package's PyPI distribution name (see
# prompts/release-launch-plan.md Item 1) -- importlib.metadata looks
# packages up by that, not by the "sorethumb" import name.
_TRACKED_PACKAGES = ("sorethumb-ml", "numpy", "scipy", "scikit-learn", "polars", "pydantic")


def dependency_versions() -> dict[str, str]:
    """Return {package: version} for every package that can affect results."""
    from importlib import metadata  # noqa: PLC0415

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions
