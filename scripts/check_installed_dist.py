#!/usr/bin/env python3
r"""Assert that an installed sorethumb-ml is the artifact it was meant to be.

Run with the *smoke venv's* interpreter, in isolated mode, after installing a
built wheel or sdist into a clean venv (``release-validation.yml`` does this on
Linux and Windows)::

    <venv>/python -I scripts/check_installed_dist.py \\
        --artifact sdist --file dist/sorethumb_ml-0.1.0.tar.gz \\
        --pyproject pyproject.toml --console-script <venv>/bin/sorethumb

Stdlib only, and ``-I`` keeps the checkout off ``sys.path``, so what is imported
is what pip installed -- never ``src/``.

It asserts that all of these agree on the version, and that the install is the
named artifact:

* the installed distribution's metadata (``importlib.metadata``);
* the runtime ``sorethumb_ml.__version__``;
* ``pyproject.toml``'s ``[project] version`` in the checkout;
* the artifact file name (``sorethumb_ml-<version>...``);
* the console script's ``--version`` output (``sorethumb <version>``);

and that pip recorded the install as coming from this exact artifact file
(``direct_url.json``: a ``.whl`` for ``--artifact wheel``, a ``.tar.gz`` for
``--artifact sdist``), that the package was imported from inside the venv, and
that the import surface that must work with no extra installed does.

Exit codes: 0 consistent; 1 one or more problems (all are listed); 2 bad usage.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

DIST_NAME = "sorethumb-ml"
_ARTIFACT_SUFFIX = {"wheel": ".whl", "sdist": ".tar.gz"}
_FILENAME_VERSION = re.compile(r"^sorethumb_ml-(?P<version>[^-]+?)(?:-[^/]*\.whl|\.tar\.gz)$")

# Imported with no extra installed: every shap/pandas/datasets import behind
# these is function-level, so a module-level one is a packaging regression.
IMPORTS_WITHOUT_EXTRAS = (
    "sorethumb_ml",
    "sorethumb_ml.explain.shap_tree",
    "sorethumb_ml.explain.gradient",
)


def artifact_version(filename: str) -> str | None:
    """Return the version in a ``sorethumb_ml-<version>...`` wheel or sdist file name, if it has one."""
    match = _FILENAME_VERSION.match(filename)
    return match["version"] if match else None


def version_problems(versions: dict[str, str | None]) -> list[str]:
    """One problem if the named versions are not all present and identical."""
    missing = sorted(source for source, version in versions.items() if not version)
    if missing:
        return [f"no version found from: {missing}"]
    if len(set(versions.values())) > 1:
        return [f"versions disagree: {dict(sorted(versions.items()))}"]
    return []


def direct_url_problems(direct_url_text: str | None, artifact: str, filename: str) -> list[str]:
    """Check pip's ``direct_url.json`` names this artifact file, and the right kind of file."""
    suffix = _ARTIFACT_SUFFIX[artifact]
    if not filename.endswith(suffix):
        return [f"--artifact {artifact} needs a {suffix} file, got {filename!r}"]
    if not direct_url_text:
        return ["no direct_url.json: pip did not record the install as coming from a local file"]
    try:
        url = str(json.loads(direct_url_text)["url"])
    except (ValueError, KeyError, TypeError):
        return [f"unreadable direct_url.json: {direct_url_text[:200]!r}"]
    installed_from = unquote(url).rstrip("/").rsplit("/", 1)[-1]
    if installed_from != filename:
        return [f"installed from {installed_from!r}, expected {filename!r}"]
    return []


def console_script_problems(output: str, returncode: int, version: str) -> list[str]:
    """``<script> --version`` must succeed and print exactly ``sorethumb <version>``."""
    if returncode != 0:
        return [f"console script --version exited {returncode}: {output.strip()[:200]!r}"]
    if output.strip() != f"sorethumb {version}":
        return [f"console script printed {output.strip()!r}, expected {f'sorethumb {version}'!r}"]
    return []


def location_problems(module_file: str, prefix: str) -> list[str]:
    """Require the package to have been imported from inside the venv, not a checkout."""
    path, root = Path(module_file).resolve(), Path(prefix).resolve()
    if root not in path.parents:
        return [f"sorethumb_ml imported from {path}, outside the venv {root}"]
    return []


def _pyproject_version(path: Path) -> str | None:
    with path.open("rb") as fh:
        project = tomllib.load(fh).get("project", {})
    if project.get("name") != DIST_NAME:
        raise SystemExit(f"{path}: [project] name is {project.get('name')!r}, expected {DIST_NAME!r}")
    version = project.get("version")
    return str(version) if version else None


def collect_problems(args: argparse.Namespace) -> list[str]:
    """Run every check, returning all problems rather than stopping at the first."""
    problems: list[str] = []
    artifact_file = Path(args.file)

    distribution = importlib.metadata.distribution(DIST_NAME)
    installed = distribution.version
    runtime = importlib.import_module("sorethumb_ml").__version__

    problems += version_problems(
        {
            "installed distribution metadata": installed,
            "runtime sorethumb_ml.__version__": runtime,
            "pyproject.toml": _pyproject_version(Path(args.pyproject)),
            "artifact file name": artifact_version(artifact_file.name),
        }
    )
    problems += direct_url_problems(
        distribution.read_text("direct_url.json"), args.artifact, artifact_file.name
    )

    sorethumb_ml = importlib.import_module("sorethumb_ml")
    problems += location_problems(str(sorethumb_ml.__file__), sys.prefix)

    for module in IMPORTS_WITHOUT_EXTRAS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- report every failure, whatever its type
            problems.append(f"import {module} failed: {type(exc).__name__}: {exc}")
    try:
        from sorethumb_ml import Config, SourceConfig, run_detection  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        problems.append(f"top-level API import failed: {type(exc).__name__}: {exc}")

    if args.console_script:
        completed = subprocess.run(  # noqa: S603 -- the path is the venv's own script, given by the caller
            [args.console_script, "--version"], capture_output=True, text=True, check=False
        )
        problems += console_script_problems(
            completed.stdout + completed.stderr, completed.returncode, installed
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact", choices=sorted(_ARTIFACT_SUFFIX), required=True)
    parser.add_argument("--file", required=True, help="the wheel or sdist file that was installed")
    parser.add_argument("--pyproject", required=True, help="the checkout's pyproject.toml")
    parser.add_argument("--console-script", help="path of the venv's sorethumb entry point")
    args = parser.parse_args(argv)

    problems = collect_problems(args)
    if problems:
        print(f"FAIL: installed {args.artifact} is not what was built:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    version = importlib.metadata.version(DIST_NAME)
    print(
        f"OK: {args.artifact} install of {DIST_NAME} {version} is consistent (metadata, runtime, pyproject, CLI)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
