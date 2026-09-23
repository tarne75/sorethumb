"""Packaging identity contract: the PyPI distribution name, and its
agreement with installed metadata and the runtime-reported version.

Pins the P0-1 rename (PyPI distribution name "sorethumb-ml" -- "sorethumb"
itself is owned by an unrelated package -- while the import package, CLI,
and repo all stay "sorethumb") so a future accidental revert or drift is
caught here, not discovered at publish time.

The third leg of "metadata, runtime version and tag agree" -- that the
value pushed as a release tag also agrees -- cannot be checked from a
pytest process: there is no tag at test-collection time, and tests must
pass identically whether or not a release is imminent. That leg is
verified separately, at actual publish time, by
.github/workflows/publish.yml's verify-and-publish job (tag vs.
pyproject.toml vs. built wheel metadata vs. imported runtime version, all
compared for the exact tagged commit). What *is* checked here, as the
part of "tag agreement" that a pytest process actually can verify
year-round: the version string is always shaped so a tag *could* agree
with it -- see the last test below.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

import sorethumb

pytestmark = pytest.mark.contract

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _pyproject() -> dict:
    with _PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_pypi_distribution_name_is_pinned_to_sorethumb_ml():
    """pyproject.toml's [project].name is "sorethumb-ml", not "sorethumb"
    (taken by an unrelated package) and not silently changed again."""
    assert _pyproject()["project"]["name"] == "sorethumb-ml"


def test_installed_distribution_metadata_matches_runtime_version():
    """importlib.metadata's record for the "sorethumb-ml" distribution
    agrees with sorethumb.__version__. sorethumb.__version__ is *derived*
    from this exact lookup (see src/sorethumb/__init__.py), so this also
    guards against the lookup ever silently falling back to the
    "0+unknown" sentinel (e.g. if the distribution name and this lookup's
    argument ever drift apart again, as already happened once for two
    other call sites during the P0-1 rename -- store/models.py's
    _TRACKED_LIBRARIES and scripts/validation/identity.py's
    _TRACKED_PACKAGES)."""
    assert sorethumb.__version__ == importlib.metadata.version("sorethumb-ml")
    assert sorethumb.__version__ != "0+unknown"


def test_version_string_is_shaped_to_agree_with_a_release_tag():
    """sorethumb.__version__ matches the exact numeric shape
    publish.yml's `on.push.tags` filter (`v[0-9]+.[0-9]+.[0-9]+`) expects
    after its `v` prefix, and that scripts/release.sh's own VERSION
    argument validation requires. A version string in any other shape
    (a pre-release/dev suffix, for example) could never be tagged in a
    way that would agree with it -- publish.yml's own verify-and-publish
    job compares the tag against this value byte-for-byte."""
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", sorethumb.__version__)
