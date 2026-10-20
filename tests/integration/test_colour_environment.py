"""A shell that forces colour must not change what the CLI tests see.

``scripts/release.sh`` runs the suite in the maintainer's own shell, where
``FORCE_COLOR`` is often set; CI never sets it. tests/conftest.py clears it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
# Asserts on the exact text of a CLI error, so escape codes would break it.
_TEXT_TEST = "tests/contract/test_cli.py::test_init_unwritable_config_location_fails_without_a_success_banner"


def test_cli_text_tests_pass_in_a_shell_that_forces_colour():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", _TEXT_TEST],
        check=False,
        cwd=_ROOT,
        env={**os.environ, "FORCE_COLOR": "3"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
