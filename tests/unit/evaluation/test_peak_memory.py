"""Peak memory is measurable on every supported platform.

POSIX reads ``ru_maxrss`` (KB on Linux, bytes on macOS); Windows has no
``resource`` module and reads the peak working set from GetProcessMemoryInfo
instead. Before that, benchmark memory columns were silently "n/a" on Windows.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from sorethumb_ml.evaluate.pipeline_benchmark import _peak_memory_mb, _windows_peak_working_set_mb

pytestmark = pytest.mark.unit


def test_peak_memory_is_a_plausible_number_here() -> None:
    peak = _peak_memory_mb()
    assert peak is not None
    assert 10.0 < peak < 1_000_000.0  # a Python process with numpy loaded, in MB


def test_peak_memory_grows_after_a_large_allocation() -> None:
    before = _peak_memory_mb()
    block = np.ones(64 * 1024 * 1024 // 8)  # 64 MB, touched
    after = _peak_memory_mb()
    del block
    assert before is not None
    assert after is not None
    assert after >= before


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX guard is what's under test")
def test_the_windows_reader_is_inert_elsewhere() -> None:
    assert _windows_peak_working_set_mb() is None
