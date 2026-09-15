"""Hypothesis property test: an interrupted atomic_write must always leave
the target file exactly as it was before the write started -- for any prior
content and any content the interrupted write was attempting, not just the
one fixed example each per-artifact test (report HTML, results Parquet,
model files, ...) already covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sorethumb import _atomic
from tests.factories.hypothesis_profiles import scaled_examples

pytestmark = pytest.mark.property


@given(
    original=st.binary(max_size=200),
    attempted=st.binary(max_size=200),
)
@settings(max_examples=scaled_examples(100))
def test_interrupted_write_leaves_prior_content_untouched(
    tmp_path_factory: pytest.TempPathFactory, original: bytes, attempted: bytes
) -> None:
    target = tmp_path_factory.mktemp("atomic") / "artifact.bin"
    _atomic.atomic_write_bytes(target, original)

    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("simulated crash between fsync and rename")

    real_replace = _atomic.os.replace
    _atomic.os.replace = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="simulated crash"):
            _atomic.atomic_write_bytes(target, attempted)
    finally:
        _atomic.os.replace = real_replace  # type: ignore[assignment]

    assert target.read_bytes() == original, "interrupted write must leave the prior content untouched"
    leftover_tmp = [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftover_tmp == [], f"a failed write must clean up its own temp file, found: {leftover_tmp}"
