"""Golden-file comparison for stable JSON/HTML/CLI structures.

A golden file pins the *entire* shape of a deterministic output so a test
catches any unintended structural drift (a renamed field, a reordered
column, a changed layout) that a handful of substring assertions would
miss. Only use this for genuinely deterministic output -- no timestamps,
run ids, or other per-invocation values baked in without normalising them
away first.

Set ``UPDATE_GOLDEN=1`` in the environment to (re)write the golden files
instead of asserting against them, after reviewing the diff of an
intentional change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"


def _normalise_json(text: str) -> str:
    """Re-serialise with sorted keys and stable indentation so key-order
    alone never causes a spurious mismatch."""
    return json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


def assert_matches_golden(actual: str, name: str, *, is_json: bool = False) -> None:
    """Assert *actual* matches ``tests/golden/<name>`` exactly (after JSON
    normalisation when ``is_json``). Regenerate with ``UPDATE_GOLDEN=1``.
    """
    path = _GOLDEN_DIR / name
    normalised = _normalise_json(actual) if is_json else actual

    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalised, encoding="utf-8")
        return

    assert path.exists(), (
        f"No golden file at {path}. Run with UPDATE_GOLDEN=1 to create it "
        f"(review the diff before committing)."
    )
    expected = path.read_text(encoding="utf-8")
    assert normalised == expected, (
        f"Output no longer matches tests/golden/{name}. If this is an "
        f"intentional change, rerun with UPDATE_GOLDEN=1 and review the diff "
        f"before committing the updated golden file."
    )
