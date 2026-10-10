"""Shared finiteness description for score arrays."""

from __future__ import annotations

import numpy as np


def nonfinite_summary(values: np.ndarray) -> str | None:
    """Describe the NaN/inf content of a float array, or return None if all finite.

    The text names each kind with its count and the position of the first bad
    value, e.g. ``"2 NaN, 1 +inf (first at index 7)"``.
    """
    finite = np.isfinite(values)
    if bool(finite.all()):
        return None
    n_nan = int(np.isnan(values).sum())
    n_pos = int(np.isposinf(values).sum())
    n_neg = int(np.isneginf(values).sum())
    parts = [f"{n} {kind}" for n, kind in ((n_nan, "NaN"), (n_pos, "+inf"), (n_neg, "-inf")) if n]
    first = int(np.flatnonzero(~finite.ravel())[0])
    return f"{', '.join(parts)} (first at index {first})"
