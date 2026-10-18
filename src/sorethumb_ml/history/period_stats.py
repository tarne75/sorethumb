"""Per-period statistics: what a period's data and scores looked like, for drift.

Each period run records, per group, a versioned summary (``period_stats``,
migration 013):

``population``
    Row count, and for every original field the run's features derive from:
    the missing count, and either 101 percentiles (numbers, dates as epoch
    seconds, booleans as 0/1) or the 20 most common values with the rest as
    ``other`` (anything else). The id and time columns are left out: they
    change every period by construction.
``scores``
    Per detector, 101 percentiles of its raw score and how many rows its own
    boundary flagged. These are on that period's self-fitted scale, so they are
    *not* comparable across periods on their own; T2's anchor run supplies a
    common scale.
``composite``
    Percentiles of the combined score, the threshold and contamination used,
    the number flagged and the ensemble policy.

Population drift compares two sets of these summaries (for example a
window's groups and periods against the prior window's) with the Population
Stability Index, per field. Summaries are combined as a mixture weighted by
row count, so periods and groups of different sizes count in proportion.
PSI on numbers is computed over the baseline's deciles, with the current side's
share in each bin read off its interpolated percentile function. With 101
stored percentiles that share is accurate to about one percentage point per
bin. Categorical PSI uses the union of both sides' top values plus ``other``.
Missing values are a bin of their own in both. The conventional bands
(below 0.1 stable, 0.1-0.25 moderate, above 0.25 large) are heuristics, not a
significance test.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

STATS_VERSION = 1
PERCENTILES = list(range(101))
TOP_K = 20
_PSI_EPS = 1e-4
PSI_BANDS = ((0.1, "stable"), (0.25, "moderate"), (math.inf, "large"))


# ---------------------------------------------------------------------------
# Computing a period's statistics
# ---------------------------------------------------------------------------


def _percentiles(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, PERCENTILES)]


def field_summary(series: pl.Series) -> dict[str, Any] | None:
    """Summarise one column; None for a type that supports neither percentiles nor counting."""
    n = len(series)
    dtype = series.dtype
    if dtype.is_nested() or dtype == pl.Object:  # lists, structs, arrays: no meaningful summary
        return None
    try:
        if dtype == pl.Boolean:
            values = series.drop_nulls().cast(pl.Float64)
        elif dtype.is_temporal():
            if dtype == pl.Date or isinstance(dtype, pl.Datetime):
                values = series.drop_nulls().dt.epoch("s").cast(pl.Float64)
            else:  # Duration / Time: a plain number of the unit
                values = series.drop_nulls().to_physical().cast(pl.Float64)
        elif dtype.is_numeric():
            values = series.drop_nulls().cast(pl.Float64)
        else:
            values = None
    except pl.exceptions.PolarsError:
        return None
    if values is not None:
        arr = values.to_numpy()
        arr = arr[np.isfinite(arr)]
        missing = n - len(arr)
        if len(arr) == 0:
            return {"kind": "numeric", "n": n, "missing": missing, "percentiles": None}
        return {
            "kind": "numeric",
            "n": n,
            "missing": missing,
            "percentiles": _percentiles(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    try:
        text = series.cast(pl.String)
    except pl.exceptions.PolarsError:
        return None
    present = text.drop_nulls()
    counts = present.value_counts(sort=True)
    top: list[tuple[str, int]] = [(str(v), int(c)) for v, c in counts.head(TOP_K).iter_rows()]
    return {
        "kind": "categorical",
        "n": n,
        "missing": n - len(present),
        "top": [list(t) for t in top],
        "other": len(present) - sum(c for _, c in top),
        "distinct": counts.height,
    }


def compute_period_stats(
    *,
    df_group: pl.DataFrame,
    fields: list[str],
    raw_scores: dict[str, np.ndarray],
    natural_flags: dict[str, np.ndarray],
    composite_score: np.ndarray,
    composite_threshold: float | None,
    contamination_used: float | None,
    n_flagged: int,
    combination: str,
    ensemble_policy: str | None,
) -> dict[str, Any]:
    """Return the versioned per-period summary for one group (see the module docstring)."""
    population: dict[str, Any] = {}
    for name in fields:
        if name in df_group.columns:
            summary = field_summary(df_group[name])
            if summary is not None:
                population[name] = summary
    scores: dict[str, Any] = {}
    for det, raw in raw_scores.items():
        arr = np.asarray(raw, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        flags = np.asarray(natural_flags.get(det, np.zeros(len(arr), dtype=bool)), dtype=bool)
        scores[det] = {
            "raw_percentiles": _percentiles(finite) if len(finite) else None,
            "natural_count": int(flags.sum()),
            "natural_rate": float(flags.mean()) if len(flags) else None,
            "scale": "self_fitted",
        }
    comp = np.asarray(composite_score, dtype=np.float64)
    comp = comp[np.isfinite(comp)]
    threshold = (
        None if composite_threshold is None or not math.isfinite(composite_threshold) else composite_threshold
    )
    return {
        "stats_version": STATS_VERSION,
        "n_rows": int(df_group.height),
        "population": population,
        "scores": scores,
        "composite": {
            "percentiles": _percentiles(comp) if len(comp) else None,
            "threshold": threshold,
            "contamination_used": contamination_used,
            "flagged": int(n_flagged),
            "combination": combination,
            "ensemble_policy": ensemble_policy,
        },
    }


# ---------------------------------------------------------------------------
# Population drift
# ---------------------------------------------------------------------------


def _mixture_cdf(parts: list[tuple[list[float], int]], x: np.ndarray) -> np.ndarray:
    """Weighted mixture of percentile-interpolated CDFs at *x* (weights = non-missing counts)."""
    total = sum(w for _, w in parts)
    out = np.zeros(len(x), dtype=np.float64)
    for pct, w in parts:
        q = np.asarray(pct, dtype=np.float64)
        p = np.asarray(PERCENTILES, dtype=np.float64) / 100.0
        # np.interp needs increasing x; collapse ties to the last percentile reaching each value.
        uq, idx = np.unique(q, return_index=True)
        last = np.concatenate([idx[1:] - 1, [len(q) - 1]])
        cdf = np.interp(x, uq, p[last], left=0.0, right=1.0)
        out += w * cdf
    return out / total if total else out


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, _PSI_EPS, None)
    a = np.clip(actual, _PSI_EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_psi(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> float | None:
    base_parts = [(s["percentiles"], s["n"] - s["missing"]) for s in baseline if s.get("percentiles")]
    cur_parts = [(s["percentiles"], s["n"] - s["missing"]) for s in current if s.get("percentiles")]
    base_n = sum(s["n"] for s in baseline)
    cur_n = sum(s["n"] for s in current)
    if not base_n or not cur_n:
        return None
    base_missing = sum(s["missing"] for s in baseline) / base_n
    cur_missing = sum(s["missing"] for s in current) / cur_n
    if not base_parts or not cur_parts:
        return _psi(np.array([base_missing, 1 - base_missing]), np.array([cur_missing, 1 - cur_missing]))
    # Baseline deciles from its mixture CDF, evaluated on a grid of every stored percentile.
    grid = np.unique(np.concatenate([np.asarray(p, dtype=np.float64) for p, _ in base_parts]))
    base_cdf = _mixture_cdf(base_parts, grid)
    edges = np.unique([float(np.interp(q, base_cdf, grid)) for q in np.linspace(0.1, 0.9, 9)])

    def shares(parts: list[tuple[list[float], int]]) -> np.ndarray:
        cdf = _mixture_cdf(parts, edges)
        return np.diff(np.concatenate([[0.0], cdf, [1.0]]))

    base = np.concatenate([[base_missing], (1 - base_missing) * shares(base_parts)])
    cur = np.concatenate([[cur_missing], (1 - cur_missing) * shares(cur_parts)])
    return _psi(base, cur)


def _categorical_psi(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> float | None:
    def counts(parts: list[dict[str, Any]]) -> tuple[dict[str, int], int, int, int]:
        c: dict[str, int] = {}
        other = missing = n = 0
        for s in parts:
            for value, k in s["top"]:
                c[value] = c.get(value, 0) + int(k)
            other += int(s["other"])
            missing += int(s["missing"])
            n += int(s["n"])
        return c, other, missing, n

    bc, bo, bm, bn = counts(baseline)
    cc, co, cm, cn = counts(current)
    if not bn or not cn:
        return None
    keys = sorted(set(bc) | set(cc))
    base = np.array([bc.get(k, 0) for k in keys] + [bo, bm], dtype=np.float64) / bn
    cur = np.array([cc.get(k, 0) for k in keys] + [co, cm], dtype=np.float64) / cn
    return _psi(base, cur)


def psi_band(psi: float) -> str:
    """Name the conventional band a PSI value falls in (a heuristic, not a test)."""
    return next(name for limit, name in PSI_BANDS if psi < limit)


def population_drift(
    baseline: list[dict[str, Any]], current: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """PSI per field between two sets of period summaries: ``{field: {"psi", "band", "kind"}}``.

    A field is compared only when both sides summarised it the same way; a
    field present on one side only, or numeric on one and categorical on the
    other, is reported with ``psi`` None and ``band`` ``"not_comparable"``.
    """
    fields: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for side, stats in ((0, baseline), (1, current)):
        for st in stats:
            for name, summary in st.get("population", {}).items():
                fields.setdefault(name, ([], []))[side].append(summary)
    out: dict[str, dict[str, Any]] = {}
    for name, (base, cur) in sorted(fields.items()):
        kinds = {s["kind"] for s in base + cur}
        if not base or not cur or len(kinds) != 1:
            out[name] = {"psi": None, "band": "not_comparable", "kind": "/".join(sorted(kinds))}
            continue
        kind = kinds.pop()
        psi = _numeric_psi(base, cur) if kind == "numeric" else _categorical_psi(base, cur)
        out[name] = {"psi": psi, "band": psi_band(psi) if psi is not None else "not_comparable", "kind": kind}
    return out
