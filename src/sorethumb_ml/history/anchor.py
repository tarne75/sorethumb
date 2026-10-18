"""The anchor run: one fixed model, calibrator and threshold that scores every period.

Each period run fits and calibrates on its own rows, so its flag rate ``r``
is measured against a bar that moves with the data. With ``history.anchor_run``
set, every period run also scores its rows with the anchor run's persisted
models, calibrators and *fixed* flag rule, giving ``r*``: the rate above an
unmoving bar. Then a change in a group's count between two windows splits
exactly into three parts (``N`` rows, ``r`` own rate, ``r*`` anchor rate)::

    Δcount = (N_t - N_{t-1}) * r_{t-1}                               volume
           + N_t * (r*_t - r*_{t-1})                                 score drift (fixed threshold)
           + N_t * ((r_t - r*_t) - (r_{t-1} - r*_{t-1}))             threshold drift (self-recalibration)

The fixed rule
--------------
Every fitting run saves its rule per group (``models/<run>/<group>/anchor_rule.json``,
plain JSON), so any fitting run can later serve as an anchor:

- ``composite``: the weighted average of calibrated scores against the run's
  own threshold (its contamination quantile, or the lowest flagged score
  under a numeric contamination), with the run's weights.
- ``intersection``/``union`` with ``contamination="auto"``: each detector's own
  fitted boundary (``natural_flag``), which the persisted model already fixes.
- ``intersection``/``union`` with a numeric contamination: per detector, the
  lowest calibrated score it flagged in the anchor run.

Score drift is also measured directly: the Population Stability Index of the
period's anchor-scale combined score against the anchor run's own, and the
share of rows above the anchor's 95th percentile (5% by construction on the
anchor's own rows).

A period is *not comparable* to the anchor, with the reason recorded, when the
anchor has no rule or no model for the group, the anchor's feature plan cannot
be applied to the period's data, or the feature schema drifted. Nothing is
guessed.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from sorethumb_ml._atomic import atomic_write_text
from sorethumb_ml.errors import ModelSchemaDriftWarning, ModelVersionMismatchWarning

if TYPE_CHECKING:
    import polars as pl

    from sorethumb_ml.store.workspace import Workspace

logger = logging.getLogger(__name__)

RULE_FILENAME = "anchor_rule.json"
RULE_VERSION = 1
_PERCENTILES = list(range(101))


# ---------------------------------------------------------------------------
# The fixed rule a fitting run saves
# ---------------------------------------------------------------------------


def build_rule(
    *,
    combination: str,
    contamination: str | float,
    weights: dict[str, float],
    threshold: float | None,
    calibrated: dict[str, np.ndarray],
    combined: np.ndarray,
    source_row: np.ndarray | None,
) -> dict[str, Any]:
    """Return a fitting run's fixed flag rule for one group (see the module docstring)."""
    from sorethumb_ml.scoring.combine import _exact_k_flags  # noqa: PLC0415

    rule: dict[str, Any] = {
        "rule_version": RULE_VERSION,
        "combination": combination,
        "contamination": contamination,
        "detectors": sorted(calibrated),
        "n_rows": len(combined),
        "combined_percentiles": [float(v) for v in np.percentile(combined, _PERCENTILES)]
        if len(combined)
        else None,
    }
    if combination == "composite":
        rule["weights"] = {d: float(w) for d, w in weights.items()}
        rule["threshold"] = threshold if threshold is not None and math.isfinite(threshold) else None
    elif not isinstance(contamination, str):
        cutoffs: dict[str, float | None] = {}
        for det, cal in calibrated.items():
            flags = _exact_k_flags(np.asarray(cal, dtype=np.float64), float(contamination), source_row)
            cutoffs[det] = float(np.asarray(cal)[flags].min()) if flags.any() else None
        rule["cutoffs"] = cutoffs
    return rule


def save_rule(ws: Workspace, run_id: str, group_key: str, rule: dict[str, Any]) -> None:
    """Write the rule next to the group's models."""
    atomic_write_text(ws.models_dir(run_id, group_key) / RULE_FILENAME, json.dumps(rule, allow_nan=False))


def load_rule(ws: Workspace, run_id: str, group_key: str) -> dict[str, Any] | None:
    """Read a run's saved rule for a group, or None when it has none (or it is unreadable)."""
    path = ws.root / "models" / run_id / group_key / RULE_FILENAME
    if not path.is_file():
        return None
    try:
        rule = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rule if isinstance(rule, dict) and rule.get("rule_version") == RULE_VERSION else None


def combined_score(rule: dict[str, Any], calibrated: dict[str, np.ndarray]) -> np.ndarray:
    """Return the anchor-scale combined score: the rule's own ranking formula (weighted mean, median or max)."""
    names = [d for d in rule["detectors"] if d in calibrated]
    matrix = np.column_stack([np.asarray(calibrated[d], dtype=np.float64) for d in names])
    if rule["combination"] == "composite":
        w = np.array([rule["weights"].get(d, 0.0) for d in names], dtype=np.float64)
        return np.asarray(matrix @ (w / w.sum()) if w.sum() > 0 else matrix.mean(axis=1))
    if rule["combination"] == "intersection":
        return np.asarray(np.median(matrix, axis=1))
    return np.asarray(matrix.max(axis=1))


def apply_rule(
    rule: dict[str, Any], calibrated: dict[str, np.ndarray], natural_flags: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Return flags under the fixed rule; None when the rule cannot flag (e.g. the anchor flagged nothing)."""
    combination = rule["combination"]
    if combination == "composite":
        if rule.get("threshold") is None:
            return None
        return combined_score(rule, calibrated) >= float(rule["threshold"])
    names = rule["detectors"]
    if "cutoffs" in rule:
        if any(rule["cutoffs"].get(d) is None for d in names):
            return None
        per = np.column_stack([np.asarray(calibrated[d]) >= float(rule["cutoffs"][d]) for d in names])
    else:
        per = np.column_stack([np.asarray(natural_flags[d], dtype=bool) for d in names])
    return per.all(axis=1) if combination == "intersection" else per.any(axis=1)


# ---------------------------------------------------------------------------
# Scoring a period against the anchor
# ---------------------------------------------------------------------------


def _not_comparable(anchor_run_id: str, reason: str) -> dict[str, Any]:
    return {"anchor_run_id": anchor_run_id, "comparable": False, "reason": reason}


def score_against_anchor(
    ws: Workspace,
    anchor_run_id: str,
    group_key: str,
    df_group: pl.DataFrame,
) -> dict[str, Any]:
    """Score one group's rows with the anchor run's models and fixed rule; the ``anchor`` section of its stats.

    Never raises: any failure is a ``comparable: False`` section with the reason.
    """
    from sorethumb_ml.features.build import apply_feature_plan  # noqa: PLC0415
    from sorethumb_ml.history.period_stats import _numeric_psi  # noqa: PLC0415
    from sorethumb_ml.store.models import load_plan, plan_digest, score_with_existing  # noqa: PLC0415

    rule = load_rule(ws, anchor_run_id, group_key)
    if rule is None:
        return _not_comparable(anchor_run_id, "the anchor run saved no fixed rule for this group")
    try:
        plan = load_plan(ws, anchor_run_id)
        space = apply_feature_plan(df_group, plan)
        matrix = space.matrix  # at the plan's dtype, exactly as detection scores it
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = score_with_existing(
                ws,
                anchor_run_id,
                group_key,
                matrix,
                space.feature_schema_hash,
                list(rule["detectors"]),
                plan_digest(plan.to_json()),
                strict=False,
            )
    except Exception as exc:  # any failure to apply or load the anchor: not comparable, never a guess
        logger.info("Anchor %s not comparable for group %s: %s", anchor_run_id, group_key, exc, exc_info=True)
        return _not_comparable(anchor_run_id, f"{type(exc).__name__}: {exc}")
    if res["drifted"] or any(issubclass(w.category, ModelSchemaDriftWarning) for w in caught):
        return _not_comparable(anchor_run_id, "the feature schema differs from the anchor run's")
    if res["missing"]:
        return _not_comparable(anchor_run_id, f"the anchor run has no model for {', '.join(res['missing'])}")
    flags = apply_rule(rule, res["calibrated"], res["natural_flags"])
    if flags is None:
        return _not_comparable(anchor_run_id, "the anchor's rule flagged no rows, so it has no threshold")

    combined = combined_score(rule, res["calibrated"])
    section: dict[str, Any] = {
        "anchor_run_id": anchor_run_id,
        "comparable": True,
        "n_rows": len(combined),
        "flagged": int(flags.sum()),
        "combined_percentiles": [float(v) for v in np.percentile(combined, _PERCENTILES)],
        "anchor_percentiles": rule.get("combined_percentiles"),
        "mean_calibrated": {d: float(np.mean(res["calibrated"][d])) for d in rule["detectors"]},
        "library_mismatch": any(issubclass(w.category, ModelVersionMismatchWarning) for w in caught),
    }
    base = rule.get("combined_percentiles")
    if base:
        p95 = base[95]
        section["tail_share"] = float(np.mean(combined > p95))
        own = {"kind": "numeric", "n": rule["n_rows"], "missing": 0, "percentiles": base}
        cur = {
            "kind": "numeric",
            "n": len(combined),
            "missing": 0,
            "percentiles": section["combined_percentiles"],
        }
        section["score_psi"] = _numeric_psi([own], [cur])
    return section
