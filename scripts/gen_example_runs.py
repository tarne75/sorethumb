#!/usr/bin/env python3
"""Generate docs/example-runs.md from validation/results.json.

Usage:
    uv run python scripts/gen_example_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = REPO_ROOT / "validation" / "results.json"
OUT_PATH = REPO_ROOT / "docs" / "example-runs.md"

# Dataset ordering matches scripts/validation/datasets.py
DATASET_NAMES = [
    "kddcup99_sa",
    "electricity",
    "weather_australia",
    "macro_us_quarterly",
    "elnino_sst",
    "sunspots_annual",
    "longley_multicollinear",
    "natops_mts",
    "basic_motions_mts",
]

DATASET_META: dict[str, dict[str, Any]] = {
    "kddcup99_sa": {
        "rows": 100_655,
        "cols": 42,
        "source": "KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample",
        "description": "Network intrusion detection. 41 connection features (numeric + categorical). "
        "`target` is a real anomaly ground truth -- the one dataset here with held-out ROC-AUC/AP.",
    },
    "electricity": {
        "rows": 45_312,
        "cols": 9,
        "source": "Harries (1999) via OpenML — Electricity dataset",
        "description": "Half-hourly Australian electricity demand 1996-1998. "
        "Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded.",
    },
    "weather_australia": {
        "rows": 690,
        "cols": 15,
        "source": "Australian Bureau of Meteorology via UCI ML Repository",
        "description": "Daily weather observations (anonymous columns A1-A14). "
        "A15 is the binary RainTomorrow label, excluded from features.",
    },
    "macro_us_quarterly": {
        "rows": 203,
        "cols": 14,
        "source": "statsmodels macrodata — US Federal Reserve",
        "description": "Quarterly US macroeconomic indicators 1959-2009 "
        "(GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices.",
    },
    "elnino_sst": {
        "rows": 61,
        "cols": 13,
        "source": "statsmodels elnino — NOAA/TOGA-TAO buoy array",
        "description": "Annual mean sea-surface temperatures across 12 Pacific buoy locations "
        "1950-2010. YEAR excluded as index.",
    },
    "sunspots_annual": {
        "rows": 309,
        "cols": 2,
        "source": "statsmodels sunspots — Royal Observatory of Belgium",
        "description": "Annual Wolf sunspot number 1700-2008. Single numeric feature after excluding YEAR.",
    },
    "longley_multicollinear": {
        "rows": 16,
        "cols": 7,
        "source": "statsmodels longley — Longley (1967)",
        "description": "Annual US macro data 1947-1962 (7 highly collinear features). "
        "16 rows only — extreme edge case. Included for completeness.",
    },
    "natops_mts": {
        "rows": 360,
        "cols": 1_225,
        "source": "UEA Time Series Classification Archive — NATOPS dataset",
        "description": "24-channel aircraft hand-signal motion capture (51 timepoints), "
        "stored wide (1,224 numeric columns). `label` excluded.",
    },
    "basic_motions_mts": {
        "rows": 80,
        "cols": 601,
        "source": "UEA Time Series Classification Archive — BasicMotions dataset",
        "description": "6-axis IMU data for 4 activities (100 timepoints x 6 channels = 600 cols). "
        "`label` excluded.",
    },
}

COMBO_ORDER = [
    "baseline",
    "baseline+ecod",
    "baseline+lof",
    "baseline+hbos",
    "baseline+ecod+lof",
    "baseline+ecod+hbos",
    "baseline+lof+hbos",
    "all6",
]

COMBO_LABELS: dict[str, str] = {
    "baseline": "if + km + oc",
    "baseline+ecod": "if + km + oc + ecod",
    "baseline+lof": "if + km + oc + lof",
    "baseline+hbos": "if + km + oc + hbos",
    "baseline+ecod+lof": "if + km + oc + ecod + lof",
    "baseline+ecod+hbos": "if + km + oc + ecod + hbos",
    "baseline+lof+hbos": "if + km + oc + lof + hbos",
    "all6": "if + km + oc + ecod + lof + hbos",
}


def _status_cell(r: dict[str, Any]) -> str | None:
    """Return a one-line status cell for a non-"success" result, or None."""
    if r["status"] == "error":
        err = (r.get("error") or "unknown")[:60]
        return f"ERROR: `{err}`"
    if r["status"] == "too_few_records":
        return f"skipped (train={r['n_train']}/holdout={r['n_holdout']} < scoring.min_records)"
    return None


def render_table(results: list[dict[str, Any]], pca: bool) -> str:
    """Render one dataset's per-combo results table for a given PCA setting."""
    pca_results = [r for r in results if r["pca"] == pca]
    if not pca_results:
        return "_No results._\n"

    by_combo = {r["combo"]: r for r in pca_results}
    has_accuracy = any(r.get("roc_auc") is not None for r in pca_results)

    header = "| Detectors | Flagged / Holdout | Rate | Time |"
    if has_accuracy:
        header += " ROC-AUC | AP |"
    lines = [header, "|---|---|---|---|" + ("---|---|" if has_accuracy else "")]

    for combo_name in COMBO_ORDER:
        r = by_combo.get(combo_name)
        label = COMBO_LABELS.get(combo_name, combo_name)
        if r is None:
            row = f"| {label} | — | — | — |"
            lines.append(row + (" — | — |" if has_accuracy else ""))
            continue

        status_cell = _status_cell(r)
        if status_cell is not None:
            row = f"| {label} | {status_cell} | — | — |"
            lines.append(row + (" — | — |" if has_accuracy else ""))
            continue

        flagged = f"{r['n_holdout_flagged']:,} / {r['n_holdout']:,}"
        pct = f"{r['holdout_flag_rate'] * 100:.2f}%"
        s = r["elapsed_seconds"]
        t = f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}m"
        row = f"| {label} | {flagged} | {pct} | {t} |"
        if has_accuracy:
            roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "n/a"
            ap = f"{r['average_precision']:.4f}" if r.get("average_precision") is not None else "n/a"
            row += f" {roc} | {ap} |"
        lines.append(row)

    return "\n".join(lines) + "\n"


def main() -> None:
    """Render validation/results.json into docs/example-runs.md."""
    if not RESULTS_PATH.exists():
        print(f"No results file found at {RESULTS_PATH}", file=sys.stderr)
        print("Run: uv run python scripts/run_validation.py", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    all_results: list[dict[str, Any]] = raw["results"] if isinstance(raw, dict) else raw

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        by_dataset.setdefault(r["dataset"], []).append(r)

    sections = [
        "# Example runs\n",
        "Validation sweep of sorethumb across nine real-world datasets "
        "(`scripts/run_validation.py`). Every case fits on a 70/30 train/held-out split "
        "and scores the held-out split via `score_forward` -- never trains and evaluates "
        "on the same rows. Every detector in a combo targets the same fixed 5% review "
        'budget (`scoring.contamination = 0.05`, `combination = "intersection"`); '
        "OneClassSVM's `nu` is derived automatically from that budget, not searched for. "
        "ROC-AUC/AP are reported only for the one dataset with a genuine anomaly ground "
        "truth (kddcup99_sa) -- every other dataset shown here has no anomaly label, so "
        "only operational metrics (rows flagged, elapsed time) are reported for it; "
        "inventing an accuracy number without ground truth would be dishonest.\n",
        "---\n",
    ]

    for name in DATASET_NAMES:
        spec = DATASET_META[name]
        results = by_dataset.get(name, [])
        sections.append(f"## {name}\n")
        sections.append(f"**Source:** {spec['source']}\n")
        sections.append(f"**Dimensions:** {spec['rows']:,} rows x {spec['cols']} columns\n")
        sections.append(f"**Description:** {spec['description']}\n")

        if not results:
            sections.append("_Not yet run._\n")
            sections.append("---\n")
            continue

        sections.append("\n### PCA disabled\n")
        sections.append(render_table(results, pca=False))

        sections.append("\n### PCA enabled\n")
        sections.append(render_table(results, pca=True))

        successful = [r for r in results if r["status"] == "success" and r["n_holdout_flagged"] > 0]
        if successful:
            best = max(successful, key=lambda r: r["n_holdout_flagged"])
            best_label = COMBO_LABELS.get(best["combo"], best["combo"])
            pca_tag = "PCA=on" if best["pca"] else "PCA=off"
            sections.append(
                f"\n**Most flagged:** `{best_label}` ({pca_tag}) — "
                f"{best['n_holdout_flagged']:,} / {best['n_holdout']:,} "
                f"({best['holdout_flag_rate'] * 100:.2f}%)\n"
            )

        sections.append("\n---\n")

    sections.append("## Notes\n")
    sections.append(
        "- `ERROR` means the run raised an unhandled exception; see `validation/results.json` for details.\n"
        '- "skipped" means the train or holdout split fell below `scoring.min_records` '
        "(default 100) -- too few rows for either half to fit/score meaningfully, not a failure.\n"
        '- `combination = "intersection"` means a row is only flagged when **all** detectors '
        "in the combo agree. Adding more detectors generally reduces the flagged count.\n"
        "- ROC-AUC/AP columns only appear for datasets with a real anomaly label "
        "(currently kddcup99_sa); `n/a` means the metric was undefined for that split "
        "(e.g. the holdout sample was single-class).\n"
    )

    out = "\n".join(sections)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Written: {OUT_PATH}")
    print(f"Datasets: {len(by_dataset)} / {len(DATASET_NAMES)}")
    total_runs = sum(len(v) for v in by_dataset.values())
    print(f"Total runs documented: {total_runs}")


if __name__ == "__main__":
    main()
