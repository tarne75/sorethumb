#!/usr/bin/env python3
"""Generate docs/example-runs.md from validation/results.json.

Usage:
    uv run python scripts/gen_example_runs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = REPO_ROOT / "validation" / "results.json"
OUT_PATH = REPO_ROOT / "docs" / "example-runs.md"

# Dataset ordering matches run_validation.py
DATASET_NAMES = [
    "kddcup99_sa", "electricity", "weather_australia", "macro_us_quarterly",
    "elnino_sst", "sunspots_annual", "longley_multicollinear", "natops_mts", "basic_motions_mts",
]

DATASET_META: dict[str, dict] = {
    "kddcup99_sa": {"rows": 100_655, "cols": 42,
        "source": "KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample",
        "description": "Network intrusion detection. 41 connection features (numeric + categorical). "
                       "Ground-truth `target` label excluded. Industry-standard anomaly benchmark."},
    "electricity": {"rows": 45_312, "cols": 9,
        "source": "Harries (1999) via OpenML — Electricity dataset",
        "description": "Half-hourly Australian electricity demand 1996–1998. "
                       "Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded."},
    "weather_australia": {"rows": 690, "cols": 15,
        "source": "Australian Bureau of Meteorology via UCI ML Repository",
        "description": "Daily weather observations (anonymous columns A1–A14). "
                       "A15 is the binary RainTomorrow label, excluded from features."},
    "macro_us_quarterly": {"rows": 203, "cols": 14,
        "source": "statsmodels macrodata — US Federal Reserve",
        "description": "Quarterly US macroeconomic indicators 1959–2009 "
                       "(GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices."},
    "elnino_sst": {"rows": 61, "cols": 13,
        "source": "statsmodels elnino — NOAA/TOGA-TAO buoy array",
        "description": "Annual mean sea-surface temperatures across 12 Pacific buoy locations "
                       "1950–2010. YEAR excluded as index."},
    "sunspots_annual": {"rows": 309, "cols": 2,
        "source": "statsmodels sunspots — Royal Observatory of Belgium",
        "description": "Annual Wolf sunspot number 1700–2008. Single numeric feature after "
                       "excluding YEAR."},
    "longley_multicollinear": {"rows": 16, "cols": 7,
        "source": "statsmodels longley — Longley (1967)",
        "description": "Annual US macro data 1947–1962 (7 highly collinear features). "
                       "16 rows only — extreme edge case. Included for completeness."},
    "natops_mts": {"rows": 360, "cols": 1_225,
        "source": "UEA Time Series Classification Archive — NATOPS dataset",
        "description": "24-channel aircraft hand-signal motion capture (51 timepoints), "
                       "stored wide (1,224 numeric columns). `label` excluded."},
    "basic_motions_mts": {"rows": 80, "cols": 601,
        "source": "UEA Time Series Classification Archive — BasicMotions dataset",
        "description": "6-axis IMU data for 4 activities (100 timepoints x 6 channels = 600 cols). "
                       "`label` excluded."},
}

COMBO_ORDER = [
    "baseline", "baseline+ecod", "baseline+lof", "baseline+hbos",
    "baseline+ecod+lof", "baseline+ecod+hbos", "baseline+lof+hbos", "all6",
]

DATASET_NOTES: dict[str, str] = {
    "sunspots_annual":
        "Only one effective feature (SUNACTIVITY) after excluding YEAR. "
        "Isolation Forest triggers a TreeSHAP fallback on 1-D data (non-fatal; "
        "heuristic attribution is used instead). Results are valid.",
    "longley_multicollinear":
        "16 rows — all detector combinations produce zero anomalies regardless of "
        "PCA setting or nu tuning. Intersection of three detectors on 16 points "
        "requires unanimous agreement that is never achieved on this dataset. "
        "Included for completeness as an extreme edge case.",
    "elnino_sst":
        "61 rows — all detector combinations produce zero anomalies. "
        "The sea-surface temperature data is too small and homogeneous for the "
        "intersection of three independent detectors to reach unanimous agreement. "
        "A union strategy would surface anomalies but is outside the scope of this validation.",
    "natops_mts":
        "1,224 numeric columns (24 channels x 51 timepoints). "
        "PCA=off: the raw feature space produces 3 consistent anomalies across all "
        "8 combos — a robust, stable signal. "
        "PCA=on: dimensionality reduction collapses the discriminating structure and "
        "produces zero anomalies across all combos. PCA is counterproductive here.",
    "basic_motions_mts":
        "80 rows, 600 numeric columns (6 channels x 100 timepoints). "
        "Zero anomalies across all combinations and both PCA settings. "
        "The dataset is too small for three independent detectors to reach "
        "intersection consensus; all activity classes contribute similar feature distributions.",
}

COMBO_LABELS: dict[str, str] = {
    "baseline":           "if + km + oc",
    "baseline+ecod":      "if + km + oc + ecod",
    "baseline+lof":       "if + km + oc + lof",
    "baseline+hbos":      "if + km + oc + hbos",
    "baseline+ecod+lof":  "if + km + oc + ecod + lof",
    "baseline+ecod+hbos": "if + km + oc + ecod + hbos",
    "baseline+lof+hbos":  "if + km + oc + lof + hbos",
    "all6":               "if + km + oc + ecod + lof + hbos",
}


def rate_str(r: dict) -> str:
    if r["status"] == "error":
        return f"ERROR"
    if r["n_total"] == 0:
        return "—"
    pct = r["rate"] * 100
    return f"{r['n_anomalies']:,} / {r['n_total']:,} ({pct:.2f}%)"


def time_str(r: dict) -> str:
    if r["status"] == "error":
        return "—"
    s = r["elapsed_seconds"]
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


def params_str(r: dict) -> str:
    parts = [f"nu={r['nu']}"]
    if r.get("warnings"):
        parts.append("⚠")
    return ", ".join(parts)


def render_table(results: list[dict], pca: bool) -> str:
    pca_results = [r for r in results if r["pca"] == pca]
    if not pca_results:
        return "_No results._\n"

    by_combo = {r["combo"]: r for r in pca_results}

    lines = []
    lines.append("| Detectors | Anomalies / Total | Rate | Time | OC-SVM nu |")
    lines.append("|---|---|---|---|---|")

    for combo_name in COMBO_ORDER:
        r = by_combo.get(combo_name)
        label = COMBO_LABELS.get(combo_name, combo_name)
        if r is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        if r["status"] == "error":
            err = r.get("error", "unknown")[:60]
            lines.append(f"| {label} | ERROR | — | — | `{err}` |")
        else:
            anomalies = f"{r['n_anomalies']:,} / {r['n_total']:,}"
            pct = f"{r['rate']*100:.2f}%"
            t = time_str(r)
            nu = r["nu"]
            warn = " ⚠" if r.get("warnings") else ""
            lines.append(f"| {label} | {anomalies} | {pct} | {t} | {nu}{warn} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"No results file found at {RESULTS_PATH}", file=sys.stderr)
        print("Run: uv run python scripts/run_validation.py", file=sys.stderr)
        sys.exit(1)

    with RESULTS_PATH.open() as f:
        all_results: list[dict] = json.load(f)

    by_dataset: dict[str, list[dict]] = {}
    for r in all_results:
        by_dataset.setdefault(r["dataset"], []).append(r)

    sections = []
    sections.append("# Example runs\n")
    sections.append(
        "Systematic validation of sorethumb across nine real-world datasets.\n"
        "Every run uses `combination = \"intersection\"` and `contamination = \"auto\"`.\n"
        "The baseline detector set is `isolation_forest + kmeans_distance + one_class_svm`;\n"
        "additional detectors are added one at a time, then in pairs, then all together.\n"
        "Each dataset is tested with PCA disabled and enabled.\n"
        "OC-SVM `nu` is tuned per dataset (smallest value that yields at least one anomaly).\n"
        "All other detector parameters are at their defaults.\n"
    )
    sections.append("---\n")

    for name in DATASET_NAMES:
        spec = DATASET_META[name]
        results = by_dataset.get(name, [])
        sections.append(f"## {name}\n")
        sections.append(f"**Source:** {spec['source']}\n")
        sections.append(f"**Dimensions:** {spec['rows']:,} rows × {spec['cols']} columns\n")
        sections.append(f"**Description:** {spec['description']}\n")

        if name in DATASET_NOTES:
            sections.append(f"> **Note:** {DATASET_NOTES[name]}\n")

        if not results:
            sections.append("_Not yet run._\n")
            sections.append("---\n")
            continue

        # Per-dataset params note
        baseline_off = next((r for r in results if not r["pca"] and r["combo"] == "baseline"), None)
        baseline_on  = next((r for r in results if r["pca"]     and r["combo"] == "baseline"), None)
        nu_off = baseline_off["nu"] if baseline_off else "—"
        nu_on  = baseline_on["nu"]  if baseline_on  else "—"
        sections.append(
            f"**Baseline detector params:** `isolation_forest` n_estimators=200 · "
            f"`kmeans_distance` k=auto · `one_class_svm` nu={nu_off} (PCA=off), nu={nu_on} (PCA=on) · "
            f"All other params at defaults.\n"
        )

        sections.append("\n### PCA disabled\n")
        sections.append(render_table(results, pca=False))

        sections.append("\n### PCA enabled\n")
        sections.append(render_table(results, pca=True))

        # Highlight best combo (most anomalies among successful runs)
        successful = [r for r in results if r["status"] == "success" and r["n_anomalies"] > 0]
        if successful:
            best = max(successful, key=lambda r: r["n_anomalies"])
            best_label = COMBO_LABELS.get(best["combo"], best["combo"])
            pca_tag = "PCA=on" if best["pca"] else "PCA=off"
            sections.append(
                f"\n**Best result:** `{best_label}` ({pca_tag}) — "
                f"{best['n_anomalies']:,} anomalies ({best['rate']*100:.2f}%)\n"
            )

        sections.append("\n---\n")

    # Warnings legend
    sections.append("## Notes\n")
    sections.append(
        "- ⚠ indicates the run completed but emitted sorethumb warnings "
        "(e.g. FeatureWidthWarning, SlowStageWarning). Results are still valid.\n"
        "- `ERROR` means the run raised an unhandled exception; see `validation/results.json` for details.\n"
        "- `combination = \"intersection\"` means a row is only flagged when **all** detectors agree. "
        "Adding more detectors generally reduces the anomaly count.\n"
        "- `contamination = \"auto\"` uses each detector's natural learned threshold. "
        "The OC-SVM `nu` parameter is the only explicitly tuned value.\n"
    )

    out = "\n".join(sections)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Written: {OUT_PATH}")
    print(f"Datasets: {len(by_dataset)} / {len(DATASET_NAMES)}")
    total_runs = sum(len(v) for v in by_dataset.values())
    print(f"Total runs documented: {total_runs}")


if __name__ == "__main__":
    main()
