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

sys.path.insert(0, str(REPO_ROOT / "src"))
from scripts.run_validation import DATASETS, COMBOS  # reuse specs  # noqa: E402

DATASET_NOTES: dict[str, str] = {
    "sunspots_annual":
        "Only one effective feature (SUNACTIVITY) after excluding YEAR. "
        "Intersection of three or more detectors on a 1-D signal is very tight; "
        "zero anomalies with some combos is expected.",
    "longley_multicollinear":
        "16 rows only. LOF's n_neighbors is clamped from 20 to 15. "
        "Results are illustrative edge-case only — conclusions should not be drawn "
        "from a 16-row dataset.",
    "natops_mts":
        "1,224 numeric columns (24 channels × 51 timepoints). "
        "PCA=off triggers FeatureWidthWarning; PCA=on reduces to a manageable subspace.",
    "basic_motions_mts":
        "600 numeric columns (6 channels × 100 timepoints). "
        "PCA=off may hit FeatureWidthWarning on strict configurations.",
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

    for combo_name, _ in COMBOS:
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

    for spec in DATASETS:
        results = by_dataset.get(spec.name, [])
        sections.append(f"## {spec.name}\n")
        sections.append(f"**Source:** {spec.source}\n")
        sections.append(f"**Dimensions:** {spec.rows:,} rows × {spec.cols} columns\n")
        sections.append(f"**Description:** {spec.description}\n")

        if spec.name in DATASET_NOTES:
            sections.append(f"> **Note:** {DATASET_NOTES[spec.name]}\n")

        if not results:
            sections.append("_Not yet run._\n")
            sections.append("---\n")
            continue

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
    print(f"Datasets: {len(by_dataset)} / {len(DATASETS)}")
    total_runs = sum(len(v) for v in by_dataset.values())
    print(f"Total runs documented: {total_runs}")


if __name__ == "__main__":
    main()
