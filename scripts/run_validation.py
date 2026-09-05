#!/usr/bin/env python3
"""Validation sweep: run all detector combos × PCA on/off × datasets.

Results are written incrementally to validation/results.json so the script
can be safely interrupted and resumed.

Usage:
    uv run python scripts/run_validation.py
    uv run python scripts/run_validation.py --dataset kddcup99_sa  # single dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sorethumb import Config, Workspace, run_detection  # noqa: E402

DATA_DIR = REPO_ROOT / "data-samples"
RESULTS_PATH = REPO_ROOT / "validation" / "results.json"
WORKDIR_BASE = REPO_ROOT / "validation" / "runs"


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    file: str
    ignore: list[str]
    source: str
    description: str
    rows: int
    cols: int


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        name="kddcup99_sa",
        file="kddcup99_sa.parquet",
        ignore=["target"],
        source="KDD Cup 1999 (UCI ML Repository) — SA subset, 10 % sample",
        description="Network intrusion detection. 41 connection features (numeric + categorical). "
        "Ground-truth `target` label excluded. Industry-standard anomaly benchmark.",
        rows=100_655,
        cols=42,
    ),
    DatasetSpec(
        name="electricity",
        file="electricity.parquet",
        ignore=["class"],
        source="Harries (1999) via OpenML — Electricity dataset",
        description="Half-hourly Australian electricity demand 1996–1998. "
        "Price and demand for NSW and Victoria plus transfer. `class` (UP/DOWN) excluded.",
        rows=45_312,
        cols=9,
    ),
    DatasetSpec(
        name="weather_australia",
        file="weather_australia.parquet",
        ignore=["A15"],
        source="Australian Bureau of Meteorology via UCI ML Repository",
        description="Daily weather observations (anonymous columns A1–A14). "
        "A15 is the binary RainTomorrow label, excluded from features.",
        rows=690,
        cols=15,
    ),
    DatasetSpec(
        name="macro_us_quarterly",
        file="macro_us_quarterly.parquet",
        ignore=["year", "quarter"],
        source="statsmodels macrodata — US Federal Reserve",
        description="Quarterly US macroeconomic indicators 1959–2009 "
        "(GDP, inflation, unemployment, interest rates). Year and quarter excluded as indices.",
        rows=203,
        cols=14,
    ),
    DatasetSpec(
        name="elnino_sst",
        file="elnino_sst.parquet",
        ignore=["YEAR"],
        source="statsmodels elnino — NOAA/TOGA-TAO buoy array",
        description="Annual mean sea-surface temperatures across 12 Pacific buoy locations "
        "1950–2010. YEAR excluded as index.",
        rows=61,
        cols=13,
    ),
    DatasetSpec(
        name="sunspots_annual",
        file="sunspots_annual.parquet",
        ignore=["YEAR"],
        source="statsmodels sunspots — Royal Observatory of Belgium",
        description="Annual Wolf sunspot number 1700–2008. Single numeric feature after "
        "excluding YEAR. Very small — edge-case/sanity dataset.",
        rows=309,
        cols=2,
    ),
    DatasetSpec(
        name="longley_multicollinear",
        file="longley_multicollinear.parquet",
        ignore=[],
        source="statsmodels longley — Longley (1967)",
        description="Annual US macro data 1947–1962 (7 highly collinear features). "
        "16 rows only — extreme edge case. Included for completeness.",
        rows=16,
        cols=7,
    ),
    DatasetSpec(
        name="natops_mts",
        file="natops_mts.parquet",
        ignore=["label"],
        source="UEA Time Series Classification Archive — NATOPS dataset",
        description="24-channel aircraft hand-signal motion capture (51 timepoints), "
        "stored wide (1224 numeric columns). `label` excluded. PCA recommended.",
        rows=360,
        cols=1_225,
    ),
    DatasetSpec(
        name="basic_motions_mts",
        file="basic_motions_mts.parquet",
        ignore=["label"],
        source="UEA Time Series Classification Archive — BasicMotions dataset",
        description="6-axis IMU data for 4 activities (100 timepoints × 6 channels = 600 cols). "
        "`label` excluded. PCA recommended.",
        rows=80,
        cols=601,
    ),
]

# ---------------------------------------------------------------------------
# Detector combos
# ---------------------------------------------------------------------------

BASE = ["isolation_forest", "kmeans_distance", "one_class_svm"]

COMBOS: list[tuple[str, list[str]]] = [
    ("baseline",           BASE),
    ("baseline+ecod",      BASE + ["ecod"]),
    ("baseline+lof",       BASE + ["lof"]),
    ("baseline+hbos",      BASE + ["hbos"]),
    ("baseline+ecod+lof",  BASE + ["ecod", "lof"]),
    ("baseline+ecod+hbos", BASE + ["ecod", "hbos"]),
    ("baseline+lof+hbos",  BASE + ["lof", "hbos"]),
    ("all6",               BASE + ["ecod", "lof", "hbos"]),
]

NU_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(dataset: DatasetSpec, pca: bool, detector_names: list[str], nu: float, workdir: Path) -> Config:
    detectors = []
    for name in detector_names:
        det: dict = {"name": name, "enabled": True}
        if name == "one_class_svm":
            det["params"] = {"nu": nu}
        detectors.append(det)

    return Config.model_validate({
        "source": {"uri": str(DATA_DIR / dataset.file)},
        "columns": {"ignore": dataset.ignore},
        "features": {"pca": pca},
        "scoring": {"contamination": "auto", "combination": "intersection"},
        "detectors": detectors,
        "run": {"workdir": str(workdir), "seed": 42},
        "explain": {"enabled": True, "max_rows": 500},
    })


# ---------------------------------------------------------------------------
# Nu tuning
# ---------------------------------------------------------------------------


def tune_nu(dataset: DatasetSpec, pca: bool, workdir: Path) -> float:
    """Try NU_CANDIDATES on the baseline, return first nu that yields > 0 anomalies."""
    print(f"    [tuning nu] ", end="", flush=True)
    for nu in NU_CANDIDATES:
        cfg = build_config(dataset, pca, BASE, nu, workdir)
        try:
            result = run_detection(cfg, no_report=True, force=True)
            n_total = sum(g.n_records for g in result.groups)
            n_anom = result.n_anomalies
            rate = n_anom / n_total if n_total > 0 else 0.0
            print(f"nu={nu}→{n_anom} ({rate:.1%}) ", end="", flush=True)
            if n_anom > 0:
                print(f"✓")
                return nu
        except Exception as e:
            print(f"nu={nu}→ERR({e}) ", end="", flush=True)
    print("no anomalies found, using 0.1")
    return 0.1


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_one(dataset: DatasetSpec, pca: bool, nu: float, combo_name: str,
            detector_names: list[str], workdir: Path) -> dict:
    cfg = build_config(dataset, pca, detector_names, nu, workdir)
    t0 = time.time()
    try:
        result = run_detection(cfg, no_report=True)
        elapsed = time.time() - t0
        n_total = sum(g.n_records for g in result.groups)
        n_anom = result.n_anomalies
        rate = n_anom / n_total if n_total > 0 else 0.0
        group_summaries = [
            {"group": g.group_label, "n_records": g.n_records,
             "n_anomalies": g.n_anomalies, "status": g.status}
            for g in result.groups
        ]
        return {
            "dataset": dataset.name,
            "pca": pca,
            "combo": combo_name,
            "detectors": detector_names,
            "nu": nu,
            "n_total": n_total,
            "n_anomalies": n_anom,
            "rate": round(rate, 6),
            "elapsed_seconds": round(elapsed, 2),
            "run_id": result.run_id,
            "status": "success",
            "warnings": result.warnings_issued,
            "groups": group_summaries,
        }
    except Exception as e:
        return {
            "dataset": dataset.name,
            "pca": pca,
            "combo": combo_name,
            "detectors": detector_names,
            "nu": nu,
            "n_total": 0,
            "n_anomalies": 0,
            "rate": 0.0,
            "elapsed_seconds": round(time.time() - t0, 2),
            "run_id": None,
            "status": "error",
            "error": str(e),
            "warnings": [],
            "groups": [],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Run only this dataset (by name)")
    args = parser.parse_args()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKDIR_BASE.mkdir(parents=True, exist_ok=True)

    # Load existing results for resumability
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as f:
            all_results: list[dict] = json.load(f)
    else:
        all_results = []

    done_keys = {(r["dataset"], r["pca"], r["combo"]) for r in all_results}

    datasets = [d for d in DATASETS if args.dataset is None or d.name == args.dataset]
    if args.dataset and not datasets:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    total_runs = len(datasets) * 2 * len(COMBOS)
    completed = 0
    print(f"Starting validation: {len(datasets)} datasets × 2 PCA settings × {len(COMBOS)} combos = {total_runs} runs")
    print(f"Already done: {len(done_keys)} runs\n")

    for dataset in datasets:
        print(f"\n{'='*65}")
        print(f"  {dataset.name}  ({dataset.rows:,} rows × {dataset.cols} cols)")
        print(f"{'='*65}")

        for pca in [False, True]:
            pca_label = "PCA=on " if pca else "PCA=off"
            print(f"\n  [{pca_label}]")

            workdir = WORKDIR_BASE / dataset.name / ("pca_on" if pca else "pca_off")
            workdir.mkdir(parents=True, exist_ok=True)

            # Determine nu: tune if baseline not yet done, else read from results
            baseline_key = (dataset.name, pca, "baseline")
            if baseline_key in done_keys:
                existing = next(r for r in all_results
                                if r["dataset"] == dataset.name
                                and r["pca"] == pca
                                and r["combo"] == "baseline")
                nu = existing["nu"]
                print(f"    [tuning nu] reusing nu={nu} from previous baseline run")
            else:
                nu = tune_nu(dataset, pca, workdir)

            for combo_name, detector_names in COMBOS:
                key = (dataset.name, pca, combo_name)
                if key in done_keys:
                    completed += 1
                    continue

                print(f"    [{combo_name:25s}] ", end="", flush=True)
                rec = run_one(dataset, pca, nu, combo_name, detector_names, workdir)
                all_results.append(rec)
                done_keys.add(key)
                completed += 1

                # Persist after every run
                with RESULTS_PATH.open("w") as f:
                    json.dump(all_results, f, indent=2)

                if rec["status"] == "success":
                    print(f"{rec['n_anomalies']:>5} / {rec['n_total']:>7} ({rec['rate']:>6.2%})  {rec['elapsed_seconds']:>6.1f}s")
                else:
                    print(f"ERROR: {rec.get('error', 'unknown')[:80]}")

    print(f"\n{'='*65}")
    print(f"Validation complete. {completed} runs. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
