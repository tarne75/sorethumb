#!/usr/bin/env python3
"""Run pre-registered ensemble-policy backtests on real labelled datasets.

See ``sorethumb_ml.evaluate.real_backtest`` for the protocol. Datasets (network;
needs the ``benchmark`` extra -- ``uv sync --extra benchmark``):

* ``kddcup99_sa`` -- KDD Cup 1999 intrusion traffic, SA subset of the 10% sample
  (mixed numeric and categorical; ~3.4% attacks). Random-subsample backtest at a
  5% review budget. Development and confirmation are disjoint 10,000-row samples.
* ``covtype_2v4`` -- UCI Covertype, class 2 (normal) vs class 4 (anomaly, ~1%),
  the ODDS formulation. Random-subsample backtest at a 1% budget; disjoint
  10,000-row samples.
* ``creditcard`` -- card transactions over two days, Sept 2013 (OpenML 42175;
  ~0.17% fraud), with a real ``Time`` column. Forward backtest: each hour is
  fitted and the next hour scored with those fits, at a 0.5% budget. Day 1 is
  development, day 2 confirmation.

Phases mirror ``scripts/run_backtest.py``: ``development`` prints the decision
and writes nothing; ``confirmation`` runs once per committed, unmodified
pre-registration, writes ``docs/benchmark-results/real-backtests/<id>.json`` and
refreshes the summary in ``docs/ensemble-policy.md``.

Usage:
    uv run python scripts/run_real_backtest.py --phase development
    uv run python scripts/run_real_backtest.py --phase confirmation tests/benchmark/preregistrations/real/<id>.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_DIR = REPO_ROOT / "tests" / "benchmark" / "preregistrations" / "real"
RESULTS_DIR = REPO_ROOT / "docs" / "benchmark-results" / "real-backtests"
DOC_PATH = REPO_ROOT / "docs" / "ensemble-policy.md"
RESULTS_LINK = "benchmark-results/real-backtests"
SAMPLE_ROWS = 10_000
SAMPLE_SEED = 2026  # fixed before any result was seen; never tuned
REPLICATES = 20


@dataclass(frozen=True)
class RealDataset:
    """One real dataset: how to load it, slice it, and backtest it."""

    id: str
    description: str
    licence: str
    provenance: str
    review_budget: float
    mode: str  # "random" | "forward"
    load: Callable[[Path], pl.DataFrame]  # -> frame with a 0/1 "label" column (and "ts" if forward)
    slice: Callable[[pl.DataFrame, str], tuple[pl.DataFrame, str]]  # (frame, phase) -> (rows, description)
    granularity: str | None = None


KDD_SA_SEED = 0  # fixes which abnormal rows sklearn's SA subset keeps


def _load_kdd(cache: Path) -> pl.DataFrame:
    from sklearn import datasets  # noqa: PLC0415

    # subset="SA" draws its abnormal rows at random: without a fixed random_state every
    # load is a different dataset, so no KDD result could be reproduced.
    bunch = datasets.fetch_kddcup99(
        subset="SA", percent10=True, as_frame=True, data_home=str(cache), random_state=KDD_SA_SEED
    )
    frame = bunch.frame
    cols: dict[str, Any] = {}
    for name in frame.columns:
        if name == "labels":
            continue
        series = frame[name]
        first = series.iloc[0]
        if isinstance(first, bytes):
            cols[name] = [v.decode() for v in series]
        else:
            cols[name] = series.astype(float).to_numpy()
    cols["label"] = (frame["labels"] != b"normal.").astype(int).to_numpy()
    return pl.DataFrame(cols)


def _load_covtype(cache: Path) -> pl.DataFrame:
    from sklearn import datasets  # noqa: PLC0415

    bunch = datasets.fetch_covtype(as_frame=True, data_home=str(cache))
    frame = bunch.frame
    frame = frame[(frame["Cover_Type"] == 2) | (frame["Cover_Type"] == 4)]
    out = pl.from_pandas(frame.drop(columns=["Cover_Type"]).astype(float))
    return out.with_columns(pl.Series("label", (frame["Cover_Type"] == 4).astype(int).to_numpy()))


def _load_creditcard(cache: Path) -> pl.DataFrame:
    from sklearn import datasets  # noqa: PLC0415

    bunch = datasets.fetch_openml(data_id=42175, as_frame=True, parser="auto", data_home=str(cache))
    frame = bunch.frame
    start = datetime(2013, 9, 1, tzinfo=UTC)  # the source says September 2013; only offsets matter
    ts = [start + timedelta(seconds=float(s)) for s in frame["Time"]]
    features = frame.drop(columns=["Time", "Class"]).astype(float)
    out = pl.from_pandas(features)
    return out.with_columns(
        pl.Series("ts", ts).dt.replace_time_zone(None),
        pl.Series("label", frame["Class"].astype(int).to_numpy()),
    )


def _random_slice(df: pl.DataFrame, phase: str) -> tuple[pl.DataFrame, str]:
    order = np.random.default_rng(SAMPLE_SEED).permutation(len(df))
    offset = 0 if phase == "development" else SAMPLE_ROWS
    rows = np.sort(order[offset : offset + SAMPLE_ROWS])
    which = "rows 1-10,000" if phase == "development" else "rows 10,001-20,000"
    return df[rows.tolist()], f"{SAMPLE_ROWS:,} random rows ({which} of a seed-{SAMPLE_SEED} permutation)"


def _day_slice(df: pl.DataFrame, phase: str) -> tuple[pl.DataFrame, str]:
    boundary = datetime(2013, 9, 2)  # noqa: DTZ001 -- ts is naive UTC
    if phase == "development":
        return df.filter(pl.col("ts") < boundary), "day 1, hourly periods"
    return df.filter(pl.col("ts") >= boundary), "day 2, hourly periods"


DATASETS: dict[str, RealDataset] = {
    d.id: d
    for d in (
        RealDataset(
            id="kddcup99_sa",
            description="KDD Cup 1999 network intrusion, SA subset of the 10% sample (~3.4% attacks).",
            licence="Freely available for research (UCI ML Repository).",
            provenance="sklearn.datasets.fetch_kddcup99(subset='SA', percent10=True, random_state=0)",
            review_budget=0.05,
            mode="random",
            load=_load_kdd,
            slice=_random_slice,
        ),
        RealDataset(
            id="covtype_2v4",
            description="UCI Covertype, class 2 vs class 4 (~1% anomalies; ODDS formulation).",
            licence="Public domain (Blackard and Dean, Colorado State University).",
            provenance="sklearn.datasets.fetch_covtype(), Cover_Type in {2, 4}",
            review_budget=0.01,
            mode="random",
            load=_load_covtype,
            slice=_random_slice,
        ),
        RealDataset(
            id="creditcard",
            description="Card transactions over two days, Sept 2013 (~0.17% fraud), time-ordered.",
            licence="Database Contents License (DbCL) v1.0, via OpenML.",
            provenance="sklearn.datasets.fetch_openml(data_id=42175)",
            review_budget=0.005,
            mode="forward",
            load=_load_creditcard,
            slice=_day_slice,
            granularity="hour",
        ),
    )
}


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 -- fixed git invocations
        ["git", *args],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registration_commit(path: Path) -> str | None:
    """Return the commit that last changed *path*; None if it is untracked or has local changes."""
    rel = str(path.resolve().relative_to(REPO_ROOT))
    try:
        _git("ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError:
        return None
    if _git("status", "--porcelain", "--", rel):
        return None
    return _git("log", "-1", "--format=%H", "--", rel) or None


def _config(ds: RealDataset, data_path: Path, workdir: Path, contamination: str = "budget") -> Any:
    from sorethumb_ml.config import Config  # noqa: PLC0415

    raw: dict[str, Any] = {
        "source": {"uri": str(data_path), "format": "parquet"},
        "run": {"workdir": str(workdir), "seed": 0},
        "columns": {"reference_column": "label"},
        "scoring": {"contamination": "auto" if contamination == "auto" else ds.review_budget},
        "explain": {"enabled": False},
    }
    if ds.mode == "forward":
        raw["columns"]["time_column"] = "ts"
        raw["history"] = {"period_granularity": ds.granularity, "roll_non_business": False}
    return Config.model_validate(raw)


def _measure(
    ds: RealDataset,
    candidate: str,
    phase: str,
    cache: Path,
    n_resamples: int,
    contamination: str = "budget",
    ranking_tolerance: float | None = None,
) -> dict[str, Any]:
    from sorethumb_ml.evaluate.user_backtest import backtest_dataset  # noqa: PLC0415

    frame, slice_desc = ds.slice(ds.load(cache), phase)
    data_path = cache / f"{ds.id}-{phase}.parquet"
    frame.write_parquet(data_path)
    with tempfile.TemporaryDirectory() as tmp:
        result = backtest_dataset(
            _config(ds, data_path, Path(tmp) / "ws", contamination),
            policies=[candidate],
            replicates=REPLICATES,
            mode=ds.mode,
            n_resamples=n_resamples,
            record=False,
            review_budget=ds.review_budget if contamination == "auto" else None,
            ranking_tolerance=ranking_tolerance if ranking_tolerance is not None else 0.10,
        )
    return {
        "slice": slice_desc,
        "rows": len(frame),
        "anomalies": int(frame["label"].sum()),
        "review_budget": ds.review_budget,
        "contamination": contamination,
        "mode": ds.mode,
        "replicates": result.replicates,
        "periods": list(result.periods),
        "data_sha256": _sha256(data_path),
        "description": ds.description,
        "licence": ds.licence,
        "provenance": ds.provenance,
        "summaries": [asdict(s) for s in result.summaries],
        "per_replicate": result.per_replicate,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, float) and value != value:  # noqa: PLR0124 -- NaN check
        return None
    if hasattr(value, "item") and not isinstance(value, list | tuple | dict):
        return _clean(value.item())
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    import sorethumb_ml  # noqa: PLC0415
    from sorethumb_ml.evaluate.real_backtest import (  # noqa: PLC0415
        decide_real,
        inject_real_results,
        load_real_preregistration,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("preregistrations", nargs="*", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "sorethumb"
        / "real-backtest",
    )
    args = parser.parse_args(argv)
    paths = args.preregistrations or [
        p for p in sorted(PREREG_DIR.glob("*.toml")) if not (RESULTS_DIR / f"{p.stem}.json").exists()
    ]
    if not paths:
        print("error: no unconfirmed pre-registration files given or found.", file=sys.stderr)
        return 2
    try:
        pres = [(p, load_real_preregistration(p)) for p in paths]
    except (ValueError, KeyError) as exc:
        print(f"error: invalid pre-registration: {exc}", file=sys.stderr)
        return 2

    confirming = args.phase == "confirmation"
    commits: dict[str, str] = {}
    if confirming:
        for path, pre in pres:
            commit = _registration_commit(path)
            if commit is None:
                print(f"error: {path} must be committed, with no local changes, first.", file=sys.stderr)
                return 2
            if (RESULTS_DIR / f"{pre.id}.json").exists():
                print(f"error: {pre.id} already has a confirmation result.", file=sys.stderr)
                return 2
            commits[pre.id] = commit

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for path, pre in pres:
        measured = {}
        for dataset_id in pre.datasets:
            print(f"{pre.id}: {dataset_id} ({args.phase})...", flush=True)
            measured[dataset_id] = _measure(
                DATASETS[dataset_id],
                pre.candidate,
                args.phase,
                args.cache_dir,
                pre.n_resamples,
                pre.contamination,
                pre.ranking_tolerance,
            )
        decision = decide_real(pre, {d: m["per_replicate"] for d, m in measured.items()})
        verdict = "OUTPERFORMS" if decision.outperforms else "does not outperform"
        print(f"\n{pre.id}: {verdict}")
        for comp in decision.comparisons:
            print(
                f"  {comp.dataset:<12} F1 diff {comp.mean_difference:+.4f} "
                f"CI [{comp.ci_low:+.4f}, {comp.ci_high:+.4f}] p_holm {comp.p_holm:.4f}"
                f"{'  improved' if comp.improved else ''}"
            )
        for reason in decision.reasons:
            print(f"  - {reason}")
        if not confirming:
            continue
        result = {
            "preregistration": asdict(pre)
            | {"path": str(path.resolve().relative_to(REPO_ROOT)), "commit": commits[pre.id]},
            "phase": args.phase,
            "datasets": measured,
            "decision": decision.as_dict(),
            "metadata": {
                "library_version": sorethumb_ml.__version__,
                "git_sha": _git("rev-parse", "HEAD"),
                "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "sample_rows": SAMPLE_ROWS,
                "sample_seed": SAMPLE_SEED,
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            },
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{pre.id}.json"
        out.write_text(json.dumps(_clean(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  written to {out.relative_to(REPO_ROOT)}")

    if confirming:
        results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
        changed = inject_real_results(DOC_PATH, results, RESULTS_LINK)
        print(
            f"Real-data summary {'updated in' if changed else 'unchanged in'} {DOC_PATH.relative_to(REPO_ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
