#!/usr/bin/env python3
"""Validation sweep: run detector combos × PCA on/off × datasets through the full pipeline.

Thin CLI over ``scripts/validation/``'s pure planner (``planner.py``,
``schema.py``, ``data.py``) and runner (``runner.py``). Each case fits on a
70/30 train/held-out split of one dataset and scores the held-out split via
``score_forward`` (never trains and evaluates on the same rows). Datasets
with a genuine anomaly ground truth (currently only kddcup99_sa; see
``scripts/validation/datasets.py``) additionally get held-out ROC-AUC/AP and
precision@k/recall@k/F1@k at a fixed review budget -- every other dataset
gets operational/pipeline-smoke metrics only (n flagged, elapsed time), never
a fabricated accuracy number for a dataset with no anomaly labels.

Results are written atomically to validation/results.json after every case,
so the script is safely interruptible. On resume, a case is only skipped if
a stored *successful* result exists whose full identity (schema version,
code revision, dataset file fingerprint, resolved config hash, seed,
dependency versions) matches what would be run now -- a failed case, or one
whose code/data/config/dependencies changed, always reruns.

Usage:
    uv run python scripts/run_validation.py
    uv run python scripts/run_validation.py --dataset kddcup99_sa --pca off
    uv run python scripts/run_validation.py --combo baseline --workers 4
    uv run python scripts/run_validation.py --fail-fast
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validation.datasets import COMBOS, DATASETS, combo_by_name, dataset_by_name  # noqa: E402
from scripts.validation.identity import code_revision, dependency_versions  # noqa: E402
from scripts.validation.planner import (  # noqa: E402
    build_cases,
    plan_run,
    read_results_file,
    write_results_atomic,
)
from scripts.validation.runner import build_config, build_identity, run_case, run_explain_check  # noqa: E402
from scripts.validation.schema import CaseKey, CaseResult, ComboSpec, DatasetSpec  # noqa: E402

DATA_DIR = REPO_ROOT / "data-samples"
RESULTS_PATH = REPO_ROOT / "validation" / "results.json"
WORKDIR_BASE = REPO_ROOT / "validation" / "runs"

_PCA_SETTINGS = {"on": [True], "off": [False], "both": [False, True]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=None, help="Run only this dataset (by name)")
    parser.add_argument("--combo", default=None, help="Run only this detector combo (by name)")
    parser.add_argument("--pca", choices=["on", "off", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new cases after the first failure this session",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel worker processes (cases use independent workspaces)"
    )
    parser.add_argument(
        "--skip-explain-check",
        action="store_true",
        help="Skip the small separate explain-validation matrix (baseline combo, PCA off, per dataset)",
    )
    return parser.parse_args()


def _print_line(result: CaseResult) -> None:
    pca_label = "on " if result.pca else "off"
    prefix = f"    [{result.dataset:22s} pca={pca_label} {result.combo:20s}]"
    if result.status == "error":
        print(f"{prefix} ERROR: {(result.error or 'unknown')[:100]}")
        return
    if result.status == "too_few_records":
        print(
            f"{prefix} skipped: train={result.n_train}/holdout={result.n_holdout} "
            f"below scoring.min_records  {result.elapsed_seconds:>6.1f}s"
        )
        return
    acc = ""
    if result.roc_auc is not None and result.average_precision is not None:
        acc = f"  ROC-AUC={result.roc_auc:.4f} AP={result.average_precision:.4f}"
    print(
        f"{prefix} {result.n_holdout_flagged:>5} / {result.n_holdout:>7} flagged"
        f"{acc}  {result.elapsed_seconds:>6.1f}s"
    )


def _lookup(case: CaseKey) -> tuple[DatasetSpec, ComboSpec]:
    ds = dataset_by_name(case.dataset)
    cb = combo_by_name(case.combo)
    assert ds is not None
    assert cb is not None
    return ds, cb


def _run_sequential(
    to_run: list[CaseKey], code_rev: str, deps: dict[str, str], fail_fast: bool, handle: _ResultHandler
) -> None:
    for case in to_run:
        if handle.failed and fail_fast:
            print("  --fail-fast: stopping before remaining cases.")
            break
        ds, cb = _lookup(case)
        result = run_case(case, ds, cb, WORKDIR_BASE, DATA_DIR, code_rev, deps)
        handle(result)


def _run_parallel(
    to_run: list[CaseKey],
    code_rev: str,
    deps: dict[str, str],
    workers: int,
    fail_fast: bool,
    handle: _ResultHandler,
) -> None:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[CaseResult], CaseKey] = {}
        for case in to_run:
            ds, cb = _lookup(case)
            fut = pool.submit(run_case, case, ds, cb, WORKDIR_BASE, DATA_DIR, code_rev, deps)
            futures[fut] = case

        fail_fast_triggered = False
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except CancelledError:
                continue  # cancelled before it started (see below); not run this session
            handle(result)
            if handle.failed and fail_fast and not fail_fast_triggered:
                fail_fast_triggered = True
                print("  --fail-fast: cancelling cases that haven't started yet.")
                for other in futures:
                    if not other.done():
                        other.cancel()


def _run_explain_matrix(dataset_filter: str | None, seed: int) -> bool:
    """Run the small, separate explain-validation matrix. Returns True if anything failed."""
    print("\nExplain-validation matrix (baseline combo, PCA off, per dataset):")
    baseline = combo_by_name("baseline")
    assert baseline is not None
    any_failed = False
    for ds in DATASETS:
        if dataset_filter and ds.name != dataset_filter:
            continue
        rec = run_explain_check(ds, baseline, seed, WORKDIR_BASE, DATA_DIR)
        status_str = "OK" if rec["status"] == "success" else f"ERROR: {str(rec['error'])[:100]}"
        print(f"    [{ds.name:22s}] {status_str}  {rec['elapsed_seconds']:>6.1f}s")
        any_failed = any_failed or rec["status"] != "success"
    return any_failed


class _ResultHandler:
    """Persists every result as it arrives and tracks whether any case failed."""

    def __init__(self, seed_results: dict[CaseKey, CaseResult]) -> None:
        self.all_results = dict(seed_results)
        self.failed = False

    def __call__(self, result: CaseResult) -> None:
        self.all_results[result.key()] = result
        write_results_atomic(RESULTS_PATH, list(self.all_results.values()))
        _print_line(result)
        # "too_few_records" is a genuine, non-failing outcome (see
        # CaseResult.status) -- only "error" counts against the exit code.
        if result.status == "error":
            self.failed = True


def main() -> None:
    """Plan, run (or reuse), and report the validation matrix; exit 1 if anything failed."""
    args = _parse_args()

    if args.dataset and dataset_by_name(args.dataset) is None:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)
    if args.combo and combo_by_name(args.combo) is None:
        print(f"Unknown combo: {args.combo}", file=sys.stderr)
        sys.exit(1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKDIR_BASE.mkdir(parents=True, exist_ok=True)

    code_rev = code_revision(REPO_ROOT)
    deps = dependency_versions()

    cases = build_cases(
        DATASETS,
        COMBOS,
        _PCA_SETTINGS[args.pca],
        [args.seed],
        dataset_filter=args.dataset,
        combo_filter=args.combo,
    )

    def identity_for_case(case: CaseKey) -> object:
        ds, cb = _lookup(case)
        source_path = DATA_DIR / ds.file
        workdir = (
            WORKDIR_BASE / ds.name / ("pca_on" if case.pca else "pca_off") / cb.name / f"seed{case.seed}"
        )
        cfg = build_config(ds, case.pca, cb, case.seed, source_path, workdir)
        return build_identity(source_path, cfg, case.seed, code_rev, deps)

    existing = {r.key(): r for r in read_results_file(RESULTS_PATH)}
    to_run, reused = plan_run(cases, existing, identity_for_case)

    print(f"Validation matrix: {len(cases)} case(s) ({len(reused)} reused, {len(to_run)} to run)")
    print(f"Code revision: {code_rev}")

    handle = _ResultHandler({**existing, **{r.key(): r for r in reused}})

    t_start = time.time()
    if args.workers <= 1:
        _run_sequential(to_run, code_rev, deps, args.fail_fast, handle)
    else:
        _run_parallel(to_run, code_rev, deps, args.workers, args.fail_fast, handle)
    elapsed = time.time() - t_start
    print(
        f"\nMain matrix: {len(to_run)} run, {len(reused)} reused in {elapsed:.1f}s. Results: {RESULTS_PATH}"
    )

    explain_failed = False
    if not args.skip_explain_check:
        explain_failed = _run_explain_matrix(args.dataset, args.seed)

    if handle.failed or explain_failed:
        print("\nOne or more required cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
