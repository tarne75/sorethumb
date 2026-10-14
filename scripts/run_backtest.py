#!/usr/bin/env python3
"""Run pre-registered ensemble-policy backtests (see ``sorethumb_ml.evaluate.backtest``).

Phases:

* ``development`` -- the candidate against its baseline on the development
  seeds (the policy matrix's). Prints the decision; writes nothing. Run it as
  often as you like while forming a hypothesis.
* ``confirmation`` -- once per pre-registration, on the disjoint confirmation
  seeds. Refused unless the pre-registration file is committed and unmodified
  (so what was registered is what is judged), and refused if a result for it
  already exists (no re-rolling: a revised hypothesis is a new
  pre-registration). Writes ``docs/benchmark-results/backtests/<id>.json``
  with full provenance and the per-seed values the verdict rests on, and
  refreshes the summary in ``docs/ensemble-policy.md``.

Usage:
    uv run python scripts/run_backtest.py --phase development
    uv run python scripts/run_backtest.py --phase confirmation tests/benchmark/preregistrations/<id>.toml

Exit codes: 0 success (whatever the verdict); 1 a cell failed; 2 a refused or
invalid request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_DIR = REPO_ROOT / "tests" / "benchmark" / "preregistrations"
BUDGETS_PATH = REPO_ROOT / "tests" / "benchmark" / "budgets.toml"
RESULTS_DIR = REPO_ROOT / "docs" / "benchmark-results" / "backtests"
DOC_PATH = REPO_ROOT / "docs" / "ensemble-policy.md"
RESULTS_LINK = "benchmark-results/backtests"


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument(
        "preregistrations",
        nargs="*",
        type=Path,
        help="pre-registration TOML files (default: every file in tests/benchmark/preregistrations)",
    )
    return parser.parse_args(argv)


def _print_decision(prereg_id: str, decision: dict[str, Any]) -> None:
    verdict = "OUTPERFORMS" if decision["outperforms"] else "does not outperform"
    print(f"\n{prereg_id}: {verdict}")
    for comp in decision["comparisons"]:
        est = comp["estimate"]
        print(
            f"  {comp['scenario']:<22} mean diff {est['mean']:+.4f} "
            f"CI [{est['ci_low']:+.4f}, {est['ci_high']:+.4f}] p_holm {comp['p_holm']:.4f}"
            f"{'  improved' if comp['improved'] else ''}"
        )
    for reason in decision["reasons"]:
        print(f"  - {reason}")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    import sorethumb_ml  # noqa: PLC0415
    from sorethumb_ml.evaluate.backtest import (  # noqa: PLC0415
        CONFIRMATION_SEEDS,
        DEVELOPMENT_SEEDS,
        STABILITY_REFITS,
        backtest_scenarios,
        decide,
        decision_as_dict,
        inject_results,
        load_preregistration,
        run_policy_seeds,
    )
    from sorethumb_ml.evaluate.budgets import load_budgets  # noqa: PLC0415
    from sorethumb_ml.evaluate.pipeline_benchmark import (  # noqa: PLC0415
        PipelineBenchmarkConfig,
        suite_version,
    )

    args = _parse_args(argv)
    paths = args.preregistrations or sorted(PREREG_DIR.glob("*.toml"))
    if not paths:
        print("error: no pre-registration files given or found.", file=sys.stderr)
        return 2
    try:
        preregs = [(path, load_preregistration(path)) for path in paths]
    except (ValueError, KeyError) as exc:
        print(f"error: invalid pre-registration: {exc}", file=sys.stderr)
        return 2

    confirming = args.phase == "confirmation"
    commits: dict[str, str] = {}
    if confirming:
        for path, prereg in preregs:
            commit = _registration_commit(path)
            if commit is None:
                print(
                    f"error: {path} must be committed, with no local changes, before its confirmation run.",
                    file=sys.stderr,
                )
                return 2
            if (RESULTS_DIR / f"{prereg.id}.json").exists():
                print(
                    f"error: {prereg.id} already has a confirmation result; a revised hypothesis needs "
                    "a new pre-registration.",
                    file=sys.stderr,
                )
                return 2
            commits[prereg.id] = commit

    seeds = CONFIRMATION_SEEDS if confirming else DEVELOPMENT_SEEDS
    budgets = load_budgets(BUDGETS_PATH)
    kinds = backtest_scenarios()
    cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def seeds_for(policy_id: str) -> dict[str, list[dict[str, Any]]]:
        if policy_id not in cache:
            print(f"Running {policy_id} on {len(kinds)} scenarios x {len(seeds)} {args.phase} seeds...")
            cache[policy_id] = run_policy_seeds(policy_id, seeds)
        return cache[policy_id]

    cfg = PipelineBenchmarkConfig(seed=seeds[0], n_seeds=len(seeds), n_bootstrap=STABILITY_REFITS)
    for path, prereg in preregs:
        try:
            base, cand = seeds_for(prereg.baseline), seeds_for(prereg.candidate)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        per_seed = {s: {"baseline": base[s], "candidate": cand[s]} for s in kinds}
        decision = decision_as_dict(decide(prereg, per_seed, kinds, budgets))
        _print_decision(prereg.id, decision)
        if not confirming:
            continue
        result = {
            "preregistration": asdict(prereg)
            | {"path": str(path.resolve().relative_to(REPO_ROOT)), "commit": commits[prereg.id]},
            "phase": args.phase,
            "seeds": list(seeds),
            "metadata": {
                "suite_version": suite_version(cfg),
                "budgets_version": budgets.version,
                "budgets_sha256": _sha256(BUDGETS_PATH),
                "library_version": sorethumb_ml.__version__,
                "git_sha": _git("rev-parse", "HEAD"),
                "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "stability_refits": STABILITY_REFITS,
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            },
            "kinds": kinds,
            "per_seed": per_seed,
            "decision": decision,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{prereg.id}.json"
        out.write_text(json.dumps(_nan_to_none(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  written to {out.relative_to(REPO_ROOT)}")

    if confirming:
        results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
        changed = inject_results(DOC_PATH, results, RESULTS_LINK)
        print(
            f"Backtest summary {'updated in' if changed else 'unchanged in'} {DOC_PATH.relative_to(REPO_ROOT)}"
        )
    return 0


def _nan_to_none(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, list | tuple | dict):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and value != value:  # noqa: PLR0124 -- NaN check
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_nan_to_none(v) for v in value]
    return value


if __name__ == "__main__":
    sys.exit(main())
