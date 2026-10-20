#!/usr/bin/env python3
"""Run the evaluation harnesses and, optionally, refresh the benchmark tables in a document.

This used to be the public ``sorethumb benchmark`` command. It is a
maintainer/release tool, not part of the product interface: it edits files, so
it lives here (a repo script, never shipped in the wheel) and takes *explicit*
paths instead of guessing them.

* ``--output-dir`` is required: the ``*.md`` / ``*.csv`` result files are
  written there, and nowhere else.
* ``--readme`` is optional: the Markdown document that holds the result markers
  (``docs/benchmarks.md`` in this repository, which also carries the headline summary)
  is edited only when you pass it. There is no default, and no path is derived
  from where the package is installed. A path inside the ``sorethumb_ml``
  package directory is refused.

Two independent suites, both on by default:

* ``--pipeline``: mixed numeric + categorical synthetic scenarios through the
  real feature pipeline (no extra dependencies).
* ``--legacy``: real datasets (KDDCup99, Covtype; network) + bare-detector
  synthetic suite. Needs the ``benchmark`` extra (``uv sync --extra benchmark``).

Off by default:

* ``--policy-matrix``: every scenario x every registered ensemble policy, with
  stability refits, checked against ``tests/benchmark/budgets.toml``. Writes
  ``policy-matrix.json`` (results plus provenance: suite version, budgets
  version, policy ids, library version, git SHA, ``uv.lock`` hash) and
  ``policy-matrix.md``; with ``--readme`` also publishes the JSON to
  ``docs/benchmark-results/`` and injects the table.

Usage:
    uv run python scripts/run_benchmark.py --output-dir benchmark_results
    uv run python scripts/run_benchmark.py --output-dir benchmark_results --readme docs/benchmarks.md
    uv run python scripts/run_benchmark.py --output-dir /tmp/bench --no-legacy --pipeline-seeds 1

Exit codes: 0 success; 1 an incomplete or errored pipeline matrix (nothing is
published); 2 bad arguments or a missing requirement.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, prog="run_benchmark.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory the *.md / *.csv result files are written to (created if needed). Required.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help=(
            "Markdown document to inject the result tables into (docs/benchmarks.md in this repo). "
            "Omit to leave every document untouched."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Legacy suite: repeat each (dataset, detector) pair over this many seeds (default 5).",
    )
    parser.add_argument(
        "--pipeline-seeds",
        type=int,
        default=3,
        help="Pipeline suite: repeat each (scenario, ablation) cell over this many seeds (default 3).",
    )
    parser.add_argument(
        "--legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the real-dataset + bare-detector suite (needs the benchmark extra and network).",
    )
    parser.add_argument(
        "--pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the full-pipeline scenario suite.",
    )
    parser.add_argument(
        "--policy-matrix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the ensemble-policy matrix (every scenario x every policy, with stability refits), "
            "check it against tests/benchmark/budgets.toml, and write policy-matrix.json/.md. "
            "With --readme, also writes docs/benchmark-results/policy-matrix.json and injects the table."
        ),
    )
    parser.add_argument(
        "--matrix-seeds", type=int, default=10, help="Policy matrix: seeds per cell (default 10)."
    )
    parser.add_argument(
        "--matrix-bootstrap",
        type=int,
        default=3,
        help="Policy matrix: stability refits per seed (default 3).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def _check_readme_is_not_package_file(readme: Path) -> str | None:
    """Return an error message if *readme* sits inside the sorethumb_ml package directory."""
    import sorethumb_ml  # noqa: PLC0415

    package_dir = Path(sorethumb_ml.__file__).resolve().parent
    resolved = readme.resolve()
    if resolved == package_dir or package_dir in resolved.parents:
        return f"--readme {readme} is inside the installed package directory ({package_dir}); refusing to edit package files."
    return None


def _run_legacy(args: argparse.Namespace) -> tuple[int, int | None]:
    """Run the legacy suite. Returns ``(n_rows, exit_code)``; exit_code is None on success."""
    try:
        import datasets  # noqa: F401, PLC0415
    except ImportError:
        print(
            "error: the legacy suite needs the benchmark extra "
            "(uv sync --extra benchmark); install it or pass --no-legacy.",
            file=sys.stderr,
        )
        return 0, 2
    from sorethumb_ml.evaluate.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        generate_metadata,
        inject_into_readme,
        run_benchmark,
        write_outputs,
    )

    print(f"Running legacy benchmark harness ({args.seeds} seed(s) per pair)...")
    metadata = generate_metadata()
    rows = run_benchmark(BenchmarkConfig(n_seeds=args.seeds))
    if args.readme is not None:
        changed = inject_into_readme(rows, args.readme, metadata)
        print(
            f"Legacy table {'injected into' if changed else 'unchanged in (or no markers found in)'} {args.readme}"
        )
    md_path, csv_path = write_outputs(rows, args.output_dir, metadata)
    print(f"Legacy results written to {md_path} and {csv_path}")
    return len(rows), None


def _run_pipeline(args: argparse.Namespace) -> tuple[int, int | None]:
    """Run the pipeline suite. Returns ``(n_rows, exit_code)``; exit_code is None on success."""
    from sorethumb_ml.evaluate.pipeline_benchmark import (  # noqa: PLC0415
        PipelineBenchmarkConfig,
        assert_complete_and_error_free,
        expected_cells,
        inject_into_readme,
        run_pipeline_benchmark,
        write_outputs,
    )

    print(f"Running full-pipeline scenario benchmark ({args.pipeline_seeds} seed(s) per cell)...")
    cfg = PipelineBenchmarkConfig(n_seeds=args.pipeline_seeds)
    rows = run_pipeline_benchmark(cfg)
    try:
        assert_complete_and_error_free(rows, expected_cells(cfg))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 0, 1
    if args.readme is not None:
        changed = inject_into_readme(rows, args.readme)
        print(
            f"Pipeline table {'injected into' if changed else 'unchanged in (or no markers found in)'} {args.readme}"
        )
    md_path, csv_path = write_outputs(rows, args.output_dir)
    print(f"Pipeline results written to {md_path} and {csv_path}")
    return len(rows), None


def _git_sha() -> str:
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _sha256(path: Path) -> str | None:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _run_policy_matrix(args: argparse.Namespace) -> tuple[int, int | None]:
    """Run the policy matrix and check it against the budgets. Returns ``(n_rows, exit_code)``."""
    from datetime import UTC, datetime  # noqa: PLC0415

    import sorethumb_ml  # noqa: PLC0415
    from sorethumb_ml.evaluate.budgets import check_rows, default_policy_report, load_budgets  # noqa: PLC0415
    from sorethumb_ml.evaluate.pipeline_benchmark import (  # noqa: PLC0415
        assert_complete_and_error_free,
        expected_cells,
        inject_policy_matrix,
        policy_matrix_config,
        run_pipeline_benchmark,
        suite_version,
        to_json,
        to_markdown,
    )
    from sorethumb_ml.scoring.policies import DEFAULT_POLICY_ID, POLICIES  # noqa: PLC0415

    budgets_path = REPO_ROOT / "tests" / "benchmark" / "budgets.toml"
    budgets = load_budgets(budgets_path)
    cfg = policy_matrix_config(n_seeds=args.matrix_seeds, n_bootstrap=args.matrix_bootstrap)
    print(f"Running policy matrix ({cfg.n_seeds} seed(s), {cfg.n_bootstrap} stability refit(s) per cell)...")
    rows = run_pipeline_benchmark(cfg)
    try:
        assert_complete_and_error_free(rows, expected_cells(cfg))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 0, 1

    report = default_policy_report(check_rows(rows, budgets), budgets, DEFAULT_POLICY_ID)
    metadata = {
        "suite_version": suite_version(cfg),
        "budgets_version": budgets.version,
        "budgets_sha256": _sha256(budgets_path),
        "policies": [p.id for p in POLICIES],
        "default_policy": DEFAULT_POLICY_ID,
        "library_version": sorethumb_ml.__version__,
        "git_sha": _git_sha(),
        "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "n_seeds": cfg.n_seeds,
        "n_bootstrap": cfg.n_bootstrap,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "default_policy_unaccepted_misses": [
            f"{c.scenario}: {c.bound} {c.value:.3f} vs {c.limit}" for c in report.unaccepted_misses
        ],
        "stale_accepted_limitations": [f"{a.scenario}: {a.bound}" for a in report.stale_acceptances],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy-matrix.json").write_text(to_json(rows, metadata), encoding="utf-8")
    (args.output_dir / "policy-matrix.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"Policy matrix written to {args.output_dir / 'policy-matrix.json'}")
    for line in metadata["default_policy_unaccepted_misses"]:
        print(f"  default policy misses an unaccepted budget: {line}")
    for line in metadata["stale_accepted_limitations"]:
        print(f"  stale accepted limitation (the default now meets it): {line}")

    if args.readme is not None:
        published = REPO_ROOT / "docs" / "benchmark-results" / "policy-matrix.json"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(to_json(rows, metadata), encoding="utf-8")
        changed = inject_policy_matrix(rows, args.readme, metadata)
        print(
            f"Policy matrix {'injected into' if changed else 'unchanged in'} {args.readme}; JSON at {published}"
        )
    return len(rows), None


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    if args.readme is not None:
        if not args.readme.is_file():
            print(f"error: --readme {args.readme} does not exist or is not a file.", file=sys.stderr)
            return 2
        problem = _check_readme_is_not_package_file(args.readme)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2

    total_rows = 0
    if args.legacy:
        n_rows, code = _run_legacy(args)
        if code is not None:
            return code
        total_rows += n_rows
    if args.pipeline:
        n_rows, code = _run_pipeline(args)
        if code is not None:
            return code
        total_rows += n_rows
    if args.policy_matrix:
        n_rows, code = _run_policy_matrix(args)
        if code is not None:
            return code
        total_rows += n_rows

    print(f"\nDone. {total_rows} result(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
