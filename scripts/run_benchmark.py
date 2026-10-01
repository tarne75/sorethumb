#!/usr/bin/env python3
"""Run the evaluation harnesses and, optionally, refresh the README's benchmark tables.

This used to be the public ``sorethumb benchmark`` command. It is a
maintainer/release tool, not part of the product interface: it edits files, so
it lives here (a repo script, never shipped in the wheel) and takes *explicit*
paths instead of guessing them.

* ``--output-dir`` is required: the ``*.md`` / ``*.csv`` result files are
  written there, and nowhere else.
* ``--readme`` is optional: the README is edited only when you pass it. There is
  no default, and no path is derived from where the package is installed. A
  path inside the ``sorethumb_ml`` package directory is refused.

Two independent suites, both on by default:

* ``--pipeline``: mixed numeric + categorical synthetic scenarios through the
  real feature pipeline (no extra dependencies).
* ``--legacy``: real datasets (KDDCup99, Covtype; network) + bare-detector
  synthetic suite. Needs the ``benchmark`` extra (``uv sync --extra benchmark``).

Usage:
    uv run python scripts/run_benchmark.py --output-dir benchmark_results
    uv run python scripts/run_benchmark.py --output-dir benchmark_results --readme README.md
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
        help="README to inject the result tables into. Omit to leave every README untouched.",
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

    print(f"\nDone. {total_rows} result(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
