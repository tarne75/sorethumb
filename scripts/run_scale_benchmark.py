#!/usr/bin/env python3
"""Run the end-to-end scale benchmark and publish its results.

Generates a mixed transaction-like dataset at each size, runs a zero-config
``sorethumb run`` on it and scores a second same-sized file forward, each in a fresh
process, and records elapsed time per stage, peak memory, feature width, detector
training samples, explanation cost and output size. See
``sorethumb_ml/evaluate/scale_benchmark.py``.

Usage:
    uv run python scripts/run_scale_benchmark.py                      # 10k, 100k, 1M; publishes
    uv run python scripts/run_scale_benchmark.py --sizes 10000 --no-publish

Needs a few GB of free disk for the 1M case (CSV inputs and the workspace); everything
is written under --scratch and removed after each size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sorethumb_ml.evaluate.scale_benchmark import (  # noqa: E402
    SCALE_SIZES,
    render_markdown,
    run_scale_benchmark,
    write_results,
)


def main() -> int:
    """Run the sizes asked for and publish (or print) the results."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SCALE_SIZES), help="Row counts to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch", type=Path, default=REPO_ROOT / "scale_benchmark_scratch")
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "docs" / "benchmark-results" / "scale.json")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "docs" / "benchmarks.md")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Print the tables instead of writing the JSON and benchmarks.md",
    )
    args = parser.parse_args()

    doc = run_scale_benchmark(tuple(args.sizes), args.scratch, seed=args.seed)
    if args.no_publish:
        print(render_markdown(doc))
    else:
        write_results(doc, args.json, args.readme)
        print(f"Wrote {args.json} and the scale section of {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
