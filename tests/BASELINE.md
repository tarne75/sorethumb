# Test-suite baseline (P1-1)

Captured 2026-09-14, immediately before the P1-1 lane refactor (`prompts/pre-release-plan.md`),
on `uv run pytest` with the config in place at that commit (`addopts = ["--strict-markers", "--tb=short"]`,
only `integration`/`benchmark` markers declared, neither applied to any test module).

## Before P1-1

- Collected: **872 tests**.
- `pytest -m "not benchmark"` (the closest existing lane at the time; `integration` was
  declared but unused, so it selected nothing): **861 passed, 11 deselected**, in **~109s**.
- Branch coverage (`--cov=sorethumb --cov-report=term-missing`, same run): **85.64%**
  (4633 statements / 1246 branches; `TOTAL` miss 654 stmts, 112 partial branches).
- Slowest tests (`--durations=15`), all in `tests/integration/` or `tests/unit/test_explain.py`
  / `test_config_wiring.py`, none over 4.8s — nothing pathologically slow yet, but nothing was
  marked `integration` either, so plain `pytest` ran the full pipeline/CLI/SQLite/report suite
  every time.

## What P1-1 changed

- Added marker definitions for `unit`, `contract`, `integration`, `property`, `slow`,
  `benchmark`, `network`, `repo_check` (`pyproject.toml`); added `--strict-config` and
  `--durations=15`; set default `addopts -m` to
  `not integration and not property and not benchmark and not slow and not network and not repo_check`.
- Applied `pytestmark` by real module behaviour (no test bodies changed):
  - `integration`: `tests/integration/{test_end_to_end,test_row_identity,test_score_forward}.py`,
    `tests/unit/{test_cli,test_config_wiring,test_history,test_report,test_store}.py`
    (real Workspace/SQLite, CLI process, full pipeline run, or report rendering).
  - `contract`: `tests/unit/test_public_api.py` (public API surface).
  - `property`: `tests/property/test_profiling_properties.py`.
  - `repo_check`: `tests/unit/test_docs_checks.py`.
  - `unit`: `tests/unit/{test_detectors,test_evaluate,test_explain,test_features,test_grouping,
    test_profiling,test_ranking,test_scoring,test_smoke}.py`.
  - `tests/unit/test_tier{1,2,3,4}_coverage.py` were deliberately left unmarked (they still run
    in the default lane): they're padding files slated for wholesale deletion in P1-3, and a
    couple of tests inside `test_tier2_coverage.py`/`test_tier3_coverage.py` do touch a real
    Workspace/CLI — not worth per-test marking on code that's about to be deleted.
  - `slow` and `network` have no members yet — nothing in the current suite meets either bar.
    They're declared for later phases (P1-6/P1-7, P3-2) to use.
- Moved the 11 tests in `test_evaluate.py` that called `run_benchmark` (real detector fitting)
  into a new `tests/benchmark/test_benchmark_harness.py`, marked `benchmark`. `test_evaluate.py`
  now only exercises pure `evaluate_scores`/`Metrics`/dataclass/formatter logic.
- `ci.yml`'s `test` job now runs `-m "not benchmark and not slow and not network"` (previously
  `-m "not integration and not benchmark"`, which had been a no-op on `integration` since the
  marker was never applied) — local integration tests still run on every PR, per the P1-1
  requirement to keep them in the required workflow.

## After P1-1

- Collected: **872 tests** (unchanged — this phase only reclassified/moved tests, none were
  added or deleted).
- Plain `pytest` (default lane: `unit` + `contract` + unmarked tier-coverage files):
  **559 passed, 313 deselected, ~5.8s**.
- `pytest -m "not benchmark and not slow and not network"` (new CI `test` job lane):
  **850 passed, 22 deselected, ~106s** (861 baseline − 11 tests moved into the `benchmark` lane).
- `pytest -m benchmark`: **22 passed** (11 pre-existing `test_accuracy_floors.py` +
  11 moved `test_benchmark_harness.py`).
- Per-marker collection counts: `unit` 364, `contract` 23, `integration` 276, `property` 5,
  `benchmark` 22, `repo_check` 10, `slow`/`network` 0. Remaining 172 = unmarked tier-coverage
  files.
- Branch coverage on the CI lane: **84.30%** (down from 85.64% — expected: the moved
  `run_benchmark` calls in `evaluate/benchmark.py` are no longer exercised by the default/CI
  lane, only by the explicit/scheduled `benchmark` lane). Still clears the `--cov-fail-under=80`
  gate.
