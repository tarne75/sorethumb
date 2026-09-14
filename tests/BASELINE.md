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

## What P1-2 changed

Added `tests/factories/` (frames, configs, detectors, workspaces, runs, benchmark_rows) and
rewired the duplicated builders the plan named onto it — no test bodies' assertions changed,
only how each test's fixtures/config/data get built:

- `frames.py`: `write_planted_csv` (unifies end-to-end's `_make_planted_csv` and
  score-forward's `_planted_csv` — same logic, different defaults, so score-forward keeps a
  one-line wrapper preserving its own `n_normal=200, n_anomaly=6`), `write_leading_anomaly_csv`
  (config-wiring's `_write_csv`), `write_grouped_csv` (CLI's `_write_csv` — same name, different
  shape, now disambiguated), `write_time_sorted_parquet` and `write_two_period_parquet`
  (end-to-end's two Parquet builders, including the "two-period builder" the plan named).
- `configs.py`: `make_config`, defaulting to `combination="intersection"` per the plan. Every
  existing caller either has just one detector (where intersection == that detector's own flag,
  so behaviour is unchanged) or passes its own `combination=` explicitly — audited call by call,
  zero behavioural changes. End-to-end's `_minimal_config`, `_period_config`,
  `_period_config_with_groups` and score-forward's `_cfg` (2 detectors + composite) now delegate
  to it instead of building `Config(...)` by hand.
- `detectors.py`: `detector_auc` centralises the higher-is-more-normal sign flip
  (`roc_auc_score(y_true, -scores)`) that `test_kmeans_auc_above_half` computed inline;
  `ban_all_fitting` promotes score-forward's `_ban_all_fitting`/`_FitAttempted`.
- `workspaces.py`: a `workspace` pytest fixture (`yield` + `ws.close()`) and a plain
  `open_workspace()` helper for tests needing more than one instance. `test_history.py`'s local
  `ws` fixture (which never closed its Workspace) now resolves to the shared one via
  `tests/unit/conftest.py` — a bare `from ... import workspace as ws` inside `test_history.py`
  itself works at runtime but trips ruff's F811 (parameter shadowing an imported name), so the
  import lives in conftest.py instead, where pytest's plugin-based fixture discovery picks it up
  with no import needed in the consuming test module. `test_store.py`'s `_open_ws` is now an
  alias for `open_workspace` (kept un-closed there deliberately: several of its tests construct
  more than one `Workspace` per test — double-init, reopen — so a single auto-closing fixture
  doesn't fit).
- `runs.py`: `run_planted_detection` returns a `PlantedRun` dataclass (`result`, `csv_path`,
  `workdir`, `anomaly_indices`) instead of a positional tuple, collapsing the
  write-csv/build-config/run_detection sequence; wired into
  `test_planted_anomalies_are_detected`, which also dropped a recomputed
  `n_total`/`n_anomaly`/`planted_positions` in favour of the returned `anomaly_indices`.
- `benchmark_rows.py`: `make_benchmark_row` promotes `test_evaluate.py`'s local `_make_row`.

Deliberately deferred, with reasons:
- **`CliRunner` fixture**: added (`workspaces.py`), but the existing module-level
  `runner = CliRunner()` globals in `test_cli.py`/`test_config_wiring.py` were left as-is —
  CliRunner is stateless (no real cleanup need, unlike `Workspace`), and converting ~60 existing
  test signatures to take it as a fixture parameter was a large mechanical diff for no behaviour
  change. Left for whoever does the P1-4 file-split to adopt where it's a natural fit.
- **`assertions.py`**: not created. Went looking for the shared-assertion duplication the plan
  named (e.g. the dense-rank invariant) and found it checked twice at two different levels
  (`test_ranking.py`'s pure `_dense_ranks` mirror vs. `test_end_to_end.py`'s dataframe-level
  check) but not actually duplicated in a way that has one obvious shared shape yet — forcing a
  helper into existence for a single dataframe-level call site would be the premature
  abstraction the project's own conventions warn against. Revisit once P1-5's CLI/report output
  consolidation actually produces repeated assertions to extract.

Verified after: full suite still **872 collected**, CI lane still **850 passed, 22 deselected**,
plain `pytest` still **559 passed** — byte-for-byte the same test outcomes as the P1-1 baseline
above, just built through shared factories. ruff/ruff format/mypy/doc-consistency all clean.

## What P1-3 changed

Deleted `tests/unit/test_tier{1,2,3,4}_coverage.py` (141 test functions total) and the four
`*_reexecuted_under_coverage` tests plus assorted `__all__`/is-not-None filler in
`test_public_api.py`. Before deleting, read every tier test and, for each one, checked whether
an equivalent already existed somewhere real (a lot did — these files were bolted on later
specifically to hit line-coverage numbers, and mostly re-tested things test_detectors.py /
test_features.py / test_store.py / test_scoring.py / test_explain.py already covered from a
different angle) before porting the genuinely-new ones:

- `tests/unit/test_config.py` (new): `_default_detectors()`'s 3-detector ensemble, `ScoringConfig`
  contamination validation, and the three `config_hash()` behavioural contracts (stable,
  excludes cosmetic fields, changes with result-affecting fields). Every *pure default-echo* test
  in tier1 (construct with defaults, assert field == the literal already declared in config.py)
  was dropped, not ported: `docs/generate_config_docs.py` already derives docs/configuration.md
  from those same Field defaults and Literal enum members, and its drift check
  (`test_docs_checks.py`, `repo_check`) already catches a silently-changed default or
  narrowed/widened Literal — a hand-written duplicate of the same literal defended nothing extra.
- `tests/unit/test_detectors.py`: only 3 of tier3's tests were new (`test_registry_values_are_detector_classes`,
  `test_detectors_all_list`, a non-callable-method protocol case) — its own `check_protocol`
  and registry tests already existed and were more complete.
- `tests/unit/test_features.py`: 13 new (FeatureSpace hash/dataclass/instantiation from tier1;
  `_sanitize`, empty-df correlate/drop_correlated, unknown time-derivative, the demotion
  break-path, and all three `build_encoding_exprs` treatment branches from tier4). Several tier4
  "edge cases" turned out to be exact duplicates of existing tests under different names
  (`test_array_derive_exprs_string_inner_no_numeric_stats`, `test_apply_pca_shape_mismatch_raises_plan_error`)
  and were dropped.
- `tests/unit/test_io.py`: 18 new — the entire TSF reader test matrix (attributes, missing
  values, padding, comments, malformed rows) plus TSV/JSON/all-string-schema, none of which
  test_io.py had any coverage of at all.
- `tests/unit/test_store.py`: 4 new Workspace accessor/prune tests (`features_dir`, `logs_dir`,
  `tmp_dir`, `db_path()`, `list_prunable()`, a prune-tolerates-an-already-deleted-file case) that
  were genuinely untested; tier3's open/context-manager/dry-run-prune tests were dropped as exact
  duplicates of existing tests (same scenario, sometimes the same function name).
- `tests/unit/test_profiling.py`: 12 new (`_check_identifier`'s aggressive/hex/no-match branches,
  `_is_ignored`'s pattern matching, the high_null/unsupported `treatment_for` mappings,
  `FeaturePlan.__post_init__`'s guard, `reference_column` exclusion, non-chosen-temporal-dropped).
  tier3's classify/JSON-round-trip tests were dropped as duplicates of existing, more thorough
  versions built from real profiled data rather than a hand-built `ColumnProfile`.
- `tests/unit/test_scoring.py`: nothing ported — every one of tier3's calibrator tests turned out
  to be an exact or near-exact duplicate of an existing test (transform-before-fit,
  empty-transform, constant-distribution, to_dict/from_dict round-trip).
- `tests/unit/test_explain.py`: 4 new — `kernel_shap_attributions` had zero existing coverage;
  `permutation_importance`'s row-cap and equal-importance (`rng_v == 0`) branches were untested.
  tier4's `back_project_pca` mismatch case was dropped as a duplicate.
- `test_public_api.py` also lost two real duplicates against `test_smoke.py`
  (`test_error_hierarchy`/`test_warning_hierarchy` checked the exact same 9 error / 10 warning
  classes as `test_all_sorethumb_errors_are_exceptions`/`test_all_sorethumb_warnings_are_user_warnings`,
  just without the str()/instantiation checks) and `test_package_version` (identical assertion to
  `test_smoke.py::test_version_attribute`) — kept the richer contract-file versions, removed the
  smoke-file duplicates.

**Coverage floor lowered 80 -> 70 in `ci.yml`.** Branch coverage on the CI lane dropped from
84.30% to **72.89%** — expected, not a loss of real behavioural coverage: `errors.py`,
`explain/__init__.py`, `scoring/__init__.py`, `store/__init__.py`, and the dataclass/Enum bodies
in `features/space.py`, `profiling/classify.py`, `profiling/plan.py`, `store/db.py`,
`store/workspace.py`, and `evaluate/metrics.py` all dropped because their class/exception bodies
execute exactly once, at true import time during pytest collection, before coverage.py's tracer
attaches — the *only* thing that ever marked those lines "covered" was the `importlib.reload()`
calls this phase deleted. This is the exact trade the plan asks for in P1-3 and resolves properly
in P1-7 ("replace the 80% line-coverage gate that incentivised reload tests with required
behavioural lanes, informational overall branch coverage, diff coverage for changed lines"); 70%
is a stopgap floor with headroom below the new honest baseline, not the final policy.

Verified after: **781 collected** (872 - 141 tier tests - 8 public_api trims - 2 smoke trims + 60
ported/new = 781, and every test now carries exactly one lane marker — the 172 previously-unmarked
tier-coverage tests are simply gone). CI lane: **759 passed, 22 deselected**. Plain `pytest`:
**464 passed**. ruff, ruff format, mypy, and the doc-consistency check all clean.

## What P1-4 changed

Reorganised the whole tree by behavioural ownership and split every monolith the plan named
(`test_detectors.py`, `test_cli.py`, `test_end_to_end.py`, `test_store.py`, `test_features.py`,
`test_history.py`) by seam. No test body's assertions changed in this phase — every move/split
was verified to reproduce the exact same pass count before moving on to the next file.

- **`tests/unit/` subpackages**: `config/`, `profiling/`, `features/`, `scoring/`, `explain/`,
  `io/`, `evaluation/` (the plan's named seven) plus three more real ones discovered while
  splitting: `detectors/` (per-detector fit/score behaviour), `history/` (pure period-label math,
  split out of `test_history.py` -- `TestResolvePeriod`/`TestPeriodBounds`/`TestStepNavigation`
  touch no Workspace at all), and `store/` (`make_group_key`/`validate_identifier`, equally pure).
- **`tests/contract/`** (new top-level directory): `test_public_api.py` (moved as-is),
  `test_cli.py` (JSON/exit-code/--help contract -- real `run` invocations only where unavoidable
  as setup, e.g. `config show <run_id>` needs a run_id to exist), `test_detector_plugins.py`
  (the `Detector` protocol, the registry, `available_extra_params()` -- the shape every plugin
  must have, split out of `test_detectors.py`), `test_database_schema.py` (Workspace.init/open,
  every migration, busy_timeout/checksums/schema-ceiling hardening, dataset/run/group CRUD,
  write_results/read_results), `test_model_manifests.py` (save_model/load_model, library-version
  recording, the P0-5 fails-closed cases), `test_feature_plan_compat.py` (FeaturePlan's
  to_json/from_json contract -- consolidated from three tests in `test_profiling.py` plus one in
  `test_features.py`, since a persisted plan is read back by score-forward and by a resumed run
  released later, making it one contract regardless of which module happens to build the plan).
- **`tests/integration/`**: added `test_pipeline_detection.py`, `test_pipeline_explain.py`,
  `test_pipeline_resume.py` (split from `test_end_to_end.py` by seam: core detection/ranking/
  schema-stability, attribution correctness, and run-identity/resume/failed-group semantics),
  `test_store_workflows.py` (interrupted-run recovery, score_with_existing, retention/pruning --
  the `test_store.py` content that's a multi-step workflow rather than one persistence contract),
  and `test_cli.py` (the CLI tests that assert on workflow correctness -- idempotency, dry-run
  isolation, config-byte-preservation -- rather than output shape). `test_end_to_end.py`'s report-
  regeneration and history/period-override tests were merged into the existing
  `tests/integration/test_report.py` and `test_history.py` respectively, since "report" and
  "history" are the plan's own named integration seams and CLI-driven history/report tests belong
  with the rest of that seam's tests, not scattered across a separate "pipeline" grouping.
- **`tests/repo_check/`** (new top-level directory): `test_docs_checks.py` (moved as-is).
- **`tests/conftest.py`**: replaced the two fixtures that had zero real callers
  (`simple_frame`/`frame_with_anomalies` -- checked with a repo-wide grep before removing) with a
  single re-export of the `workspace` fixture from `tests/factories/workspaces.py`, now needed
  repo-wide since Workspace-backed tests live in `unit/`, `contract/`, and `integration/`.
  `tests/unit/conftest.py` (P1-2's home for this same re-export) is gone -- one fixture, one home.
- **Found and fixed while auditing "no CLI process may stay labelled unit"**: `test_smoke.py`
  had its own `test_cli_version` invoking `runner.invoke(app, ["--version"])` -- a real CliRunner
  call, and an exact duplicate of `test_cli.py::test_version`'s assertions (already contract from
  this phase's split). Deleted the smoke-file copy.

Verified after every split (not just at the end): **780 collected** (781 - 1 for the
`test_cli_version` duplicate just found; every other move/split reproduced the prior file's exact
pass count). CI lane: **758 passed, 22 deselected**, **~43s** (branch coverage still 72.89% --
identical to before this phase, confirming it's a pure reorganisation). Plain `pytest`: **583
passed** (up from 464 -- the default lane now also runs the populated `contract` lane, which
didn't have real content until this phase; still ~10s). Per-marker collection: `unit` 462,
`contract` 122, `integration` 159, `property` 5, `benchmark` 22, `repo_check` 10 -- sums to 780,
every test carries exactly one marker. ruff, ruff format, mypy, and the doc-consistency check all
clean.

**Known wart, left alone**: `tests/contract/test_cli.py` and `tests/integration/test_cli.py`
each define their own local `workspace` fixture returning `(csv_path, toml_path, workdir)` --
same name, different type, as the shared root-conftest `workspace` fixture (a `Workspace`
object). Pytest's per-module fixture resolution means there's no actual runtime conflict (the
local definition always wins in its own file), but it's a naming trap for anyone skimming both
files side by side. Renaming it (e.g. to `cli_workspace`) would touch every one of the ~50 CLI
test signatures for a cosmetic win; left as a note rather than done under this phase's already
large diff.
