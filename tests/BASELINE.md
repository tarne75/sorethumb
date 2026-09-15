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

## What P1-5 changed

- **Detector contract table** (`tests/contract/test_detector_plugins.py`): a `DetectorSpec`
  dataclass table, one row per built-in detector, with fields for min_rows, whether `seed`
  actually changes the fit (verified against each wrapper's `# noqa: ARG002` on `seed` --
  isolation_forest/kmeans_distance pass it to sklearn's `random_state`, one_class_svm/lof/
  ecod/hbos ignore it entirely), the natural_flag mechanism, training cap, extra-params
  support, and attribution kind. Four parametrized tests run over the table: class-vars match,
  fit/score at the documented minimum row count, the higher-is-more-normal orientation
  invariant, seed-sensitivity, and (new) a save_model/load_model round-trip -- which ECOD,
  HBOS and LOF had *never* been individually tested through before; only the shipped
  three-detector ensemble was. Removed the 26 now-subsumed generic per-detector tests
  (fit_score_shape, scores_are_floats, natural_flag_shape[_and_dtype], class_vars, and the
  four bare `higher_more_normal` mean-comparison duplicates) from
  `tests/unit/detectors/test_detectors.py`, keeping every detector-*specific* test (CBLOF,
  elbow-k, ECOD/HBOS feature contributions, LOF's small-dataset clamp, OCSVM's exact
  zero-hyperplane threshold, extra_params forwarding) exactly as it was.
- **`tests/unit/evaluation/test_evaluate.py`**: collapsed 9 `evaluate_scores()` tests that
  each called it with near-identical inputs to check one field into
  `test_evaluate_scores_on_random_input` (one call, one coherent assertion block), and 4
  "perfect detector" tests into `test_perfect_detector_maximises_every_metric` the same way.
- **`tests/integration/test_report.py`**: collapsed 4 `render_report()` calls with the exact
  same fixture (checking file existence, run_id, dataset_uri, and no-external-refs
  separately) into one golden-file comparison plus one focused
  `test_html_is_self_contained_no_external_refs` (kept separate and explicitly named because
  it's a security-relevant invariant, not just another field); collapsed 3 identical
  `write_group_csv()` calls (name, columns, row count) into one. Every formula-injection and
  escaping test was left untouched, per the explicit instruction to retain those.
- **`test_explanation_references_perturbed_column`**: replaced its `pytest.skip("explain not
  enabled or reason columns absent")` escape hatch with a hard assertion that `reason_1`
  exists (explain is enabled by default in this fixture, so the skip was pure dead code that
  could have silently hidden a real regression).
- **`test_fit_apply_schema_is_stable`**: this test's whole point is proving fit and apply agree
  on the feature schema hash *when demotion fires* -- but its fixture (`high_card`, 80
  categories) never actually triggered demotion: at cardinality 80 with the default
  `one_hot_max_cardinality=20`, that column was already frequency-encoded, never one-hot, so
  `compute_demotions` had nothing to demote and the hash comparison passed vacuously regardless
  of whether demotion logic worked at all. Rebuilt the fixture with three genuinely one-hot
  categorical columns and a deliberately restrictive `max_feature_width=10`, and added
  `assert plan.demoted_columns` (plus the expected `FeatureWidthWarning`) before comparing
  hashes, so the test now fails loudly if demotion stops firing instead of passing either way.
- **CLI tests must not query `store._conn`**: found one violation in each of two files.
  `tests/integration/test_cli.py::_totals_period_labels` used raw
  `SELECT DISTINCT period_label FROM totals`; rewritten to load the same `Config` the CLI
  loaded and call the public `Store.totals_for_periods()`. `test_run_dry_run_registers_dataset_and_run_but_fits_nothing`
  checked six raw `SELECT COUNT(*)` queries; rewritten onto `Store.dataset_snapshots()`,
  `Store.list_runs()` and `Store.all_run_groups()` (dropping the redundant standalone `totals`
  count, since zero run_groups already implies zero totals) plus a filesystem check for models.
  Every other `store._conn` usage in the tree is in `tests/contract/` (the store/manifest
  contract files) or `tests/integration/test_history.py` / `test_store_workflows.py` (seeding
  scenario state for a workflow test) -- both legitimate, not CLI tests.
- **Private estimator checks audited**: `._model`/`._quantile_values` access exists only in
  `tests/contract/test_model_manifests.py` (clearly a manifest contract file) and in
  `tests/unit/detectors/test_detectors.py` functions all named `test_extra_params_forwarded_to_*`
  / `test_extra_params_via_registry_kwargs` -- already self-describing as adapter-contract
  checks, so nothing needed renaming or moving.
- **Golden files** (new `tests/golden/` + `tests/factories/golden.py`): `cli_detectors.json`
  and `cli_config_schema.json` (both fully deterministic CLI JSON output -- no timestamps or
  run ids involved) replace weak substring checks in `test_detectors_json_output`/
  `test_config_schema_emits_json`; `report_index.html` pins the full rendered report from the
  file's existing fixed `_RUN_META`/`_group()` fixtures. `assert_matches_golden()` normalises
  JSON (`sort_keys=True`) before comparing so key-order alone can't cause a spurious failure,
  and supports `UPDATE_GOLDEN=1` to regenerate after reviewing an intentional diff.

Verified after every change: **767 collected** (780 - 13 net, despite the detector table adding
48 new tests -- the 26 removed detector duplicates plus the evaluate.py/report.py
consolidations outweighed it). CI lane **745 passed, 22 deselected**, coverage still **72.89%**
(identical to P1-4 -- confirms no behavioural drift, just better-organised and higher-quality
assertions over the same code paths). ruff, ruff format, mypy, and the doc-consistency check all
clean.

## What P1-6 changed

**Deferred to P2, by explicit user decision**: two items on the plan's acceptance-test list
depend on production code that doesn't exist yet. "Empty intersections producing a specific
actionable warning" needs P2-4's `ZeroAnomalyWarning` (grepped the whole tree -- doesn't exist).
"Explicit contamination selects exactly k rows under ties" needs P2-3's exact-k selector --
`combine.py` still uses `np.quantile(..., 1.0 - c)` threshold selection today, which can select
more or fewer than k rows when scores tie at the boundary. Writing tests for either now would
mean either committing a known-failing test or pulling P2's production changes forward out of
order; asked the user, who chose to defer both and keep this phase test-only. Revisit when
P2-3/P2-4 land.

**Fixed a real anti-pattern already in the suite**: `tests/property/test_profiling_properties.py`
had three `@given` tests using a silent `return` to skip invalid examples (`if n_null > n_total:
return`) -- exactly what the plan says never to do, since Hypothesis's shrinking/statistics
can't distinguish "discarded" from "passed" the way `assume()` lets it. Replaced all three with
`assume(...)`.

**Acceptance scenarios -- audited first, most already existed**: grepped for each of the nine
named scenarios before writing anything new. Eight were already covered by P0/P1-1..P1-5 work
(failed groups persisted as failed, atomic config-scoped history, the three-way intersection
vote, dense ranks, score-forward's five corruption/mismatch rejections, null-vs-empty-vs-literal
groups, source-stable row ids). The ninth -- "reports containing planted IDs/reasons" -- was a
real gap: the existing report tests checked the report wasn't *blank*, never that it actually
*contains* the specific planted rows and a real reason. Added
`test_report_surfaces_planted_row_ids_and_reasons` (`tests/integration/test_report.py`): runs a
real planted-anomaly detection with reporting on, and asserts the sibling group CSV's `row_id`
set matches the flagged rows exactly (not a substring guess) and that `reason_1` is populated.

**New property tests** (`tests/property/`, all using `assume()`/constrained strategies, never
silent returns):
- `test_profiling_properties.py`: classification *and* treatment are invariant to row
  permutation; exact row duplication preserves `null_ratio` and classification/treatment.
  (`cardinality_ratio` is deliberately *not* asserted invariant under duplication -- doubling
  the population without adding new distinct values mathematically halves unique/total by
  construction; Hypothesis found this immediately when a first draft of the test asserted it,
  which was the test's bug, not the product's.)
- `test_calibration_properties.py` (new): `Calibrator.transform()` is bounded to [0, 1],
  monotone non-increasing in the raw score, tie-aware (identical raw scores calibrate
  identically), and `to_dict()`/`from_dict()` round-trips to identical transform output.
- `test_feature_properties.py` (new): `fit_features`/`apply_feature_plan` always preserve row
  count and produce a fully finite matrix, across make_frame's space of column types, nulls,
  and correlation.
- `test_grouping_properties.py` (new): `make_group_key` is invariant to the order its dict was
  built in (already guaranteed by `json.dumps(..., sort_keys=True)`, now regression-tested
  directly), and distinguishes `None`/`""`/`"None"` and same-string-form different types (`1`
  vs `"1"`).
- `test_detector_properties.py` (new): `natural_flag` is batch-invariant -- a row's flag must
  not depend on what else was scored alongside it -- for the five detectors that already
  promise it (isolation_forest, one_class_svm, lof, ecod, hbos) via a threshold fixed at fit
  time. `kmeans_distance` is deliberately excluded: its Tukey-fence natural boundary is
  recomputed from whatever batch is passed to `natural_flag()`, so it does not yet promise this
  (that's P2-1's fix).
- `test_atomic_write_properties.py` (new): for arbitrary prior/attempted byte content, an
  interrupted `atomic_write` always leaves the target exactly as it was and cleans up its own
  temp file -- generalises the fixed-example versions of this check already in
  `test_database_schema.py`/`test_model_manifests.py`/`test_report.py` into one property test
  against `_atomic.py` directly.

**New concurrency scenario** (`tests/integration/test_store_workflows.py`):
`test_concurrent_write_results_does_not_corrupt_the_workspace` -- eight threads writing results
for the same `(run_id, group_key)` at once; every thread either succeeds cleanly or raises a
real exception (never hangs or corrupts), and afterward the workspace is fully readable with
exactly one writer's rows as the final state (atomic_write's rename is all-or-nothing, so the
last one to replace() wins wholesale, never an interleaved mix).

Verified after every change: **782 collected** (767 + 15 net: 18 property tests including the 3
fixed ones, +1 concurrency scenario, +1 report acceptance test). CI lane **760 passed, 22
deselected**, coverage **72.92%** (up slightly from 72.89% -- the new property tests exercise a
few previously-uncovered lines in `calibrate.py`/`workspace.py`/`_atomic.py`). ruff, ruff
format, mypy, and the doc-consistency check all clean.

## What P1-7 changed

Rebuilt CI around the lane taxonomy from P1-1..P1-6, and audited the full "Test-refactor
acceptance criteria" list (`prompts/pre-release-plan.md` line 59) end to end -- 15 of 17
criteria fully hold; one required an actual fix (below); one is knowingly partial and carried
to P3-2, documented rather than silently dropped.

**Audit finding: `slow` and `network` markers were declared but empty.** Both had zero members
since P1-1 ("declared for later phases to use") -- a literal violation of "every declared marker
selects a meaningful, non-empty lane." The only genuinely network-dependent thing in the whole
tree was `docs/check_readme_snippets.py`, invoked directly by a CI step, never through pytest.
Fixed by adding `tests/benchmark/test_real_dataset_smoke.py`: fits `isolation_forest` on the
real, network-fetched `kddcup99_sa` and `covtype` datasets (capped to 20k rows via `max_rows`,
one seed) and asserts ROC-AUC beats random. Marked `benchmark`, `slow`, *and* `network` --
it's genuinely all three (fits a real detector, fetches over the network, and is too slow for
every-PR use), which is exactly the kind of test these two lanes exist for. Both lanes are now
non-empty (2 tests each) without inventing filler.

**Windows CI: knowingly not added, per a standing decision.** P1-7's text asks for "local
integration on Linux and Windows." A separate, earlier remediation effort already decided twice
(2026-09-08, recorded in project memory) to leave Windows out of the matrix -- it needs its own
phase for SQLite file-locking and path-separator behaviour, not a drive-by addition inside a
CI-policy phase. The new `integration` job stays Linux-only; the comment in `ci.yml` says why.
This is a disclosed, deliberate gap against the plan's literal wording, not an oversight.

**`.github/workflows/ci.yml` rebuilt around nine named jobs** (was five):
- `lint` -- unchanged (ruff, format, mypy, doc-check).
- `fast-tests` -- `-m "unit or contract"`, matrix `{ubuntu, macos} x {3.11, 3.12, 3.13}`. This is
  both a required PR job and the deterministic-compatibility-across-Python-versions run the
  plan asks for; they were the same matrix already, just now scoped to the right marker
  expression and named for what it actually checks.
- `integration` -- `-m integration`, Linux only (see above), required on every PR.
- `property` -- `-m property`, `HYPOTHESIS_PROFILE=ci`, required on every PR.
- `repo-check` -- `-m repo_check`, required on every PR. (Previously these 10 tests ran
  unnamed inside the old monolithic `test` job; they now have their own required lane, matching
  the plan's "repository/docs checks" bullet as a job of its own, distinct from `lint`'s direct
  script invocation.)
- `benchmark-smoke` -- `pytest tests/benchmark/test_accuracy_floors.py -m benchmark`, required
  on every PR. Narrow and fast on purpose: synthetic data, one seed, ~2.5s. This is the "one
  tolerance-banded benchmark smoke case that can catch a score-direction regression" the plan
  asks for; it already existed as `test_default_ensemble_beats_random` +
  `test_detector_clears_roc_auc_floor` (added when the `kmeans_distance`/CBLOF fix needed a
  regression guard), just never run on PRs before -- the full `benchmark` marker was
  schedule-only.
- `coverage` -- new. Runs `unit or contract or integration or property or repo_check` together
  with `--cov-report=xml`/`--cov-report=term` (informational; no `--cov-fail-under`), uploads to
  Codecov, then on `pull_request` events runs `uvx diff-cover coverage.xml
  --compare-branch=origin/<base> --fail-under=85` -- required, changed-lines-only. This replaces
  the old blanket `--cov-fail-under=70` gate (itself already a stopgap down from 80%, see P1-3)
  with exactly what the plan asks for: informational overall branch coverage, required diff
  coverage on changed lines.
- `build` -- unchanged (sdist/wheel, twine check, package-data assertion, clean-venv smoke).
- `docs-snippets` -- moved off `push`/`pull_request`; now `if: github.event_name == 'schedule'
  || github.event_name == 'workflow_dispatch'`. It downloads a real dataset via the README
  quickstart and was running on every push despite being the one job the plan explicitly wants
  kept "manual or nightly."
- `full-nightly` -- new, schedule/manual only. `-m "benchmark or slow or network or property"`
  with `HYPOTHESIS_PROFILE=nightly`, 30-minute budget. This is the "full accuracy, memory and
  slow tests on a frozen canonical environment on schedule" job; there is currently no
  meaningful separate "memory" signal to run (`peak_rss_mb` was removed as misleading in P0-7,
  see `tests/benchmark/test_benchmark_harness.py::test_run_benchmark_row_has_no_peak_rss_mb`) --
  a real one is P3-2 scope (measuring true peak process memory across a rebuilt harness), not
  invented here.
- Every job has an explicit `timeout-minutes` (5-30, sized to what was actually measured
  locally: fast-tests ~11s, integration ~36s, property ~6s, repo-check ~2s, benchmark-smoke
  ~3s, coverage ~2min) and inherits `--durations=15` from `addopts`, so slowest-test output and
  collected/passed/deselected counts land in every job's log -- satisfying "emit lane runtime,
  collected-test count and slowest tests" without extra tooling.

**Hypothesis profiles made explicit** (new `tests/factories/hypothesis_profiles.py`). All 16
`@settings(max_examples=N)` call sites across the six `tests/property/*.py` files previously
hardcoded a fixed count each; none of them actually varied with any profile, so "bounded for CI
vs larger for dev/nightly" wasn't real even though the plan asked for it explicitly. Replaced
every hardcoded `N` with `scaled_examples(N)`: a `HYPOTHESIS_PROFILE` env var (`ci` | `dev` |
`nightly`, default `dev`) selects a multiplier (1x / 3x / 10x) applied to each test's original
baseline, so per-test relative tuning (cheap tests still get more examples than ones that fit a
real detector) is preserved while the *scale* is now genuinely profile-driven. `deadline=None`
on all three registered profiles -- several of these tests fit real estimators inside an
example, and wall-clock deadlines were a flakiness risk waiting to happen, not a real signal.
CI sets `HYPOTHESIS_PROFILE=ci`; `full-nightly` sets `nightly`; local runs default to `dev`.
Verified property lane still passes at both `ci` (18 passed, 5.6s) and `dev` (18 passed, 10.1s).

**Full acceptance-criteria audit** (all 17, against the current tree):

| # | Criterion | Status |
|---|---|---|
| 1 | Plain `pytest` = fast/deterministic/network-free | Holds (574/784 collected, 210 deselected) |
| 2 | Every marker non-empty | Fixed this phase (`slow`/`network` were 0; now 2 each) |
| 3 | Local integration required on PRs | Fixed this phase (`integration` job, no `if` gate) |
| 4 | No tier/reload/coverage-only tests remain | Holds (grep clean outside this file) |
| 5 | No workspace/CLI/SQLite/serialization/report/benchmark-fit in `tests/unit/` | Holds (grep clean) |
| 6 | Shared factories replace duplicated builders | Holds (`tests/factories/`, from P1-2) |
| 7 | Every resource fixture uses `yield`; conftest has only used fixtures | Holds (root `conftest.py` re-checked) |
| 8 | One parameterised case per new detector | Holds (`DETECTOR_SPECS`, from P1-5) |
| 9 | Required regressions cannot dynamically skip | Holds (no `pytest.skip`/`mark.skip` in unit/contract/integration/property/repo_check) |
| 10 | CLI tests avoid `store._conn` | Holds (grep clean, from P1-5) |
| 11 | Private estimator state only in named adapter contracts | Holds (only hit outside `tests/contract/` was `tree_shap_attributions`, a false positive on `.tree_`) |
| 12 | Every P0 fix has a regression | Holds for the runtime fixes P0-1..P0-7 (grep-matched by keyword per item); P0-8/9/10 are docs/CI/metadata, not runtime behaviour, so a pytest regression doesn't apply to them |
| 13 | Default config exercises `combination="intersection"` | Holds (`tests/factories/configs.py`, from P1-2) |
| 14 | PR benchmark smoke detects inverted direction; floors include varying-density + tolerance bands | **Partial.** Smoke fixed this phase (`benchmark-smoke` job). A varying-density synthetic dataset does not exist yet -- `test_accuracy_floors.py` already documents excluding LOF from its floors for exactly this reason. Building one is P3-2 scope ("varying-density regimes") and is deliberately not done here, same disposition as the two P1-6 deferrals |
| 15 | Explicit bounded-CI / larger-dev/nightly Hypothesis profiles | Fixed this phase (`hypothesis_profiles.py`) |
| 16 | Informational overall branch coverage; required >=85% diff coverage | Fixed this phase (`coverage` job) |
| 17 | Required lanes report runtime budget + slowest tests | Fixed this phase (`timeout-minutes` + `--durations=15` per job) |

Final collected count: **784** (782 + 2, the new real-dataset smoke test). Lane counts: unit
422, contract 152, integration 158, property 18, slow 2, benchmark 24, network 2, repo_check 10.
Plain `pytest` (unit + contract): 574 passed. `-m integration`: 158 passed in ~36s. `-m property`
at the `ci` profile: 18 passed in 5.6s. ruff, ruff format, mypy, and the doc-consistency check
all clean.
