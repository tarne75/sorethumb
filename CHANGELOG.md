# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-21

First public release. The Fixed / Compatibility entries record corrections
made during pre-release hardening; there is no prior published version to
diff against.

### Added

- Detector `extra_params`: an escape hatch to pass arbitrary constructor kwargs
  straight to the underlying sklearn estimator (`n_jobs`, `max_features`, KMeans
  `tol`/`max_iter`/`algorithm`, LOF `leaf_size`/`metric`/`p`, OCSVM `tol`/
  `shrinking`, …). Set it in config as
  `params = { extra_params = { n_jobs = 4 } }`. Keys are validated at detector
  construction against the estimator's real parameter list (a typo fails there,
  not deep in sklearn); reserved keys (`random_state`, `contamination`,
  `novelty`, `n_clusters`) and keys already exposed as wrapper arguments are
  rejected with `ConfigError`. `get_params()` now includes an `extra_params`
  entry on every detector. `ecod`/`hbos` have no underlying estimator and reject
  any non-empty `extra_params`. `sorethumb init` writes every accepted key
  (with its scikit-learn default) into the starter config, commented out, and
  the [configuration reference](docs/configuration.md#detector-extra_params)
  lists them per detector. Each wrapper class exposes
  `available_extra_params()` for programmatic discovery.
- Accuracy-floor benchmark (`tests/benchmark/test_accuracy_floors.py`, marker
  `benchmark`): each guarded detector must clear a committed per-detector
  ROC-AUC floor on the network-free synthetic datasets. Restores a real
  accuracy-regression signal — the scheduled `benchmark` CI job previously
  selected zero tests — and locks in the `kmeans_distance` CBLOF fix.
- `KMeansDetector.large_cluster_coverage` (constructor argument, default `0.90`)
  tunes the CBLOF reference set — the fraction of training rows the "large"
  clusters must cover. Surfaced in `get_params()` alongside the resolved
  `n_large_clusters`; the fitted reference centroids are exposed as the
  `large_centroids` property.
- Packaging metadata: PyPI trove classifiers, keywords, and `[project.urls]`
  (Homepage, Repository, Documentation, Changelog, Issues). `py.typed` ships in
  the wheel.
- Release workflow (`.github/workflows/publish.yml`): a `vMAJOR.MINOR.PATCH`
  tag builds the sdist + wheel, runs `twine check`, verifies the tag matches
  `project.version`, and publishes to PyPI via trusted publishing (OIDC — no
  API token). Needs a one-time PyPI trusted-publisher entry and a `pypi`
  deployment environment.
- Store durability hardening. `PRAGMA busy_timeout` (30s) is now set on every
  connection, so a second process writing to the same workspace waits for the
  lock instead of failing instantly with "database is locked". Each migration
  runs under an explicit `BEGIN IMMEDIATE`/`COMMIT` (SQLite only auto-wraps
  DML, not DDL, so a bare `with conn:` around `CREATE`/`ALTER`/`DROP` was
  never really transactional); a checksum of each bundled migration file is
  recorded and verified on replay, and opening a workspace whose schema is
  newer than the installed sorethumb understands now fails closed rather than
  limping on with a stale schema. Bundled migrations 001-005 are all
  idempotent (`IF NOT EXISTS` / `ADD COLUMN` de-duplication), so a crash
  mid-migration can be safely retried. HTML/JSON reports and CSV exports are
  now written atomically (shared `sorethumb._atomic` module, previously only
  used for model artifacts), so a reader can never open a half-written file.
- Native, exact per-feature attributions for ECOD and HBOS. Both detectors
  already define their score as an unweighted average of independent
  per-feature terms (an empirical-CDF tail probability for ECOD, a histogram
  bin log-density for HBOS); `feature_contributions()` on each detector
  returns those terms directly (`explain/native.py`), so summing the
  attribution recovers the score with zero error — a new `attribution_kind`
  value, `"exact"`, reflects that this is a stronger guarantee than
  TreeSHAP's `"model_specific"` (additivity unverified). `blend()`'s tier
  logic is now a proper ranking (`exact` > `model_specific` > `heuristic`):
  the blended tag is the weakest of the contributing sources', not a binary
  all-or-nothing check.
- Realised per-detector flag rates surfaced. `ScoreEnsemble.combine()` returns a
  new `per_detector_rates` (the fraction each detector's own heuristic boundary
  flagged, for every input detector); `GroupSummary` carries it as
  `detector_flag_rates` (plus `dropped_detectors`), the run summary prints it,
  and `sorethumb run --json` reports it per group. `contamination="auto"` is the
  median of these — showing them stops the flagged count being read as "how many
  anomalies you have".
- Ensemble bad-member guard. Before weighting or combining, `ScoreEnsemble`
  drops any detector whose score *ranking* is anti-correlated with the ensemble
  median (Spearman's rho < −0.15) — a member fighting the consensus rather than
  adding a diverse-but-consistent view. Previously nothing measured any detector
  against the others, so such a member voted (and, in `intersection`, could veto
  every anomaly) with full weight. Runs for every weighting / combination, needs
  ≥ 3 members, and won't drop more than `k − 2` (a run where most members look
  bad means the consensus itself is unreliable). Dropped detectors are listed in
  the `combine()` result's new `dropped_members` and appear in `weights` with
  0.0; their per-detector score columns are still recorded.
- `source.dataset_id`: a stable logical identity for a dataset, held constant
  across snapshots. All history (periods, per-group totals, runs) is now keyed on
  it. When unset it is derived from `source.uri` (`<file stem>-<12 hex of the
  path>`, query strings ignored). New `io.fingerprint.logical_dataset_id` /
  `snapshot_fingerprint`; `RunResult` / `sorethumb run --json` gain `snapshot_fp`.
- `dataset_snapshot` table (migration 005): one row per observed content+schema
  version of a dataset, with `first_seen` / `last_seen`. `dataset.snapshot_fp`
  points at the most recent. `Store.dataset_snapshots(dataset_fp)` lists them.
- `sorethumb report [run_id]` is implemented — it previously created an empty
  directory and rendered nothing. It rebuilds `reports/<run_id>/index.html` and
  the per-group CSV siblings purely from persisted state (run row, FeaturePlan,
  per-group results Parquet); defaults to the latest run. New public API
  `sorethumb.render_report_for_run(ws, run_id)`; `Store.get_run_group` and
  `store.results.results_path` back it.
- CI doc-consistency gates. `docs/generate_config_docs.py --check` now also fails
  on prose docs that reference an unknown `section.field`, state a config default
  that no longer matches the schema, or contain a broken relative Markdown link.
  New `docs/check_readme_snippets.py` (run by a `docs-snippets` CI job) executes
  every fenced README block: `python` blocks run in a subprocess, `toml` config
  fragments are validated against `Config`, and each `sorethumb <cmd>` / `--flag`
  is checked against the live CLI.
- `sorethumb score --from-run RUN_ID` is now real (it previously ignored
  `--from-run` and did a full fitted run). It loads the source run's persisted
  FeaturePlan and, per group, its per-detector models and calibrators; applies
  the plan to the new data (`apply_feature_plan`); scores each unpickled
  detector without re-fitting; reuses the persisted calibrators so scores map
  onto the source run's reference distribution; and writes a new, distinct
  `score_…` run recording `source_run_id` (migration 004). Schema drift and
  library-version drift are detected per group (`--strict` → errors). New
  public API `sorethumb.score_forward(config, source_run_id)`; the fitted plan
  is persisted at `models/<run_id>/plan.json` (`save_plan` / `load_plan`).
- Full-pipeline benchmark harness (`evaluate/pipeline_benchmark.py`,
  `sorethumb benchmark`): a named taxonomy of synthetic anomaly types
  (point, local, contextual, clustered, masking, swamping, varying-density)
  with mixed numeric and categorical columns through the real feature
  pipeline (encoding, scaling, optional PCA, detectors, calibration,
  ensemble), never training and evaluating on the same rows. Every
  (scenario, ablation) cell runs over several seeds and reports mean ± a 95%
  confidence interval; `peak_memory_mb` is a real OS-tracked high-water mark
  from an isolated subprocess per cell. `sklearn:*` rows compare against bare
  sklearn detectors with no sorethumb pipeline. `sorethumb benchmark` now runs
  this suite alongside the legacy real-dataset one (each independently
  disable-able with `--no-pipeline` / `--no-legacy`) and injects both result
  tables into the README from one documented command.

### Changed

- TreeSHAP's `attribution_kind` is `model_specific`, not `exact`. TreeSHAP runs
  with `check_additivity=False` (IsolationForest's `score_samples` isn't the
  strict SHAP sum of base value + contributions, so additivity checking would
  raise on every call) — so additivity is unverified at runtime, and even
  where the decomposition holds exactly it's exact for *path length*, which
  IsolationForest's score is a nonlinear transform of. Neither supports
  calling the result `exact`. `blend()`'s tier logic is renamed to match
  (`model_specific` iff every contributing source is `model_specific`, else
  `heuristic`); nothing sorethumb computes is tagged `exact` any more. See
  `docs/explanations.md` and `docs/approximations.md`.
- Persisted models now record the fit-time versions of Python, sorethumb,
  scikit-learn, numpy, scipy and joblib in `manifest.json`. `load_model` and
  `score_with_existing` compare them against the current environment and emit
  `ModelVersionMismatchWarning` (or, in strict mode, raise
  `ModelVersionMismatchError`) so a dependency upgrade can no longer change
  scores silently.
- CI adds a `build` job: builds the distribution, `twine check`s it, asserts
  the package data (`py.typed`, SQL migrations) is shipped, then installs the
  wheel into a clean virtualenv and smoke-tests `sorethumb --version` and
  `import sorethumb`.
- CI declares a weekly `schedule` trigger so the `benchmark` job (guarded by
  `if: github.event_name == 'schedule'`) can actually run — restoring the
  accuracy-regression signal.
- README marked pre-release (0.1.0); install instructions build from a clone.
- Statistical contract sharpened. The README, `docs/adapting-to-your-data.md`,
  `docs/configuration-examples.md`, `docs/approximations.md`, the
  `scoring.contamination` field description, the run summary and `--json` output
  no longer present `contamination` / `contamination="auto"` as a prevalence
  estimate. `contamination` is a **review budget** (the size of the shortlist you
  will look at); `"auto"` is a heuristic median of per-detector cut-offs that
  disagree. The CLI summary line "Total anomalies: N" is now "Flagged for review:
  N (X% of M rows) — … not an estimate of true prevalence"; `--json` groups gain
  `n_flagged` alongside the retained `n_anomalies`.
- `_pipeline`: the per-group ledger/status bookkeeping and the
  ensemble→threshold→explain→write tail are factored into shared helpers
  (`_execute_group`, `_finalize_group`) used by both `run_detection` and
  `score_forward`, so the two paths cannot diverge.
- `run_id` derivation now folds in the snapshot fingerprint, so a changed source
  snapshot produces a fresh run (no resume against stale results) while period /
  totals history keys on the stable `dataset_id` alone. `config_hash` excludes
  `source.dataset_id` (an organisational label, not a result-affecting param).
- `Calibrator` is self-calibration only. The `mode="reference"` path
  (`__init__(mode=...)`, `fit(reference_scores=...)`) was unreachable —
  `run_detection` hardcoded `mode="self"` and `ScoringConfig` had no field to
  select otherwise — so it is removed. `to_dict` no longer emits `"mode"`;
  `from_dict` ignores it in payloads written by older versions.
- README differentiators #2 / #4 and the `sorethumb backfill` help + docs
  corrected: `backfill` fits and self-calibrates each period independently, so a
  `sorethumb history` trend reflects *relative* period-to-period movement, not an
  absolute anomaly level on one scale. `sorethumb score --from-run RUN_ID` stays
  the single-scale cross-run path. (No prior release claimed otherwise in a
  shipped changelog.)
- Dependency and lockfile reproducibility: `uv.lock` is now enforced (`uv lock
  --check` + `uv sync --frozen`) in every CI/publish job; `setup-uv` is pinned
  to an exact commit and Hatchling is constrained. shap, numba, matplotlib,
  pandas and datasets moved out of the mandatory install into `explain` /
  `report` / `benchmark` / `dev` optional-dependency extras; TreeSHAP/KernelSHAP
  degrade to the gradient method with a clear warning when `explain` isn't
  installed, rather than raising.
- Zero-config default workspace changed from `.` to a dedicated
  `./sorethumb-workspace/` directory, so a first `sorethumb run <file>` never
  scatters `sorethumb.db`/`models`/`results`/`reports`/`logs` beside the source
  data or loose in the current directory. A pre-existing workspace at `.`
  (the old default) is detected and refused rather than silently orphaned —
  pass `--workdir .` explicitly to keep using it. `sorethumb init` now creates
  this same directory (previously a mismatched, hidden `.sorethumb_workspace`)
  and fills in the starter config's `run.workdir` to match it.
- The PyPI distribution name is `sorethumb-ml`, not `sorethumb` — the
  latter is owned by an unrelated package. Only the published/installed
  name changes (`pip install sorethumb-ml`); the import package
  (`import sorethumb`), the CLI command (`sorethumb run` etc.), the GitHub
  repo, and every on-disk convention (`sorethumb.toml`,
  `sorethumb-workspace/`, `sorethumb.db`) are unaffected, the same way
  `pip install beautifulsoup4` gives you `import bs4`. Publication also
  moved from OIDC trusted publishing to a PyPI API token
  (`secrets.PYPI_TOKEN`), since trusted publishing needs the PyPI project
  to already exist to configure a trusted publisher against it — impossible
  for this project's first-ever publish; the `pypi` GitHub Environment now
  also requires a manual reviewer approval before any publish reaches PyPI.

### Removed

- Dropped the unsupported `s3://` example from the README; only local paths and
  `http(s)://` URLs are accepted (an `s3://` URI already raised
  `SourceError: Unsupported URI scheme`).
- `history/totals.py`'s `compute_totals` (and `ledger.py`'s
  `periods_missing_groups`, `db.py`'s `groups_seen_for_dataset`): all dead
  code left over from the history redesign, never called from any production
  path. The real, authoritative completion path is
  `_pipeline._record_period_history` / `Store.record_period_completion`,
  built directly from each group's `GroupSummary`. Removing `compute_totals`
  also removed the now-unreachable `PopulationMismatchWarning`.

### Fixed

- `sorethumb run --detectors` silently rewrote `sorethumb.toml`. When an
  existing config file was overridden with `--detectors`/`-d`, `run()` called
  `_write_minimal_toml`, which regenerates a full starter TOML from the
  live `Config` object and writes only `source.uri`, `run.workdir` and
  `detectors` back in — every other section (`columns`, `profiling`,
  `features`, `scoring`, `explain`, `history`, `report`, comments, and any
  nested `params`/`extra_params`) was silently discarded and replaced with
  defaults. `--detectors` now only replaces `cfg.detectors` for the current
  invocation in memory; it never writes to the config file. The one
  remaining auto-write path (prompting to save a starter config when `run`
  is invoked with a data-file argument and no `sorethumb.toml` exists yet)
  is unaffected, since there is no existing file for it to destroy.
- `kmeans_distance` no longer lets a tight anomaly cluster capture its own
  centroid and be scored normal — the previous behaviour ranked planted
  anomalies as the *most* normal records (benchmark ROC-AUC ~0.0002). Scoring
  is now CBLOF-style: every row is measured against the nearest *large-cluster*
  centroid, where "large" is the set of clusters that together cover
  `large_cluster_coverage` (default `0.90`) of the training data; small
  clusters — anomaly sub-groups included — are excluded from the reference set.
  Consequence of the 0.90 default: the detector assumes anomalies are ≲ 10 % of
  the data. Lower `large_cluster_coverage` for a higher true anomaly rate;
  raise it toward `1.0` to be stricter.
- Artefact pruning no longer matches a failed run to its files by a path
  substring (`instr(path, run_id)`), which could delete another run's files
  when one `run_id` was a substring of another. Artefacts now carry their
  owning `run_id` (migration 003) and the prune query joins on equality.
- The migration runner now strips full-line `--` comments before splitting on
  `;`, so prose containing a semicolon in a migration comment can no longer
  truncate the following statement.
- Scaler no longer clamps a genuine sub-1.0 spread up to `1.0`, which flattened
  fine-grained columns to near-zero variance. A spread is now used as-is; only
  an effectively-zero spread (constant or degenerate column) falls back to
  `1.0`.
- Standard-mode scaling fits mean and std over each column's 1st–99th percentile
  range, so a handful of anomalous rows can no longer inflate the centre or the
  spread used to standardise the normal bulk. (Robust mode already used
  outlier-resistant median/IQR.)
- Score-forward artifact loading did not fail closed. `sorethumb score
  --from-run` accepted a source run of any status (`running`, `failed`, or
  even another score-forward run) as long as its row existed;
  `store.models.load_model` defaulted a missing manifest to `{}` and a
  missing calibrator file to a freshly-constructed, unfitted `Calibrator()`
  instead of raising, and fell back to pre-namespacing shared filenames
  (`manifest.json`/`calibrator.json`) left over from before per-detector
  namespacing. None of it verified that the manifest actually described the
  file sitting next to it, or that the file hadn't been corrupted or swapped
  since it was written. `score_forward` now rejects a source run whose
  status isn't `complete` and rejects a source run that is itself a
  score-forward run (one never persists its own models, so there would be
  nothing trustworthy to load). `save_model` records a SHA-256 digest of the
  estimator and calibrator files it just wrote (manifest `file_digests`);
  `load_model` now requires the manifest and calibrator file to exist (never
  defaults or falls back to a legacy filename — this is the only supported
  on-disk format for the first release), verifies the manifest's
  `run_id`/`group_key`/`detector_name` match what was requested, verifies an
  optional caller-supplied `plan_digest` against the manifest's (rejecting a
  model fitted against a different `FeaturePlan`), and re-hashes the
  estimator/calibrator files before `joblib.load`/parsing them, comparing
  against the recorded digests. Any of these identity or digest checks
  failing raises a new `ModelIntegrityError` unconditionally (not gated on
  `run.strict` — there is no safe degraded behaviour for a tampered or
  corrupted pickle); `score_with_existing` lets it propagate instead of
  treating it the same as a detector that was simply never fitted (that
  case, a genuinely absent model, remains a lenient per-detector skip, not a
  failure).
- Null group values were silently dropped instead of processed, and a
  fallback `row_id` (used whenever no `id_column` is configured) collided
  across groups. `_slice_group_frame` filtered a group by casting the column
  to `Utf8` and comparing it to a pre-stringified value; casting a null to
  `Utf8` stays null, so `== "None"` never matched a genuine null group and
  every one of its rows was silently excluded from every group's slice. The
  same premature `str(...)` also meant a null value, the empty string, and
  the literal string `"None"` all collapsed to the same group key. Both
  `run_detection` and `score_forward` now keep each group column's *typed*
  value all the way through (`group_values: dict[str, Any]`); `_slice_group_frame`
  matches `None` with `.is_null()` and compares every other value against its
  own dtype instead of a string cast, and `make_group_key`
  (`store/workspace.py`) hashes the typed JSON encoding directly — `null`,
  `""`, and `"None"` are distinct tokens there, so the three group identities
  can no longer collide. Separately, the fallback `row_id` (`FeatureSpace.row_ids`,
  used when no `id_column` is configured) was `np.arange(len(df))` recomputed
  fresh inside *each* group's own feature space — every group's flagged rows
  started back at `row_id=0`, so two different groups' anomalies could carry
  the same `row_id` and joining results back to the source frame was
  ambiguous. The raw source frame is now stamped with one stable global row
  index (`_stamp_source_row_id`, an internal reserved column excluded from
  profiling/features via `INTERNAL_ROW_ID_COLUMN`) before any period filter,
  group filter, or time sort in both pipelines; `fit_features` /
  `apply_feature_plan` read it back as `row_ids` instead of a fresh
  positional range, so it survives every later slice and reorder and stays
  unique and joinable across groups. `schema_fingerprint` ignores this
  reserved column so it never counts as schema drift for dataset identity or
  score-forward's plan-apply check. An explicit `id_column` remains
  authoritative whenever configured.
- `combination="intersection"`/`"union"` did not actually require every
  configured detector's vote. The bad-member guard in `ScoreEnsemble.combine`
  (which drops a detector whose score ranking is anti-correlated with the
  ensemble median) ran before the combination strategy was even checked, so a
  configured three-way intersection could silently become a two-way one — the
  decision rule changed without any visible config change. The guard now only
  drops members for `combination="composite"`, where removing an outlier from
  a weighted average doesn't change the decision rule; for
  `"intersection"`/`"union"` every configured vote is kept and a new
  `AntiCorrelatedMemberWarning` is raised instead (promoted to an error under
  `run.strict`, via the existing `SorethumbWarning` mechanism).
- `rank` could be assigned to a row `combination="intersection"`/`"union"`
  never actually flagged, and a genuinely flagged row could be left at rank 0.
  `_finalize_group` derived `rank` from a fresh global sort of
  `composite_score` and ranked the top `n_anomalies` rows by that score —
  but for the set-operation combinations, `anomaly_flag` comes from an
  independent per-detector threshold vote, not from `composite_score`
  (`min`/`max` across detectors), so the globally highest-scoring rows are
  not guaranteed to be the flagged ones. `rank` is now derived from the same
  flagged-only, score-ordered array already used for attribution ordering
  (`_flagged_idx_by_score_desc`), so `rank > 0` if and only if a row is
  flagged, always.
- History completion was neither atomic nor scoped to a configuration.
  `totals` rows preserved `config_hash` (migration 002), but every reader —
  `last_complete_period_label`, `completed_group_keys`, `groups_seen_for_dataset`,
  `totals_for_periods`, `periods_missing_groups`, `iter_pending_periods`,
  `compute_rolling_windows` — ignored it, so two configurations processing the
  same `period_label` could make each other look done, or have their totals
  silently summed together in a rolling-window trend. Separately, "is this
  period done?" was answered by "does the totals table have any row for it?",
  so a period where only 1 of 3 groups succeeded (2 failed) still read as
  fully complete and was never retried. A new `period_execution` table
  (migration 006), keyed on `(dataset_fp, period_label, config_hash)`, is now
  written atomically alongside that attempt's totals rows
  (`Store.record_period_completion`, one `BEGIN IMMEDIATE`/`COMMIT`) and is
  the sole source of truth for completion: `complete=1` only when every group
  discovered in that run reached a non-failed terminal status. Every history
  reader now takes an explicit `config_hash` (no default), and `sorethumb
  backfill` / `sorethumb history` display the config hash they're scoped to.
  `_record_period_history` also now writes totals for resumed (`skipped`)
  groups, not just `success`/`too_few_records` — needed so a crash between a
  group being marked complete in the run ledger and history ever being
  recorded doesn't permanently lose that group's contribution; the next
  retry now records it. The dead `calibration_modes_for_periods` lookup
  (diffed a `calibration_mode` config field removed when self-calibration
  became the only mode, so it always read as "no break") is replaced by
  `other_config_hashes_for_periods`: `WindowResult.calibration_break` now
  fires when a window's span also has totals recorded under a *different*
  configuration — real provenance that the trend may be an incomplete
  picture, instead of a check that could never trip. (`groups_seen_for_dataset`
  and `periods_missing_groups` were themselves later removed as dead code —
  see Removed — once the redesign settled and backfill's real pending-period
  path never ended up calling them.)
- A group that failed without raising (e.g. "no detector produced scores",
  or a score-forward group whose requested detector was never persisted in
  the source run) was written to the `run_group` ledger as `status='complete'`
  regardless of the `GroupSummary.status` the group body actually returned.
  A later call with the same inputs (same deterministic `run_id`) then saw
  that group as already done, skipped re-executing it, and could report the
  run as complete without the group ever having genuinely succeeded.
  `_execute_group` now persists `failed` when the body reports `status="failed"`
  (so a retry re-executes it), and `complete` for both `success` and
  `too_few_records` (both genuine terminal outcomes).
- `evaluate_scores` returned `0.0` for ROC-AUC/AP when a population has only
  one class present (both are mathematically undefined there). `0.0` reads as
  a real, terrible score — indistinguishable from a model that actively
  anti-ranks — and silently drags down any mean/std computed over it (as the
  benchmark harness's multi-seed aggregation does). Both now return
  `float("nan")`, which correctly propagates through aggregation instead of
  masquerading as data.
- Benchmark harness leaked the labels into their own evaluation. `run_benchmark`
  set the precision@k / recall@k / F1 operating point to
  `contamination = y.mean()` (the true label rate), which makes
  `k = round(n_total * contamination)` equal `n_positives` exactly — precision@k
  and recall@k then share the same numerator *and* denominator, so all three
  metrics reduce to one number (`n_true_at_k / n_positives`), which is exactly
  why every README benchmark row showed identical values in those three
  columns. Fixed: the harness now evaluates every dataset at one fixed 5%
  review budget (`_REVIEW_BUDGET` in `evaluate/benchmark.py`), chosen
  independently of any dataset's true rate. `evaluate_scores` also warns at
  runtime if it's ever called with a `contamination` that makes `k ==
  n_positives`, so this can't silently regress. Each (dataset, detector) pair
  is now run over `n_seeds` seeds (`BenchmarkConfig.n_seeds`, `sorethumb
  benchmark --seeds`, default 5) and reported as mean ± standard deviation,
  not a single draw; `BenchmarkRow` gained a `*_std` field per metric plus
  `n_seeds`.
- ECOD and HBOS explanations went through finite-difference gradients
  (`explain/gradient.py`), whose default 1%-of-std perturbation routinely
  produced an exact all-zero (or noisy) attribution vector for precisely the
  rows that matter most. Both detectors' scores are a discrete rank/bin
  lookup (empirical-CDF tail rank for ECOD, histogram bin for HBOS) with no
  sub-resolution structure: a genuinely anomalous, far-tail row's small
  perturbation routinely lands in the exact same rank or bin as the
  unperturbed row. `_compute_attributions` now dispatches ECOD/HBOS to their
  own exact decomposition instead (see Added); finite-difference gradients
  are restricted to OneClassSVM and LOF, the two detectors whose score
  responds continuously to a small perturbation, and an unrecognised
  detector type is skipped (logged) rather than defaulting to gradient.
- PCA back-projection could mislabel or fabricate reasons when PCA is on
  (`features.pca=true`). `back_project_pca` was called with
  `n_features=len(plan.output_features)` — a column count/identity fixed
  *before* width-control demotion and correlation-drop, not the width
  `plan.pca_components`' loadings were actually fit on. A mismatch raised
  (caught) or, when the counts happened to coincide, silently mapped
  attributions to the wrong original columns; either way the failure was
  logged at DEBUG and the fallback attributed in raw PCA-component space
  (`pc_6="high"` is not a reason a user can act on). Separately,
  `plan.output_features` / `derived_to_original` were never recomputed after
  demotion, so the plan misdescribed its own matrix (still listing one-hot
  dummies for a column that now emits one frequency feature) independent of
  PCA. Fixes: `FeaturePlan` gains `pre_pca_feature_names`, the exact
  column list snapshotted immediately before the PCA step (after demotion and
  correlation reduction) — back-projection now uses that, not
  `output_features`. `fit_features` recomputes `output_features` /
  `derived_to_original` after demotion so both describe the real matrix. A
  back-projection failure (bad shape, or any other exception) now logs at
  WARNING and marks every flagged row in the group `attribution_kind
  ="unavailable"` with `reason_1="unavailable (PCA back-projection failed)"`
  — a failure here isn't per-row, so nothing in the group gets a trustworthy
  original-column attribution.
- Explanations beyond `explain.max_rows` were fabricated, not omitted.
  `gradient_attributions` / `kernel_shap_attributions` (used for every detector
  without a native attributor — OneClassSVM, ECOD, LOF, HBOS) silently truncate
  to `explain.max_rows`; `_pipeline.py` zero-padded every row past the cap back
  to full length, and `top_n_reasons`' descending sort, given an all-zero
  vector, returned the first `top_n` columns in dict-insertion order —
  indistinguishable from a real (if weak) reason. Two fixes: flagged rows are
  now attributed in `composite_score`-descending order, so a truncated run
  explains the most anomalous rows first and only the least anomalous flagged
  rows are left uncovered; and `_compute_attributions` now tracks, per row,
  whether *any* source actually computed a value for it. A row with zero
  covering sources gets `attribution_kind="unavailable"` and
  `reason_1="unavailable (beyond explain.max_rows)"` instead of a fabricated
  `column=value` reason.
- shap is not installed unless the `explain` extra is requested: TreeSHAP and
  KernelSHAP now catch the resulting `ImportError` and fall back to the
  gradient method with a `FallbackAttributionWarning`, instead of the run
  failing.
- The apply path was unguarded against schema drift. `apply_feature_plan` never
  checked `plan.schema_fingerprint` against the incoming data, and `apply_scaler`
  silently passed through any column with no fitted scale parameters. Together,
  a drifted input (a column renamed, removed, or retyped since the plan was
  fitted) could reach a distance-based detector with one column left on its raw
  scale — dominating the matrix — with no error anywhere.
  `apply_feature_plan` now raises `PlanError` immediately if the input's
  `schema_fingerprint` doesn't match the plan's (same schema, different values —
  the real score-forward case — is unaffected). `apply_scaler` now raises
  `PlanError` if a column it's asked to scale has no fitted parameters, instead
  of leaving it unscaled; `apply_feature_plan` was adjusted to drop
  correlation-reduced columns *before* scaling (mirroring `fit_features`) so a
  column removed by `correlation_reduction` — which never needed a scale
  parameter in the first place — doesn't trip the new check.
- `scoring.weighting = "agreement"` was broken. It compared each detector's
  `natural_flag` to the row-wise majority vote; at realistic contamination
  almost no row is flagged, so the "majority" is all-normal, every detector
  scores ~0.98, weights collapse to equal — and a detector that flags *nothing*
  scores 1.0 and gets the *largest* weight. It now weights each detector by the
  leave-one-out Spearman rank correlation between its calibrated scores and the
  consensus ranking of the other detectors; anti-correlated or constant-score
  detectors get weight 0, and it falls back to equal when nothing correlates.
  `ScoreEnsemble` also now logs a warning when a non-`equal` weighting is set
  with a non-`composite` combination (`intersection` / `union` ignore weights).
- `kmeans_distance` (CBLOF) scoring OOM'd on wide/large inputs. `score_samples`
  and `centroid_attributions` built the `(n, n_large, d)` difference tensor
  (`X[:, None, :] - centres[None, :, :]`) — ~2 GB at 1M rows × 5 centroids × 50
  features, before taking the norm. Both now call a shared
  `_nearest_large_centroid` helper backed by
  `sklearn.metrics.pairwise.euclidean_distances`, which forms only the
  `(n, n_large)` distance matrix. Numerically identical (argmin unchanged,
  distances match to float round-off); benchmark accuracy floors are unmoved.
- `detectors[*].train_row_cap` did nothing — the per-group fit loop always used
  the detector class's `default_train_row_cap` and never read the config field,
  so the Scale guide's "set `train_row_cap`" advice had no effect. The loop now
  uses the configured cap when set, falling back to the class default. The three
  contradictory OneClassSVM caps (config `_default_detectors()` said 20000, the
  class and every doc said 25000) are reconciled onto the class value as the
  single source of truth: `_default_detectors()` and the `sorethumb init` starter
  no longer hard-code the numbers, and `train_row_cap` is now purely an override.
  Its docstring drops the wrong "None means the full training set" (it never did).
- Documentation reconciled against the code before a fresh clone reads it:
  `docs/configuration-examples.md` called `composite` the default combiner (it is
  `intersection`); `docs/adapting-to-your-data.md` gave `profiling.null_ratio_flag`
  a default of `0.30` (it is `0.0`) and called `MemoryBudgetError` a "memory
  warning"; the README described `run.max_memory_mb` as purely advisory when a
  run whose *projected* feature-matrix size exceeds it aborts with
  `MemoryBudgetError` before any fit; `docs/index.md` linked a non-existent
  `architecture.md` and omitted `configuration-examples.md`; and the README
  60-second quickstart forced KDDCup99's categorical byte columns into an
  all-`Float64` Polars schema (`ComputeError`) — it now uses `as_frame=True` +
  `pl.from_pandas`, decodes the bytes columns, and takes 20k rows so the run
  finishes in about a minute.
- `sorethumb run --dry-run` claimed "nothing will be written". It in fact opens
  the workspace (applying migrations), registers the dataset (`dataset` +
  `dataset_snapshot` rows) and the run (`run` row, left in status `running`) —
  it just fits no models and writes no results, history or report. The CLI
  message, `--dry-run` help, `run_detection` docstring and `docs/cli_reference.md`
  now say exactly that.
- `docs/check_readme_snippets.py` verified README `--flag`s by parsing rendered
  `--help` text, which `rich` truncates at narrow terminal widths — so the
  `docs-snippets` job and `test_docs_checks` failed in CI (80-col) while passing
  locally. It now introspects the click command tree directly.
- Eleven documented config fields were inert — now wired, one behavioural test
  each (`tests/unit/test_config_wiring.py`):
  `run.max_rows` (deterministic head-truncation of the input + `SampleTruncatedWarning`),
  `run.reuse_models` (score with a persisted model for the group+detector instead
  of refitting; also dropped from `config_hash` as an execution-only knob),
  `run.strict` (promotes every `SorethumbWarning` — including in-group ones — to a
  failure via `warnings.filterwarnings("error", …)`),
  `run.slow_stage_seconds` (`SlowStageWarning` around the load and feature-fit
  stages and per group, not just the OCSVM fit),
  `explain.enabled` (skips attribution entirely),
  `explain.kernel_shap` (routes non-tree detectors to KernelSHAP),
  `explain.permutation_importance` (runs an extra per-detector cross-check),
  `report.formats` (selects `html` / `csv` / `json`; new `index.json` writer),
  `report.open_after` (opens the report via `webbrowser`),
  `report.rolling_windows` (default windows for `sorethumb history`),
  `source.cache` (`false` returns a single overwritten `uncached_data.*` and
  never writes a fingerprint-keyed cache dir).
- Model persistence: a group's `calibrator.json` and `manifest.json` were not
  namespaced by detector, so in a multi-detector group the second `save_model`
  overwrote the first's calibrator and manifest. `load_model` /
  `score_with_existing` then returned the wrong detector's calibrator (garbage
  calibrated scores). Files are now `<detector>.calibrator.json` /
  `<detector>.manifest.json`; `load_model` falls back to the old names for
  workspaces written by an earlier version. The bug was latent because the
  fit path uses in-memory calibrators and single-detector runs are unaffected.
- Model-side writes (`.joblib`, `.calibrator.json`, `.manifest.json`,
  `plan.json`) are now atomic — a sibling temp file, `fsync`, then
  `os.replace` — so a crash mid-write can no longer leave a half-written file
  for a later score-forward run to load.
- Historical period selection: a `period_label` override (every
  `sorethumb backfill` label) is now resolved to concrete
  `[period_from, period_to)` bounds and the dataset is filtered to that window.
  Previously the filter ran only for the auto-resolved current period, so every
  backfilled label processed the *entire* dataset. The window bounds are
  coerced to the time column's dtype (Date / Datetime / tz-aware Datetime /
  Utf8), which also fixes the auto path for real temporal columns — Polars will
  not compare a temporal column to a string. New `history.periods.period_bounds`.
- History ledger: `run_detection` now writes the `period` row and one `totals`
  row per processed group (population + anomaly count, built from the
  `GroupSummary` objects, not the flagged-only results frame). Nothing was
  writing these, so `sorethumb backfill` re-queued every period on each run and
  `sorethumb history` had nothing to aggregate. `too_few_records` groups are
  recorded with a zero count so they stop re-queueing; `skipped` / `failed`
  groups are left untouched. `score_forward` deliberately does not write history.
- Backfill was leaking weekend periods into a range even with
  `history.roll_non_business = true` configured: `resolve_backfill_range`'s
  three branches walked labels via calendar-day arithmetic
  (`period_range`/`step_back`/`step_forward`), none of which have any weekday
  awareness, so a window whose *span* crossed a weekend still produced
  Saturday/Sunday labels. A new `filter_non_business` is applied once,
  uniformly, after all three branches build their range.
- `sorethumb backfill` now creates the workspace if it does not exist yet
  (matching `sorethumb run`), instead of failing with a `StoreError`.
- Dataset identity was the full-content+schema fingerprint
  (`content[:32]_schema[:16]`), so appending a day of rows to the source started
  a brand-new dataset and orphaned every prior `period` / `totals` / `run` row —
  backfill re-bootstrapped from scratch each time the file grew. Identity is now
  the stable logical `dataset_id`; the content+schema fingerprint is recorded
  per snapshot instead (see Added). `sorethumb backfill` / `sorethumb history` no
  longer read the source file at all — they resolve identity from config.
- `Calibrator.transform` tie handling: it interpolated (`np.interp`) over the
  10 000-point quantile grid, which is undefined on the flat segments a
  repeated reference value produces — a tied score could land anywhere in its
  band. It now uses an explicit mid-rank empirical CDF
  (`(#{ref < s} + 0.5·#{ref == s}) / n` via `searchsorted`), so a tied value
  maps to the midpoint of its band. Monotonicity and the constant-reference →
  0.5 guard are unchanged.
- Repeat runs no longer blank a report. A skipped group (already `complete` in
  the ledger) returned a `GroupSummary` with `n_anomalies=0` and no results
  path, so a second run of the same deterministic `run_id` re-rendered the
  report — and the per-group CSVs — with every group empty, destroying the good
  one. Skipped groups now carry their persisted count + results path, and the
  report is rendered from the store (`render_report_for_run`) rather than from
  in-memory run state. The provenance block also now shows the real
  `dataset_fp` (was blank).
- The published README benchmark table was generated by an earlier version
  of `sorethumb.evaluate.benchmark` that derived the precision@k/recall@k/F1
  review budget from the labels being scored
  (`k = round(n_total * y.mean())`), which forces `k == n_positives` and
  collapses all three metrics into one number — every row in the old table
  showed identical `precision_at_k`/`recall_at_k`/`f1_at_contamination`
  values. The table has been removed pending a fresh run of the now-fixed
  harness (fixed 5% review budget, already in place from the prior
  label-leak fix) rather than leave numbers up that read as more
  informative than they are. Two further honesty gaps in the harness are
  fixed alongside this: `BenchmarkRow`'s ROC-AUC/AP (and their `_std`
  columns) render `"n/a"`, not `"nan"`, when a seed's sample was
  single-class, and are averaged across seeds with `nanmean`/`nanstd` so
  one degenerate seed doesn't collapse the whole row's mean to NaN; and the
  before/after RSS delta (`peak_rss_mb`, a `psutil`-measured process-memory
  snapshot around a single fit/score call — not a real peak, and noisy
  under GC/allocator behaviour) has been removed rather than kept as a
  number nobody could act on. `sorethumb benchmark` now also records a
  `BenchmarkMetadata` (generation timestamp, platform, Python, sorethumb,
  numpy, scipy, scikit-learn versions) alongside every run, written to
  `benchmark_metadata.json` and prepended to the Markdown table, so a
  published number can be tied to the environment that produced it.
- `sorethumb.__version__` was a second hard-coded copy of `pyproject.toml`'s
  version, and `Store.insert_run`'s `library_version` fallback was a third
  (a stale literal `"0.1.0"` that neither of `run_detection`/`score_forward`
  ever overrode, so every run's report footer always showed `"0.1.0"`
  regardless of what was actually installed). `__version__` is now derived
  from installed distribution metadata (`importlib.metadata.version`);
  both pipeline entry points now pass it explicitly to `insert_run`, and the
  now-pointless hard-coded fallback there is removed.

### Security

- CSV report cells and column names are neutralised against spreadsheet formula
  injection: a leading `=`, `+`, `-`, `@`, or control character is prefixed with
  `'` before the frame is written.
- Documented the persisted-model trust boundary. A workspace's fitted
  estimators and calibrators are `joblib`/pickle files; `sorethumb score
  --from-run` and `run.reuse_models` unpickle them with no sandboxing, which
  is arbitrary code execution, not safe data loading. The SHA-256 file
  digests checked on load (see `load_model`) are an integrity check — they
  catch corruption or a swapped file — not a security boundary: a
  deliberately malicious file carries its own matching digest. New README
  "Honest limitations" entry and `SECURITY.md` section spell this out; the
  `score --from-run` CLI help/docstring and `docs/cli_reference.md` no
  longer imply the digest check makes a third-party workspace safe to load.
- `sorethumb workspace reset` could delete unrelated data. It resolved
  `run.workdir` to a path and ran `shutil.rmtree(path, ignore_errors=True)`
  without ever verifying that path was actually a sorethumb workspace — a
  typo, a bad unattended `--workdir`/config, or a resolved-symlink surprise
  could point it at a home directory, a git repository root, or worse, and
  `ignore_errors=True` meant a partial failure still reported success.
  `reset` now refuses outright, regardless of `--yes`, unless the target
  actually opens as a real workspace (has a `sorethumb.db` marker) — and
  separately, always refuses a filesystem/drive root, the home directory,
  the current working directory, a git repository root, or a suspiciously
  shallow path, even if one of those happened to contain a marker file.
  Deletion failures now propagate as a non-zero exit instead of being
  silently swallowed.

### Compatibility

- Identity digests widened from 64-bit (16 hex) to 128-bit (32 hex):
  `run_id`, `group_key`, `dataset_fp`, `config_hash`, and the model plan digest.
  Run IDs and group directories therefore have new names — existing workspaces
  start fresh runs rather than resuming. No migration provided.
- `attribution_kind == "exact"` is now reserved for ECOD/HBOS's native
  per-feature decomposition (`explain/native.py`) — it no longer appears for
  IsolationForest/TreeSHAP results, which read `"model_specific"` instead
  (see Changed). Update any downstream filter/comparison on the literal
  string `"exact"` accordingly.
- One-time re-baseline for existing workspaces: the first run after upgrading
  computes the new logical `dataset_id`, which will not match the old
  content+schema `dataset_fp`, so history recorded before the upgrade stays under
  the old key and is not visible to `sorethumb history` / backfill for the new
  id. Migration 005 seeds `dataset_snapshot` from existing `dataset` rows so
  their snapshot history is preserved. Set `source.dataset_id` explicitly to pin
  identity going forward.
- `pip install sorethumb` installs an unrelated package, not this one —
  install `pip install sorethumb-ml` instead (see Changed). No prior
  release of this project ever shipped under the `sorethumb` PyPI name to
  migrate away from; `import sorethumb` and the `sorethumb` CLI command are
  unaffected.

### Docs

- Regenerated the README benchmark table. `kmeans_distance` on the synthetic
  datasets now reads ROC-AUC 1.00 (was 0.08 / 0.0002) — the table still carried
  pre-CBLOF-fix numbers. Other detectors' accuracy metrics are unchanged.
- `docs/models.md`, `docs/explanations.md` and the README detector table now
  describe `kmeans_distance` as CBLOF (distance to the nearest large-cluster
  centroid, not its own) and document `large_cluster_coverage` and its ~10 %
  contamination ceiling.
- `docs/models.md` "How anomaly scoring works" corrected: `composite_score` is
  1 = most anomalous, 0 = normal (it was documented inverted).
- README images and relative documentation links converted to absolute
  `github.com` URLs so both render correctly on GitHub and on PyPI (which has
  no filesystem context to resolve a relative link against). Fixed a wrong
  sample report path in the CLI quickstart transcript. Adopted PEP 639
  license metadata (`license = "Apache-2.0"` + `license-files`, dropping the
  now-redundant OSI classifier). `docs/approximations.md` and the README's
  Honest limitations gained entries for correlation pruning, PCA's
  low-variance risk, contaminated fitting, capped-detector train/score
  mixing, and zero-inflated scaling.

[Unreleased]: https://github.com/tarne75/sorethumb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarne75/sorethumb/releases/tag/v0.1.0
