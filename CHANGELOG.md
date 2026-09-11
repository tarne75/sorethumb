# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Fixed

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
  shipped changelog; `[0.1.0]` is not yet tagged.)

### Compatibility

- `attribution_kind == "exact"` no longer appears in results Parquet, CLI
  output, or reports — TreeSHAP rows now read `"model_specific"` (see Changed).
  Update any downstream filter/comparison on the literal string `"exact"`.
- One-time re-baseline for existing workspaces: the first run after upgrading
  computes the new logical `dataset_id`, which will not match the old
  content+schema `dataset_fp`, so history recorded before the upgrade stays under
  the old key and is not visible to `sorethumb history` / backfill for the new
  id. Migration 005 seeds `dataset_snapshot` from existing `dataset` rows so
  their snapshot history is preserved. Set `source.dataset_id` explicitly to pin
  identity going forward.

## [0.1.0] - 2026-09-09

First tagged release. The Fixed / Compatibility entries record corrections made
during pre-release hardening; there is no prior published version to diff
against.

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

### Changed

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

### Removed

- Dropped the unsupported `s3://` example from the README; only local paths and
  `http(s)://` URLs are accepted (an `s3://` URI already raised
  `SourceError: Unsupported URI scheme`).

### Fixed

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

### Security

- CSV report cells and column names are neutralised against spreadsheet formula
  injection: a leading `=`, `+`, `-`, `@`, or control character is prefixed with
  `'` before the frame is written.

### Compatibility

- Identity digests widened from 64-bit (16 hex) to 128-bit (32 hex):
  `run_id`, `group_key`, `dataset_fp`, `config_hash`, and the model plan digest.
  Run IDs and group directories therefore have new names — existing workspaces
  start fresh runs rather than resuming. No migration provided.

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

[Unreleased]: https://github.com/tarne75/sorethumb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarne75/sorethumb/releases/tag/v0.1.0
