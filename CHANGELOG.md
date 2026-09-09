# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

- Model persistence: a group's `calibrator.json` and `manifest.json` were not
  namespaced by detector, so in a multi-detector group the second `save_model`
  overwrote the first's calibrator and manifest. `load_model` /
  `score_with_existing` then returned the wrong detector's calibrator (garbage
  calibrated scores). Files are now `<detector>.calibrator.json` /
  `<detector>.manifest.json`; `load_model` falls back to the old names for
  workspaces written by an earlier version. The bug was latent because the
  fit path uses in-memory calibrators and single-detector runs are unaffected.

### Changed

- `_pipeline`: the per-group ledger/status bookkeeping and the
  ensemble→threshold→explain→write tail are factored into shared helpers
  (`_execute_group`, `_finalize_group`) used by both `run_detection` and
  `score_forward`, so the two paths cannot diverge.

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
