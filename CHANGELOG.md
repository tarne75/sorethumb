# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

### Security

- CSV report cells and column names are now neutralised against spreadsheet
  formula injection: a leading `=`, `+`, `-`, `@`, or control character is
  prefixed with `'` before the frame is written.

### Fixed

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

### Compatibility

- Identity digests widened from 64-bit (16 hex) to 128-bit (32 hex):
  `run_id`, `group_key`, `dataset_fp`, `config_hash`, and the model plan digest.
  Run IDs and group directories therefore have new names — existing workspaces
  will start fresh runs rather than resuming. Pre-release, no migration provided.

### Changed

- Persisted models now record the fit-time versions of Python, sorethumb,
  scikit-learn, numpy, scipy and joblib in `manifest.json`. `load_model` and
  `score_with_existing` compare them against the current environment and emit
  `ModelVersionMismatchWarning` (or, in strict mode, raise
  `ModelVersionMismatchError`) so a dependency upgrade can no longer change
  scores silently.
- CI declares a weekly `schedule` trigger so the `benchmark` job (guarded by
  `if: github.event_name == 'schedule'`) can actually run — restoring the
  accuracy-regression signal.
- README marked pre-release (0.1.0); install instructions now build from a
  clone rather than implying a published PyPI package.

### Removed

- Dropped the unsupported `s3://` example from the README; only local paths and
  `http(s)://` URLs are accepted (an `s3://` URI already raised
  `SourceError: Unsupported URI scheme`).
