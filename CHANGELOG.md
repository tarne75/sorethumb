# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security

- CSV report cells and column names are now neutralised against spreadsheet
  formula injection: a leading `=`, `+`, `-`, `@`, or control character is
  prefixed with `'` before the frame is written.

### Changed

- Persisted models now record the fit-time versions of Python, sorethumb,
  scikit-learn, numpy, scipy and joblib in `manifest.json`. `load_model` and
  `score_with_existing` compare them against the current environment and emit
  `ModelVersionMismatchWarning` (or, in strict mode, raise
  `ModelVersionMismatchError`) so a dependency upgrade can no longer change
  scores silently.

### Removed

- Dropped the unsupported `s3://` example from the README; only local paths and
  `http(s)://` URLs are accepted (an `s3://` URI already raised
  `SourceError: Unsupported URI scheme`).
