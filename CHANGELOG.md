# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-21

First public release.

### Added

- **Names**: install the `sorethumb-ml` distribution (`pip install sorethumb-ml`;
  `sorethumb` on PyPI is an unrelated package), import `sorethumb_ml`, and run
  the `sorethumb` command. Third-party detectors register through the
  `sorethumb_ml.detectors` entry-point group. Python 3.11–3.13 on Linux,
  macOS and Windows (x64); the package ships `py.typed`.
- **Pipeline**: profile every column, encode, impute and derive features, fit
  an ensemble of detectors, percentile-calibrate their scores, combine them into
  a ranking and a flag decision, and explain each flagged row in terms of the
  original columns. It ranks a review shortlist: `scoring.contamination` is a
  review budget, not an estimate of how many anomalies the data contains, and
  each run shows the per-detector flag rates alongside the flagged count.
- **Detectors**: `isolation_forest`, `kmeans_distance` (CBLOF-style) and
  `one_class_svm` form the default ensemble; `ecod`, `hbos` and `lof` are also
  available. A detector's `extra_params` forwards extra constructor keywords to
  the underlying scikit-learn estimator.
- **Detector output contract**: `score_samples` must return a finite 1-D numeric
  `ndarray` with one value per input row, and `natural_flag` a 1-D `bool`
  `ndarray` of the same length. Every call is checked immediately, for a
  freshly fitted detector, a `run.reuse_models` model and a `score --from-run`
  model alike. A violation, or an exception raised inside a plugin's method,
  fails the group with a `DetectorError` naming the detector, the failing
  condition, the expected contract and the actual dtype/shape, and the model is
  not saved. `ScoreEnsemble.combine()` and `Calibrator.fit()`/`transform()` also
  reject malformed or non-finite input themselves. See
  [Writing a detector plugin](docs/models.md#writing-a-detector-plugin).
- **Ensemble combination**: `intersection` (the default: every configured
  detector must flag a row; see Known limitations), `union`, and `composite`
  (a weighted average with a guard that drops an anti-correlated member, and
  optional agreement-based weighting).
- **Explanations**: up to `explain.top_n` reasons per flagged row
  (`reason_1`, `reason_2`, ...), labelled per row by `attribution_kind`:
  `exact` for ECOD and HBOS, `model_specific` (TreeSHAP) for Isolation Forest
  with the `explain` extra installed, and `heuristic` otherwise (centroid
  distance, input gradient, or the reference-scaled deviation used for rows far
  outside the data). See `docs/explanations.md`.
- **CLI**: `init`, `inspect`, `run`, `score`, `report`, `backfill`, `history`,
  `runs`, `show`, `anomalies`, `explain-plan`, `detectors`, `config`
  (`check`, `schema`, `show`) and `workspace` (`ls`, `du`, `prune`, `vacuum`,
  `migrate`, `reset`). `sorethumb run data.csv` works with no config file, and
  the read-only commands after it need none either. Runs are resumable and
  idempotent, tracked in a SQLite workspace. Exit codes are stable (`0` ok,
  `1` runtime failure, `2` pre-flight rejection, `3` not found, `4` partial
  success) and every `--json` command emits one document per outcome; see
  `docs/cli_reference.md`.
- **Score-forward**: `sorethumb score --from-run RUN_ID` scores new data with a
  fitted run's persisted models and calibrators, without refitting, so scores
  stay on one scale across calls. A config whose fit-time settings differ from
  the source run's is rejected.
- **History**: `sorethumb history` shows rolling-window trends and
  `sorethumb backfill` fills missing periods, both scoped to the dataset and
  the exact configuration that produced each period.
- **Reports**: self-contained HTML plus CSV and JSON, re-renderable from
  persisted state with `sorethumb report`.
- **Windows** (10/11 x64): every command, the Python API and reports work as
  on Linux and macOS, from PowerShell, Windows PowerShell 5.1 or cmd.exe; CI
  runs the unit, contract and integration suites on `windows-latest` for
  every change, plus an install of the built wheel. Spellings of one Windows
  file that differ only in letter case, separators or a `file://` prefix
  (`C:\Data\x.csv`, `c:/data/x.csv`, `file:///C:/Data/x.csv`) are one dataset
  for history and one configuration for run ids and model reuse; a
  drive-relative path such as `C:data.csv` is rejected with a clear message.
  `~` in `--config`, `--workdir` and `init` paths is expanded by sorethumb
  itself, so it works in cmd.exe too, and a command started with stdin from
  `NUL` (a scheduled task, `subprocess.DEVNULL`) never prompts.
- **Files held open by another program** (a spreadsheet, a file-sync client,
  antivirus): replacing or deleting a report, model or downloaded file is
  retried briefly, then fails with a `FileInUseError` naming the file.
  `sorethumb workspace prune` deletes everything else it can, keeps a locked
  file indexed for the next prune, and exits non-zero.
- **Workspace checks before a run**: without Windows long-path support, `init`,
  `run` and `score` stop before doing any work (exit code `2`,
  `PathTooLongError`) when the workspace is too deep for the 260-character
  path limit. A workspace on a network filesystem, a mapped drive or UNC
  path, or in a OneDrive, Dropbox, Google Drive or iCloud folder gets a
  one-line warning.
- **Text encodings**: a CSV that isn't UTF-8 (Excel's plain "CSV" on Windows
  writes cp1252) fails with a message saying so and how to fix it.
  `report.csv_bom` starts report CSVs with the byte-order mark Excel needs to
  show non-ASCII text. Output redirected to a pipe or file is UTF-8 unless
  `PYTHONIOENCODING` says otherwise, and every `--json` output is
  ASCII-escaped, so it is lossless under any output encoding.
- **Extras**: `explain` (shap, numba) for TreeSHAP and KernelSHAP attributions;
  `benchmark` (datasets, pandas) for the benchmark harnesses.
- **Docs**: `docs/stability.md` sets out the pre-1.0 API and stability policy;
  `docs/benchmarks.md` holds the benchmark results and their accuracy floors.

### Security

- CSV report output is neutralised against spreadsheet formula injection.
- `sorethumb workspace reset` refuses any target that does not open as a
  sorethumb workspace, and always refuses a filesystem root, the home
  directory, the current directory, a git repository root or a very shallow
  path. It deletes only the entries sorethumb creates, so other files in the
  directory are kept, and removes a symlink or Windows directory junction
  among them without following it. The typed confirmation accepts the path
  however it is spelled: quoted (as Windows Explorer's "Copy as path" gives
  it), with a trailing separator, or on Windows in a different letter case
  or with forward slashes.
- Credentials are never sent over plaintext HTTP. `source.auth` or
  `user:password@` in an `http://` `source.uri` is rejected at config load, and
  a credentialed request is refused before it is sent on every hop, including
  the first request of an HTTP-to-HTTPS redirect. There is no loopback
  exception.
- A config validation error never echoes a `user:password@` from
  `source.uri`, and download error messages redact it.
- Authenticated HTTP(S) downloads (`source.auth`) send the `Authorization`
  header only to the configured origin: never across a redirect to another host
  or port. An HTTPS-to-HTTP redirect is refused, and so is a request or redirect
  to a loopback, link-local, private or reserved address.
- An unexpected CLI crash never prints local variables, so a traceback cannot
  reveal the download credential.
- **Persisted-model trust boundary**: a workspace's fitted models are
  `joblib`/pickle files, and `score --from-run` and `run.reuse_models` unpickle
  them with no sandboxing. That is arbitrary code execution, not safe data
  loading. The SHA-256 digests checked on load catch corruption or a swapped
  file; they are **not** a security boundary against a deliberately malicious
  one. Only use workspaces you created or fully trust. See
  [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

### Known limitations

- **The default `combination="intersection"`** flags a row only when every
  configured detector flags it independently. That is deliberately
  conservative and, on some data shapes, can flag very few true anomalies or
  none. See `docs/approximations.md`.
- **Without the `explain` extra**, Isolation Forest explanations are
  `heuristic` rather than `model_specific`.
- Every detector fits on the same unlabelled data it scores; there is no
  held-out "known normal" reference. Independent runs are not on a common score
  scale; use `score --from-run` for a comparable trend. See
  `docs/approximations.md` for this and the other approximations (PCA,
  correlation pruning, row-capped detector training, zero-inflated scaling).
- **The `swamping` benchmark scenario** found no false-positive inflation from
  unlabelled training contamination; measured on matched data, the effect runs
  the other way (mildly reduced sensitivity).
- Real-dataset benchmark figures (KDDCup99, Covtype) are capped at 20,000 rows
  for CI runtime, as `docs/benchmarks.md` states where the tables appear.
- **Platforms**: native Windows on ARM is not supported (pyarrow publishes no
  Windows ARM64 wheel; an x64 Python under emulation works), and on any OS a
  workspace needs a local drive, not a network share or synced folder. See
  the README's Supported platforms section.

[Unreleased]: https://github.com/tarne75/sorethumb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarne75/sorethumb/releases/tag/v0.1.0
