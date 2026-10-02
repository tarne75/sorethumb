# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **Windows**: every write of a model, result, report or downloaded source
  failed with `OSError: [Errno 9] Bad file descriptor`, because the file was
  flushed to disk through a read-only handle, which Windows rejects.
- **Redirected output**: a command whose output went to a pipe or file could
  stop with `UnicodeEncodeError` when the text included a character outside
  the stream's encoding (for example a column named `温度` on Windows, where
  pipes default to cp1252). Redirected output is now UTF-8 unless
  `PYTHONIOENCODING` says otherwise, an unencodable character is replaced
  rather than aborting, and every `--json` output is ASCII-escaped, so it is
  lossless under any output encoding.
- **Files held open by another program** (Windows): replacing a report, model
  or downloaded file that a spreadsheet, a file-sync client or antivirus
  software has open is retried briefly, then fails with a `FileInUseError`
  naming the file, and a failed report now says why in the run summary
  instead of only "see the log". `workspace reset` and `workspace prune`
  retry deletions the same way.
- **Log and database handles** no longer outlive the command that opened
  them, so `workspace reset` can delete `logs/` on Windows, in-process
  callers no longer keep logging into the first workspace they used, and a
  log rotation that loses a race with another process keeps logging instead
  of reporting "--- Logging error ---" for every later record.
- **Prompts on Windows**: `sorethumb run data.csv` started with stdin from
  `NUL` (`< NUL`, a scheduled task, `subprocess.DEVNULL`) offered to save a
  config, read end-of-file and aborted; Windows reports the null device as a
  terminal. Only a real console is now treated as interactive.
- **`file://C:/...` URIs** (two slashes, as often typed by hand) resolve to
  the drive path instead of an invalid `\\C:\...` share path.
- **`workspace reset`** removes a directory junction inside the workspace
  without following it (it used to stop halfway), deletes read-only files,
  and accepts the typed confirmation however the path is spelled: quoted (as
  Windows Explorer's "Copy as path" gives it), with a trailing separator, or
  on Windows in a different letter case or with forward slashes. The prompt
  now shows the exact path to type.

### Added

- **Windows path-length preflight**: without long-path support Windows caps a
  path at 260 characters, and sorethumb creates files about 112 characters
  below the workspace directory. `init`, `run` and `score` now stop before
  doing any work (exit code 2, `PathTooLongError`) when the workspace is too
  deep, instead of failing with a bare "file not found" minutes into a run.

### Changed

- **Windows source paths**: spellings of one Windows file that differ only
  in letter case, separators or a `file://` prefix (`C:\Data\x.csv`,
  `c:/data/x.csv`, `file:///C:/Data/x.csv`) are now one dataset for history
  and one configuration for run ids and model reuse. Paths on Linux and
  macOS, relative paths and URLs are identified exactly as before. A
  drive-relative path such as `C:data.csv` is now rejected with a clear
  message instead of "Unsupported URI scheme 'c'".

## [0.1.0] - 2026-09-21

First public release.

### Added

- **Names**: install the `sorethumb-ml` distribution (`pip install sorethumb-ml`;
  `sorethumb` on PyPI is an unrelated package), import `sorethumb_ml`, and run
  the `sorethumb` command. Third-party detectors register through the
  `sorethumb_ml.detectors` entry-point group. Python 3.11–3.13 on Linux and
  macOS; the package ships `py.typed`.
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
  directory are kept.
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
- **Windows is untested and known to fail** across most CLI paths; see the
  README's and SECURITY.md's Supported platforms sections.

[Unreleased]: https://github.com/tarne75/sorethumb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarne75/sorethumb/releases/tag/v0.1.0
