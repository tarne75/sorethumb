# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  each run shows the per-detector flag rates alongside the flagged count. How many
  rows are flagged depends on the ensemble policy as well as on `contamination`.
  By default (`intersection` at `"auto"`) it is whatever every detector's own
  boundary agrees on, not a number you choose.
- **Input data**: a file with a single column, or with every column as text
  (all-categorical data), is valid input. A CSV/TSV is refused as
  mis-delimited only when another delimiter character appears in the header
  and the same number of times in most data rows; an all-text schema is
  re-scanned (up to 1,000,000 rows) and refused, naming
  `source.read_options['infer_schema_length']`, if a numeric column was hidden
  by blank leading rows. A source path that is a directory is rejected with a
  clear message. A column is never dropped for having few distinct values
  (only a constant one is), since a rare level of a low-cardinality column can
  be exactly the anomaly; `profiling.near_constant_distinct` can opt in to
  dropping them. `columns.include_only` restricts
  the features to the listed columns plus `time_column`, `group_by`,
  `id_column` and `reference_column`; a listed name not in the dataset is a
  `ConfigError`.
- **Categorical columns**: up to `features.one_hot_max_cardinality` levels are
  one-hot encoded, more are frequency-encoded. One-hot dummies are not scaled
  like numeric columns: two rows that differ only in one category are
  `features.categorical_distance` apart (default 0.5, in the interquartile-range
  units numeric features use), so a categorical column of noise cannot dominate
  the distance-based detectors (KMeans, One-Class SVM, LOF) while a rare level
  still stands out. Isolation Forest is unaffected.
- **Dates and times stored as text**: a text column holding ISO dates or
  datetimes, a time of day, or a day/month/year date is parsed into a real
  date/time before profiling, so it gets derived features (`hour`,
  `dayofweek`, ...) and a text `columns.time_column` filters period windows by
  date rather than as a string. Every date/time column gets derived features,
  not only `time_column`. `group_by`, `id_column` and `reference_column`
  columns, and columns an `ignore` pattern or `include_only` excludes, are
  left as text. An ambiguous day/month order (every day and month <= 12) is
  never guessed: the column stays text, with a warning that `run.strict`
  does not turn into an error. `columns.datetime_formats` forces a column's
  format, or `"epoch_s"`/`"epoch_ms"` for a Unix-timestamp integer or float
  column. A value that fails to parse becomes missing. `Duration` columns
  become a numeric seconds feature. `run`, `score --from-run`, `inspect`,
  `explain-plan` and the Python API's `build_feature_plan`/`apply_feature_plan`
  all parse the same way, and the parse is stored with the run so
  `score --from-run` replays it exactly.
- **Detectors**: `isolation_forest`, `kmeans_distance` (CBLOF-style) and
  `one_class_svm` form the default ensemble; `ecod`, `hbos` and `lof` are also
  available. A detector's `extra_params` forwards extra constructor keywords to
  the underlying scikit-learn estimator. On discrete features (for example
  all-categorical data: at most 50 distinct feature rows, and at most half
  as many as there are rows), `kmeans_distance` gives each distinct row its
  own cluster so a rare category is flagged; its `k_max` defaults to automatic
  (8 on continuous data), and an explicitly set `k_max` is a hard cap.
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
  optional agreement-based weighting). Each is a versioned *policy*
  (`intersection@1`, `union@1`, `composite-equal@1`, `composite-agreement@1`,
  `composite-manual@1`) recorded on every run (`run --json`, `show`); a change
  to what a policy flags is a new version. The default is a deliberate design
  decision and is not changing for now; see "Where things stand" in
  `docs/ensemble-policy.md` for the evidence and its limits.
- **Explanations**: up to `explain.top_n` reasons per flagged row
  (`reason_1`, `reason_2`, ...), labelled by `attribution_kind`: `exact` for ECOD
  and HBOS, `model_specific` for Isolation Forest (interventional TreeSHAP, with
  the `explain` extra) and `heuristic` otherwise (centroid distance, input
  gradient, KernelSHAP, or the reference-scaled deviation used where a score is
  flat). See `docs/explanations.md`.
  - **Detection first.** Each group's detection output (`anomalies.parquet`:
    scores, flags, ranks) is written, and its SHA-256 recorded as
    `detection_digest`, before any explanation is computed. Nothing the
    explanation stage does, including failing, changes which rows are flagged.
    Detection always uses `features.dtype`, whatever the explain settings.
  - **Explained against normal rows.** Every method measures a flagged row
    against a reference of rows no detector flags (falling back to the less
    anomalous half of the unflagged rows, and to no explanation when neither has
    10 rows), never against the anomalies. A fitting run saves the reference
    summary (`models/<run>/<group>/reference.json`, plain JSON), and
    `score --from-run` explains against it.
  - **Per-row status.** Each flagged row has an `explain_status` (`ok`,
    `partial`, `unavailable`) and an `explain_issue` saying what is missing; an
    unavailable row has no `reason_<k>`. Each group records `explain_status`,
    `n_explained` and `n_explain_unavailable` (`run --json`, `show`).
  - **The record behind each reason** (`attributions.parquet`): original field,
    raw value, reference value and direction, signed contribution (positive
    toward anomaly) and the magnitude reasons are ranked by, each detector's
    method, weight and share, the transformed features behind the field, every
    approximation caveat (`approximation_flags`), `explainer_version`
    (`explain@2`) and `reference_id`. Shown by `sorethumb anomalies --detail` and
    the HTML report's "Explanation detail" section; `index.json` carries it under
    `attributions`.
  - **Failures are reported, not hidden.** An explanation failure is a warning
    with a stable code (`explain.stage_failed`, `explain.backend_failed`,
    `explain.no_signal`, `explain.reference_unavailable`, ...), never a group
    failure. A failed explanation stage makes the run exit `partial`, as does any
    explanation warning under `run.strict`.
- **Structured warnings**: every warning carries a stable `code`, its stage and,
  where they apply, the detector and rows affected. They are stored with each
  group, shown with their codes in the run summary, `show` and the report, and
  listed in `run --json` (`warnings`).
- **CLI**: `init`, `inspect`, `run`, `score`, `report`, `backfill`, `history`,
  `backtest`, `evidence`, `runs`, `show`, `anomalies`, `explain-plan`, `detectors`,
  `config` (`check`, `schema`, `show`) and `workspace` (`ls`, `du`, `prune`,
  `vacuum`, `migrate`, `reset`). `sorethumb run data.csv` works with no config file, and
  the read-only commands after it need none either. Runs are resumable and
  idempotent, tracked in a SQLite workspace. Exit codes are stable (`0` ok,
  `1` runtime failure, `2` pre-flight rejection, `3` not found, `4` partial
  success) and every `--json` command emits one document per outcome; see
  `docs/cli_reference.md`. A group is selected by its label (`run --only-group`,
  `show --group`, `history --group`; the last two also take its key).
- **Reproducible fits**: KMeans (the detector, its choice of k, and the explanation
  reference) fits on one OpenMP thread, so the same data and seed give the same
  centroids every run; scikit-learn's parallel partial sums otherwise varied by a
  rounding step, which on data with many duplicate rows moved whole tied blocks across
  the KMeans flag fence (up to 90 rows flagged more or fewer between identical KDD Cup
  1999 runs). About 18% more KMeans fit time at its 200,000-row training cap.
  `threadpoolctl` is now a declared dependency. The real-data backtest pins the KDD
  SA subset's random draw (`random_state=0`), which was different on every load.
- **Uninformative detector boundaries**: at `scoring.contamination = "auto"`, a
  detector whose own boundary flagged more than half the rows it was fitted on is left
  out of the flag vote (the intersection or union, and the composite's
  median-of-rates cut), with an `UninformativeBoundaryWarning`
  (`scoring.boundary_uninformative`); its scores still rank. On a table of
  low-cardinality categoricals, Isolation Forest's boundary flagged every row (so
  Isolation Forest alone flagged every row, and `union` did too) and One-Class SVM's
  sat on every row, where the side a row fell on was floating-point noise and differed
  by platform. The fit-time rate is saved in each model's manifest
  (`boundary_flag_rate`), so `score --from-run` applies the same verdict; models
  saved before have none and keep their vote. `GroupSummary.abstaining_detectors` and
  `groups[].abstaining_detectors` in `--json` name them. A numeric contamination is
  unaffected.
- **Reading a run: the ranking first**: every `run` and `score` summary ends with
  the top-ranked rows, flagged or not, and each group keeps its top 20 by combined
  score in `ranked.parquet` (`GroupSummary.top_ranked`, `groups[].top_ranked` in
  `--json`, `sorethumb anomalies --ranked`, and in the report). At the default
  `scoring.contamination = "auto"` the detectors' own boundaries set how many rows
  are flagged, so every run says the size was not chosen (`shortlist_size_chosen:
  false` in `--json`), and a workspace's first run, or any run without a
  `sorethumb.toml`, says it prominently with the count it came to and how to set a
  review budget. The default policy is unchanged. A run, `anomalies` or a report
  with no flagged rows says that is the policy's decision, not a finding that the
  data holds no anomalies, and shows the ranking; a missing results file is
  reported as missing.
- **Score-forward**: `sorethumb score --from-run RUN_ID` scores new data with a
  fitted run's persisted models and calibrators, without refitting, so scores
  stay on one scale across calls. A config whose fit-time settings differ from
  the source run's is rejected.
- **History**: `sorethumb backfill` fills missing periods and `sorethumb history`
  reports how the data is changing, scoped to the dataset and the exact
  configuration that produced each period. A period window that selects no rows
  fails immediately, naming the window, the time column and the data's available
  range. See `docs/trends.md`.
  - Four signals, each with a verdict per window (`ok`, caveats with reasons, or
    n/a with why): **population** drift (PSI per input field, and volume; no
    model involved), **scores** and **threshold** (with an anchor run, below),
    and **counts** (the anomaly rate per window with Wilson and Newcombe 95%
    intervals, coverage and matched groups).
  - Under `composite` with a numeric contamination the rate is set by the review
    budget, so no change is shown.
  - **Anchor run** (`history.anchor_run`): each period run also scores its rows
    with one fixed run's models, calibrators and flag rule, so a change in
    anomaly count splits exactly into volume, score drift and threshold drift.
    Every fitting run saves its rule (`anchor_rule.json`, plain JSON), so any
    complete fitting run can be an anchor.
  - Each period run records per-period statistics (`period_stats`): per input
    field, its missing count and percentiles or top values; per detector, score
    percentiles and flag count.
  - `history --json` prints the whole report; each period run's HTML report shows
    it per group, and `index.json` carries it under `trend`.
- **Reports**: self-contained HTML plus CSV and JSON, re-renderable from
  persisted state with `sorethumb report`.
- **Windows** (10/11 x64): every command, the Python API and reports work as
  on Linux and macOS, from PowerShell, Windows PowerShell 5.1 or cmd.exe; CI
  runs the unit, contract and integration suites on `windows-latest` for
  every change, plus installs of the built wheel and sdist. Spellings of one Windows
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
  `benchmark` (datasets, pandas, pyarrow) for the benchmark harnesses.
- **Docs**: `docs/stability.md` sets out the pre-1.0 API and stability policy;
  `docs/benchmarks.md` holds the benchmark results and their accuracy floors,
  and an ensemble policy matrix: every scenario under every policy, with
  precision, recall, ranking, stability and false-positive figures checked
  against versioned per-anomaly-type budgets, where every budget the default
  misses is listed as an accepted limitation with its reason.
  `docs/ensemble-policy.md` sets out the backtest protocol a candidate policy
  must pass before it can be said to outperform the default (pre-registration,
  disjoint confirmation seeds, paired bootstrap with Holm correction, no budget
  or stability regressions, confirmation in rounds of fresh seeds) and
  publishes each confirmation result. An end-to-end scale benchmark
  (`scripts/run_scale_benchmark.py`) runs a zero-config `sorethumb run` and a
  `score --from-run` at 10,000, 100,000 and 1,000,000 rows, and publishes time per
  stage, peak memory, feature width, each detector's training sample, explanation
  cost and output size. `docs/explanations.md` and
  `docs/trends.md` say what each explanation and trend figure means and where it
  stops.
- **`sorethumb backtest`**: compares every registered ensemble policy with the
  default on your own data, using `columns.reference_column` labels, and records
  each verdict in the workspace (migration 011, `policy_evidence`). A `run` or
  `score` with a non-default policy that has no `outperforms` record for the
  dataset and configuration warns (`UnevidencedPolicyWarning`); `run.strict`
  refuses it (`UnevidencedPolicyError`, exit code 2). The evidence a run relied
  on is recorded as `run.policy_evidence_id` and in `--json` output, and shown by
  `sorethumb show` and the run summary; `sorethumb evidence` lists the recorded
  evidence and which records apply to the current configuration.
  `sorethumb backtest --forward` tests out of time instead: each period is fitted
  and the next scored with those fits, as `score --from-run` does. A candidate
  must also rank the anomalies no worse than the default (mean average precision
  within `--ranking-tolerance`, 10%): decision rule 2, which `run` requires of the
  evidence it relies on. With `scoring.contamination = "auto"`, `--review-budget`
  states the budget flags are judged against.
- **Real-data policy evidence**: the pre-registered backtest protocol applied to
  KDD Cup 1999 (SA), Covertype (2 vs 4) and a time-ordered credit-card fraud
  dataset, with disjoint development and confirmation slices; results in
  `docs/ensemble-policy.md`.
  Composite policies must be paired with a numeric `scoring.contamination`: at
  `"auto"`, `composite-agreement@1` flags several times the review budget and
  does not outperform the default (pre-registered, synthetic and real).
  `docs/ensemble-policy.md` gives a recommended setup for users with a review
  budget, and qualifies the default's precision-first rationale: it holds at the
  shipped `"auto"`, and the default's real weakness is unpredictable volume.

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
  or port. An HTTPS-to-HTTP redirect is refused, and so is a redirect to a
  different host that resolves to a loopback, link-local, private, reserved
  or multicast address (such as a cloud metadata endpoint). The host you
  configure yourself is never blocked, even an internal one; see SECURITY.md.
- An unexpected CLI crash never prints local variables, so a traceback cannot
  reveal the download credential.
- Source URI query values are never stored, logged or reported. The persisted
  and displayed form keeps the scheme, host, non-default port, path and query
  key names and replaces every query value with `REDACTED` (no name list, so
  `client_secret`, `jwt`, `accessKey` and unlisted names are covered); userinfo
  becomes `***@` and the fragment is dropped. This covers `dataset.source_uri`,
  `run.config_json`, `RunResult.dataset_uri`, `run --json`, `show`, reports,
  the `httpx` request log line and download errors, with no opt-out. A new
  `source_digest` (SHA-256 of the canonical full URI) on `dataset` and `run`
  keeps query-differing sources distinguishable. A workspace created by a
  pre-release build is scrubbed when opened (migration 009; see SECURITY.md for
  what it does not reach). Without
  `source.dataset_id`, a run whose query key set differs from the dataset's
  previous run is refused, naming the `dataset_id` that keeps its history.
- **Persisted-model trust boundary**: a workspace's fitted models are
  `joblib`/pickle files, and `score --from-run`, `run.reuse_models` and
  `history.anchor_run` unpickle them with no sandboxing. That is arbitrary code execution, not safe data
  loading. The SHA-256 digests checked on load catch corruption or a swapped
  file; they are **not** a security boundary against a deliberately malicious
  one. Only use workspaces you created or fully trust. See
  [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

### Known limitations

- **The default `combination="intersection"`** flags a row only when every
  configured detector flags it independently. That is deliberately
  conservative and, on some data shapes, can flag very few true anomalies or
  none. At the default `scoring.contamination = "auto"` you do not control how
  many rows it flags: 0.18x to 8.4x a review budget on the real datasets tested.
  With a fixed review budget, `composite-agreement@1` at that budget is the
  evidenced choice. See `docs/ensemble-policy.md` and `docs/approximations.md`.
- **Policy evidence has limits.** It covers a maintainer-written synthetic suite
  and three real datasets (one confirmation slice each), with the default three
  detectors at stated review budgets. `rule:discrete-budget@2` won on synthetic
  data only and is not a selectable setting. No policy recovers contextual
  anomalies. Full list: "Where things stand" in `docs/ensemble-policy.md`.
- **Without the `explain` extra**, Isolation Forest explanations are
  `heuristic` rather than `model_specific`. Explanations are exact only for ECOD
  and HBOS; TreeSHAP decomposes Isolation Forest's path length, not its score.
- **`run.max_memory_mb` is a projection, not a limit**: it refuses a run whose
  estimated feature-matrix size exceeds it, but cannot cap actual memory use.
- **How much a categorical column counts is a prior.** One-hot dummies sit at
  a fixed scale (`features.categorical_distance`, 0.5 interquartile ranges per
  change of category), chosen so uninformative categories cannot swamp the
  distance-based detectors; it is not learned from your data. See
  `docs/approximations.md`.
- Every detector fits on the same unlabelled data it scores; there is no
  held-out "known normal" reference. Independent runs are not on a common score
  scale: `sorethumb history` separates score drift from threshold drift only
  with an anchor run, and `score --from-run` reuses a fitted run's scale. See
  `docs/approximations.md` for this and the other approximations (PCA,
  correlation pruning, row-capped detector training, zero-inflated scaling).
- **The `swamping` benchmark scenario** found no false-positive inflation from
  unlabelled training contamination; measured on matched data, the effect runs
  the other way (mildly reduced sensitivity).
- Real-dataset benchmark figures (KDDCup99, Covtype) are capped at 20,000 rows
  for CI runtime, as `docs/benchmarks.md` states where the tables appear.
- **Dataset size**: measured end to end up to 1,000,000 rows (with the defaults, 6.9
  minutes and 4.9 GB peak memory on an 8-core, 16 GB laptop); larger datasets are
  untested. Explanations are most of the time and memory, and peak memory is set by
  loading, profiling and explanations, not by the projected feature matrix that
  `run.max_memory_mb` checks. See the README's Scale guide.
- **Platforms**: native Windows on ARM is untested (the core installs from
  Windows ARM64 wheels, but no CI job runs there, and the `benchmark` extra's
  pyarrow has no such wheel; an x64 Python under emulation is the tested
  route), and on any OS a workspace needs a local drive, not a network share or
  synced folder. See the README's Supported platforms section.

[Unreleased]: https://github.com/tarne75/sorethumb/commits/main
