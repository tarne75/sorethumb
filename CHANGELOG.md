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
  clear message. Binary columns are always kept;
  `profiling.near_constant_distinct` (default 3) drops a column with exactly
  3 distinct values, and setting it to 2 keeps those too (for a rare level
  of a 3-value category that is itself the anomaly). `columns.include_only` restricts
  the features to the listed columns plus `time_column`, `group_by`,
  `id_column` and `reference_column`; a listed name not in the dataset is a
  `ConfigError`.
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
  the exact configuration that produced each period. A period window that
  selects no rows fails immediately, naming the window, the time column and
  the data's available range.
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
  `benchmark` (datasets, pandas) for the benchmark harnesses.
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
  publishes each confirmation result.
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
- **Detection and explanation separated**: each group's detection output
  (`anomalies.parquet`: scores, flags, ranks) is written, and its SHA-256
  recorded as `detection_digest`, before any explanation is computed.
  Explanations go to `explanations.parquet`, joined back by `row_id` for
  `sorethumb anomalies` and the report, and each group records an
  `explain_status` (`ok`, `partial`, `failed`, `not_requested`,
  `no_flagged_rows`), `n_explained` and `n_explain_unavailable` (migration 012;
  `run --json`, `show`).
  - An explanation failure is a warning (`ExplanationFailedWarning`,
    `AttributionBackendWarning`) and never fails the group or changes its
    detection output.
  - A failed explanation stage makes the run exit `partial`. Under `run.strict`,
    any explanation warning does too, where it used to fail the group and lose
    its detection results.
  - Flagged rows that nothing explained used to show `attribution_kind = "none"`.
    They are now reported as a failed explanation.
  - Detection no longer depends on the explain settings. Turning explanations on
    used to upcast the detectors' input to float64, which shifted their scores
    slightly. Detection now always uses `features.dtype`.
- **Structured warnings**: every project warning carries a stable `code` (e.g.
  `explain.backend_failed`, `explain.rows_beyond_max_rows`), its stage and,
  where they apply, the detector and rows affected. Warnings are persisted as
  records in `run_group.warnings_json`, shown with their codes in the run
  summary, `show` and the report, and listed in `run --json` (`warnings`).
  Older plain-string warnings read back as `legacy`.
  - Explanation problems that used to be silent are now warnings: a detector
    with no attribution method, a failed PCA back-projection (previously only
    logged), rows beyond `explain.max_rows`, and a reference that includes
    flagged rows.
  - Each explained row records `explain_status` (`ok`, `partial`,
    `unavailable`) and `explain_issue`. An unavailable row's `reason_1` is now
    null; it used to hold the text "unavailable (...)".
- **Explanation record**: `attributions.parquet` records, for every reason, the
  original field, its raw value, the reference value and the row's direction
  against it, the signed contribution (positive toward anomaly) alongside the
  magnitude reasons are ranked by, each detector's method, weight and share, the
  transformed features behind the field, every approximation caveat that applies
  (`approximation_flags`), the `explainer_version` (`explain@1`) and the
  `reference_id`. `sorethumb anomalies --detail` shows it, as does each
  group's "Explanation detail" section in the HTML report (`index.json` carries
  the records under `attributions`). Ranking and the `reason_<k>` labels are
  unchanged.
  - A row is now `partial`, with `explain_issue` naming the detectors, when a
    detector that counts toward the score says nothing about that row: it is
    beyond that detector's `explain.max_rows`, or its attribution there is zero.
    Such rows used to show as `ok`.
  - A row every source attributes as zero is now `unavailable`
    (`no_attribution_signal`, warning `explain.no_signal`). It used to list its
    first fields as reasons, all with zero weight.
  - A group whose explanation stage fails no longer keeps an earlier attempt's
    explanation files.
- **Explanations measured against a consensus-normal reference** (`explain@2`):
  - The reference is the rows no detector flags, falling back to the less
    anomalous half of the unflagged rows. It used to be all unflagged rows, or
    the whole group, flagged rows included, when fewer than 10 were unflagged.
    When neither population has 10 rows, the flagged rows are `unavailable`
    (`no_reference`, warning `explain.reference_unavailable`) rather than
    explained against anomalies.
  - Isolation Forest attributions use interventional TreeSHAP with that
    reference as background, with shap's additivity check on. They used to be
    path-dependent with the check off.
  - KMeans centroid attributions are divided by each dimension's spread in the
    reference.
  - A fitting run saves the reference summary as
    `models/<run>/<group>/reference.json`; `score --from-run` explains against
    it, falling back with a warning when there is none.
  - Explanation values and, in some rows, reasons change from `explain@1`. The
    flags `treeshap_path_dependent` and `reference_includes_flagged_rows` are
    gone; `treeshap_path_length` and `reference_lower_half` are new.
  - Within one attribution source, gradient rows and marginal-deviation rows
    are judged separately when deciding whether a row's attribution is
    negligible, so a small real gradient is no longer discarded because the
    same source attributed a far-out row by marginal deviation.
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
- **Honest rolling-window trends** (`sorethumb history`, report):
  - Each trend states its basis. Under `composite` with a numeric
    contamination the rate is set by the review budget, so no change is shown.
  - The trend ends at the latest *elapsed* period with totals (new
    `--period`), and shows coverage per window.
  - Only groups present in both windows are compared (new `--all-groups`
    turns this off), and the composition is shown.
  - Rates carry Wilson 95% intervals and the change a Newcombe interval;
    `Δ %` is withheld below 10 prior anomalies. Low volume is flagged for every
    window, not only the shortest.
  - Comparability caveats now also cover mixed library versions, mixed
    ensemble policies, incomplete coverage and a changed group mix.
  - The HTML report's trend table is now filled in for runs with a period; it
    was never populated before.
  - **Per-period statistics** (migration 013, `period_stats`), recorded with each
    period's totals:
    - per original field: missing count, plus percentiles or top values;
    - per detector: score percentiles and flag count;
    - the composite threshold, contamination and flag count.
  - `sorethumb history` adds a **population drift** table. Per window it shows
    the change in row count and the fields that moved most by Population
    Stability Index, with conventional heuristic bands. No model is involved,
    so self-calibration cannot hide or invent a change.
  - **Anchor run** (`history.anchor_run`). Each period run also scores its
    rows with one fixed run's models, calibrators and flag rule. `sorethumb
    history` then splits each window's change in anomaly count exactly into
    volume, score drift (the rate above a fixed threshold moved) and threshold
    drift (self-recalibration moved the bar). It also shows anchor-scale score
    PSI and tail share.
    - Every fitting run saves its rule (`anchor_rule.json`, plain JSON).
    - Periods the anchor cannot score are left out, with the reason.
    - The anchor is not part of the config hash but is part of the run id, so
      setting it keeps one trend while giving re-run periods fresh runs.
  - **Four-section `sorethumb history`**: a verdict summary, then
    1 Population, 2 Scores, 3 Threshold and 4 Counts. Each section has a
    verdict per window: ok, caveats with reasons, or n/a with why.
    - `history --json` prints the whole report.
    - Each period run's HTML report shows the same sections per group, and its
      `index.json` carries them under `trend`.
    - New guide: [docs/trends.md](docs/trends.md).

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
  keeps query-differing sources distinguishable. Migration 009 irreversibly
  scrubs values stored by earlier versions and compacts the database (best
  effort: skipped with a warning if another process has it open); logs and
  reports written by earlier versions are not touched (see SECURITY.md). Without
  `source.dataset_id`, a run whose query key set differs from the dataset's
  previous run is refused, naming the `dataset_id` that keeps its history.
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
