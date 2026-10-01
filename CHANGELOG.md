# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-21

First public release. There is no prior published version to diff against,
so the Fixed/Security/Compatibility entries below record corrections made
during pre-release hardening rather than changes from an earlier release.

### Added

- **Core pipeline**: profile every column, encode/impute/derive features,
  fit an ensemble of anomaly detectors, percentile-calibrate their scores,
  combine them into a ranking and a flag decision, and explain each flagged
  row in terms of the original columns.
- **Detectors**: `isolation_forest`, `kmeans_distance` (CBLOF-style), and
  `one_class_svm` form the default three-detector ensemble; `ecod`, `hbos`,
  and `lof` are also available. A detector `extra_params` escape hatch
  forwards arbitrary constructor keywords to the underlying scikit-learn
  estimator.
- **Ensemble combination**: `intersection` (the default — every configured
  detector must independently flag a row; see Known limitations),
  `union`, and `composite` (a weighted average with an anti-correlated-member
  guard and an optional agreement-based weighting scheme).
- **Explanations**: a per-record, original-column reason for every flagged
  row — exact for ECOD/HBOS (their score is already an additive sum of
  per-feature terms), TreeSHAP-based (`model_specific`) for Isolation
  Forest, and heuristic (centroid distance or finite-difference gradient)
  for the rest.
- **CLI**: `run`, `inspect`, `anomalies`, `report`, `history`, `backfill`,
  `score --from-run`, `benchmark`, `init`, `config`, and `workspace`,
  backed by a SQLite-based workspace store with resumable, idempotent,
  ledger-tracked execution.
- **`sorethumb score --from-run RUN_ID`**: score new data against a
  previously fitted run's persisted models and calibrators with no
  refitting, so scores stay on one comparable scale across calls (see
  Compatibility for what this requires of the calling config).
- **`sorethumb report`**: re-render a run's HTML/JSON/CSV report purely
  from persisted state, independent of the process that produced the run.
- **`sorethumb history` / `sorethumb backfill`**: rolling-window trend
  aggregation and historical-period backfilling, both scoped correctly to
  dataset identity and the exact configuration that produced each period.
- **Two benchmark harnesses**, both run via `scripts/run_benchmark.py` (a
  maintainer script that needs explicit `--output-dir`/`--readme` paths; it
  is deliberately not part of the `sorethumb` command): a
  full-pipeline synthetic-scenario suite (point/local/contextual/clustered/
  masking/swamping/varying-density anomaly types through the real feature
  pipeline) and a real-dataset (KDDCup99, Covtype) plus legacy-synthetic
  suite — each with committed, CI-gated accuracy floors.
- **Packaging**: published to PyPI as `sorethumb-ml` (see Compatibility);
  ships `py.typed`; a tag-triggered, manually-approved publish workflow,
  with a rehearsal path against TestPyPI.

### Changed

- **Statistical framing**: `sorethumb` is a ranker and review-budget
  selector, not a prevalence estimator. `contamination` is documented and
  surfaced throughout as a review budget, not a measurement of how many
  anomalies the data contains; realised per-detector flag rates are always
  shown alongside the flagged count.
- **Default ensemble ranking fix**: the `intersection` combination's
  continuous ranking now uses the per-row median across detectors instead
  of `min()`, so one badly-misranking detector can no longer drag the whole
  ensemble's ranking below random. The *flag* decision itself (an AND of
  every detector's independent vote) is unchanged by this and stays
  deliberately conservative — see Known limitations.
- The zero-config default workspace moved from the current directory to a
  dedicated `./sorethumb-workspace/`, so a first run never scatters files
  beside the source data.
- `kmeans_distance` scores CBLOF-style (distance to the nearest
  *large*-cluster centroid), so a tight anomaly cluster can no longer
  capture its own centroid and be scored as normal.
- `Calibrator` supports self-calibration only; the never-reachable
  "reference" mode was removed.

### Removed

- The unsupported `s3://` source-URI example (only local paths and
  `http(s)://` were ever actually supported).
- Several confirmed-dead code paths left over from the history/totals
  redesign.

### Fixed

Pre-release hardening surfaced and fixed real issues across the codebase
before anything shipped:

- **Attribution scale and background come from a normal reference**: the gradient and KernelSHAP explainers now take target rows separately from a required `reference` matrix. Perturbation step sizes and the KernelSHAP background were previously derived from the flagged rows themselves, so an explanation changed with the other rows flagged in the same run and a lone flagged row had a zero-variance fallback step. The pipeline passes the unflagged rows as the reference (the whole population when fewer than 10 are unflagged) and computes attributions only for flagged rows.
- **Attribution backend failures are reported**: when a detector's attribution backend raised, the failure was logged at debug level only and the detector silently contributed nothing. It now emits one `AttributionBackendWarning` per detector and group (cause in the message, the traceback in a warning-level log record); other detectors' explanations for the group are preserved, and `run.strict` promotes the warning to a group failure like any other project warning.
- **Cohort contrast is comparable across column kinds**: `compute_contrast` now scores numeric columns by the two-sample KS statistic and categorical columns by total variation distance, both bounded in `[0, 1]` and independent of cohort size, so one table can rank both. The previous scores (|Cohen's d| + KS, unbounded, versus a max category lift that used `frequency x unflagged row count` for categories absent from the unflagged cohort) were not comparable and let weak categorical evidence outrank strong numeric separation. `stat_name` is now `ks_statistic` / `total_variation`.
- **Missing and non-finite values**: float `NaN` is now treated as missing
  (imputed like a null) instead of leaking into scaling, and the feature
  matrix no longer replaces anything non-finite with `0.0` after scaling --
  that made a `+Inf`/`-Inf` row look perfectly typical. Infinities in
  columns that feed the matrix, and impossible derived values such as a
  `float32` overflow, now fail with a clear `PlanError`.
- **Data identity and history correctness**: row identity (`id_column`) is
  validated for existence/uniqueness before any filtering; a group
  selector matching nothing now fails loudly instead of silently
  completing; history and backfill completion are now atomic and scoped to
  the exact configuration that produced them; dataset identity survives
  the source file growing over time.
- **Model persistence integrity**: persisted models and calibrators are
  digest-verified and namespaced per detector; a score-forward source run
  must be genuinely complete and not itself a score-forward run; all
  model-side writes are atomic.
- **Config and CLI correctness**: generated `sorethumb.toml` files are now
  always valid TOML; `--detectors` no longer silently rewrites the rest of
  an existing config file to defaults; previously-inert configuration
  fields are now wired up; one canonical `config_hash` is used everywhere
  instead of two silently-different values.
- **Explanation correctness**: a detector with zero ensemble weight can no
  longer supply a row's entire displayed explanation; rows beyond
  `explain.max_rows` are marked unavailable instead of given a fabricated
  reason; a PCA back-projection failure fails closed instead of
  mislabelling; ECOD/HBOS get exact native attributions instead of a noisy
  finite-difference approximation.
- **HTTP source fetching**: the documented `read_options` schema-inference
  override now actually works; HTTP caching is validator-aware (ETag/
  Last-Modified, genuine 304 reuse) instead of a "cache hit" that still
  fully re-downloaded every time.
- **HBOS scoring**: a value far outside the fitted histogram range is now
  treated as unseen (scored at the density floor) instead of inheriting a
  dense edge bin's score.
- **Benchmark evidence**: the `swamping` scenario now uses a genuinely
  matched clean/contaminated comparison (see Known limitations for what it
  found); the harness refuses to publish an incomplete or errored result
  matrix; real-dataset numbers come from a reproducible, provenance-carrying
  run instead of an empty placeholder.
- **Packaging**: the source distribution no longer ships local dev-tool
  state or caches; macOS is now actually integration-tested, matching its
  advertised support.

### Security

- CSV report output is neutralised against spreadsheet formula injection.
- `sorethumb workspace reset` refuses to delete a target unless it
  genuinely opens as a sorethumb workspace, and always refuses an
  obviously-wrong target (a filesystem root, home directory, git repo
  root) even if one happened to contain a marker file.
- Authenticated HTTP(S) downloads (`source.auth`) scope the `Authorization`
  header to the exact configured origin — it is never sent across a
  redirect to a different host or port, and an HTTPS→HTTP downgrade
  redirect is refused outright.
- **Persisted-model trust boundary** (read before using `score --from-run`
  or `run.reuse_models` against a workspace you didn't create yourself): a
  workspace's fitted models are `joblib`/pickle files, unpickled with no
  sandboxing — this is arbitrary code execution, not safe data loading.
  The SHA-256 digests checked on load catch corruption or a swapped file;
  they are **not** a security boundary against a deliberately malicious
  file. See [SECURITY.md](https://github.com/tarne75/sorethumb/blob/main/SECURITY.md).

### Known limitations

- **Default `combination="intersection"`** requires every configured
  detector to independently agree before flagging a row. Even after the
  ranking fix above, this stays deliberately conservative on some data
  shapes and can produce very few (or zero) true-positive flags — a
  consequence of requiring unanimous agreement among individually-weaker
  boundaries, not a bug. See `docs/approximations.md`.
- **The `swamping` benchmark scenario** found no false-positive-inflating
  effect from unlabelled training contamination in this pipeline —
  measured properly, the effect runs the other way (a mild masking /
  reduced-sensitivity effect). This is reported as an honest empirical
  finding, not the scenario's originally-assumed premise.
- Benchmark evidence in the README covers only the shipped default
  ablation for the full-pipeline suite, and real-dataset (KDDCup99/Covtype)
  numbers are capped at 20,000 rows for practical CI runtime — both
  disclosed inline where the tables appear.
- **Windows is untested and known-broken** (a real, unresolved `OSError`
  across most CLI paths, confirmed by a rehearsal run) — see the README's
  and SECURITY.md's Supported platforms sections.
- Every detector fits on the same, unlabelled data it then scores — there
  is no held-out "known normal" reference set, since that is what makes
  this unsupervised in the first place. See `docs/approximations.md` for
  this and other approximation-driven limitations (PCA, correlation
  pruning, capped-detector train/score mixing, zero-inflated scaling).

### Compatibility

- **PyPI distribution renamed to `sorethumb-ml`** (`pip install
  sorethumb-ml`) — `sorethumb` on PyPI is an unrelated package. The CLI
  command, GitHub repo, and every on-disk convention (`sorethumb.toml`,
  `sorethumb.db`, `sorethumb.log`, the `sorethumb-workspace/` directory)
  are unaffected.
- **Import package renamed to `sorethumb_ml`** (`import sorethumb_ml`,
  previously `import sorethumb`) to match the PyPI distribution name and
  avoid the two diverging. The `sorethumb.detectors` third-party
  entry-point group is likewise now `sorethumb_ml.detectors`. The CLI
  command stays `sorethumb`.
- **`score --from-run` compatibility**: a config that claims different
  fit-time settings than what the source run actually persisted
  (`columns`/`profiling`/`features`, or a fit detector's `params`/
  `train_row_cap`) is now rejected outright, rather than silently ignored
  in favour of what was actually loaded.
- Identity digests widened from 64-bit to 128-bit (`run_id`, `group_key`,
  `dataset_fp`, `config_hash`); no migration is needed, since nothing
  shipped before this release for a workspace to migrate from.
- `attribution_kind == "exact"` is reserved for ECOD/HBOS's native
  decomposition; Isolation Forest/TreeSHAP results read `"model_specific"`
  instead.

[Unreleased]: https://github.com/tarne75/sorethumb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tarne75/sorethumb/releases/tag/v0.1.0
