# Limitations

What sorethumb does not do, and where its approximations bite. The
[README](../README.md#what-it-does-not-do) has the short version; this is the full
list, each point with the page that measures or explains it.

- `run.max_memory_mb` caps the *projected* feature-matrix size (rows × encoded
  columns × dtype bytes): a run whose estimate exceeds it aborts with
  `MemoryBudgetError` before any model is fitted. It is not a live RSS ceiling —
  Python cannot impose one — so actual peak memory can still exceed the budget.
- The composite score is an interpretable ranking score, not a calibrated probability.
- `contamination` is a review budget, not a prevalence estimate. What it does depends
  on the policy. Under a composite policy a number N means *exactly* the top N% of the
  ranking. Under the default `intersection` it means each detector's top N%, and only
  rows all of them flag are kept, typically far fewer than N%. At `"auto"` (the
  default), each detector uses its own heuristic boundary (see the realised rates in
  the run output); a composite then cuts at the median of those rates, which flagged
  several times a review budget in the backtests. None of this says how many genuine
  anomalies the data holds; only labelled data can tell you that.
- Detectors fit on the same (unlabelled) data they then score — there is no
  held-out "known normal" reference set, since that is what makes this
  unsupervised in the first place. A detector whose row cap is smaller than
  the group (e.g. `one_class_svm`'s default 25,000) fits on a random
  subsample but still scores every row, so a rare pattern absent from that
  subsample by chance may never shape the fitted boundary. See
  [docs/approximations.md](approximations.md).
- ECOD and HBOS are the only detectors with a truly exact attribution — their
  score is already an additive sum of per-feature terms, so decomposing it is
  zero-error by construction. Isolation Forest yields model-specific
  (interventional TreeSHAP) attributions, which decompose path length rather
  than the score itself; KMeans and One-Class SVM/LOF are heuristic (centroid
  distance, or a finite-difference gradient restricted to the two detectors
  whose score responds continuously to a small perturbation). See
  [docs/explanations.md](explanations.md).
- Feature reduction can discard exactly the signal that matters. Correlation
  pruning (`features.correlation_threshold`, default 0.95) drops one column
  from each highly-correlated pair entirely — invisible to every detector if
  the anomaly is specifically a *broken* relationship between that pair.
  PCA (opt-in, off by default) selects components by variance explained
  across the whole dataset, which is a property of the normal bulk, not of
  where a specific anomaly deviates — a run can clear
  `pca_min_explained_variance` with no warning and still have discarded the
  one dimension that would have flagged a particular row. See
  [docs/approximations.md](approximations.md).
- Robust scaling (the default) can leave a zero-inflated column — mostly
  zeros, with a real signal in the nonzero minority — effectively unscaled:
  its median and IQR are both 0, so the degenerate-column guard sets its
  scale to 1.0 instead of dividing by zero, leaving its raw magnitude to
  dominate or be drowned out in distance-based detectors regardless of how
  informative it actually is. See
  [docs/approximations.md](approximations.md).
- Self-calibration maps every run's scores to roughly uniform on [0, 1] by
  construction, so two independently-fitted runs — including the per-period runs
  `sorethumb backfill` produces — are not on a common scale. A raw rate trend
  mixes real change with each period's moving threshold. `sorethumb history`
  reports population drift without a model, and with `history.anchor_run` scores
  every period against one fixed run to separate score drift from threshold
  drift. For one-off comparisons, reuse a fitted run:
  `sorethumb score --from-run RUN_ID`.
- The default ensemble policy (`intersection@1`: every detector must flag a row, each at
  its own boundary under the default `scoring.contamination = "auto"`) is a deliberate
  design decision. It can be precise, but how many rows it flags is unpredictable: on the
  real datasets tested it ranged from 0.18x to 8.4x a review budget.
  Some anomaly types (local, varying-density, contextual) can go unflagged. If you have a
  review budget, the evidence favours `composite-agreement@1` with `scoring.contamination`
  set to that budget; never pair a composite with `"auto"`. Choose on measured evidence,
  not intuition: `sorethumb backtest` measures every policy against the default on your own
  labelled data, and a run with a non-default policy that lacks that evidence warns
  (`run.strict` refuses it). The default is not changing for now, by decision; the
  evidence, what it does and does not cover, and the reasons are in
  [docs/ensemble-policy.md](ensemble-policy.md#where-things-stand-2026-10-15).
- Unsupervised anomaly ≠ the thing you care about. The library ranks statistical oddity;
  whether an odd record is *interesting* is a domain judgement it cannot make.
- **A workspace is executable, not just data.** Persisted runs store fitted
  estimators and calibrators as `joblib`/pickle files, which execute arbitrary
  code on load. `sorethumb score --from-run` (and `run.reuse_models`, which
  reloads a persisted model within the same run) unpickle those files with no
  sandboxing. SHA-256 file digests catch accidental corruption or a
  misplaced/swapped file — they do **not** make an untrusted pickle safe,
  since a deliberately crafted malicious file carries its own matching
  digest. Only load a workspace you created yourself or that came from a
  source you fully trust; never point these at a workspace received from
  someone else without inspecting it first. See [SECURITY.md](../SECURITY.md).
- Fetching `source.uri` over http(s) is hardened against the obvious cases
  (a redirect pivoting to a cloud metadata endpoint or another internal
  host, an unbounded response), not a general-purpose sandbox for an
  untrusted remote server; a configured `Authorization` header is only ever
  sent over HTTPS to the exact origin `source.uri` names (credentials with
  an `http://` URI are rejected), and a redirect that would downgrade
  HTTPS to HTTP is refused outright — see
  [SECURITY.md](../SECURITY.md).
