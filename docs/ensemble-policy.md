# Ensemble policy: the default is a design decision

sorethumb combines several detectors into one flag decision. The rule it uses is the
*ensemble policy*, identified as `name@version` (for example `intersection@1`) and
recorded on every run.

## The default: `intersection@1`, by design

A row is flagged only when every configured detector flags it on its own boundary.
This is the default deliberately, not by accident:

- **Precision first.** A flagged row is one every detector independently found unusual,
  so a short review list is mostly worth reviewing.
- **Bounded false positives.** No single detector's quirks can add rows on its own.
- **Explainable.** "All three detectors agreed" is a reason a reviewer can act on.

## The limitations it accepts

Unanimity is conservative. On some anomaly types it flags few or none of the true
anomalies even when the combined ranking is reasonable. The full-pipeline benchmark
([benchmarks.md](benchmarks.md)) shows flag recall of 0.0 on `local` and
`varying_density` and about 0.02 on `contextual`; see
[approximations.md](approximations.md). These shortfalls are accepted in exchange for
the precision above. They are not hidden, and they are not a reason to change the
default silently.

## The evidence

The [ensemble policy matrix](benchmarks.md#ensemble-policy-matrix) measures every
registered policy on every benchmark scenario: precision and recall of the actual
flags, ranking, stability under refits, and the false-positive budget. Each cell
is checked against versioned per-anomaly-type budgets
([`tests/benchmark/budgets.toml`](../tests/benchmark/budgets.toml)). Every budget
the default misses is listed there under `accepted_limitations` with its reason,
and a scheduled CI job fails if the default misses one that is not listed, or
meets one that still is.

## Choosing a different policy

`scoring.combination` (and `scoring.weighting` for `composite`) selects another
registered policy: `union@1`, `composite-equal@1`, `composite-agreement@1` or
`composite-manual@1`. Choose one on evidence, not intuition: measure the candidate
against the default on labelled data that resembles yours, comparing the precision and
recall of the actual flag decision, the ranking (average precision), stability across
refits, and the false-positive rate, and prefer a policy only where it wins clearly
without breaching a false-positive budget the default meets.

## Backtests: when a candidate may be said to outperform

A benchmark matrix shows how policies behave; it cannot by itself justify switching,
because a rule tuned while looking at those numbers will look good on them. A candidate
policy, or a data-dependent *candidate rule* (`rule:name@version`, which picks a
registered policy from properties of the fitted feature plan, never from labels or
scores), is judged by a fixed protocol (`sorethumb_ml.evaluate.backtest`):

1. **Pre-registration.** The comparison is written down and committed before it is run
   on confirmation data: candidate and baseline, the primary metric, the hypothesis
   scenarios, whether all or any of them must improve, the significance level and the
   stability tolerance
   ([`tests/benchmark/preregistrations/`](../tests/benchmark/preregistrations/)). The
   result records the file's sha256 and commit, so it cannot be edited afterwards
   unnoticed.
2. **Seed split.** Candidates are developed on the development seeds (the policy
   matrix's, 0-9) and judged once on 20 disjoint confirmation seeds (10000-10019). A
   confirmation run is refused for an uncommitted pre-registration, and refused again
   once a result exists: a revised hypothesis is a new pre-registration.
3. **Paired statistics.** Candidate and baseline run on the same seeds, so on the same
   data and split. Each hypothesis scenario's per-seed differences in the primary metric
   get a paired bootstrap estimate (10,000 resamples), and the p-values are
   Holm-corrected across the hypothesis scenarios.
4. **Decision rule.** The candidate *outperforms* only if the required hypothesis
   scenarios improve (positive mean difference, Holm-adjusted p ≤ alpha), it meets on
   every scenario each budget the baseline meets, and its stability is nowhere more than
   the tolerance below the baseline's.

A candidate that outperforms is eligible to ship as a supported, versioned option with
its evidence attached. It does not become the default by this process alone (see
below). Candidate rules stay out of the configuration until they win.

### Results

<!-- backtest-results-start -->
<!-- backtest-results-end -->

## Versioning

A policy's version identifies its behaviour. Any change to which rows a policy flags or
the score it produces, a bug fix included, bumps the version, and a golden test pins
each version's output so this cannot happen silently. A run records the id it used, so
results stay attributable after the code moves on.

## Changing the default

The default changes only when a candidate policy wins under a pre-registered,
backtested comparison, in a minor version, with a CHANGELOG entry. Never as a side
effect of a benchmark run.
