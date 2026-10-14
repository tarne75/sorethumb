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
<!-- AUTO-GENERATED — do not edit manually; run `python scripts/run_backtest.py --phase confirmation` to add a result. -->

| pre-registration | candidate | baseline | primary metric | verdict | seeds |
| --- | --- | --- | --- | --- | --- |
| [`composite-agreement-v1`](benchmark-results/backtests/composite-agreement-v1.json) | `composite-agreement@1` | `intersection@1` | flag_f1 | **does not outperform** | 20 |
| [`discrete-composite-v1`](benchmark-results/backtests/discrete-composite-v1.json) | `rule:discrete-composite@1` | `intersection@1` | flag_f1 | **does not outperform** | 20 |

**`composite-agreement-v1`** -- Thresholding the agreement-weighted composite score at the review budget improves
flag F1 over unanimity on the anomaly types where the ensemble ranking is good but the
unanimous flag under-flags: point, clustered, masking and irrelevant_columns.

| hypothesis scenario | mean difference | 95% CI | p (Holm) | improved |
| --- | --- | --- | --- | --- |
| point | +0.5244 | [+0.4475, +0.5938] | 0.0008 | yes |
| clustered | +0.3108 | [+0.2463, +0.3807] | 0.0008 | yes |
| masking | +0.2560 | [+0.2281, +0.2829] | 0.0008 | yes |
| irrelevant_columns | +0.1539 | [+0.1031, +0.2038] | 0.0008 | yes |

Why it does not outperform:

- budget regression on local: max_flag_false_positive_rate 0.05 met by the baseline (0.0145) but not the candidate (0.0540)
- budget regression on swamping_clean: max_flag_false_positive_rate 0.05 met by the baseline (0.0080) but not the candidate (0.0533)
- budget regression on swamping_contaminated: max_flag_false_positive_rate 0.05 met by the baseline (0.0127) but not the candidate (0.0533)
- budget regression on varying_density: max_flag_false_positive_rate 0.05 met by the baseline (0.0140) but not the candidate (0.0525)

**`discrete-composite-v1`** -- On data whose kept columns are all categorical or boolean, unanimity flags almost none of
the anomalies even when the ranking is reasonable. Switching to the equal-weight composite
there, and only there, improves flag F1 on all_categorical; on every other scenario the
rule runs the baseline unchanged.

| hypothesis scenario | mean difference | 95% CI | p (Holm) | improved |
| --- | --- | --- | --- | --- |
| all_categorical | +0.1084 | [+0.0755, +0.1395] | 0.0002 | yes |

Why it does not outperform:

- budget regression on all_categorical: min_average_precision 0.15 met by the baseline (0.1693) but not the candidate (0.1490)
<!-- backtest-results-end -->

### What the confirmations show (2026-10-14)

Neither candidate outperforms `intersection@1`, and both results came out as their
pre-registrations predicted from the development seeds:

- **`composite-agreement@1`** catches far more anomalies where the ensemble ranks them
  well (flag F1 +0.15 to +0.52 on point, clustered, masking and irrelevant columns). But
  a composite always fills the review budget. Where it cannot rank the anomalies (local,
  varying density, the swamping pair), it fills the budget with normal rows, and its
  false-positive rate rises from 0.01 to 0.054, past the 0.05 budget the default meets.
  This is the trade the default is designed not to make.
- **`rule:discrete-composite@1`** improves flag F1 on all-categorical data (+0.11) with
  no false-positive, volume or stability cost, and is identical to the default
  everywhere else. It fails on one budget: the composite's *ranking* is slightly worse
  there than the default's combined score (average precision 0.149 against the 0.15
  budget, where the default reaches 0.169). The gain comes from where the threshold sits,
  not from a better ranking. A later candidate could test that observation under its own
  pre-registration; this result cannot be reinterpreted as a win.

## Versioning

A policy's version identifies its behaviour. Any change to which rows a policy flags or
the score it produces, a bug fix included, bumps the version, and a golden test pins
each version's output so this cannot happen silently. A run records the id it used, so
results stay attributable after the code moves on.

## Changing the default

The default changes only when a candidate policy wins under a pre-registered,
backtested comparison, in a minor version, with a CHANGELOG entry. Never as a side
effect of a benchmark run.
