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

## Choosing a different policy on your data

`scoring.combination` (and `scoring.weighting` for `composite`) selects another
registered policy: `union@1`, `composite-equal@1`, `composite-agreement@1` or
`composite-manual@1`. Choose one on evidence, not intuition (`composite-agreement@1`
has published backtest evidence on the synthetic suite; see
[Round 2](#round-2-2026-10-14-budgets-v3)). That evidence is about known anomaly
types, though. Whether a policy is better on *your* data is measured by
`sorethumb backtest` ([CLI reference](cli_reference.md#sorethumb-backtest)):

1. Set `columns.reference_column` to a 0/1 label column (used only for evaluation,
   never for fitting) and `scoring.contamination` to the fraction of rows you can
   review.
2. Run `sorethumb backtest`. It fits your configured pipeline on random subsamples
   of your rows, applies the default and every candidate to the same fits, and
   decides each candidate with the same rule as the synthetic backtests:
   significantly higher flag F1 (paired bootstrap, Holm-corrected across all the
   candidates tested at once, so trying several and keeping the best is accounted
   for), no more wrong flags than the review budget where the default stays within
   it, and flags no less stable.
3. Each verdict is recorded in the workspace for this dataset and this
   configuration (everything but the policy itself and settings that cannot change
   a flag).

A `run` or `score` with a non-default policy then looks for an `outperforms` record
in that scope and records its id on the run (`policy_evidence_id`, also shown by
`sorethumb show` and the run summary; `sorethumb evidence` lists every record and
which apply to your current configuration). Without one, the
run warns and proceeds; under `run.strict` it refuses before anything is registered.
The policy is never switched for you: the choice stays yours, made visible and
checkable. Without labels, `sorethumb backtest` reports flag volume and stability
only; its `not_assessable` verdict does not satisfy the check, because accuracy was
not measured.

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
   matrix's, 0-9) and judged once on 20 disjoint confirmation seeds. Confirmations run in
   rounds: once a round's results are published its seeds have been seen, so the next
   round uses fresh ones (round 1: 10000-10019, closed; round 2: 20000-20019, open). A
   confirmation run is refused for an uncommitted pre-registration, for a closed round,
   and again once a result exists: a revised hypothesis is a new pre-registration.
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
| [`composite-agreement-v2`](benchmark-results/backtests/composite-agreement-v2.json) | `composite-agreement@1` | `intersection@1` | flag_f1 | **outperforms** | 20 |
| [`discrete-budget-v1`](benchmark-results/backtests/discrete-budget-v1.json) | `rule:discrete-budget@1` | `intersection@1` | flag_f1 | **does not outperform** | 20 |
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

**`composite-agreement-v2`** -- Re-test of composite-agreement-v1 under budgets v3, which state the false-positive bound in
whole rows (a policy may spend its review budget, never more) instead of a rate over normal
rows that an exact-budget policy could breach with no extra flags. Same hypothesis: the
agreement-weighted composite improves flag F1 over unanimity on point, clustered, masking
and irrelevant_columns. Judged on fresh seeds, because round 1's have been seen.

| hypothesis scenario | mean difference | 95% CI | p (Holm) | improved |
| --- | --- | --- | --- | --- |
| point | +0.4514 | [+0.3659, +0.5335] | 0.0008 | yes |
| clustered | +0.4181 | [+0.3473, +0.4867] | 0.0008 | yes |
| masking | +0.2362 | [+0.2070, +0.2645] | 0.0008 | yes |
| irrelevant_columns | +0.1415 | [+0.0864, +0.1967] | 0.0008 | yes |

**`discrete-budget-v1`** -- discrete-composite-v1 improved flag F1 on all-categorical data but ranked worse than the
default (AP 0.149 vs 0.169): the gain came from where the flag threshold sits, not from the
composite's ranking. This rule keeps the default's ranking (intersection@1's combined score
over every detector) and on all-discrete data flags exactly the review budget's top rows by
it. Hypothesis: flag F1 on all_categorical improves over the default with average precision
unchanged; every other scenario runs the default unchanged.

| hypothesis scenario | mean difference | 95% CI | p (Holm) | improved |
| --- | --- | --- | --- | --- |
| all_categorical | +0.1721 | [+0.1320, +0.2133] | 0.0002 | yes |

Why it does not outperform:

- stability regression on all_categorical: flag_jaccard_stability 0.6970 vs baseline 0.7600 (tolerance 0.05)

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

### Round 1 (2026-10-14, budgets v2)

Neither candidate outperforms `intersection@1`, and both results came out as their
pre-registrations predicted from the development seeds:

- **`composite-agreement@1`** catches far more anomalies where the ensemble ranks them
  well (flag F1 +0.15 to +0.52 on point, clustered, masking and irrelevant columns). But
  a composite always fills the review budget. Where it cannot rank the anomalies (local,
  varying density, the swamping pair), it fills the budget with normal rows, and its
  false-positive rate rises from 0.01 to 0.054, past the 0.05 budget the default meets.
  Most of that breach, though, is in how budgets v2 stated the bound (see the correction
  below).
- **`rule:discrete-composite@1`** improves flag F1 on all-categorical data (+0.11) with
  no false-positive, volume or stability cost, and is identical to the default
  everywhere else. It fails on one budget: the composite's *ranking* is slightly worse
  there than the default's combined score (average precision 0.149 against the 0.15
  budget, where the default reaches 0.169). The gain comes from where the threshold sits,
  not from a better ranking. A later candidate could test that observation under its own
  pre-registration; this result cannot be reinterpreted as a win.

**Correction after round 1: budgets v3.** The documented principle is that a policy's
wrong flags may not exceed its review budget. Budgets v2 encoded it as a false-positive
*rate* over normal rows capped at the review budget, which is a fraction of *all* rows.
`composite-agreement@1` flags exactly the budget, rounded to whole rows (13 of 252). Where
none of those rows is an anomaly, its rate is 13/240 = 0.054: a breach with no extra
flags at all. Budgets v3 states the bound in rows (`false_positive_budget_ratio` ≤ 1.0).
The round-1 verdicts stand: they were judged under v2, which is archived byte-for-byte
(`tests/benchmark/budgets-archive/v2.toml`). The fix is tested only through new
pre-registrations on round 2's fresh seeds. A bound changed after seeing a result is
exactly what this protocol guards against, so the change gets no retroactive effect.

### Round 2 (2026-10-14, budgets v3)

Both verdicts went against the expectations pre-registered from the development seeds,
which is why confirmation is a separate step:

- **`composite-agreement@1` outperforms `intersection@1`** (`composite-agreement-v2`).
  Flag F1 improves by +0.14 to +0.45 on point, clustered, masking and irrelevant
  columns, and it meets every budget the default meets, stability included. The
  average-precision and stability dips seen on the development seeds did not recur on
  the fresh seeds. What it trades away is unchanged: it always spends the review
  budget, so where the ensemble cannot rank the anomalies its flags are mostly normal
  rows, still within the budget. It is now a policy with published evidence. The
  default stays `intersection@1` by design (see below), and choosing the composite for
  your data should rest on evidence from data like yours.
- **`rule:discrete-budget@1` does not outperform** (`discrete-budget-v1`). Keeping the
  default's ranking and flagging the review budget's top rows on all-discrete data
  improves flag F1 by +0.17 with average precision unchanged. But its flagged set is
  less stable across refits (Jaccard 0.70 against the default's 0.76, beyond the 0.05
  tolerance). Rows near the budget cut-off change between refits, which a unanimous
  vote mostly avoids by flagging fewer rows.

## Versioning

A policy's version identifies its behaviour. Any change to which rows a policy flags or
the score it produces, a bug fix included, bumps the version, and a golden test pins
each version's output so this cannot happen silently. A run records the id it used, so
results stay attributable after the code moves on.

## Changing the default

The default changes only when a candidate policy wins under a pre-registered,
backtested comparison, in a minor version, with a CHANGELOG entry. Never as a side
effect of a benchmark run.
