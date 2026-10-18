# Trends: what `sorethumb history` can and cannot tell you

Each period run fits its own detectors and calibrates them on that period's
own rows. A trend across periods is therefore a sequence of **independently
calibrated snapshots**. One anomaly rate per period cannot say *why* it moved.
`sorethumb history` (and the trend section of each period run's report) splits
the question into four signals. Each has its own unit and its own verdict per
window.

| Section | Question | Unit | Needs |
|---|---|---|---|
| 1. Population | Did the input data change? | PSI per field; row count | per-period statistics |
| 2. Scores | Do the *same* models score the data differently? | share above a fixed threshold; score PSI | an anchor run |
| 3. Threshold | Did each period's own recalibrated threshold move? | own rate minus anchor-scale rate; rows | an anchor run |
| 4. Counts | How many rows did each period flag, and why did that change? | anomaly rate with 95% interval; rows | totals (anchor for the split) |

## Windows

Each figure compares a *current* window, the last *w* periods ending at the
reference period, with the *prior* window, the *w* periods before it
(`--window`, default `report.rolling_windows`).

- **Reference period.** The most recent period with totals that has already
  ended, not the one in progress. Pick another with `--period`.
- **Matched groups.** Only groups present in both windows are compared, so a
  group appearing or disappearing cannot move a figure on its own (a
  Simpson's-paradox artefact). `--all-groups` turns this off.
- **Coverage.** Periods with totals out of periods expected, per window.
  Weekend days are not expected when `history.roll_non_business` is set.

## 1. Population

Every period run records, per group, a summary of its input data:

- the row count;
- per original field, the missing count, plus either 101 percentiles (numbers,
  dates, booleans) or the 20 most common values and the rest.

The id and time columns are left out, since they change every period by
construction. Population drift is the **Population Stability Index** per field
between the two windows:

- Numbers use the earlier window's deciles; categories use the union of both
  sides' common values plus "other".
- Missing values are a bin of their own.
- Summaries are combined as a mixture weighted by row count.

PSI on numbers is reconstructed from the stored percentiles, accurate to about
a percentage point per bin. The bands (below 0.1 stable, 0.1–0.25 moderate,
above 0.25 large) are conventional heuristics, **not a significance test**.

No model is involved, so self-calibration can neither hide nor invent a
population change. A population change need not change anything the models
care about, though; section 2 says whether it did.

## 2. Scores, and the anchor run

Self-calibrated scores sit near-uniformly on [0, 1] in every period by
construction, so a period's own scores cannot show score drift. That takes a
**fixed scale**: set `history.anchor_run` to a completed fitting run of the
same dataset, and each period run also scores its rows with that run's models,
calibrators and fixed flag rule:

| `scoring.combination` | `scoring.contamination` | Fixed rule |
|---|---|---|
| `composite` | any | the anchor's weights and combined-score threshold |
| `intersection` / `union` | `"auto"` | each detector's own fitted boundary (fixed by the persisted model) |
| `intersection` / `union` | a number | each detector's calibrated cut-off in the anchor run |

Every fitting run saves its rule (`models/<run>/<group>/anchor_rule.json`), so
any complete fitting run can be an anchor. Section 2 shows:

- **Anchor rate**: the share of rows above the anchor's fixed threshold, with a
  Wilson 95% interval, and its change with a Newcombe interval.
- **Score PSI**: the current window's anchor-scale combined scores against the
  anchor run's own.
- **Above anchor p95**: the share of rows scoring above the anchor's own 95th
  percentile, which is 5% on the anchor's own rows.

A period the anchor cannot score is left out, with the reason listed: the
anchor has no model or rule for the group, its feature plan doesn't fit the
data, the feature schema changed, or a model file is corrupt. This never fails
the run.

`history.anchor_run` is not part of the config hash, so setting or changing it
keeps one trend. Periods already run are scored against it only when re-run
(`sorethumb backfill --force-period LABEL`, or `sorethumb run` for that
period). The anchor's models are unpickled on every period run, so the same
trust rules apply as for `score --from-run` (see [SECURITY.md](../SECURITY.md)).

## 3. Threshold

With an anchor, a period's own flag rate `r` (its own recalibrated threshold)
and its anchor-scale rate `r*` (a fixed threshold) are measured on the same
rows. The **gap** `r − r*` is how far the period's own threshold sits from the
anchor's. A change in the gap is **threshold drift**: self-recalibration
absorbing a real change, or creating an apparent one.

Under `composite` with a numeric contamination, the own rate is fixed by the
review budget. Threshold drift then mirrors score drift by construction: every
real change is absorbed. The verdict says so.

## 4. Counts

The own anomaly rate per window, volume-weighted (anomalies over rows), with a
Wilson 95% interval. The change has a Newcombe 95% interval, and `*` marks an
interval that excludes 0. These intervals treat rows as independent and cover
sampling noise only. `Δ %` is withheld when the prior window has fewer than 10
anomalies.

Under `composite` with a numeric `contamination`, every period flags the same
fraction, so the rate is set by the review budget and no change is shown.

With an anchor, the change in count splits exactly into three terms (`N` rows,
`r` own rate, `r*` anchor rate):

```
Δcount = (N_t − N_{t−1})·r_{t−1}                               volume
       + N_t·(r*_t − r*_{t−1})                                 score drift (fixed threshold)
       + N_t·[(r_t − r*_t) − (r_{t−1} − r*_{t−1})]             threshold drift
```

The terms are computed per group and summed. Only cells (group, period)
scored against one anchor on both sides are used, and the terms add up
exactly to the change in those cells. More than one anchor in the span gives
no split.

## Verdicts

Each section has a verdict per window:

- **ok**: nothing known to weaken the comparison.
- **caveats**: comparable, with reasons listed. For example:
  - `group_mix_changed` or incomplete coverage;
  - `other_config_in_span`: another configuration has totals in the span,
    excluded here;
  - `mixed_library_versions` or `mixed_ensemble_policies` across the span;
  - `low_volume`: below `history.low_volume_threshold`;
  - cells left out of the anchor split;
  - a budget-fixed rate.
- **n/a**: the figure cannot be computed. For example: no per-period statistics
  (periods run before they existed), no anchor run set, or no period scored
  against it.

## Machine-readable output

`sorethumb history --json` prints the whole report: per window, the population,
anchor and count figures and the four verdicts with their reasons. Each period
run's `index.json` report carries the same object per group under `trend`.

## Limits

- **The anchor is a fixed scale, not ground truth.** A model fitted on one
  period encodes that period's notion of normal, including any anomalies it
  contained. Choose an anchor period you accept as representative, and re-anchor
  deliberately, not as a reaction to the trend.
- **Intervals cover sampling noise only.** Within a period, rows of one group
  are rarely independent; treat the intervals as a lower bound on uncertainty.
- **PSI is reconstructed from summaries** and is accurate to about a percentage
  point per bin. Its bands are conventions.
- **Periods run before a feature existed don't have it.** Per-period statistics
  and anchor scores are recorded from the run that introduced them onwards. Re-run
  a period to add them.
