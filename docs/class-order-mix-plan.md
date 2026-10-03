# Within-class order at the matched horizon: which class carries the learned order's loss, and does recency carry information beyond the bit?

Status: pre-registration. No replay named below has been run. Committed
before any implementation, with the [evaluation-window](tail-window-check-plan.md)
and [leaf-eligibility](leaf-matched-horizon-plan.md) plans; the
implementation is reviewed and committed before the run.

## Why

The [matched class-order control](matched-horizon-findings.md) gave the
frozen `next_use` ranker the exact reuse class at the cell's matched horizon
`h*` on residents. Within that class the ranker's order was a consistent loss
against recency in 8/12 trace × cell (−0.3 to −1.1 points), where within the
600-second class it had been a gain in 9/12; recency within the matched
class reached the exact label's eviction gain in every new cell. Two things
are not separated by those two arms:

- whether the learned order's loss sits among the candidates that are reused
  within `h*` (the reusable class, where the exact label would order by next
  use) or among those that are not (the non-reusable class, whose next use
  is beyond `h*` or absent);
- whether recency within a class carries selection information beyond the
  bit, or the bit alone does the work and the learned order only adds harm.

## Question and predictions

With admission by the frozen ranker, the exact class at `h*` on residents and
the sampling unchanged: in which class does the learned order lose to
recency, and how does a uniformly random order within the class compare with
both?

Predictions, fixed before the run, on five-seed means:

1. The loss sits in the reusable class: in each of the eight trace × cell
   where the learned order within the matched class was a consistent loss
   (conversation 0.25%×1, 0.25%×4, 1%×1, 2%×1; tool-agent 0.25%×1, 1%×1,
   1%×4, 2%×1), `D_in < 0` and `D_out > D_in`, where `D_in` and `D_out` are
   defined below.
2. Recency carries information beyond the bit: `D_rand < 0` in at least 8 of
   the 12 trace × cell, consistently over seeds.

## Fixed surface

- Traces, capacities, seeds 0–4, L1, L2 store, hit rule, sizes, mechanism
  `all16`, hooks and decision statistics, `h*` per cell: as the matched
  class-order control.
- Arms, all with admission by the frozen `next_use` ranker through
  `errorloc.HybridOverride` unchanged and the store's key
  `((reusable within h*, within-class key), last_group)`:
  - `mix_out_learned`: the within-class key is the ranker's score for a
    non-reusable candidate and a constant for a reusable one, so the
    non-reusable class is ordered by the ranker and the reusable class by
    recency;
  - `mix_in_learned`: the reverse, the reusable class by the ranker and the
    non-reusable class by recency;
  - `random_within_class`: the within-class key is a uniform draw in [0, 1)
    per candidate per decision, from a random stream seeded by the replay
    seed and the arm name and separate from the store's sampling stream,
    so the candidate sets are those the recency arm would see under the
    same store.
  The ranker inside the mixed arms is the one object the override consults,
  observed once, as in `horizonctl.ReuseClassScorer`. 3 arms × 12 trace ×
  cell × 5 seeds = 180 replays.
- References, published rows and not rerun: `learned`, `evict_label`
  (`results/paper/error_location_001/`); `evict_binary_h*_learned` and
  `evict_binary_h*_recency` (`results/paper/matched_class_order_001/`);
  `label_binary_h*` (`results/paper/horizon_control_001/`, `horizon_fill_001/`).
- Differences, seed-paired, in points: `D_out = U(mix_out_learned) −
  U(evict_binary_h*_recency)`, `D_in = U(mix_in_learned) −
  U(evict_binary_h*_recency)`, `D_rand = U(random_within_class) −
  U(evict_binary_h*_recency)`, `D_learned = U(evict_binary_h*_learned) −
  U(evict_binary_h*_recency)` (published).

## Required checks

- Identifiers of every replay equal the published reference rows'; the
  Phase 0.98b identities hold; the class statistic of the matched control
  (no resident eviction discards a state reusable within `h*` while a
  sampled resident is not; no override outside a first round) is zero in
  every replay.
- Unit tests on constructed traces, decision by decision: a mixed scorer
  with the ranker on both classes equals `evict_binary_h*_learned`, with the
  ranker on neither equals `evict_binary_h*_recency`; the random arm's
  candidate sets equal the recency arm's on a trace where both make the
  same choices, and two replays of the random arm with the same seed are
  identical while the sampling stream is untouched (the recency arm replayed
  beside it is unchanged).
- No existing replay path or runner is changed; the existing tests pass.

## Readings, fixed before the run

Five-seed means; "consistent" means the same sign in all five seed-paired
values. Every value of every arm is reported, with `R_h*` of each.

1. **Which class.** Per trace × cell: `D_in`, `D_out`, their seed signs and
   readings; the count of the eight cells named above in which `D_in < 0`
   and `D_out > D_in` (prediction 1), and the same two conditions over all
   12 descriptively. Also whether `D_in + D_out` is within the seed interval
   of `D_learned` (additivity, descriptive).
2. **Recency beyond the bit.** Per trace × cell: `D_rand`, its seed signs and
   reading; the count of consistent losses out of 12 (prediction 2); and
   the seed-paired comparison of the random arm with the learned order,
   `U(random_within_class) − U(evict_binary_h*_learned)`, descriptive, which
   says whether the learned order is above, at or below a random order.

## Interpretation boundaries

- `h*` is the best of grids run on these traces, read after the fact; the
  bit is exact; the comparison is sampled-16 greedy, not an optimum.
- "Not reusable within `h*`" includes states reused after `h*`; nothing here
  says they are worthless.
- The random arm describes one random stream per seed; seed intervals
  describe sampling-seed and random-stream variability together.
- Every arm reads the trace's future and none is a policy; nothing here
  triggers a feature, a target or a fit.

## Execution order

1. Commit this plan with its two companions, alone.
2. Implement a separate module and runner on the matched class-order
   runner's replay call; tests on constructed traces; no change to an
   existing path. Review and commit before any replay.
3. No separate smoke; the run carries the identifier and class checks.
4. Run once with 10 workers into `results/paper/class_order_mix_001/`,
   recording plan and code commits and the hashes of the traces and of every
   table read.
5. Report, including failed predictions.
