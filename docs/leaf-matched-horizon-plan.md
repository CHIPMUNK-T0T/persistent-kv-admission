# The matched-horizon bit and the class order under leaf eligibility

Status: pre-registration. No replay named below has been run. Committed
before any implementation, with the [evaluation-window](tail-window-check-plan.md)
and [class-order mix](class-order-mix-plan.md) plans; the implementation is
reviewed and committed before the run.

## Why

Under `leaf16` (an arrival with a cached child is not a candidate; only
leaves are sampled), the horizon control found the frozen ranker's gap to the
exact label eviction-located in 12/12, as under `all16`. It did not run the
reuse bit at any horizon or the class-order arms under `leaf16`. Whether the
bit at the capacity-matched horizon reaches the exact `next_use` label, and
whether the learned order within the matched class loses to recency, could
be products of `all16`, where a state whose ancestor was evicted can remain
and the arrival is nearly always a candidate.

## Question and predictions

With `h*` carried over from `all16` unchanged, and the references the
`leaf16` rungs: does the exact bit at `h*` with recency reach the `leaf16`
exact `next_use` label; does the frozen ranker given the class at `h*` reach
the `leaf16` exact label's eviction gain; and does its order within the class
lose to recency?

Predictions, fixed before the run, on five-seed means:

1. `S_h*` under `leaf16` is at most 0.10 in 12/12 trace × cell.
2. `R_h*(evict_binary_h*_learned)` under `leaf16` is at least 0.9 in at least
   10/12; the reading at 0.25%×1 is not predicted.
3. The learned order within the matched class is a consistent loss against
   recency in at least 8/12.

## Fixed surface

- Traces, capacities, seeds 0–4, L1, L2 store, hit rule, sizes, hooks and
  decision statistics: as the horizon control's Part 3. Mechanism `leaf16`
  (`l2_eligibility="leaf"`, width 16). `h*` per cell as under `all16`; no
  re-tuning.
- Arms under `leaf16`: `label_binary_h*` (the label rung's scorer at `h*`,
  as `horizonctl` and `horizonfill` build it), `evict_binary_h*_learned` and
  `evict_binary_h*_recency` (as the matched class-order control builds
  them), and the `label` rung rerun as a reproduction anchor. 4 arms × 12
  trace × cell × 5 seeds = 240 replays.
- References, published rows and not rerun: `lru`, `learned`, `label` under
  `leaf16` from `results/paper/mechanism_control_001/`; `evict_label` under
  `leaf16` from `results/paper/horizon_control_001/`.

## Required checks

- The 60 `label` replays reproduce the published `leaf16` `label` rows
  exactly (`avoided_prefill_tokens` and the counter digest). Nothing is
  published otherwise.
- Identifiers of every replay equal the published `leaf16` reference rows';
  every replay has zero present-but-unusable tokens; the Phase 0.98b
  identities hold; the class statistic (no resident eviction discards a
  state reusable within `h*` while a sampled resident is not; no override
  outside a first round) is zero in every replay.
- Unit tests on constructed traces under `leaf16`: with admission by the
  exact `h*` label the recency arm is the `label_binary_h*` rung decision by
  decision; the `all16` and `leaf16` arms differ only in eligibility.
- No existing replay path or runner is changed; the existing tests pass.

## Readings, fixed before the run

Five-seed means; "consistent" means the same sign in all five seed-paired
values. Every value is reported beside the published `all16` value of the
same arm.

1. **Horizon under leaf eligibility.** `S_h* = (U(label) − U(label_binary_h*))
   / (U(label) − U(lru))` with the `leaf16` rungs; count of `≤ 0.10` out of
   12 (prediction 1).
2. **Class order under leaf eligibility.** `R_h*(a) = (U(a) − U(learned)) /
   (U(evict_label) − U(learned))` with the `leaf16` references, for both
   matched arms; count of `R_h*(learned order) ≥ 0.9` out of 12 (prediction
   2).
3. **Order within the class.** `U(evict_binary_h*_learned) −
   U(evict_binary_h*_recency)`, seed-paired; count of consistent losses out
   of 12 (prediction 3).
4. **Admission, descriptive.** `U(label_binary_h*) − U(evict_binary_h*_recency)`
   under `leaf16`, seed-paired, beside the `all16` value and the share of
   decisions in which the arrival is a candidate.

## Interpretation boundaries

- `h*` is the `all16` best of grids run on these traces; under `leaf16` the
  best horizon may differ, and nothing here looks for it.
- `leaf16` is a control on eligibility, not a proposed mechanism.
- Every arm reads the trace's future and none is a policy.

## Execution order

1. Commit this plan with its two companions, alone.
2. Implement a separate runner on the matched class-order and horizon
   runners' replay calls with `leaf16` eligibility; tests on constructed
   traces; no change to an existing path. Review and commit before any
   replay.
3. No separate smoke; the run carries the 60 reproduction replays.
4. Run once with 10 workers into `results/paper/leaf_matched_horizon_001/`,
   recording plan and code commits and the hashes of the traces and of every
   table read.
5. Report, including failed predictions.

## Addendum, before any replay

Written when the review of the implementation found that one required
check cannot be done as worded, and committed before any replay.

- **Reproduction without a counter digest.** The published `leaf16` `label`
  rows of `results/paper/mechanism_control_001/replay_seeds.csv` carry no
  `counters_sha256` or `decision_sha256` column (that run predates the
  digests). The reproduction check therefore compares, for each of the 60
  `label` replays, `avoided_prefill_tokens` and **every counter column the
  published row carries**, exactly (everything but identity, timing, memory
  and annotation columns; 87 columns per row), and requires that the replay
  counters `l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`,
  `measured_requests`, `l1_avoided_tokens`, `l2_avoided_tokens`,
  `avoided_prefill_tokens`, `extra_avoided_tokens`,
  `l2_present_unusable_tokens`, `l2_present_unusable_blocks`,
  `l2_admissions`, `l2_rejections`, `l2_evictions`, `l2_decisions` and
  `l1_evictions` are among them. Any digest a published row does carry is
  compared as well. This run's own digests are recorded in
  `reproduction.csv`. Counters the published rows never carried (block
  counts, byte-seconds, depth bins) cannot be checked.
- **Class statistic on `label_binary_h*` as well.** The statistic is attached
  to the three `h*` arms and required zero in each; it is not attached to the
  `label` anchor, whose key is the `next_use` label (at `h* = 600 s` a victim
  exactly 600 s from reuse may tie with a never-reused state, so zero is not
  guaranteed there; those victims are covered by the label identities of the
  error-location statistics).
- **Identifiers against every reference of the cell.** Each replay is
  compared with every published row of its trace × cell × seed read by the
  run, `leaf16` and `all16` alike (11 rows), not only the `leaf16` rows.
- **Prediction 2 counts 0.25% × 1.** The reading at 0.25% × 1 is not
  predicted, but it is counted in the "at least 10 of 12".
- **Published `all16` values recomputed and checked.** The `all16` columns
  beside every reading are recomputed from the published per-seed rows and
  must equal the published `horizon.csv`, `fill.csv`, `class_order.csv`,
  `order.csv` and `admission.csv` values before any replay.
