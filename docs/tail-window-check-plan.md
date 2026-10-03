# Evaluation-window check: do the matched-horizon readings hold when the last 600 seconds of the trace are excluded?

Status: pre-registration. No replay named below has been run. This plan is
committed before any implementation, in one commit with the
[class-order mix](class-order-mix-plan.md) and the
[leaf-eligibility](leaf-matched-horizon-plan.md) plans; the implementation is
reviewed and committed before the run.

## Why

Every exact-label arm treats a state with no later occurrence in the trace
as "not reused". At the end of a finite trace that is knowledge of the end,
which no causal policy has. The evaluation window of every two-tier replay
runs from the split at 2,122,199 ms to the trace end at 3,536,999 ms, 1,414.8
seconds, and its last 600 seconds are 42.4% of it. The saved-log diagnoses
excluded those 600 seconds (a decision is eligible only when its labels are
observable); the replay utilities behind the horizon control, its fill-in and
the [matched-horizon follow-ups](matched-horizon-findings.md) counted them.
Whether the readings of those controls depend on the tail has not been read.

## Question and predictions

With traces, arms, capacities, seeds, mechanism and the matched horizon `h*`
of each cell unchanged, do the readings hold on the window from the split to
600 seconds before the trace end?

Predictions, fixed before the run, on the head window:

1. `S_h* ≤ 0.10` in 12/12 trace × cell (on the full window it is at most
   0.028).
2. `R_h*(evict_binary_h*_learned) ≥ 0.9` in the same 10/12 trace × cell as
   on the full window, and below it at 0.25%×1 on both traces.
3. The five-seed mean of `U(evict_binary_h*_learned) − U(evict_binary_h*_recency)`
   has the sign it has on the full window in 12/12.
4. The five-seed mean of `U(label_binary_h*) − U(evict_binary_h*_recency)`
   has the sign it has on the full window in 12/12.

## Fixed surface

- Traces, capacities (six cells), seeds 0–4, L1, L2 store, hit rule, sizes,
  mechanism `all16`, hooks and decision statistics: as the horizon control
  and the matched class-order control. `h*`: 60 s at 0.25%×1, 150 s at
  0.25%×4 and 1%×1, 300 s at 2%×1, 600 s at 1%×4 and 2%×4.
- Arms, every one rerun: `lru`, `learned`, `label`, `evict_label`,
  `label_binary_h*`, `evict_binary_h*_learned`, `evict_binary_h*_recency`,
  built exactly as the error-location control, the horizon control (and its
  fill-in, for 150 s) and the matched class-order control build them. 7 arms
  × 12 trace × cell × 5 seeds = 420 replays.
- Windows. `full`: the published evaluation window, split to trace end.
  `head`: split to `trace end − 600,000 ms` inclusive, counted by
  `onpolicy.LabelWindowUtilityCollector(trace, split_ms, end_ms − 600,000)`
  as the on-policy run counted its label window, attached beside the
  attribution collector through a request hook that calls both (new code,
  no existing path changed). `tail`: `full − head`, by subtraction of the
  integer token counters. `U` on a window is extra avoided prefill tokens
  (avoided minus L1-avoided) in points of that window's input tokens. No
  window depends on `h`; `h*` is not chosen again.

## Required checks

- Every replay reproduces its published row exactly (`avoided_prefill_tokens`
  and the counter digest; the decision digest where the published row
  carries one): `lru`, `learned`, `label`, `evict_label` from
  `results/paper/error_location_001/`; `label_binary_60/300/600` from
  `results/paper/horizon_control_001/` and `label_binary_150` from
  `results/paper/horizon_fill_001/`; the two class-order arms from
  `results/paper/matched_class_order_001/`. Nothing is published otherwise.
- The head window's counters are consistent: `head ≤ full` in every counter,
  head `measured_requests` equal across the seven arms of a trace × cell ×
  seed (the request stream and the window are arm-independent), and head
  `requested_tokens` and `l1_avoided_tokens` equal across them.
- The request hook that calls both collectors is read-only: a replay with
  and without it gives identical counters (unit test on constructed traces),
  and the attribution identities of Phase 0.98b hold in every replay.
- Unit tests on constructed traces: the head counters equal those of a
  replay whose trace is cut at the window end only where the cut cannot
  change decisions (a trace with no request after the cut), and equal
  `LabelWindowUtilityCollector`'s published arithmetic; tail = full − head.

## Readings, fixed before the run

Each reading is computed on `head` and reported beside `full` (recomputed
from the reproduced rows, equal to the published values) and `tail`.

1. **Horizon.** `S_h* = (U(label) − U(label_binary_h*)) / (U(label) − U(lru))`
   per trace × cell; count of `S_h* ≤ 0.10` out of 12, against prediction 1.
2. **Class order.** `R_h*(a) = (U(a) − U(learned)) / (U(evict_label) − U(learned))`
   for both matched arms; count of `R_h*(learned order) ≥ 0.9` out of 12 and
   the per-cell agreement with the full-window reading, against prediction 2.
3. **Order within the class** and 4. **admission**: the seed-paired
   differences of the matched class-order control's readings 2 and 3 on
   `head`, with seed signs; the per-cell agreement of the sign of the mean
   with the full window, against predictions 3 and 4; the categorical
   reading (consistent gain / loss / mixed) beside it, descriptive.
5. **Tail share, descriptive.** For `G = U(label) − U(learned)` and for each
   difference above, the share of the full-window token difference that
   falls in the tail, beside the tail's share of input tokens.

## Interpretation boundaries

- A difference between `head` and `tail` mixes the exact labels' knowledge
  of the end with any change of the workload over time; nothing here
  attributes it to either.
- The head window is 814.8 seconds of a 1,414.8-second window; seed
  intervals describe sampling-seed variability only.
- `h*` was read from full-window replays of the same traces; it is kept, not
  re-chosen, so the head readings are a sensitivity check of published
  readings, not a new selection.
- Every arm reads the trace's future and none is a policy.

## Execution order

1. Commit this plan, with the two companion plans, alone.
2. Implement a separate runner on the matched class-order runner's replay
   call, with the second collector and the window arithmetic; tests on
   constructed traces; no change to an existing replay path or runner.
   Review and commit before any replay.
3. No separate smoke: every arm is a published one rerun, and the run
   carries 420 reproduction checks without which nothing is published.
4. Run once with 10 workers from a tree whose execution source is the code
   commit, into `results/paper/tail_window_check_001/`, recording plan and
   code commits and the hashes of the traces and of every table read.
5. Report the readings next to the published ones, including failed
   predictions.
