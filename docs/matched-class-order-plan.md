# Class order at the matched horizon: the frozen ranker given the exact reuse class at the horizon that works for the cell

Status: pre-registration. No replay named below has been run. This plan is
committed before any implementation, in one commit with the
[matched-horizon diagnosis](matched-horizon-diagnosis-plan.md); the
implementation is reviewed and committed before the run.

## Why, and what was already seen

Reading 2 of the [horizon and class-order control](horizon-control-findings.md)
gave the frozen `next_use` ranker the exact 600-second reuse class on
residents (`evict_binary_learned`: admission by the ranker, eviction by
`(reusable within 600 s, ranker score, last_group)`). It recovered 94–104% of
what the exact label recovers on eviction in the six trace × cell where that
class is itself sufficient (`S_600 ≤ 0.10`), and 33–61% in the other six. The
registered name of that reading attributed the shortfall to the ranker's
order within the class; the findings withdrew the attribution, because
recency within the class recovered no more and reading 1 with its fill-in
showed that 600 seconds is the wrong boundary in those six. Published values,
conversation / tool-agent:

| cell | horizon that works, `h*` | `S_h*` | `R(evict_binary_learned)` at 600 s | `R(evict_binary_recency)` at 600 s |
|---|---:|---:|---:|---:|
| 0.25%×1 | 60 s | 0.028 / 0.001 | 0.350 / 0.330 | −0.156 / −0.133 |
| 0.25%×4 | 150 s | −0.035 / −0.028 | 0.605 / 0.558 | 0.321 / 0.291 |
| 1%×1 | 150 s | −0.036 / −0.036 | 0.470 / 0.450 | 0.439 / 0.456 |
| 1%×4 | 600 s | −0.001 / 0.000 | 0.998 / 0.987 | 1.006 / 1.001 |
| 2%×1 | 300 s | −0.038 / −0.031 | 0.938 / 0.956 | 0.872 / 0.917 |
| 2%×4 | 600 s | 0.001 / −0.003 | 1.043 / 1.042 | 0.999 / 1.003 |

`h*` is the horizon of the grids run that minimises `S_h`, read after those
runs, the same on both traces. An arm with the reuse class at `h*` was not
run.

## Question and prediction

With the exact reuse class at the horizon that works for the cell given on
residents, and the frozen ranker keeping admission and the order within the
class, does the arm reach the exact label's gain on eviction?

Prediction, fixed before the run: **yes in all six trace × cell where
`h*` < 600 s**: `R_h*(evict_binary_h*_learned) ≥ 0.9` in each. In the other
six the arm is the published `evict_binary_learned` and must reproduce it.
The two counts are reported side by side and never merged: "n/6 new at the
matched horizon" and "6/6 reproduced at 600 s".

## Fixed surface

- Traces, capacities, seeds, evaluation window, L1, L2 store, hit rule,
  sizes, mechanism (`all16`), hooks and decision statistics: as Part 2 of the
  horizon control.
- Arms, at `h = h*(cell)`: `evict_binary_h_learned` and
  `evict_binary_h_recency`, built exactly as `horizonctl.class_order_setup`
  builds the published arms with the exact reuse label at `h` in place of
  600 s: admission by the frozen `next_use` ranker through
  `errorloc.HybridOverride` unchanged; eviction by the store's key
  `(reusable within h s, ranker score, last_group)` for the learned order and
  `(reusable within h s, last_group)` for recency.
- Trace × cell: all 12; seeds 0–4. Replays: 2 arms × 12 × 5 = 120, of which
  the 60 at the six trace × cell with `h*` = 600 s are the published arms
  rerun as a reproduction check.
- References, published rows and not rerun: `lru`, `learned`, `label`,
  `evict_label` under `all16` from `results/paper/error_location_001/`;
  `label_binary_h*` from `results/paper/horizon_control_001/` (60, 300,
  600 s) and `results/paper/horizon_fill_001/` (150 s); the published
  `evict_binary_learned` and `evict_binary_recency` at 600 s from
  `results/paper/horizon_control_001/`.

## Required checks

- The 60 replays at `h*` = 600 s reproduce the published class-order rows of
  `results/paper/horizon_control_001/replay_seeds.csv` exactly
  (`avoided_prefill_tokens` and the replay's counter digest). Nothing is
  published otherwise.
- `l1_avoided_tokens`, `requested_tokens`, the capacities and the
  compulsory-absent charge of every replay equal those of the published
  reference rows of its trace × cell × seed; the request partition and
  per-block identities of Phase 0.98b hold in every replay.
- In every replay of both arms, the number of resident evictions that
  discard a state reusable within `h*` while a sampled resident is not is
  zero, as constructed; every overridden decision is a first-round decision
  in which the arrival is a candidate.
- Unit tests on constructed traces: at `h` = 600 s each new arm equals the
  published arm decision by decision; at another `h` the store's key orders
  as `(reusable within h, score, last_group)`; with admission by the exact
  `h`-second reuse label, the recency arm is the `label_binary_h` rung.
- No existing replay path or runner is changed; the existing tests pass.

## Readings, fixed before the run

`U(a)` is extra avoided prefill tokens over L1 alone, in points of window
input tokens, on five-seed means; "consistent" means the same sign in all
five seed-paired values. Every value of every arm is reported.

1. **Class order at the matched horizon.**
   `R_h*(a) = (U(a) − U(learned)) / (U(evict_label) − U(learned))`. A trace ×
   cell reads *reuse identification at the matched horizon suffices with the
   ranker's order* when `R_h*(evict_binary_h*_learned) ≥ 0.9` (exactly 0.9
   suffices), and *the ranker's order costs at the matched horizon*
   otherwise. Counted out of 6 (new) and out of 6 (reproduced), against the
   prediction of 6/6 new.
2. **Order within the matched class.** `U(evict_binary_h*_learned) −
   U(evict_binary_h*_recency)`, seed-paired: consistent gain, consistent loss
   or mixed, out of 12; `R_h*(evict_binary_h*_recency)` beside it. Context:
   at 600 s the ranker's order was a consistent gain in 9/12.
3. **Admission at the matched horizon.** `U(label_binary_h*) −
   U(evict_binary_h*_recency)`: the two arms evict residents by the same key
   and differ in who decides rejection; seed-paired sign, out of 12,
   descriptive. Context: at 600 s the ranker's admission was a consistent
   loss in 9/12, by 0.15–1.67 points.
4. **Matched versus 600 s, descriptive.** In the six trace × cell with
   `h*` < 600 s: `U(evict_binary_h*_learned) − U(evict_binary_learned)` and
   the same for recency, seed-paired.

## Interpretation boundaries

- `h*` is the best of grids run on these traces, read after the fact; the
  reading says what an exact class at that horizon does for this ranker, not
  that the horizon can be chosen before the fact or that the class can be
  learned.
- `evict_label` is the exact `next_use` label applied greedily to a sampled
  decision; it is the comparison the horizon control used, not an optimum.
- The ranker's admission is the published relative rule on the arrival
  against sampled residents; nothing is said about an absolute admission
  filter.
- Every arm reads the trace's future and none is a policy; nothing here
  triggers a feature, a target, a fit or a deployed policy.
- Seed intervals describe sampling-seed variability only.

## Execution order

1. Commit this plan, alone with the matched-horizon diagnosis plan.
2. Implement a separate runner on `horizonctl.class_order_setup` with the
   horizon as a parameter and the horizon control's replay call, with tests
   on constructed traces; no change to an existing replay path or runner.
   Review and commit before any replay.
3. No separate smoke: the arms are the published ones with another
   parameter, and the run carries 60 reproduction replays without which
   nothing is published.
4. Run once with 10 workers from a tree whose execution source is the code
   commit, into `results/paper/matched_class_order_001/`, recording plan and
   code commits and the hashes of the traces and of every table read.
5. Report the four readings next to the horizon control's, including a
   failed prediction.

## Addendum, before any replay

Written when the review of the implementation found a miscount above, and
committed before any replay. The "Why" section counts six trace × cell where
the 600-second class fell short in reading 2 of the horizon control. The
table of `h*` puts 2%×1 at 300 s as well, so the trace × cell with `h*` <
600 s are **eight** — 0.25%×1, 0.25%×4, 1%×1 and 2%×1 on both traces — and
those with `h*` = 600 s are **four** (1%×4 and 2%×4 on both traces), that is
40 reproduction replays, not 60. The table rule stands: `h*` is the
minimiser of `S_h` on the grids run, and 2%×1 has `S_300` below `S_600`.

Reading 1 is therefore counted out of 8 new and out of 4 reproduced, and the
prediction is `R_h* ≥ 0.9` in all eight. Because 2%×1 already read
"suffices" at 600 s (`R` 0.938 / 0.956), the prediction is easier to meet
there; the count over the six trace × cell that read "the ranker's order
costs" at 600 s is reported beside the eight, and the two are never merged.
Reading 4 covers the eight trace × cell with `h*` < 600 s. The 120 replays,
the arms, the map and the other readings are unchanged.
