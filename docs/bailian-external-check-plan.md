# External-workload check on the Qwen-Bailian traces: the matched horizons transplanted, and a sufficing horizon within a fixed grid

Status: pre-registration **draft**, for review. No replay named below has been run; no implementation exists; nothing is fitted. Written after the [input and conversion audit](bailian-input-audit.md) and the [handoff](next-phase-handoff-20261003.md), and before any smoke. The draft becomes a pre-registration when it is agreed and committed alone; the implementation is then reviewed and committed before any replay.

## Why

Every result so far is on two Mooncake traces of one deployment family (about 59 minutes each). The [horizon control](horizon-control-findings.md) and its fill-in found, under `all16`, that an exact one-bit reuse label ("reused within `h` seconds") with recency tie-break reaches the sampled greedy exact `next_use` label (within 10% of its gain over sampled LRU) at a horizon that grows with L2 capacity — 60, 150, 300 and 600 s over the six cells — with the horizon read from grids run on the same traces. The [three checks](matched-horizon-checks-findings.md) found the reading stable off the last 600 s and weaker under `leaf16` (9/12 at the transplanted horizon). Whether any of this holds on an independent workload is untested. This plan asks two separate questions on the four Qwen-Bailian traces, which are a second workload family of one provider, not four independent deployments.

## Questions and predictions

**A. Transplant.** With the Mooncake cell → `h*` table carried over unchanged (0.25%×1: 60 s, 0.25%×4: 150 s, 1%×1: 150 s, 1%×4: 600 s, 2%×1: 300 s, 2%×4: 600 s) and the same capacity fractions of each trace's own working set, does the exact bit at `h*` with recency reach each mechanism's greedy exact `next_use` label on the Bailian traces? This tests the table, not a rule that derives a horizon from capacity.

**B. Existence within a fixed grid.** Over the grid `G = {6, 15, 60, 150, 300, 600, 1200}` s, fixed here, is there a horizon at which the bit with recency reaches the label, and does that horizon grow with L2 capacity? The best grid point is a post-hoc choice and is reported as one.

**R. Recency beyond the bit.** Does a uniformly random order within the reuse class lose to recency within the class, as it did on Mooncake (12/12)?

Predictions, fixed before any replay, on five-seed means over the primary window, counted over the *evaluable* trace × cell (defined below; at most 24 per mechanism):

1. **A, `all16`.** `S_h*` ≤ 0.10 in at least 12 of 24. Reasoning recorded now: at the same capacity fraction the Bailian L2 holds more seconds of input traffic than the Mooncake L2 (the audit's working set and token rate; by the ratio of working-set bytes to input tokens per second, 1.2–2.1× the conversation trace's and 1.7–3.1× the tool-agent trace's), so the transplanted horizons are expected to be short for the smaller cells; the two 600-second cells are the grid's old top. Failure of this prediction does not refute the usefulness of a reuse class on this workload and does not stop the later steps.
2. **A, direction of failure, `all16`.** Among the cells where the transplanted `h*` does not suffice, the best grid horizon is longer than `h*` in more cells than it is shorter. (Reported as 0/0 if prediction 1 holds in 24/24.)
3. **B, `all16`.** Some grid horizon suffices (`min_h S_h` ≤ 0.10) in at least 20 of 24.
4. **B, monotonicity, `all16`.** Within a trace, the best grid horizon is nondecreasing in L2 bytes over the five distinct L2 levels (0.25%×1 < 0.25%×4 = 1%×1 < 2%×1 < 1%×4 = 2%×4; equal levels compared as one, taking the smaller best), in at least 3 of 4 traces.
5. **R, `all16`.** `D_rand = U(label_binary_random_h*) − U(label_binary_h*)` is negative in all five seeds in at least 16 of 24.
6. **B, `leaf16`.** Some grid horizon suffices in at least 16 of 24 under `leaf16` (on Mooncake the transplanted `h*` sufficed in 9/12 under `leaf16`; no `leaf16` grid has been run).

Everything else below is reported without a prediction: the `leaf16` transplant counts, the `leaf16` random-order reading, the calibration-half reading, the granularity control, every absolute utility.

## Inputs, fixed

- The four converted traces of `data/raw/qwen_bailian_512/` with the hashes in their manifests (`bailian_toc_trace`, `bailian_tob_trace`, `bailian_thinking_trace`, `bailian_coder_trace`; 512-token blocks, milliseconds, prefix-chained sequential ids), read by the unmodified `load_mooncake_trace`. The [audit](bailian-input-audit.md) records their identity, the conversion's integrity and what the 16→512 coarsening changes. The raw files are not modified; the converter is not changed.
- All four traces are in. A trace is excluded only if the loader rejects it or an audit check fails; neither is expected. A trace × cell is **evaluable** when the five-seed mean of `U_m(label) − U_m(lru)` on the primary window is at least 1.0 point of window input tokens under that mechanism; otherwise it is "no headroom", reported and left out of every count. The rule is fixed here, before any utility is seen.
- Capacity: L1 = `round(f × W)` and L2 = `round(f × k × W)` bytes with `W` the trace's packed working set (`gap.working_set_bytes`, 2048 bytes per token, partial final blocks charged by their token count), `f` ∈ {0.0025, 0.01, 0.02}, `k` ∈ {1, 4}: the six published cells on each trace's own working set. The plan does not treat equal fractions as equal budgets. From the audit, at 512 tokens (MiB; L2 = L1 at ×1):

  | trace | `W` (GiB) | L1 at 0.25% / 1% / 2% | L2 at 0.25%×4 / 1%×4 / 2%×4 |
  |---|---:|---|---|
  | To-C | 95.4 | 244 / 977 / 1,955 | 977 / 3,909 / 7,819 |
  | To-B | 196.6 | 503 / 2,013 / 4,025 | 2,013 / 8,051 / 16,102 |
  | thinking | 55.5 | 142 / 568 / 1,137 | 568 / 2,274 / 4,548 |
  | coder | 173.2 | 443 / 1,773 / 3,547 | 1,773 / 7,094 / 14,188 |
  | Mooncake conversation / tool-agent | 173.0 / 166.2 | 443 / 1,771 / 3,543 ; 425 / 1,702 / 3,404 | 1,771 / 7,086 / 14,171 ; 1,702 / 6,808 / 13,616 |

- Audit facts the plan takes as given: To-B has no session links (every record a root) and short inputs, with 25.8% of its input tokens in partial final 512-token blocks (To-C 10.3%, thinking 4.9%, coder 4.3%, Mooncake 2.2–2.6%); a child request almost never carries its parent's last 16-token block (93–94% of children on the three traces with sessions), so at 512 tokens a child reuses its parent's prefix only up to the previous 512-token boundary; the coarsening removes 7.8 / 19.0 / 2.8 / 3.2 points of repeatable input (To-C / To-B / thinking / coder) and enlarges the working set 1.19× / 1.41× / 1.05× / 1.09×. Simultaneous requests are rare (at most 3 per timestamp), so the no-hit-between-simultaneous-requests rule is nearly inactive, where on Mooncake every request is in a batch of up to 47.
- Split and label horizon as every replay runner: the split at 60% of the trace span (`decisionpop.horizon_for`, label horizon 600 s, 24 snapshots), L1 `lru`, the published hit rule, sizes, store and sampled-16 eviction. Mechanisms `all16` (`l2_eligibility="all"`, width 16) and `leaf16` (`"leaf"`, 16), each with its own references. Seeds 0–4 for every sampled arm.
- **Windows.** Every replay runs from the split to the trace end and is counted on three windows by read-only request hooks (`tailwindow.LabelWindowUtilityCollector`, one per window, chained): the **primary window** `W = [split, end − 1,200,000 ms]`, inside which every label of every grid horizon is observed without knowledge of the trace end; the **full window** `[split, end]`, the runners' published window; and the two halves of `W` by time, `W1 = [split, mid]` and `W2 = (mid, end − 1,200 s]` with `mid` the midpoint of `W`, for the calibration-half reading. Readings and predictions are on `W`; the full window is reported beside them. No window changes with the horizon, and no input is cut.
- Utility `U` is extra avoided prefill tokens over L1 alone, reported in tokens and in points (100 × tokens / the window's input tokens), for every arm, mechanism and window. Heap references `H_off` (the heap offline `next_use` comparator) and `H_lru` (heap LRU) are run per trace × cell (deterministic, no seed; `run_decision_population`'s heap path) and reported as context, with the headroom `T = H_off − U_m(lru)`.

## Arms

Per trace × mechanism × cell × seed, all with the store's first minimum of the key and no override, built by the modules that built their published rows:

- `lru`: the sampled LRU rung (`mechanism control`, `l2_policy="lru"`).
- `label`: the sampled greedy exact `next_use` label at the decision (`mechanism control`'s `label` rung).
- `label_binary_h` for `h ∈ G`: the exact bit at `h` with recency, key `(binary_h, last_group)` (`horizonfill.arm_setup`: `ExactLabelScorer(trace, h, "binary")`, `l2_policy="learned"`). `h = h*` is the transplant arm of A; the other six points are B.
- `label_binary_random_h*`: the exact bit at `h*` with a uniform draw in [0, 1) per candidate per decision inside both classes, key `((binary_h*, u), last_group)`: `classmix.RandomClassScorer(ExactLabelScorer(trace, h*, "binary"), seed, arm)` used as the store's scorer for admission and eviction alike, its stream seeded from the replay seed and the arm name and separate from the store's sampling stream (so its candidate sets are the recency arm's where the decisions coincide).

Counts: A = 4 traces × 2 mechanisms × 6 cells × 5 seeds × 4 arms (`lru`, `label`, `label_binary_h*`, `label_binary_random_h*`) = 960; B = 4 × 2 × 6 × 5 × 6 further grid points = 1,440; heap references 4 × 6 × 2 = 48 deterministic runs. 2,400 sampled replays in all, before the granularity control.

## Readings, fixed before the run

Five-seed means on `W` unless stated; "consistent" means the same sign in all five seed-paired values. Every value is reported with its full-window value, its seed min and max, the absolute `U` of every arm in tokens and points, and `T`.

1. **Transplant (A).** `S_h*(m) = (U_m(label) − U_m(label_binary_h*)) / (U_m(label) − U_m(lru))` per trace × cell × mechanism; "suffices" when ≤ 0.10; counts out of the evaluable cells (prediction 1 for `all16`; `leaf16` reported).
2. **Grid (B).** `S_h(m)` at every `h ∈ G`; `h_best = argmin_h S_h` (ties to the smaller `h`), `min_h S_h`, the count of cells with `min_h S_h` ≤ 0.10 (predictions 3 and 6), the direction of failure (prediction 2) and the monotonicity of `h_best` in L2 bytes (prediction 4). Each cell is classified, in this order: "transplant suffices" (reading 1), "re-tuned grid point suffices" (reading 1 fails, reading 2 holds), "none in the grid" (both fail), "no headroom". Nothing is said about horizons outside `G` or other rules.
3. **Recency beyond the bit (R).** `D_rand` seed-paired, its seed signs and reading; the count of consistent losses (prediction 5 for `all16`; `leaf16` reported).
4. **Calibration half, derived, descriptive.** For each trace × cell × mechanism, `h_cal = argmin_h S_h` computed on `W1` (ties to the smaller `h`), then `S_{h_cal}` on `W2` and whether ≤ 0.10, beside `min_h S_h` on `W2`. The replays are the same; only the counting windows differ. This says what choosing a horizon from the first half of the window would have given on the second; it chooses nothing on `W2`.
5. **Granularity control (C), descriptive.** See below.
6. **Mooncake side by side, descriptive.** The published Mooncake values of readings 1–3 on their full window, beside the Bailian values, as context; they are not pooled.

Denominators: a ratio whose denominator is below 1.0 point is not computed ("no headroom"); a ratio is never clipped. `S` above 1 or below 0 is reported as is.

## Granularity control (C)

The 16→512 coarsening changes which shared prefixes are representable, the state sizes and the working set that the capacity fractions divide; the audit quantifies all three and the repeatable tokens lost, and To-B is the trace it changes most (19 points of repeatable input; working set 1.41×). One minimal control, on To-B, at 16-token blocks through the unmodified converter (`--block-tokens 16`) and loader (`block_size=16`), mechanism `all16`, arms `lru`, `label`, `label_binary_h*`, seeds 0–4, at the **same absolute bytes** as To-B's six 512-token cells (budget set (ii); L1 and L2 in bytes as in the table above): 6 × 3 × 5 = 90 replays, at most 4 workers (the loader holds 4.6 million states; the audit's direct walk used 1.8 GiB, the loader will use more). Readings, descriptive: `S_h*` and the absolute `U` of each arm at 16 and at 512 under the same bytes, and the agreement of "suffices" between the two granularities per cell. The fractions-of-the-16-token-working-set budget set (i) is not run: it would change budget and granularity together. The sampled-16 mechanism itself behaves differently when it samples 16 of millions of states, so this is a sensitivity control, not a ground truth at 16 tokens; it is not pooled with A or B. If the smoke's per-replay time at 16 tokens exceeds 20 minutes, the control is reduced to seeds 0–2 (54 replays); if a worker needs more than 6 GiB, the worker count is halved; both reductions are recorded before the run.

## Required checks

- The [audit](bailian-input-audit.md) passes (identities equal the manifests, the conversion is reproduced byte for byte, the loader accepts every trace).
- **Reproduction anchor on Mooncake.** Before any Bailian replay, the new runner replays `lru`, `label`, `label_binary_60` and `label_binary_600` at conversation 0.25%×1 and 1%×4 under `all16` and `leaf16` (2 cells × 2 mechanisms × 4 arms × 5 seeds = 80 replays) and must reproduce the published rows exactly: `avoided_prefill_tokens` and the counter and decision digests where published (`horizon_control_001`, `error_location_001`), every published counter column where not (`mechanism_control_001`'s `leaf16` rows). Nothing is published otherwise.
- Identifiers (`l1_capacity_bytes`, `l2_capacity_bytes`, `requested_tokens`, `l1_avoided_tokens`, `absent_compulsory_tokens`) equal across every arm of a trace × cell × seed, on every window; the window counters are consistent (head at most full; the halves sum to `W`); the Phase 0.98b identities hold; under `leaf16` no replay has a present-but-unusable token; every decision is seen with its final victim; no override anywhere; the random arm's draw count equals its candidate count and its stream is its own (the sampling stream untouched: its `lru`-side identifiers equal the recency arm's).
- Unit tests on constructed traces: the random arm with its draw replaced by a constant equals `label_binary_h*` decision by decision; three window collectors on one replay sum correctly; the 16-token path through converter and loader on a constructed file.
- No existing replay path, runner, converter or test is changed; the existing tests pass.

## Smoke, resources and stop conditions

- Smoke, after the anchor: 1 seed × 1 cell (1%×1) × `all16` × the four A arms on each Bailian trace (16 replays) and the three C arms on To-B at 16 tokens at one budget (3 replays), recording seconds and peak RSS per replay. The smoke rows are discarded.
- Estimate from Mooncake (18–120 s per replay at 290–410 thousand block occurrences): the Bailian traces carry 107–505 thousand occurrences at 512 tokens, so about 10–150 s per sampled replay; 2,400 replays on 10 workers ≈ 2–5 hours. C: To-B at 16 tokens carries 9.96 million occurrences (24× the tool-agent trace), so about 5–30 minutes per replay; 90 replays at 4 workers ≈ 2–12 hours. The memory of the 16-token loader is the unknown; the smoke measures it.
- Stop conditions, fixed now: if a smoke replay exceeds 15 minutes or 3 GiB RSS at 512 tokens, the run is not started and the plan is re-issued; if only the 16-token smoke exceeds 10 minutes or 6 GiB, C is reduced as stated above; if a worker dies during the run, the run is reported as incomplete and not published in part. The run is one invocation per part (A+B together, C separately) into `results/paper/bailian_external_check_001/` and `results/paper/bailian_granularity_control_001/`, recording plan and code commits and the hashes of the traces, manifests and every table read.

## What is deferred, and why

- **Model transplant.** The frozen Mooncake rankers are per-trace fits (`pi0` `next_use` for `conversation_trace` and for `toolagent_trace`); applying one to a Bailian trace requires choosing which, and its features and normalisation were fixed on Mooncake. The eviction-located and class-order readings need `learned`, `evict_label` and the class arms (hybrids with the ranker's admission), about 5 arms × 240 = 1,200 further replays per mechanism. This is a separate sub-analysis, "transplant diagnostic of a fixed model", to be pre-registered after A/B with the model assignment fixed before the run; its failure would not read as "history cannot supply the bit".
- **History-based fit (⑤).** No fit is run here, so no learner's result can leak into the horizon or the rule. ⑤ fixes eligibility, budget, horizon and admission/tie-break first, then asks whether a learner given past information increases avoided tokens; it must fix the mechanism, because the within-class order reading reverses between `all16` and `leaf16` on Mooncake.

## Interpretation boundaries

- The four traces are one provider's workload family sampled over two hours; they are not four independent deployments and are not pooled with Mooncake.
- 512-token blocks on a 16-token trace: the audit's losses (including the parent's last block never being reused by its child) apply to every arm equally, and C bounds the effect on the readings on To-B only, the trace most affected.
- `h*` and `G` are Mooncake's; "transplant suffices" is a statement about this table on these traces, "re-tuned suffices" means a horizon within `G` chosen after the fact, "none in the grid" says nothing about horizons outside `G` or other rules.
- Every arm reads the trace's future; none is a policy; the comparator is sampled-16 greedy, not an optimum.

## Execution order

1. Review of this draft (Astra, the user); agreed → committed alone as the pre-registration, with the audit.
2. Implementation of a separate runner on the horizon-control / mechanism-control / tail-window replay calls, with the random arm and the three-window counting; unit tests; no change to an existing path; review and commit.
3. The Mooncake reproduction anchor (80 replays), then the smoke (19 replays); both reported before the run.
4. A+B once with 10 workers; C once with at most 4 workers, after A+B.
5. Report, including failed predictions and the classification of every cell.

## Open issues for review

1. Grid: the handoff's candidate `{6, 15, 60, 150, 300, 600}` is extended by 1,200 s here (the Bailian L2 holds more seconds of traffic at the same fraction, so the old top may be short); keeping 6 s costs 240 replays and anchors the low end. Either change costs ±240 replays.
2. Primary window `end − 1,200 s` (28% of the trace at 2 h) follows from the grid's top; the alternative is `end − 600 s` with the 1,200-second arm counted only where its label is observed, which breaks the common window.
3. Prediction 1's threshold (12 of 24) is a judgement; the reasoning is recorded and the count is reported whatever the threshold.
4. Whether the model-transplant sub-analysis follows A/B automatically or waits for their result.
5. The evaluable rule (1.0 point of headroom) is new; the Mooncake denominators were 6–31 points.
