# Working-set ratio check: does a reuse working set locate the capacity reversals?

Status: pre-registration. The ratio below has not been computed for any cell.
The sign outcomes it is compared with are already published; what is fixed
here, before the ratio is known, is its definition and the rule that turns it
into a predicted sign.

## Question and scope

Several effects in this repository change sign with capacity on the real
traces: 2-hit admission against LRU, and arrival protection against no
protection. The [handoff](strategy-handoff-20260927.md) lists a capacity
reversal as candidate A and asks for an explanation that existing work does
not already give.

EfficientAgent ([arXiv 2609.33762](https://arxiv.org/abs/2609.33762)) gives one
for a host KV tier: a chunk survives in an LRU tier of capacity `C` when the
distinct other state referenced between two of its uses fits in `C` (its
Equation 4); the *reuse working set* is that amount at the scale of the
workload; with `γ_H` = working set / capacity, declining writes helps when
`γ_H > 1` and hurts when `γ_H ≤ 1` (its Sections 4 and 5.5, Proposition 1).

Question, fixed before the computation: does the same ratio, computed from the
L1 victim stream of this simulator, place the published sign changes where
they occur? If it does, the capacity dependence here is an instance of that
explanation and is not a separate contribution. If it does not, the mismatch
is recorded with the cells and ratios at which it fails.

This is arithmetic on a victim stream and on published tables. It runs no L2
policy, fits nothing and proposes nothing.

## Definitions

- **Offer.** With L1 heap LRU at fraction `f`, every L1 eviction offers one
  state to L2. The offer stream depends on `f` only (the same for every L2
  arm; this invariant is checked in every replay of earlier phases).
- **Return.** The return of an offer of state `s` is the first later request
  whose prefix contains `s`. An offer with no later such request never returns.
- **Reference model.** An admit-all exclusive tier that keeps offers in offer
  order: an offer stays until it returns or is pushed out by newer offers. Its
  *stack distance* `D` is the largest total size, at any instant between the
  offer and its return, of the newer offers that have not yet returned. The
  offer is still held at its return if and only if `D + size ≤ C`. Sizes are
  the packed sizes of the simulator.
- **Working set.** `W(f)` is the token-weighted median of `D + size` over the
  offers whose return falls in the evaluation window (the last 40% of the
  trace), one value per trace and L1 fraction. The token-weighted quartiles
  are reported beside it.
- **Ratio.** `γ(cell) = W(f) / C_L2(cell)`.

EfficientAgent's closed form `(A − 1)·N̄·β` needs agent identities and a pool
size, which the Mooncake traces do not carry; the trace-based stack distance
it is derived from is used instead. The reference model orders by offer time,
whereas the simulator's LRU orders by last request; it ignores the tree hit
rule. It is a scale, as in the source.

## Outcomes compared, all already published

Per real trace × cell (12):

1. Heap `lru_2hit` minus heap `lru`
   (`results/paper/decision_population_replay_seeds.csv`, deterministic).
2. Sampled `lru_2hit_s` minus `lru_s`, seed-paired over five seeds (same
   file), read as consistent gain, consistent loss or mixed.
3. Arrival protection `all` minus no protection, seed-paired
   (`results/paper/phase1_intervention_pairs_seeds.csv`), read the same way.

Predicted signs, from the source's rule: where `γ > 1`, declining writes
helps — outcomes 1 and 2 positive, outcome 3 negative; where `γ ≤ 1`, the
opposite.

## Readings, fixed before the computation

1. **Agreement.** For each outcome, the number of trace × cell whose observed
   sign equals the predicted sign, out of 12; a mixed or zero observation is a
   miss. Every miss is listed with its `γ`.
2. **Verdict per outcome.** *Located* if 12/12. *Located outside the
   transition* if every miss has `0.5 ≤ γ ≤ 2`. *Not located* otherwise.
3. **Threshold, descriptive.** The interval of thresholds on `γ` that would
   maximise agreement for each outcome, and that maximum. A maximum of 12 at a
   threshold away from 1 means the ordering by `γ` is right and the scale is
   off; a maximum below 12 means the sign pattern is not monotone in `γ`.
4. **Declinable share, descriptive.** Per cell, the share of offered bytes in
   the evaluation window whose offer never returns or has `D + size > C_L2` —
   the writes Proposition 1 says an LRU tier can decline without losing a hit
   — next to the rejection share of the `label` and `learned` rungs under
   `all16` from `results/paper/mechanism_control_001/replay.csv`.

Interpretation boundaries. A verdict of *located* says the published reversal
is accounted for by a known working-set argument on these two traces; it does
not validate EfficientAgent's system, whose tier, workload and write path
differ. A verdict of *not located* does not refute it either, for the same
reason and because the reference model is a scale. Three L1 fractions give
three values of `W` per trace and twelve cells in all; agreement counts are
descriptive. Nothing here triggers a new policy or experiment.

## Execution order

1. Commit this plan alone.
2. Implement the stack-distance computation and the comparison with tests on
   constructed streams (peak outstanding size, offers that never return,
   returns outside the window, the equality `D + size ≤ C` against a brute-force
   admit-all tier on small cases). Review and commit before the real traces
   are processed.
3. Run once from a clean tree into `results/paper/working_set_ratio_001/`,
   recording plan and code commits and trace hashes.
4. Report the ratios, the three agreement counts and verdicts, and the
   descriptive tables in a findings document, including misses.
