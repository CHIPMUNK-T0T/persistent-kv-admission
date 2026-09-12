# The Predictability–Retention Gap: Where the Unrecovered Headroom Lives

Phase 0.5 left one question open. Future reuse of a KV state is predictable from
its history (AUC 0.86–0.94 for the 600 s label on the real traces), yet no causal
policy recovered more than 0.248 of the LRU-to-offline headroom under a byte
budget. This document reports the decomposition experiment that asked *why*,
using the existing replay engine and no new policy.

All numbers come from one run of `scripts/run_predictability_gap.py` over the
pinned Mooncake FAST'25 conversation, tool-agent, and synthetic traces, seven
budgets (0.1% to 10% of packed unique-state bytes), **five seeds** per sampled
arm, and the same single hyperparameter set as Phase 0.5
(`results/paper/gap_run_config.json`). Review artifacts are in `results/paper/`
(`oracle_replay.csv`, `oracle_decomposition.csv`,
`candidate_set_prediction.csv`, `population_ladder.csv`,
`standardization_transfer.csv`, `coefficient_similarity.csv`, `fig6`–`fig8`).
Reported values are seed means; 95% confidence intervals (Student-t, n = 5) are
in the CSVs and are quoted where they matter.

## 1. Design

### Arms

Every arm below runs through the **same prefix-closed cache and the same
sampled-leaf eviction** (16 random retained leaves re-scored at the decision,
recency as the secondary key). A difference between two arms is therefore a
difference in what the arm knows or optimises, never in how it evicts.

| arm | knows | ranks by |
|---|---|---|
| `lru`, `lfu` | history | recency; lifetime frequency |
| `learned_history` | history | the Phase 0.5 ranker (`fixed_self`, 23 features, fitted on the trace's own training window), rows standardised against the *retained* population |
| `learned_history_observed_norm` | history | the same ranker, rows standardised against the *observed* population it was fitted on |
| `online` | history | the Phase 0.5 online AdaGrad learner |
| `oracle_binary` | future | **the exact training label**: reused within the fit horizon (600 s real, 300 s synthetic) |
| `oracle_binary_h60`, `oracle_binary_h300` | future | the same binary label at other horizons |
| `oracle_binary_per_byte` | future | binary label ÷ state bytes |
| `oracle_count` | future | number of reuses within the fit horizon |
| `oracle_next_use_sampled` | future | exact next-use time (the offline comparator's score function) under sampled eviction |
| `offline_next_use` | future | the Phase 0 heap comparator: exact next-use time, exact search over all leaves (closure = 1 by definition) |
| `lru_heap`, `lfu_heap` | history | exact-search references |

`HeadroomClosure = (Policy − LRU) / (OfflineNextUse − LRU)` in avoided prefill
tokens over the evaluation window, with the same seed's sampled LRU as floor.

### Gaps

Each gap is computed within a seed, then averaged:

- **signal gap** = `oracle_binary − learned_history`: loss from not knowing the label.
- **objective gap** = `oracle_next_use_sampled − oracle_binary` (also per-byte and count variants): loss from the label being the wrong target even when known perfectly.
- **candidate-search gap** = `offline_next_use − oracle_next_use_sampled`: loss from ranking 16 sampled leaves instead of all leaves.

Prefix dependency is a constraint every arm shares, the comparator included, and
is not a separate arm.

### Decision-population measurement

Every sampled eviction decision of the logged arms is recorded (reservoir of
20,000 decisions per run, evaluation window only), and ranking quality is
computed on those candidate sets: pooled AUC over all candidate rows,
within-decision AUC (pairs ranked against each other in one decision), positive
prevalence, and the **regret rate**: the share of decisions in which the evicted
leaf is reused within the horizon although a candidate that is not was
available. Labels are uncensored: a decision counts at horizon H only if H
seconds of trace remain. Separately, at the 24 shared test snapshots of the
prediction protocol, the same scorer is evaluated on nested populations:
observed states (the protocol's population) ⊃ retained states ⊃ retained leaves.

## 2. Perfect knowledge of the training label: the answer depends on the budget

HeadroomClosure of `oracle_binary` at the fit horizon (`fig6`):

| budget | conversation (600 s) | tool-agent (600 s) | synthetic (300 s) |
|---|---:|---:|---:|
| 0.1% | 0.06 | 0.68 | 0.05 |
| 0.25% | 0.13 | 0.22 | 0.06 |
| 0.5% | 0.21 | 0.20 | 0.12 |
| 1% | 0.43 | 0.44 | 0.21 |
| 2% | 0.72 | 0.76 | 0.23 |
| 5% | 0.86 | 0.96 | 0.29 |
| 10% | 0.80 | 0.94 | 0.50 |

**Below 1% of the working set, a perfect predictor of the label the learned arms
were trained on recovers 5–21% of the headroom** (tool-agent at 0.1% is the one
exception, where every arm including LFU is already above 0.7). **At 2% and
above on the real traces it recovers 72–96%.** Synthetic stays below 0.3 up to
5% and reaches 0.50 only at 10%.

So the pre-registered gate does not return one case. Against the rule fixed
before the run (≥ 0.7 signal problem, ≤ 0.3 objective problem, otherwise
decompose):

- **budgets ≤ 0.5%: objective problem** on all three traces;
- **budgets ≥ 2% on the real traces: signal problem**;
- **1%: mixed** (0.43 / 0.44 / 0.21);
- **synthetic: objective problem at every budget up to 5%.**

Full arm comparison at the three budgets Phase 0.5 reported (seed means; CI
half-widths are ≤ 0.02 unless marked):

| trace / budget | lfu | learned | learned (obs-norm) | online | oracle bin 60 s | oracle bin 300 s | oracle bin fit | bin ÷ bytes | oracle count | oracle next-use (sampled) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation 0.25% | 0.19 | 0.06 | 0.18 | 0.06 | **0.72** | 0.16 | 0.13 | 0.15 | 0.55 | 0.94 |
| conversation 1% | 0.22 | 0.19 | 0.24 | 0.12 | 0.46 | **0.57** | 0.43 | 0.44 | 0.62 | 0.87 |
| conversation 5% | −0.21 | 0.08 | 0.09 | −0.01 | 0.23 | 0.64 | **0.86** | 0.86 | 0.86 | 0.82 |
| tool-agent 0.25% | 0.27 | 0.14 | 0.03 (±0.25) | 0.16 | **0.76** | 0.25 | 0.22 | 0.25 | 0.58 | 0.94 |
| tool-agent 1% | 0.21 | 0.20 | 0.17 (±0.08) | 0.11 | 0.47 | **0.56** | 0.44 | 0.44 | 0.62 | 0.86 |
| tool-agent 5% | −0.27 | 0.13 | 0.13 | 0.00 | 0.29 | 0.75 | **0.96** | 0.96 | 0.96 | 0.90 |
| synthetic 0.25% | 0.12 | −0.01 | 0.03 | 0.06 | **0.12** | — | 0.06 | 0.12 | 0.33 | 1.00 |
| synthetic 1% | 0.08 | −0.04 | −0.21 | 0.04 | **0.28** | — | 0.21 | 0.23 | 0.31 | 1.00 |
| synthetic 5% | −0.10 | −0.13 | −0.23 | −0.08 | **0.76** | — | 0.29 | 0.33 | 0.44 | 0.96 |

(Synthetic's fit horizon is 300 s, so its "oracle bin fit" column is the 300 s label.)

## 3. Decomposition

Seed-paired gaps in closure units (`fig7`, `oracle_decomposition.csv`):

| trace / budget | recovered by learned | signal gap | objective gap (next-use − binary) | per-byte gap | count gap | candidate-search gap |
|---|---:|---:|---:|---:|---:|---:|
| conversation 0.25% | 0.06 | 0.06 | **0.81** | 0.03 | 0.42 | 0.06 |
| conversation 1% | 0.19 | 0.24 | **0.43** | 0.00 | 0.19 | 0.14 |
| conversation 5% | 0.08 | **0.77** | −0.04 | 0.00 | 0.00 | 0.18 |
| tool-agent 0.25% | 0.14 | 0.08 | **0.73** | 0.03 | 0.36 | 0.06 |
| tool-agent 1% | 0.20 | 0.25 | **0.42** | −0.01 | 0.18 | 0.14 |
| tool-agent 5% | 0.13 | **0.83** | −0.06 | 0.00 | 0.00 | 0.11 |
| synthetic 0.25% | −0.01 | 0.07 | **0.94** | 0.06 | 0.27 | 0.00 |
| synthetic 1% | −0.04 | 0.24 | **0.79** | 0.02 | 0.10 | 0.00 |
| synthetic 5% | −0.13 | 0.42 | **0.67** | 0.04 | 0.15 | 0.04 |

Three readings.

**Value per byte is not where the headroom is.** `oracle_binary_per_byte` is
within 0.01 of `oracle_binary` at every budget ≥ 0.5% on the real traces, within
0.06 on synthetic, and adds at most 0.10 at 0.1%. Under the packed size model each state's value per hit is
proportional to its own bytes, so a binary label divided by bytes only changes
the order among states of unequal size, and that order does not matter much
here.

**The candidate-search loss is real but secondary.** Ranking 16 sampled leaves
instead of all leaves costs 0.00–0.18 of closure; it peaks at 1–5% on the real
traces (0.11–0.18) and is zero on synthetic below 2% because the cache holds
fewer than 16 leaves there. Sampled eviction was not the reason the causal arms
failed.

**The dominant term switches with budget.** At small budgets the objective gap
is 0.7–0.9: the binary label is known perfectly and still loses most of the
headroom. At 5% on the real traces the objective gap vanishes (the 600 s binary
label even edges the sampled next-use rule) and the signal gap is 0.77–0.83:
the label is the right target and the causal predictor is what fails.

## 4. Why the binary label fails at small budgets: its horizon does not match the cache

The horizon-sensitivity arms explain the objective gap. Closure of the binary
oracle by label horizon on the real traces:

| budget | best horizon | conversation 60 s / 300 s / 600 s | tool-agent 60 s / 300 s / 600 s |
|---|---|---|---|
| 0.1% | 60 s | **0.57** / 0.07 / 0.06 | **0.90** / 0.72 / 0.68 |
| 0.25% | 60 s | **0.72** / 0.16 / 0.13 | **0.76** / 0.25 / 0.22 |
| 0.5% | 60 s | **0.55** / 0.26 / 0.21 | **0.58** / 0.25 / 0.20 |
| 1% | 300 s | 0.46 / **0.57** / 0.43 | 0.47 / **0.56** / 0.44 |
| 2% | 300 s | 0.41 / **0.87** / 0.72 | 0.42 / **0.89** / 0.76 |
| 5% | 600 s | 0.23 / 0.64 / **0.86** | 0.29 / 0.75 / **0.96** |
| 10% | 600 s | 0.19 / 0.62 / **0.80** | 0.26 / 0.77 / **0.94** |

**The best label horizon grows monotonically with the budget**: 60 s below 1%,
300 s at 1–2%, 600 s at 5% and above. On synthetic, where reuse intervals are
short, the 60 s label is best at every budget (0.76 vs 0.29 at 5%).

The mechanism is visible in the decision logs. Among the candidates the
`oracle_binary` arm ranks at 0.25% and 1%, **88–97% are positive** under the
600 s label (`candidate_set_prediction.csv`, `prevalence`): almost everything the
small cache holds will be reused within ten minutes, so a label that says only
"reused within ten minutes" cannot tell the cache which of them to drop, and the
decision falls back to the recency tie-break. What the cache needs to know at
that budget is what it can keep *until the next use*, i.e. reuse within roughly
its own residence time. At 5% on the real traces the prevalence among candidates
falls to 0.34–0.48 and the 600 s label becomes discriminative again.

This is also the mechanism behind the budget inversion reported in Phase 0.5:
LFU (long memory) wins at small budgets in Phase 0.5's terms because its
candidates happen to be mostly positive anyway, and collapses at large budgets
(−0.21 / −0.27 / −0.10 at 5%, −0.66 / −1.02 / −0.17 at 10%) where the long
horizon matters and lifetime frequency ranks live leaves *inversely*
(within-decision AUC 0.40–0.47 at 1–5%, below chance).

A reuse-count label (`oracle_count`) recovers part of the objective gap
(0.55–0.58 at 0.25% against 0.13–0.22 for the binary label) but not what the
next-use time recovers (0.94), so magnitude helps and timing helps more.

## 5. Why the causal predictor fails at large budgets: the skill is not on the decision population

The Phase 0.5 hypothesis that prediction skill lives in separating dead states
from live ones, not in ranking live candidates, is now a measurement (`fig8`,
`population_ladder.csv`, `candidate_set_prediction.csv`). AUC of the
`learned_history` score for the fit-horizon label, same scorer, same instants:

| population | conversation 0.25% / 1% / 5% | tool-agent 0.25% / 1% / 5% |
|---|---|---|
| observed states, protocol scoring (n = 40,000, prevalence 0.05 / 0.04) | 0.86 | 0.94 |
| observed states, replay scoring | 0.84 / 0.86 / 0.86 | 0.93 / 0.93 / 0.94 |
| retained states (prevalence 0.64 / 0.58 / 0.34; 0.64 / 0.61 / 0.35) | 0.48 / 0.59 / 0.67 | 0.54 / 0.59 / 0.68 |
| retained leaves | 0.49 / 0.56 / 0.63 | 0.58 / 0.57 / 0.63 |
| eviction candidates (within-decision) | 0.57 / 0.57 / 0.63 | 0.62 / 0.59 / 0.63 |

**AUC 0.86–0.94 on observed states becomes 0.57–0.63 on the leaves the eviction
actually ranks.** Prevalence goes from 4–5% to 28–64%: the cache has already
removed the easy negatives. The regret rate makes the consequence concrete: the
learned arm evicts a leaf that will be reused within 600 s, although a leaf that
will not was available, in **29–32% of avoidable decisions at 0.25–1% and 18–19%
at 5%** — the same as LRU (32% / 32% / 20–21%) and LFU (31% / 29% / 24%).
The oracle's regret is 0 by construction.

Two controls bear on whether this is fixable with history features:

- The `online` arm draws its training examples from the retained population, so
  it is fitted on the right population. Its within-decision AUC is 0.52–0.60 and
  its closure is never better than the fixed model's.
- `learned_history_observed_norm` removes a train/deploy mismatch discovered
  during this phase: `learned_history` standardises feature rows against the
  retained population while the ranker was fitted on rows standardised against
  the observed population, and for a linear model that changes the effective
  weights (w_i / scale_i). The control faithfully reproduces the protocol score
  (observed-population AUC 0.86 / 0.94 / 0.67 exactly) and raises small-budget
  closure on conversation (0.17–0.21 vs 0.01–0.12) but is unstable on tool-agent
  (±0.13–0.25 at ≤ 0.5%) and worse on synthetic. Its decision-population AUC is
  0.57–0.69, in the same band. The normalisation choice moves the learned arm around
  within the LFU band; it does not move it toward the oracle.

The synthetic trace is different in kind: the learned model is at or below
chance on its own candidates (0.48 / 0.67 / 0.49) and LFU's small-budget win
there comes from candidates that are 88% positive.

## 6. Standardisation robustness of the transfer result

Concern C-2 from Phase 0.5 was that the cross-workload transfer might be
manufactured by per-decision-point standardisation. Transfer AUC at 300 s with
the same rankers under three standardisations (`standardization_transfer.csv`):

| fitted → evaluated | per-decision | train-set (source scales) | none |
|---|---:|---:|---:|
| conversation → tool-agent | 0.940 | 0.940 | 0.938 |
| tool-agent → conversation | 0.879 | 0.881 | 0.878 |
| conversation → conversation | 0.881 | 0.881 | 0.878 |
| tool-agent → tool-agent | 0.939 | 0.940 | 0.937 |
| synthetic → synthetic | 0.673 | 0.600 | 0.787 |

Coefficient cosine conversation ↔ tool-agent: 0.979 / 0.984 / 0.990. The
real-trace transfer is unchanged to three decimals under every variant, so the
transfer is a property of the two workloads, not of the preprocessing. C-2 is
refuted. The only sensitive cell is synthetic's self-fit, which improves without
standardisation (0.79) and is the least trustworthy row of the matrix in any
case. The transferred structure remains what Phase 0.5's revised wording says:
frequency plus recency with near-equal weight.

## 7. Seed dispersion

224 sampled-arm cells over five seeds: median 95% CI half-width on closure is
0.002, and only 8 cells exceed 0.05, all of them learned or online arms
(worst: `learned_history_observed_norm` on tool-agent at 0.25%, ±0.25). The
oracle arms and the baselines are stable to the third decimal. The budget
inversion of LFU, the ranking of the oracle arms, and every gap sign reported
above hold in each seed individually. The Phase 0.5 concern that closure numbers
came from a single seed is closed; the single-seed Phase 0.5 numbers were within 0.024 of the seed means for
every arm the two runs share.

Seeds are now genuinely repeatable: the retained-set containers were changed from
sets to insertion-ordered dicts because Python's per-process string-hash
randomisation made the sampled candidate draw, and so every seeded result,
unrepeatable across processes.

## 8. Required statements

### Confirmed

- **A perfect predictor of the current training label recovers 5–21% of the
  headroom below 1% of the working set and 72–96% at 2–10% on the real
  traces.** The gap's cause is budget-dependent.
- **The objective gap is a horizon mismatch.** The best binary label horizon
  grows with the budget (60 s → 300 s → 600 s from 0.1% to 10%). At small
  budgets 88–97% of the leaves a cache holds will be reused within 600 s, so
  the fixed 600 s label cannot separate them; a next-use-time score recovers
  0.82–1.00 at every budget.
- **The population mismatch is real.** The same learned score falls from AUC
  0.86–0.94 on observed states to 0.57–0.63 on the eviction candidates, with
  prevalence rising from 4–5% to 28–64%; its regret rate equals LRU's.
- **Value per byte is not where the headroom is** (per-byte gap ≤ 0.01 on the
  real traces at every budget ≥ 0.5%, ≤ 0.06 on synthetic).
- **Sampled eviction is not why causal arms fail** (search gap 0.00–0.18).
- **The cross-workload transfer survives every standardisation variant** (real-trace cells unchanged to three decimals; cosine ≥ 0.98).
- **Results are seed-stable** (median CI half-width 0.002 over 224 cells).

### Refuted

- "Better prediction of the 600 s reuse label would fix retention." Refuted
  below 1%: the perfect label recovers ≤ 0.21 there.
- "The learned score's poor closure is a modelling failure on the global task."
  Refuted: the global task is solved (0.86–0.94); the decision task is not
  (0.57–0.63), and the online learner fitted on the decision population does no
  better.
- "The headroom is in value per byte." Refuted at every budget ≥ 0.5%.
- "The headroom is in the eviction search width." Refuted as the main term.
- "Transfer between the real workloads is an artefact of per-decision
  standardisation" (Phase 0.5 concern C-2). Refuted.
- "LFU's small-budget win reflects a good ranking of live leaves." Refuted: its
  within-decision AUC is below 0.5 at ≥ 1%; it wins where nearly every candidate
  is positive and loses everywhere else.

### Unresolved

- **Whether a causal predictor can approach the short-horizon or next-use
  oracles on the live population.** Phase 0.5 measured the 60 s label at AUC
  0.90 / 0.94 on observed states, but that is the global task; its
  decision-population AUC is not yet measured.
- **Which budget regime a real persistent tier sits in.** The traces span 59
  minutes. A fixed capacity is a smaller fraction of a longer trace's working
  set, which argues that persistent tiers are usually in the small-budget
  (residence time ≪ label horizon) regime, but that is an inference, not a
  measurement, and the timescale limit from Phase 0 still applies.
- **Prefix dependency** was held constant across arms and not separated.
- **Why the 600 s binary oracle beats the sampled next-use rule at 5%** on the
  real traces (0.86 vs 0.82, 0.96 vs 0.90). Greedy furthest-next-use is not
  optimal under variable sizes and prefix closure; the comparator is a headroom
  estimate, not a bound, and this is a case where it is beaten.
- **Synthetic** behaves unlike either real trace everywhere in this study.

### Research decision

The gate returned different cases in different budget regimes, so the decision
is stated per regime and in order.

1. **Research 1 continues, with the target changed.** This is the first step
   regardless of regime, because the fixed-horizon binary label is the wrong
   objective wherever the cache's residence time is short, and the count and
   next-use oracles are at least as good as the binary oracle everywhere except
   5%. The target moves from "reused within 600 s" to a residence-time-matched
   horizon, a reuse count, or the next-use time. This is a change of what is
   predicted, not a new policy; the first measurement is the causal
   predictability of those targets **on the decision population**, using the
   candidate logs this phase produces.
2. **Research 2 now has grounds, in the large-budget regime.** At 2–10% on the
   real traces the label is right (oracle 0.72–0.96) and the history-based
   predictor's decision-population AUC is ~0.6, unchanged by training on that
   population or by the normalisation control. The information the eviction
   decision needs among live states is not in single-state reuse history. The
   target for Research 2 is precise: raise within-decision AUC for the
   long-horizon label above the ~0.6 that history features reach, with the
   oracle closure as the ceiling.
3. **Order.** Step 1 before step 2. The target definition determines what a
   semantic signal would have to predict, and step 1 is measurable from
   existing logs in hours.

## 9. Method limits

Avoided tokens are exact-prefix trace estimates. The offline comparator has
future knowledge and greedy leaf eviction; the 5% cells where the binary oracle
beats it under sampled eviction show it is not a bound. Oracle arms break ties
by recency; a different tie-break would change their absolute closure, not the
horizon ordering, which is driven by prevalence. Candidate-set metrics use a
20,000-decision reservoir per run; pooled AUC mixes score scales across time and
the within-decision AUC is the decision-relevant number. The population ladder
caps the observed population at 40,000 states, as the prediction protocol did.
The online arm's 600 s training horizon still exceeds synthetic's usable
horizon. The real traces are 59 minutes long; nothing here speaks to
hours-to-days residence times except by the inference noted above.
