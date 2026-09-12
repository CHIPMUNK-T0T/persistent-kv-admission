# Target Change Inside Research 1 (Phase 0.9)

Question: the gap decomposition (`docs/predictability-retention-gap.md`) said
that below 1% of the working set the fixed 600 s binary label is the wrong
objective, that the best label horizon grows with the budget, and that reuse
count and next-use time are better targets for a perfect predictor. Does the
**causal** ranker inherit that when only its target changes?

Everything else is held fixed: the 23 history features, the per-decision
standardisation, the L2 penalty and its scaling, the training window and
embargo, the deployment path (`FixedModelScorer`, cached-population
normalisation as in Phase 0.5), the prefix-closed sampled-leaf eviction with
16 candidates, the seeds, and the budgets. No policy is introduced and nothing
is tuned per trace. Pipeline: `scripts/run_target_change.py`. Artifacts:
`results/paper/target_change.csv`, `target_change_config.json`,
`fig10_target_change.png`; per-seed rows and raw JSONL in
`results/target_change/` (untracked).

## 1. Design

### Targets

| arm | target | fit |
|---|---|---|
| `learned_binary_h600` | reused within 600 s (Phase 0.5 target; identical to `learned_history` of Phase 0.75, cell for cell) | L2 logistic, case-control 1:10 |
| `learned_binary_h300`, `learned_binary_h60` | reused within 300 s / 60 s | same |
| `learned_count_h60`, `learned_count_h600` | log(1 + number of uses within 60 s / 600 s) | L2 ridge, closed form, same penalty scale |
| `learned_next_use_h600` | −log(1 + min(seconds to next use, 600)); never used again sits at the floor | same |
| `learned_matched` | switches between the fitted binary rankers by the cache's residence time (below) | no parameter |

Fit horizons are clipped by the trace as in every earlier phase: the synthetic
trace supports 300 s at most, so its 600 s fits become `count_h300` and
`next_use_h300`, and `binary_h600` does not exist there. The graded targets
are fitted on a uniform quota sample of the observed population (about
144,000 rows per fit); the binary targets on a 1:10 case-control sample of the
same snapshots.

### Horizon-matched arm

Little's law gives the mean residence time of an inserted byte as
capacity ÷ byte insertion rate. `HorizonMatchedScorer` tracks the cumulative
bytes inserted into the cache since the start of the replay, divides the
capacity by that rate, and scores with the fitted binary ranker whose horizon
is nearest in log space. It has no parameter. The horizons it can choose from
are the ones fitted: 60, 300, 600 s.

### Protocol

Five seeds (0–4) for every sampled arm, budgets 0.1 / 0.25 / 1 / 2 / 5% of the
packed unique-state bytes, three traces, 665 replays. `HeadroomClosure`,
seed-paired differences, and 95% CIs (Student-t over seeds) as in Phase 0.75.
LRU and LFU are re-run through the same sampled eviction with the same seeds;
the heap offline-next-use comparator is the denominator.

## 2. Closure by target

Mean HeadroomClosure over five seeds. CI half-widths are ≤ 0.012 on the real
traces except where marked; the exact values are in `target_change.csv`.
Oracle columns are the Phase 0.75 arms with the same target (perfect knowledge
of that target, same eviction).

### conversation

| budget | LFU | binary 600 s | binary 300 s | binary 60 s | count 60 s | count 600 s | next use | matched | oracle 60 s / 600 s / count / next use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1% | 0.168 | 0.009 | 0.008 | 0.017 | 0.099 | 0.079 | 0.075 | 0.017 | 0.57 / 0.06 / 0.48 / 0.99 |
| 0.25% | 0.187 | 0.064 | 0.045 | 0.101 | 0.145 | 0.149 | 0.148 | 0.101 | 0.72 / 0.13 / 0.55 / 0.94 |
| 1% | 0.215 | 0.193 | 0.189 | 0.230 | 0.197 | 0.231 | **0.249** | 0.230 | 0.46 / 0.43 / 0.62 / 0.87 |
| 2% | 0.157 | 0.253 | 0.256 | **0.270** | 0.209 | 0.241 | 0.254 | 0.270 | 0.41 / 0.72 / 0.77 / 0.85 |
| 5% | −0.213 | 0.084 | 0.070 | 0.089 | 0.050 | 0.084 | 0.076 | 0.071 | 0.23 / 0.86 / 0.86 / 0.82 |

### tool-agent

| budget | LFU | binary 600 s | binary 300 s | binary 60 s | count 60 s | count 600 s | next use | matched | oracle 60 s / 600 s / count / next use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1% | 0.825 | 0.792 | 0.783 | 0.790 | 0.790 | 0.810 | 0.814 | 0.790 | 0.90 / 0.68 / 0.89 / 1.00 |
| 0.25% | 0.271 | 0.139 | 0.120 | 0.172 | 0.116 | 0.240 (±0.015) | **0.253** | 0.172 | 0.76 / 0.22 / 0.58 / 0.94 |
| 1% | 0.214 | 0.195 | 0.179 | 0.211 | 0.083 (±0.022) | 0.222 | **0.244** | 0.211 | 0.47 / 0.44 / 0.62 / 0.86 |
| 2% | 0.146 | 0.252 | 0.251 | **0.256** | 0.124 | 0.225 | 0.241 | 0.256 | 0.42 / 0.76 / 0.80 / 0.85 |
| 5% | −0.274 | **0.128** | 0.114 | 0.107 | −0.133 (±0.015) | 0.060 | 0.095 | 0.114 | 0.29 / 0.96 / 0.96 / 0.90 |

### synthetic

No causal arm exceeds 0.09 at any budget. `count_h60` is the only arm that
never falls below LRU (0.013–0.059); `next_use_h300` reaches 0.087 at 1%;
the binary arms are negative from 0.25% up (down to −0.28 at 2% for 60 s).
LFU is 0.06–0.12 at ≤ 1%. The oracles for the same targets are 0.31–0.44
(count) and 0.96–1.00 (next use), so this trace remains a pure signal problem
for history features, as in every earlier phase.

## 3. Seed-paired differences

Difference in closure between an arm and the 600 s binary target, paired by
seed, mean ± 95% CI:

| trace | arm | 0.1% | 0.25% | 1% | 2% | 5% |
|---|---|---:|---:|---:|---:|---:|
| conversation | next use | +0.066 ±0.000 | +0.084 ±0.003 | +0.056 ±0.005 | +0.001 ±0.003 | −0.009 ±0.010 |
| conversation | count 600 s | +0.071 ±0.000 | +0.085 ±0.005 | +0.038 ±0.005 | −0.013 ±0.004 | −0.000 ±0.005 |
| conversation | count 60 s | +0.091 ±0.000 | +0.081 ±0.014 | +0.004 ±0.008 | −0.044 ±0.008 | −0.035 ±0.008 |
| conversation | binary 60 s | +0.008 ±0.000 | +0.037 ±0.004 | +0.037 ±0.004 | +0.016 ±0.004 | +0.005 ±0.009 |
| tool-agent | next use | +0.022 ±0.000 | +0.114 ±0.004 | +0.050 ±0.006 | −0.011 ±0.009 | −0.033 ±0.004 |
| tool-agent | count 600 s | +0.018 ±0.000 | +0.101 ±0.013 | +0.028 ±0.007 | −0.027 ±0.011 | −0.068 ±0.014 |
| tool-agent | count 60 s | −0.002 ±0.000 | −0.023 ±0.006 | −0.112 ±0.021 | −0.128 ±0.003 | −0.261 ±0.013 |
| tool-agent | binary 60 s | −0.002 ±0.000 | +0.033 ±0.006 | +0.016 ±0.006 | +0.004 ±0.008 | −0.021 ±0.012 |

(At 0.1% the five seeds coincide: with so few retained leaves the sampled
candidate set is the whole leaf set and the replay is deterministic.)

The same arms against LFU, paired by seed:

| trace | arm | 0.1% | 0.25% | 1% | 2% | 5% |
|---|---|---:|---:|---:|---:|---:|
| conversation | next use | −0.093 | −0.039 ±0.004 | +0.034 ±0.005 | +0.097 ±0.002 | +0.289 ±0.004 |
| conversation | binary 60 s | −0.151 | −0.086 ±0.004 | +0.015 ±0.004 | +0.112 ±0.006 | +0.302 ±0.009 |
| tool-agent | next use | −0.011 | −0.018 ±0.003 | +0.031 ±0.003 | +0.095 ±0.004 | +0.369 ±0.011 |
| tool-agent | count 600 s | −0.015 | −0.030 ±0.011 | +0.009 ±0.003 | +0.079 ±0.007 | +0.333 ±0.010 |

## 4. What the target change buys, and what it does not

**Where the oracle said the target was the problem (≤ 0.25%), changing the
target helps, by a small absolute amount.** On conversation the count and
next-use targets lift closure from 0.01–0.06 to 0.08–0.15; on tool-agent at
0.25% from 0.14 to 0.24–0.25. Every one of those differences is outside its
seed CI. At 1% the next-use target is the best causal arm on both real traces
(0.249 / 0.244, +0.05 to +0.06 over the 600 s label) and is above LFU there
(+0.03). At 2% the targets are within ±0.02 of each other; at 5% nothing beats
the 600 s label and the count targets are worse (tool-agent: −0.07 and −0.26).

**It does not close the gap.** Three ways to see the same thing:

- *Against LFU.* Below 1% LFU is still ahead of the best causal target on both
  real traces (conversation −0.04 to −0.09, tool-agent −0.01 to −0.03 at
  0.25%). The best causal arm still beats LRU and LFU only from 1% upward,
  where it already did with the old target.
- *Against the same-target oracle.* The residual, oracle − learned for the
  same target, is 0.39–0.40 for count at 0.1–1% on conversation and
  0.34–0.40 at 0.25–1% on tool-agent; for next use it is 0.62–0.92 on
  conversation and 0.62–0.69 on tool-agent at 0.25–1%. The causal ranker
  reaches 16–37% of its own count oracle and 8–29% of its own next-use oracle
  at ≤ 1% on conversation, and 36–42% / 27–28% at 0.25–1% on tool-agent.
  Once the target is right, the causal side is the limit.
- *Best cell overall.* The best closure of any causal arm at any budget from
  0.25% up moves from 0.253 (600 s label) to 0.270 (60 s label, 2%,
  conversation) and from 0.252 to 0.256 on tool-agent. (At 0.1% on tool-agent
  every arm, LFU included, sits at 0.78–0.83 with the 60 s oracle at 0.90;
  that cell is not a target question.)

So the objective gap measured in Phase 0.75 was real, and the causal ranker
does recover part of it by predicting a better target. But the part it
recovers is small because the ranker cannot rank the better target well on
the decision population either: the gap that remains after the target change
is a signal gap on the new target. This is consistent with §5b of the gap
document, where a higher-capacity model on the same 23 features did not rank
the eviction candidates better than the linear one.

## 5. The horizon-matched arm

Little's-law residence times at the end of the replay, real traces:

| budget | 0.1% | 0.25% | 1% | 2% | 5% |
|---|---:|---:|---:|---:|---:|
| residence (s) | 2 | 6 | 25 | 52 | 141 |

(Synthetic: 0–20 s.) The insertion rate includes re-insertions of evicted
states, so the residence time is short: at ≤ 2% the nearest fitted horizon is
always 60 s and the matched arm is the 60 s arm to three decimals. At 5% it
uses 300 s for 88–90% of the groups and lands between or below the fixed arms
(conversation 0.071 vs 0.089 / 0.084 for 60 s / 600 s; tool-agent 0.114 vs
0.107 / 0.128). The residence time that this estimate produces is much
shorter than the label horizon the oracle sweep found best (60 s at ≤ 0.25%,
300 s at 1–2%, 600 s at 5%): the mean residence of an inserted byte is not
the survival time of the states worth keeping. As specified, with these three
horizons, the matched arm adds nothing over the fixed 60 s label. It was not
re-tuned.

## 6. What the fitted rankers look like

Standardised coefficient vectors of the fits used in the replay (refitted
deterministically; stored in `target_change_config.json`), cosine similarity,
conversation / tool-agent:

| pair | cosine |
|---|---:|
| binary 60 s vs binary 600 s | 0.83 / 0.90 |
| binary 600 s vs next use | 0.65 / 0.67 |
| binary 600 s vs count 600 s | 0.65 / 0.46 |
| binary 600 s vs count 60 s | 0.34 / 0.16 |
| count 600 s vs next use | 0.82 / 0.79 |

The binary fits are recency + fan-out + windowed frequency (the 60 s fit adds
"recent uses over lifetime"). The 60 s count fit is dominated by lifetime
frequency, which is why it behaves like LFU: best of the causal arms at 0.1%
on conversation, and the arm that collapses at 5% (tool-agent −0.13, LFU
−0.27). The 600 s count and next-use fits are 600 s-window frequency, recency,
and age; they are the two arms that gain most at 0.25–1%.

## 7. Required statements

### Confirmed

- **The objective gap is partly recoverable by a causal model.** With the
  same features and ranker, the count and next-use targets recover
  0.07–0.09 more headroom than the 600 s binary target at 0.1–0.25% on
  conversation and 0.10–0.11 at 0.25% on tool-agent, outside the seed CIs.
- **Next-use time is the best causal target at 1%** on both real traces
  (0.249 / 0.244), the first budget where a causal arm beats LFU.
- **The horizon ordering carries over.** The 60 s binary target is better
  than 600 s at 0.25–2% on both real traces (+0.02 to +0.04) and no better at
  5%; the count targets are better below 1% and worse at 5%.
- **Results are seed-stable** (median CI half-width 0.004, maximum 0.022,
  over the 90 sampled-arm cells on the real traces).

### Refuted

- "Changing the target closes the gap." The best causal arm reaches 16–37%
  of its own count oracle and ≤ 29% of its own next-use oracle at ≤ 1% on
  conversation, is still below LFU below 1% on both real traces, and its best
  cell moves from 0.253 to 0.270.
- "A Little's-law residence estimate selects the right label horizon." With
  the three fitted horizons it selects 60 s at every budget ≤ 2% and adds
  nothing over the fixed 60 s arm; at 5% it is between or below the fixed
  arms.
- "The synthetic trace becomes learnable with a better target." No arm
  exceeds 0.09.

### Unresolved

- **The remaining gap is a signal gap on the new target, on the decision
  population.** Whether it is the features or the fit population has not
  been separated for the graded targets: the ranker here is fitted on the
  observed population, and the candidate logs (which record next-use deltas)
  allow the regression analogue of the §5b check. Not run.
- **The 5% regime on the real traces is untouched**: the best causal arm is
  0.08–0.13 against oracles of 0.82–0.96, for every target.
- **Which budget regime a real persistent tier sits in** is still an
  inference from 59-minute traces (Phase 0.75, Unresolved).
- The sample-width (search) gap of 0.11–0.18 at 1–5% is unchanged and
  untested.

### Research decision

The Phase 0.9 gate asked for a causal target that recovers materially more
headroom than the 600 s binary target where the oracle said the target was the
problem. It is met in direction and not in magnitude: the target change is
kept (next-use time at 1%, count or next-use below 1%, 600 s binary at 5%),
and the same causal ranker on the new targets still leaves most of the
headroom, below LFU at small budgets. Within Research 1 the next question is
therefore no longer "which target" but "why the history features cannot rank
the right target on the eviction candidates", which the candidate logs can
address without a new policy. That choice is recorded in
`docs/experiment-plan.md` when it is made.

## 8. Method limits

Everything from the gap document applies: exact-prefix token estimates, a
greedy comparator that is not a bound (the 5% cells where the 600 s oracle
beats it), recency tie-breaks, 16-candidate sampled eviction. Specific to this
phase: the graded targets are fitted by ridge regression on a uniform sample
of the observed population while the binary targets use case-control sampling,
so the two families differ in row weighting as well as target; the graded
targets clip at the horizon, so "never used again" and "used after 600 s" are
the same value; the matched arm's residence estimate counts every insertion
including re-insertions; and the horizon set is the three values the traces
support, with nothing below 60 s. The real traces are 59 minutes long.
