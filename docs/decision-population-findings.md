# Phase 0.97 — Decision-population-matched retention learning

**Question.** Does training on the actual retention-decision population
convert global reuse predictability into retention utility?

Phases 0.5–0.95 established that the same 23-feature causal history
representation predicts reuse well on the observed population (AUC
0.86–0.94) and progressively worse as the population narrows to the states
a cache actually decides about, and that its global-trained ranking does not
beat LRU / LFU under a byte budget. Two explanations were left entangled:
the representation lacks the signal that separates hard candidates, or the
model was fitted on the wrong distribution. This phase changes only the
training population and measures both predictive quality and replay utility
on the same budget axis. The design, metrics, and interpretation cases were
fixed in `docs/experiment-plan.md` before any result existed.

## 1. Design

**Setting (fixed, Phase 0.95).** Heap-LRU L1 at 0.25 / 1 / 2% of the
working set; every L1 eviction is offered to an exclusive union L2 of 1 × or
4 × L1; tree hit rule; evaluation window = the last 40% of each trace
(`measure_from_ms = split_ms`); packed size model, 2048 bytes per token.

**L2 mechanism for scored arms (fixed).** Every arm whose score varies with
time runs through one sampled mechanism: when the store is over capacity
after admitting the arriving victim, the decision set is the arriving victim
plus a uniform sample of up to 16 other residents; all are scored at that
moment and the first minimum leaves. If the arriving victim is the minimum
in the first round it is rejected. Generic policies (LRU, LFU, 2-hit) are run
through the same mechanism (`lru_s`, `lfu_s`, `lru_2hit_s`); their heap
versions and the heap offline comparator (`offline_next_use`, prefix-first
ties) are kept as references. Five seeds drive the sampling.

**Representation, model, targets (fixed).** The 23 causal features of
`temporal.py`; `LogisticRanker` (Newton, L2 = 0.01) for the binary target,
`RidgeRanker` (L2 = 0.01) for the graded targets; targets exactly as in
Phase 0.9: binary reuse within H, `log1p(count within H)`,
`−log1p(min(seconds to next use, H))`, H = 600 s (300 s on synthetic by the
existing usability rule). Time split at 60% of the trace with a horizon
embargo: training rows satisfy `t + H ≤ split`, test rows `split ≤ t` and
`t + H ≤ end`.

**Training populations (the only varied factor).**

| population | rows | standardisation at fit | scoring at replay |
|---|---|---|---|
| A_pd (control, Phase 0.9 exact) | all states observed so far at 24 training snapshots, case-control 10 : 1 for binary | per snapshot z-score | `FixedModelScorer`, population normaliser over the L2 residents |
| A_none | the same rows | raw features, ranker-internal training statistics | raw features, ranker statistics |
| B (L1 victims) | every L1 eviction in the training split, features at eviction | raw, ranker statistics | raw, ranker statistics |
| C_lru / C_lfu | decision sets (arriving victim + sampled residents) logged while sampled L2-LRU / L2-LFU runs on the training split of the same cell, reservoir cap 40,000 decisions | raw, ranker statistics | raw, ranker statistics |
| C_union | C_lru rows plus C_lfu rows whose (time, state) does not occur in C_lru | raw, ranker statistics | raw, ranker statistics |

A_none exists so that the population effect (A_none → B → C) is not
confounded with the standardisation choice (A_pd → A_none). Behaviour
policies that generate C are causal; the policy identity is not a feature;
no future-aware policy generates training data. Rows are capped at 150,000
per fit by uniform subsampling (seed 0).

**Predictive evaluation (secondary).** Every fitted ranker × target is scored
on three test-split populations: observed (Phase 0.5 shared snapshots),
victims (test-split eviction events of the same L1 budget), and candidates
(test-split decision sets of the LRU and of the LFU behaviour log of the
same cell). Binary: pooled AUC, within-decision AUC (micro / macro), and the
share of decisions whose evicted state has the lowest label. Graded: pooled
Spearman, mean within-decision Spearman over decisions with ≥ 2 distinct
labels (constant-label decisions counted separately), the same eviction
share. This off-policy evaluation is a common condition for every model;
because a learned policy changes the retained set, every sampled arm is
also evaluated on-policy at seed 0: the decisions it actually faced, scored
with the values the store used, including the share of evictions that
removed a state reused within H while a never-reused-within-H candidate was
available. Train-split candidate metrics and fit convergence (Newton
iterations, ridge condition number) are recorded so that a null result can
be told from a fitting failure; the LRU and LFU keys' own within-decision
metrics on their logs are the generic references.

**Replay utility (primary).** Extra avoided prefill tokens over L1 alone as a
share of evaluation-window input tokens, and headroom closure
= (arm − lru_s) / (heap offline − lru_s) per seed, mean and 95% CI over
five seeds.

**Interpretation cases, fixed before the run and amended before any result
was inspected** (per target, real traces). R = within-decision AUC (macro,
binary) or mean within-decision Spearman (graded, decisions with ≥ 2
distinct labels) on the test-split decisions of the sampled L2-LRU
behaviour log of the cell; R_high = 0.70 / 0.30; closure differences are
against A_none. Case A, mismatch was the problem: closure(C) ≥
closure(A_none) + 0.10 and ≥ the best generic L2 in ≥ 4 of 6 cells. Case B,
prediction adequate but not converted: R(C) ≥ R_high or R(C) − R(A_none) ≥
0.05, while closure improves by < 0.05 in most cells. Case C, representation
is the limit: R(C) < R_high, R(C) − R(A_none) < 0.05, the fit converged with
train-split R also < R_high, and closure(C) < best generic + 0.05.
Otherwise unresolved. The on-policy R of each learned arm on its own
decisions (seed 0) is reported next to the off-policy R; a gap above 0.05
flags the off-policy diagnosis.

Pipeline: `scripts/run_decision_population.py`; outputs in
`results/paper/decision_population_*.csv`, `fig14_decision_population.png`,
`fig15_population_ladder.png`.
