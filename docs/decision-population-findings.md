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

## 2. The run

Commit `379aa20`, `python3 scripts/run_decision_population.py data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl data/raw/synthetic_trace.jsonl --workers 24`, 2026-09-13, 5,561 s wall clock on 24 workers (1,962 replays in 5,335 s). Nothing in the code, grid, thresholds, seeds or arms changed between the pre-registration amendment and the end of the run. Checks made on the outputs before any interpretation:

- The four heap reference arms (LRU, LFU, 2-hit, offline) reproduce the Phase 0.95 tree / union rows exactly in all 72 trace × cell × arm combinations.
- The A_pd control reproduces the Phase 0.9 coefficients to 0.0 for all three targets on both real traces (the synthetic Phase 0.9 coefficients were never stored, so that cell is unverifiable).
- All 207 fits converged (logistic ≤ 8 Newton steps, ridge condition number ≤ 1,276); no NaN in any predictive, on-policy or replay row.
- `admissions + rejections + already_held = L1 evictions` in all 2,250 per-seed replay rows; in all 1,134 on-policy rows the recorded arg-min equals the state actually evicted (rate 1.000); the LRU and LFU keys' on-policy metrics on their own decisions differ from their off-policy metrics on the logged decisions by ≤ 0.007, so the two evaluations agree when the policy is the same.
- No closure spread across seeds above 0.10 (largest 0.098, toolagent 2% × 4, B / count).

## 3. Replay utility (primary metric)

Headroom closure = (arm − sampled L2-LRU) / (heap offline − sampled L2-LRU), mean over five seeds; CI95 half-widths are ≤ 0.02 for every learned arm on the real traces except at 2% × 4 (≤ 0.045). The next-use target is shown for the main-comparison populations because it is the best or within 0.02 of the best target for each population in most cells; all targets are in `decision_population_replay.csv` and fig14.

| trace | L1 × L2 | best heap generic | best sampled generic | A_none nxt | B nxt | C_lru nxt | C_lfu nxt | C_union nxt | best learned (arm) | offline, share of input |
|---|---|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.193 (LFU) | 0.117 (LFU) | 0.169 | 0.152 | 0.173 | 0.104 | 0.112 | 0.173 ± 0.006 (C_lru nxt) | 8.4 |
| conversation | 0.25% × 4 | 0.228 (LFU) | 0.195 (LFU) | 0.211 | 0.202 | 0.190 | 0.181 | 0.192 | 0.228 ± 0.004 (A_none cnt) | 21.2 |
| conversation | 1% × 1 | 0.208 (2-hit) | 0.152 (LFU) | 0.193 | 0.184 | 0.135 | 0.149 | 0.186 | 0.208 ± 0.009 (A_pd cnt) | 22.1 |
| conversation | 1% × 4 | 0.136 (2-hit) | 0.085 (2-hit) | 0.121 | 0.066 | 0.016 | 0.101 | 0.141 | 0.141 ± 0.011 (C_union nxt) | 34.4 |
| conversation | 2% × 1 | 0.179 (2-hit) | 0.115 (2-hit) | 0.110 | -0.295 | 0.011 | 0.117 | 0.157 | 0.157 ± 0.003 (C_union nxt) | 27.4 |
| conversation | 2% × 4 | 0.148 (LRU) | 0.000 (LRU) | 0.169 | -1.071 | 0.176 | 0.175 | 0.181 | 0.208 ± 0.008 (A_pd cnt) | 32.2 |
| toolagent | 0.25% × 1 | 0.186 (LFU) | 0.110 (LFU) | 0.161 | 0.126 | 0.143 | 0.098 | 0.111 | 0.185 ± 0.005 (A_pd cnt) | 6.5 |
| toolagent | 0.25% × 4 | 0.222 (LFU) | 0.185 (LFU) | 0.205 | 0.198 | 0.169 | 0.166 | 0.177 | 0.218 ± 0.004 (A_none cnt) | 15.6 |
| toolagent | 1% × 1 | 0.229 (2-hit) | 0.164 (2-hit) | 0.196 | 0.151 | 0.118 | 0.140 | 0.170 | 0.206 ± 0.005 (A_pd nxt) | 15.4 |
| toolagent | 1% × 4 | 0.169 (2-hit) | 0.104 (2-hit) | 0.117 | 0.062 | 0.002 | 0.107 | 0.128 | 0.132 ± 0.014 (A_pd bin) | 22.4 |
| toolagent | 2% × 1 | 0.190 (2-hit) | 0.129 (2-hit) | 0.090 | -0.306 | -0.054 | 0.088 | 0.128 | 0.128 ± 0.008 (C_union nxt) | 18.9 |
| toolagent | 2% × 4 | 0.193 (LRU) | 0.000 (LRU) | 0.222 | -1.495 | 0.250 | 0.237 | 0.266 | 0.266 ± 0.009 (C_union nxt) | 20.4 |

The same cells in extra avoided prefill tokens over L1 alone, as a share of evaluation-window input tokens:

| trace | L1 × L2 | lru_s | best heap generic | best learned | offline | learned − best heap |
|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.1 | 1.7 (LFU) | 1.6 | 8.4 | -0.2 |
| conversation | 0.25% × 4 | 1.1 | 5.7 (LFU) | 5.7 | 21.2 | +0.0 |
| conversation | 1% × 1 | 2.0 | 6.2 (2-hit) | 6.2 | 22.1 | +0.0 |
| conversation | 1% × 4 | 13.2 | 16.1 (2-hit) | 16.2 | 34.4 | +0.1 |
| conversation | 2% × 1 | 7.8 | 11.3 (2-hit) | 10.9 | 27.4 | -0.4 |
| conversation | 2% × 4 | 19.7 | 21.6 (LRU) | 22.3 | 32.2 | +0.7 |
| toolagent | 0.25% × 1 | 0.8 | 1.9 (LFU) | 1.8 | 6.5 | -0.0 |
| toolagent | 0.25% × 4 | 1.5 | 4.7 (LFU) | 4.6 | 15.6 | -0.1 |
| toolagent | 1% × 1 | 1.4 | 4.6 (2-hit) | 4.3 | 15.4 | -0.3 |
| toolagent | 1% × 4 | 9.6 | 11.7 (2-hit) | 11.3 | 22.4 | -0.5 |
| toolagent | 2% × 1 | 5.8 | 8.3 (2-hit) | 7.4 | 18.9 | -0.8 |
| toolagent | 2% × 4 | 14.1 | 15.3 (LRU) | 15.8 | 20.4 | +0.5 |

- **No learned L2, on any training population or target, closes more than 0.27 of the headroom in any real cell**; the best learned arm sits at 0.13–0.27 and leaves 73–87% of the offline − LRU gap in every cell. In input tokens the best learned arm is within −0.8 to +0.7 points of the best generic heap policy. Its mean is above that policy in five cells: conversation 0.25% × 4 (+0.005), 1% × 1 (+0.006), 1% × 4 (+0.10), 2% × 4 (+0.74) and toolagent 2% × 4 (+0.47); the first two are inside the five-seed CI95 (±0.05 / ±0.19), the other three are outside it. At 2% × 4 heap LFU and 2-hit fall below LRU.
- **The candidate-trained rankers do not improve on the global one.** C_lru, the population that matches the pre-registered R, has the lowest closure of the C variants: below A_none in 9–12 of 12 cells per target, and below sampled LRU (negative) at 1% × 4, 2% × 1 and 2% × 4 on both traces for the binary and count targets (toolagent 2% × 1 for next-use). No C variant exceeds A_none by more than 0.05 in any cell on the binary or next-use targets (C_union is within ±0.05 of A_none in 11 of 12 next-use cells). The only larger gains are on the count target at toolagent 2% × 4 (+0.18 to +0.47), where A_none's count fit is the outlier described in §6.
- **The victim-trained ranker B collapses at L1 = 2%** (closure −0.27 to −0.35 at 2% × 1 and −0.86 to −1.87 at 2% × 4 on the two traces) although it tracks A_none within 0.05 at 0.25%. This was not pre-registered; its cause was not examined here.
- The sampled mechanism itself costs closure relative to the heap versions of the same policy: 0.00–0.19 for LRU, 0.03–0.14 for LFU, 0.02–0.08 for 2-hit, growing with the budget. The learned arms run on the sampled mechanism, so at 2% × 4 the learned closure of 0.19–0.27 is largely the recovery of the sampling loss (heap LRU is at 0.15–0.19 on the same scale).

## 4. Predictive quality: off-policy and on-policy

R is the pre-registered ranking metric (within-decision AUC macro for binary; mean within-decision Spearman over decisions with ≥ 2 distinct labels for the graded targets), evaluated off-policy on the test-split decisions of the sampled L2-LRU behaviour log of the same cell, and on-policy on the decisions the learned arm itself produced at seed 0, scored with the values the store used. "C_lru train" is C_lru's R on its own training decisions. The last column is the LRU key's own R on its log and on its own on-policy stream.

**binary (R_high = 0.70)**

| trace | L1 × L2 | A_none off / on | B off / on | C_lru off / on | C_lfu off / on | C_union off / on | C_lru train | LRU key off / LRU on |
|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.725 / 0.575 | 0.741 / 0.395 | 0.741 / 0.414 | 0.646 / 0.241 | 0.715 / 0.271 | 0.709 | 0.499 / 0.499 |
| conversation | 0.25% × 4 | 0.713 / 0.616 | 0.740 / 0.529 | 0.739 / 0.517 | 0.666 / 0.581 | 0.699 / 0.582 | 0.700 | 0.502 / 0.504 |
| conversation | 1% × 1 | 0.720 / 0.611 | 0.731 / 0.564 | 0.736 / 0.513 | 0.664 / 0.596 | 0.700 / 0.609 | 0.692 | 0.507 / 0.505 |
| conversation | 1% × 4 | 0.738 / 0.669 | 0.744 / 0.660 | 0.745 / 0.633 | 0.724 / 0.684 | 0.736 / 0.678 | 0.695 | 0.562 / 0.559 |
| conversation | 2% × 1 | 0.731 / 0.629 | 0.716 / 0.554 | 0.741 / 0.611 | 0.694 / 0.655 | 0.725 / 0.657 | 0.690 | 0.534 / 0.534 |
| conversation | 2% × 4 | 0.744 / 0.695 | 0.691 / 0.624 | 0.747 / 0.703 | 0.737 / 0.704 | 0.743 / 0.705 | 0.701 | 0.605 / 0.608 |
| toolagent | 0.25% × 1 | 0.744 / 0.581 | 0.757 / 0.394 | 0.758 / 0.417 | 0.683 / 0.244 | 0.710 / 0.263 | 0.722 | 0.487 / 0.488 |
| toolagent | 0.25% × 4 | 0.726 / 0.617 | 0.751 / 0.534 | 0.752 / 0.523 | 0.702 / 0.579 | 0.715 / 0.575 | 0.712 | 0.501 / 0.499 |
| toolagent | 1% × 1 | 0.728 / 0.611 | 0.744 / 0.546 | 0.747 / 0.510 | 0.702 / 0.602 | 0.716 / 0.600 | 0.705 | 0.504 / 0.503 |
| toolagent | 1% × 4 | 0.754 / 0.670 | 0.761 / 0.653 | 0.763 / 0.648 | 0.750 / 0.688 | 0.756 / 0.678 | 0.715 | 0.561 / 0.563 |
| toolagent | 2% × 1 | 0.746 / 0.636 | 0.748 / 0.459 | 0.755 / 0.594 | 0.725 / 0.664 | 0.744 / 0.662 | 0.705 | 0.537 / 0.537 |
| toolagent | 2% × 4 | 0.760 / 0.709 | 0.728 / 0.550 | 0.768 / 0.720 | 0.761 / 0.720 | 0.764 / 0.718 | 0.731 | 0.618 / 0.618 |

**count (R_high = 0.30)**

| trace | L1 × L2 | A_none off / on | B off / on | C_lru off / on | C_lfu off / on | C_union off / on | C_lru train | LRU key off / LRU on |
|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.351 / 0.033 | 0.377 / -0.045 | 0.380 / -0.018 | 0.222 / -0.294 | 0.345 / -0.317 | 0.322 | -0.006 / -0.006 |
| conversation | 0.25% × 4 | 0.354 / 0.231 | 0.390 / 0.131 | 0.394 / 0.089 | 0.275 / 0.077 | 0.322 / 0.054 | 0.324 | 0.006 / 0.008 |
| conversation | 1% × 1 | 0.366 / 0.216 | 0.377 / 0.075 | 0.381 / -0.007 | 0.268 / 0.140 | 0.317 / 0.115 | 0.307 | 0.013 / 0.009 |
| conversation | 1% × 4 | 0.370 / 0.312 | 0.348 / 0.257 | 0.383 / 0.210 | 0.333 / 0.335 | 0.365 / 0.328 | 0.307 | 0.090 / 0.087 |
| conversation | 2% × 1 | 0.364 / 0.292 | 0.308 / 0.131 | 0.376 / 0.179 | 0.283 / 0.293 | 0.332 / 0.293 | 0.300 | 0.051 / 0.051 |
| conversation | 2% × 4 | 0.337 / 0.288 | 0.213 / 0.141 | 0.343 / 0.263 | 0.325 / 0.297 | 0.334 / 0.296 | 0.286 | 0.142 / 0.147 |
| toolagent | 0.25% × 1 | 0.381 / 0.067 | 0.395 / -0.039 | 0.403 / 0.006 | 0.257 / -0.328 | 0.338 / -0.336 | 0.338 | -0.026 / -0.023 |
| toolagent | 0.25% × 4 | 0.378 / 0.278 | 0.390 / 0.148 | 0.405 / 0.138 | 0.325 / 0.071 | 0.345 / 0.064 | 0.332 | 0.003 / 0.001 |
| toolagent | 1% × 1 | 0.390 / 0.266 | 0.396 / 0.009 | 0.395 / -0.032 | 0.314 / 0.115 | 0.347 / 0.120 | 0.324 | 0.008 / 0.005 |
| toolagent | 1% × 4 | 0.370 / 0.315 | 0.368 / 0.228 | 0.406 / 0.223 | 0.369 / 0.345 | 0.390 / 0.337 | 0.333 | 0.089 / 0.091 |
| toolagent | 2% × 1 | 0.354 / 0.304 | 0.371 / -0.018 | 0.392 / 0.169 | 0.321 / 0.299 | 0.360 / 0.293 | 0.320 | 0.053 / 0.055 |
| toolagent | 2% × 4 | 0.308 / 0.290 | 0.261 / -0.028 | 0.362 / 0.291 | 0.351 / 0.317 | 0.358 / 0.315 | 0.317 | 0.156 / 0.156 |

**next-use (R_high = 0.30)**

| trace | L1 × L2 | A_none off / on | B off / on | C_lru off / on | C_lfu off / on | C_union off / on | C_lru train | LRU key off / LRU on |
|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.322 / 0.052 | 0.359 / 0.009 | 0.359 / 0.052 | 0.277 / -0.244 | 0.342 / -0.151 | 0.312 | -0.005 / -0.007 |
| conversation | 0.25% × 4 | 0.315 / 0.177 | 0.367 / 0.183 | 0.371 / 0.110 | 0.280 / 0.083 | 0.338 / 0.073 | 0.314 | -0.000 / 0.001 |
| conversation | 1% × 1 | 0.334 / 0.159 | 0.357 / 0.187 | 0.362 / 0.009 | 0.278 / 0.105 | 0.331 / 0.084 | 0.294 | 0.008 / 0.005 |
| conversation | 1% × 4 | 0.349 / 0.279 | 0.307 / 0.206 | 0.370 / 0.200 | 0.341 / 0.292 | 0.362 / 0.285 | 0.289 | 0.094 / 0.090 |
| conversation | 2% × 1 | 0.340 / 0.220 | 0.323 / -0.014 | 0.363 / 0.159 | 0.297 / 0.229 | 0.352 / 0.233 | 0.280 | 0.053 / 0.053 |
| conversation | 2% × 4 | 0.331 / 0.285 | 0.219 / 0.075 | 0.338 / 0.283 | 0.326 / 0.298 | 0.334 / 0.299 | 0.277 | 0.144 / 0.148 |
| toolagent | 0.25% × 1 | 0.350 / 0.074 | 0.374 / -0.085 | 0.377 / -0.045 | 0.279 / -0.211 | 0.351 / -0.164 | 0.329 | -0.025 / -0.025 |
| toolagent | 0.25% × 4 | 0.331 / 0.197 | 0.368 / 0.140 | 0.375 / 0.060 | 0.320 / 0.083 | 0.347 / 0.080 | 0.319 | -0.002 / -0.006 |
| toolagent | 1% × 1 | 0.347 / 0.174 | 0.366 / 0.221 | 0.372 / 0.025 | 0.318 / 0.086 | 0.348 / 0.103 | 0.306 | 0.003 / 0.000 |
| toolagent | 1% × 4 | 0.364 / 0.276 | 0.293 / 0.184 | 0.391 / 0.212 | 0.371 / 0.290 | 0.384 / 0.275 | 0.318 | 0.091 / 0.094 |
| toolagent | 2% × 1 | 0.349 / 0.231 | 0.352 / -0.034 | 0.374 / 0.109 | 0.325 / 0.222 | 0.370 / 0.233 | 0.301 | 0.055 / 0.057 |
| toolagent | 2% × 4 | 0.347 / 0.300 | 0.272 / 0.042 | 0.359 / 0.301 | 0.350 / 0.319 | 0.356 / 0.317 | 0.310 | 0.158 / 0.157 |

- **Off-policy, on the LRU behaviour log, the global and the LRU-log-trained rankers clear R_high everywhere**: A_none, A_pd and C_lru in 36 of 36 trace × cell × target evaluations, C_union in 34, B in 30, C_lfu (trained on the LFU log) in 21. C_lru adds 0.00–0.06 to A_none there and is the best model on that log in every cell; C_lfu is best on the LFU log (fig15). The gain of matching the training population to the evaluation log is real but ≤ 0.05.
- **On-policy, on the decisions each learned arm creates for itself, R is far lower**: 0.02–0.67 below the off-policy value of the same model, beyond the pre-registered 0.05 tolerance in 28–36 of 36 evaluations for A_none, B, C_lru and C_union, and below R_high in 30–36 of 36 for those populations (4–6 of the 6 cells for every target). The gap is largest at 0.25% × 1, where the candidate-trained arms rank their own decision sets inversely to reuse (binary AUC 0.24–0.27, graded Spearman −0.15 to −0.34) and even A_none falls to 0.58 / 0.03–0.07. The gap shrinks with the budget and is 0.02–0.08 at 2% × 4 (B excepted: 0.07–0.29).
- The behaviour keys do not show this gap (≤ 0.007), so it is a property of the learned policies' streams, not of the logger. It is consistent with the retained set being selected by the same score that is later asked to rank it: residents are the states the ranker scored high, so the residents that turn out not to be reused accumulate, and the arriving victim, which the ranker scores low, is the candidate most often reused next. This mechanism was not tested and is stated as the reading most consistent with the numbers, not as a finding.
- The share of on-policy evictions that removed a state reused within H while a never-reused-within-H candidate was available (binary label, seed 0):

| trace | L1 × L2 | LRU | LFU | 2-hit | A_none | B | C_lru | C_lfu | C_union |
|---|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | 0.32 | 0.29 | 0.52 | 0.31 | 0.31 | 0.31 | 0.32 | 0.32 |
| conversation | 0.25% × 4 | 0.32 | 0.28 | 0.48 | 0.28 | 0.28 | 0.28 | 0.29 | 0.28 |
| conversation | 1% × 1 | 0.31 | 0.27 | 0.44 | 0.27 | 0.27 | 0.27 | 0.28 | 0.27 |
| conversation | 1% × 4 | 0.21 | 0.24 | 0.16 | 0.18 | 0.19 | 0.21 | 0.19 | 0.19 |
| conversation | 2% × 1 | 0.24 | 0.25 | 0.29 | 0.21 | 0.26 | 0.23 | 0.22 | 0.22 |
| conversation | 2% × 4 | 0.12 | 0.18 | 0.02 | 0.10 | 0.20 | 0.11 | 0.10 | 0.10 |
| toolagent | 0.25% × 1 | 0.32 | 0.28 | 0.56 | 0.31 | 0.31 | 0.31 | 0.32 | 0.32 |
| toolagent | 0.25% × 4 | 0.32 | 0.28 | 0.50 | 0.27 | 0.28 | 0.28 | 0.29 | 0.28 |
| toolagent | 1% × 1 | 0.30 | 0.27 | 0.46 | 0.27 | 0.27 | 0.28 | 0.27 | 0.27 |
| toolagent | 1% × 4 | 0.20 | 0.23 | 0.13 | 0.17 | 0.19 | 0.21 | 0.18 | 0.18 |
| toolagent | 2% × 1 | 0.23 | 0.24 | 0.29 | 0.21 | 0.27 | 0.24 | 0.21 | 0.21 |
| toolagent | 2% × 4 | 0.10 | 0.17 | 0.00 | 0.09 | 0.21 | 0.10 | 0.08 | 0.08 |

  The learned arms are at the sampled-LRU / LFU level (0.27–0.32) at 0.25% and 1% × 1 and within 0.03 of sampled LRU at the larger budgets (B excepted); only heap-style 2-hit at 2% × 4 is clearly lower, at the price of admitting almost nothing.

## 5. Pre-registered interpretation

Per target on the two real traces, cells out of 6 (C = candidate-trained population; "closure gain" is against A_none; "best sampled" is the best of the three generic sampled arms, the pre-registered floor):

| trace | target | C | Case A cells | R ≥ R_high | R − R(A_none) ≥ 0.05 | closure gain < 0.05 | train R < R_high | closure < best sampled + 0.05 | on-policy R < R_high | max |on − off| |
|---|---|---|---|---|---|---|---|---|---|---|
| conversation | binary | C_lru | 0/6 | 6/6 | 0/6 | 6/6 | 3/6 | 6/6 | 5/6 | 0.33 |
| conversation | binary | C_lfu | 0/6 | 2/6 | 0/6 | 6/6 | 6/6 | 5/6 | 5/6 | 0.40 |
| conversation | binary | C_union | 0/6 | 4/6 | 0/6 | 6/6 | 6/6 | 5/6 | 5/6 | 0.44 |
| conversation | count | C_lru | 0/6 | 6/6 | 0/6 | 6/6 | 2/6 | 6/6 | 6/6 | 0.40 |
| conversation | count | C_lfu | 0/6 | 2/6 | 0/6 | 6/6 | 6/6 | 5/6 | 5/6 | 0.52 |
| conversation | count | C_union | 0/6 | 6/6 | 0/6 | 6/6 | 6/6 | 5/6 | 5/6 | 0.66 |
| conversation | next_use | C_lru | 0/6 | 6/6 | 1/6 | 6/6 | 4/6 | 4/6 | 6/6 | 0.35 |
| conversation | next_use | C_lfu | 0/6 | 2/6 | 0/6 | 6/6 | 6/6 | 5/6 | 6/6 | 0.52 |
| conversation | next_use | C_union | 0/6 | 6/6 | 0/6 | 6/6 | 6/6 | 4/6 | 6/6 | 0.49 |
| toolagent | binary | C_lru | 0/6 | 6/6 | 0/6 | 6/6 | 0/6 | 6/6 | 5/6 | 0.34 |
| toolagent | binary | C_lfu | 0/6 | 5/6 | 0/6 | 6/6 | 4/6 | 5/6 | 5/6 | 0.44 |
| toolagent | binary | C_union | 0/6 | 6/6 | 0/6 | 6/6 | 4/6 | 5/6 | 5/6 | 0.45 |
| toolagent | count | C_lru | 0/6 | 6/6 | 1/6 | 5/6 | 0/6 | 6/6 | 6/6 | 0.43 |
| toolagent | count | C_lfu | 1/6 | 5/6 | 0/6 | 5/6 | 5/6 | 5/6 | 4/6 | 0.58 |
| toolagent | count | C_union | 1/6 | 6/6 | 0/6 | 5/6 | 4/6 | 5/6 | 4/6 | 0.67 |
| toolagent | next_use | C_lru | 0/6 | 6/6 | 0/6 | 6/6 | 0/6 | 5/6 | 5/6 | 0.42 |
| toolagent | next_use | C_lfu | 0/6 | 5/6 | 0/6 | 6/6 | 4/6 | 5/6 | 5/6 | 0.49 |
| toolagent | next_use | C_union | 0/6 | 6/6 | 0/6 | 6/6 | 3/6 | 5/6 | 5/6 | 0.51 |

- **Case A (mismatch was the problem): refuted** for every candidate population on every target. No C variant reaches closure(A_none) + 0.10 together with the best generic sampled arm in more than 1 of 6 cells (0 of 6 in 16 of the 18 combinations). The refuted statement is the pre-registered one: with the same features and model, replacing the global training set by the decision sets that sampled L2-LRU / L2-LFU generate does not convert predictive quality into retention utility. It does not refute population mismatch in general: the mismatch between the training decisions and the decisions the learned policy creates for itself (§4) is untreated by this design.
- **Case B (prediction adequate, retention not converted): met by its letter** for C_lru on all three targets and both traces (off-policy R ≥ R_high in 6 of 6 cells, closure gain over A_none < 0.05 in 5–6 of 6) and for C_union on the graded targets; **flagged in every target** by the pre-registered on-policy check: the on-policy R of the same arm is more than 0.05 below its off-policy R in 34 of 36 C_lru evaluations (largest gap per target 0.33–0.43; up to 0.67 for C_union) and below R_high in 33 of 36. The premise "prediction adequate" holds on the LRU behaviour log and does not hold on the decisions the learned policy actually faces.
- **Case C (representation is the limit): not established.** R(C) ≥ R_high off-policy, the train-split R of C_lru is ≥ R_high in 27 of 36 evaluations, and the fits converged. The evidence needed for Case C (low ranking quality on the decision population even in-sample) is absent off-policy and present only on-policy, where it is confounded with the selection effect above.
- **Verdict: the primary question is answered (Refuted); the secondary diagnosis is Unresolved between B and C.** Which of "adequate prediction, not converted" and "insufficient representation" describes the learned L2 depends on which decision population is asked. The three populations have different roles: the common behaviour log compares rankers on one problem, each policy's own decisions diagnose what it faces in operation, and replay measures the policy. A policy's own decision sets differ in difficulty from policy to policy, so a low on-policy R by itself does not establish a representation limit, and a whole-set ranking metric weights every pair equally whereas utility depends on the one state actually dropped. The pre-registered rule that only Case C justifies non-history signal is therefore not satisfied.

## 6. Other observations, not pre-registered

- A_none and A_pd (raw features versus Phase 0.9 per-decision standardisation, same rows) are within ±0.10 closure in 35 of 36 real cells; the exception is toolagent 2% × 4 / count, where A_none is at −0.28 and A_pd at +0.26 (CI95 ±0.016 / ±0.016). The cause was not examined.
- C_lru has the highest off-policy R on the LRU log in every cell and the lowest replay closure of the C variants, negative in 6 of 12 cells on the binary and on the count target. This is the sharpest single instance of off-policy ranking quality and on-policy utility disagreeing.
- On the synthetic trace every learned arm but one is below sampled LRU (closure −0.66 to +0.03; the exception is C_lfu / count at 0.25% × 1, +0.027 ± 0.010) while the heap generics are at +0.03 to +0.26, and A_none's off-policy R on the synthetic candidate sets is below chance (AUC 0.43–0.46; A_pd 0.57–0.65). The synthetic trace is outside the pre-registered judgement and is reported as observed.
- The on-policy metric is measured at seed 0 only; the five-seed closure CIs show that the replay side is stable, but the on-policy R has no dispersion estimate.
- **Present-but-unusable tokens.** The replay records, per request, the L2 blocks that were present but could not be used because an ancestor was missing (`l2_present_unusable_tokens`, share of input). Phase 0.95 found this to be exactly zero for heap LRU / LFU / 2-hit, and it is zero for them here. Under the sampled mechanism it is not zero for any arm: sampled LRU 0.05–2.5%, sampled LFU 0.4–2.3%, sampled 2-hit 0.1–1.5%, the heap offline comparator 0.0–1.1% (its known tie-break effect), and the learned arms 0.2–3.9% (next-use target, five-seed means). In most cells the learned arms are at the level of sampled LRU in the same cell (conversation 1% × 4: sampled LRU 2.28%, A_none 1.62%, C_lru 2.34%); the victim-trained B is highest where it collapses (2.0–3.8%, against 1.1–1.8% for sampled LRU), which accounts for part, not all, of its loss (its extra avoided share falls by 8–15 points there). Sampling breaks the monotone-chain guarantee that made the heap policies' dependency cost zero, because the descendant of a sampled ancestor may not be in the sample. The share is an upper bound on what an ancestor-aware decision could recover, not an estimate of it, and it separates two questions the ranking metrics conflate: whether the retained state is reused, and whether the retained KV is usable when it is.

## 7. Claim limits

One linear ranker (logistic / ridge, L2 = 0.01) on the 23 causal history features; two real traces plus one synthetic; H = 600 s (300 s synthetic); sampled decision width 16; five seeds for replay, one for on-policy; behaviour logs from sampled L2-LRU and L2-LFU only, reservoir-capped at 40,000 decisions per log; rows capped at 150,000 per fit; test-split evaluation with the horizon embargo. The closure scale is bounded by the sampled mechanism (its own loss relative to heap policies is up to 0.19); a learned arm run through an exact structure would need to be re-measured. "Best generic" in the pre-registered rule is the best sampled generic arm; against the heap generics the learned arms are at or below parity except at 2% × 4. Nothing here evaluates semantic or embedding features, neural rankers, iterated on-policy training, or any policy other than the five populations listed.

## 8. Decision

The decision-population hypothesis is refuted within this design: matching the training population to the retention decision does not convert the history representation's predictive quality into L2 utility, and the best learned L2 leaves 73–87% of the offline headroom in every real cell, at parity with the best generic heap policy. The B-versus-C diagnosis is Unresolved because the off-policy and on-policy views of "prediction adequate" disagree by more than the pre-registered tolerance. Per the pre-registration, this does not license non-history signal, additional heuristics, or a new policy. What the results point to instead is a narrower question about the decisions themselves: which evictions (rejecting the arriving victim, or removing a resident) led to lost reuse or to present-but-unusable KV, on the few cells where the loss is largest. That diagnostic is recorded in the plan as the candidate next step and is not scheduled here.


## 9. Phase 0.98 — Eviction-decision attribution (diagnostic)

Pre-registered in `docs/experiment-plan.md` (Phase 0.98) before the run;
commit `bafce0a`, `scripts/run_decision_attribution.py`, 330 replays in
1,346 s on 24 workers (1,405 s in all). Nothing is fitted or changed: the
Phase 0.97 arms are rebuilt by the Phase 0.97 code and replayed with three
read-only hooks. Every one of the 330 replays reproduces its Phase 0.97
`avoided_prefill_tokens` exactly; the per-request partition (L2 hits + root
loss + present-unusable + downstream-absent = every block beyond the L1
prefix) and the equality of the present-unusable total with the replay's own
counter are asserted in every replay; unexplained root losses (a previously
requested state with no removal record) are 0 over the grid. Cells: 0.25% × 1,
1% × 4, 2% × 4 on the two real traces; arms: sampled LRU (the mechanism
control), sampled LFU, sampled 2-hit, and A_none / B / C_lru / C_union on the
next-use and the binary target; five seeds; window = the last 40% of each
trace. All shares below are of window input tokens, five-seed means; the
binary-target arms are in the CSVs and summarised in the text.

### 9.1 Where the reuse was lost

Root loss = tokens of the first block beyond the L1 prefix that L2 does not
hold, charged to the last decision that removed it (rejection of the arriving
victim, eviction as a resident, or compulsory: never offered); present-unusable
= blocks after it that L2 holds but the tree rule cannot use, charged to the
same decision; downstream absent = blocks after it that L2 does not hold,
not charged. "Decision loss" = rejected + evicted root loss + the two
present-unusable classes; the difference to sampled LRU is per seed
(CI95 over seeds); "dominant" is the pre-registered ≥ 50% rule on that
difference; "wasted" = present-unusable / (L2 hits + present-unusable), the
share of what L2 retained and was asked for that could not be used.

| trace | L1 × L2 | arm | L2 hits | root: rejected / evicted / compulsory | unusable after rejection / eviction | downstream absent | decision loss − sampled LRU | dominant | wasted |
|---|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | sampled LRU | 0.12 | 0.02 / 1.86 / 2.52 | 0.00 / 0.06 | 90.9 | +0.00 ± 0.00 | not worse | 0.34 |
| conversation | 0.25% × 1 | sampled LFU | 1.09 | 1.56 / 0.33 / 2.51 | 0.08 / 0.48 | 89.4 | +0.51 ± 0.07 | rejected | 0.34 |
| conversation | 0.25% × 1 | sampled 2-hit | 0.40 | 0.85 / 1.03 / 2.52 | 0.00 / 0.15 | 90.5 | +0.08 ± 0.02 | rejected | 0.27 |
| conversation | 0.25% × 1 | A_none | 1.52 | 1.37 / 0.51 / 2.53 | 0.12 / 0.54 | 88.9 | +0.59 ± 0.05 | rejected | 0.30 |
| conversation | 0.25% × 1 | B | 1.38 | 1.12 / 0.73 / 2.54 | 0.01 / 0.32 | 89.4 | +0.25 ± 0.03 | rejected | 0.19 |
| conversation | 0.25% × 1 | C_lru | 1.55 | 1.16 / 0.69 / 2.55 | 0.02 / 0.42 | 89.1 | +0.35 ± 0.04 | rejected | 0.22 |
| conversation | 0.25% × 1 | C_union | 1.05 | 1.31 / 0.55 / 2.54 | 0.03 / 0.21 | 89.8 | +0.16 ± 0.01 | rejected | 0.19 |
| conversation | 1% × 4 | sampled LRU | 13.21 | 0.00 / 1.03 / 3.26 | 0.00 / 2.28 | 74.7 | +0.00 ± 0.00 | not worse | 0.15 |
| conversation | 1% × 4 | sampled LFU | 9.64 | 0.90 / 0.86 / 2.61 | 0.55 / 1.02 | 78.9 | +0.02 ± 0.24 | rejected | 0.14 |
| conversation | 1% × 4 | sampled 2-hit | 15.01 | 1.36 / 0.18 / 2.81 | 0.00 / 1.16 | 74.0 | -0.61 ± 0.17 | not worse | 0.07 |
| conversation | 1% × 4 | A_none | 15.78 | 0.01 / 1.33 / 2.98 | 0.01 / 1.61 | 72.8 | -0.36 ± 0.32 | not worse | 0.09 |
| conversation | 1% × 4 | B | 14.60 | 0.00 / 1.20 / 3.10 | 0.00 / 2.81 | 72.8 | +0.70 ± 0.32 | evicted | 0.16 |
| conversation | 1% × 4 | C_lru | 13.56 | 0.46 / 1.05 / 2.83 | 0.26 / 2.08 | 74.3 | +0.54 ± 0.33 | rejected | 0.15 |
| conversation | 1% × 4 | C_union | 16.19 | 0.06 / 1.33 / 2.93 | 0.12 / 1.21 | 72.7 | -0.59 ± 0.28 | not worse | 0.08 |
| conversation | 2% × 4 | sampled LRU | 19.70 | 0.00 / 0.56 / 3.70 | 0.00 / 1.76 | 66.2 | +0.00 ± 0.00 | not worse | 0.08 |
| conversation | 2% × 4 | sampled LFU | 12.36 | 0.69 / 0.78 / 2.85 | 1.09 / 1.16 | 72.9 | +1.40 ± 0.07 | rejected | 0.15 |
| conversation | 2% × 4 | sampled 2-hit | 17.01 | 1.33 / 0.03 / 2.95 | 0.00 / 0.31 | 70.2 | -0.64 ± 0.10 | not worse | 0.02 |
| conversation | 2% × 4 | A_none | 21.81 | 0.00 / 0.65 / 3.61 | 0.00 / 1.40 | 64.4 | -0.26 ± 0.12 | not worse | 0.06 |
| conversation | 2% × 4 | B | 6.28 | 1.23 / 0.30 / 2.80 | 3.02 / 0.76 | 77.5 | +2.99 ± 0.04 | rejected | 0.38 |
| conversation | 2% × 4 | C_lru | 21.90 | 0.00 / 0.62 / 3.64 | 0.00 / 1.43 | 64.3 | -0.26 ± 0.15 | not worse | 0.06 |
| conversation | 2% × 4 | C_union | 21.97 | 0.00 / 0.65 / 3.61 | 0.00 / 1.28 | 64.4 | -0.39 ± 0.13 | not worse | 0.05 |
| toolagent | 0.25% × 1 | sampled LRU | 0.78 | 0.01 / 1.32 / 3.91 | 0.00 / 0.05 | 58.5 | +0.00 ± 0.00 | not worse | 0.06 |
| toolagent | 0.25% × 1 | sampled LFU | 1.41 | 1.08 / 0.26 / 3.91 | 0.07 / 0.30 | 57.6 | +0.33 ± 0.01 | rejected | 0.21 |
| toolagent | 0.25% × 1 | sampled 2-hit | 1.03 | 0.62 / 0.71 / 3.91 | 0.00 / 0.11 | 58.2 | +0.06 ± 0.02 | rejected | 0.10 |
| toolagent | 0.25% × 1 | A_none | 1.71 | 0.93 / 0.40 / 3.92 | 0.08 / 0.39 | 57.2 | +0.41 ± 0.04 | rejected | 0.21 |
| toolagent | 0.25% × 1 | B | 1.50 | 0.86 / 0.46 / 3.92 | 0.01 / 0.28 | 57.6 | +0.23 ± 0.03 | rejected | 0.16 |
| toolagent | 0.25% × 1 | C_lru | 1.60 | 0.85 / 0.47 / 3.92 | 0.02 / 0.32 | 57.4 | +0.27 ± 0.02 | rejected | 0.17 |
| toolagent | 0.25% × 1 | C_union | 1.42 | 0.93 / 0.39 / 3.92 | 0.03 / 0.15 | 57.8 | +0.12 ± 0.02 | rejected | 0.11 |
| toolagent | 1% × 4 | sampled LRU | 9.56 | 0.00 / 0.68 / 4.46 | 0.00 / 1.64 | 46.8 | +0.00 ± 0.00 | not worse | 0.15 |
| toolagent | 1% × 4 | sampled LFU | 6.69 | 0.59 / 0.65 / 3.98 | 0.41 / 0.77 | 50.1 | +0.09 ± 0.14 | rejected | 0.15 |
| toolagent | 1% × 4 | sampled 2-hit | 10.89 | 0.99 / 0.09 / 4.12 | 0.00 / 0.77 | 46.3 | -0.47 ± 0.16 | not worse | 0.07 |
| toolagent | 1% × 4 | A_none | 11.07 | 0.01 / 0.93 / 4.23 | 0.01 / 1.29 | 45.6 | -0.09 ± 0.19 | not worse | 0.10 |
| toolagent | 1% × 4 | B | 10.37 | 0.00 / 0.80 / 4.36 | 0.00 / 1.98 | 45.7 | +0.45 ± 0.17 | evicted | 0.16 |
| toolagent | 1% × 4 | C_lru | 9.59 | 0.33 / 0.73 / 4.13 | 0.16 / 1.41 | 46.8 | +0.32 ± 0.20 | rejected | 0.14 |
| toolagent | 1% × 4 | C_union | 11.21 | 0.03 / 0.96 / 4.19 | 0.05 / 1.13 | 45.6 | -0.14 ± 0.16 | not worse | 0.10 |
| toolagent | 2% × 4 | sampled LRU | 14.07 | 0.00 / 0.32 / 4.79 | 0.00 / 1.25 | 40.8 | +0.00 ± 0.00 | not worse | 0.08 |
| toolagent | 2% × 4 | sampled LFU | 8.66 | 0.43 / 0.57 / 4.16 | 0.81 / 1.00 | 45.6 | +1.25 ± 0.06 | rejected | 0.17 |
| toolagent | 2% × 4 | sampled 2-hit | 11.81 | 0.94 / 0.01 / 4.21 | 0.00 / 0.10 | 44.2 | -0.52 ± 0.10 | not worse | 0.01 |
| toolagent | 2% × 4 | A_none | 15.48 | 0.00 / 0.41 / 4.71 | 0.00 / 0.93 | 39.7 | -0.22 ± 0.10 | not worse | 0.06 |
| toolagent | 2% × 4 | B | 4.53 | 0.60 / 0.45 / 4.12 | 1.25 / 1.60 | 48.7 | +2.33 ± 0.12 | rejected | 0.39 |
| toolagent | 2% × 4 | C_lru | 15.66 | 0.00 / 0.40 / 4.72 | 0.00 / 0.99 | 39.5 | -0.18 ± 0.06 | not worse | 0.06 |
| toolagent | 2% × 4 | C_union | 15.76 | 0.00 / 0.39 / 4.73 | 0.00 / 0.89 | 39.5 | -0.29 ± 0.08 | not worse | 0.05 |

- **The pre-registered decision loss is a narrow band.** Rejected, evicted
  and present-unusable tokens together are 1.0–5.3% of input for every arm,
  compulsory roots 2.5–4.8%, and downstream-absent blocks 39–91%. Where an
  arm is ≥ 1 point of input below sampled LRU in L2 hits, the decision loss
  accounts for 22–41% of the shortfall (B at 2% × 4: 22–27% of a 9.0–13.4
  point shortfall; C_lru / binary at 1% × 4: 34–41% of 1.9 points) and the
  rest is downstream-absent. The root-only attribution charges one block per
  broken chain; the blocks behind it were also removed, each by its own
  decision, and are not charged. The dominant-failure labels therefore
  describe the root block of each broken chain, not the whole chain.
- **At 0.25% × 1 every arm other than sampled LRU reads "rejected"**,
  including sampled LFU and 2-hit. They reject 71–99 thousand arrivals in the
  window (sampled LRU: 2–3 thousand), which moves loss from resident
  evictions (−0.6 to −1.5 points) to rejections (+0.6 to +1.6); the net
  decision loss is +0.06 to +0.96 points above sampled LRU while their L2
  hits are +0.1 to +1.4 points above it. The ≥ 50% rule names the largest
  positive component when the components have opposite signs. Rejected
  arrivals are reused within H at 0.26–0.30, evicted residents at 0.42–0.53
  (Table 9.3), so rejecting is the less costly of the two decisions in that
  cell.
- **At 1% × 4 C_lru reads "rejected" on both traces and both targets**
  (+0.3 to +0.8 points; 15–19 thousand rejections against ≤ 3 thousand for
  A_none and C_union), B / next-use reads "evicted" (+0.45 / +0.70, from
  present-unusable after evictions), A_none and C_union are not worse than
  sampled LRU.
- **At 2% × 4 B reads "rejected" on both traces and targets** (+2.3 to +3.0
  points, CI ≤ 0.12): 23–31 thousand rejections, of which 22–24% are reused
  within H, and 1.2–3.0 points of present-unusable tokens *after a
  rejection* (sampled LRU: 0.00). B rejects arriving ancestors of blocks it
  holds: L1 evicts leaves first, so an ancestor arrives after its
  descendants, and B scores it below the residents. Its wasted-retention
  share is 0.28–0.39 against 0.08 for sampled LRU. C_lru / binary reads
  "evicted" (+0.85 / +0.90); A_none, C_lru / next-use and C_union are not
  worse than sampled LRU.

### 9.2 Orphaning: the present-but-unusable KV is mechanism-borne

At every resident eviction, the blocks of the evicted state's L2-resident
descendants (unusable from that instant under the tree rule); window only.

| trace | L1 × L2 | arm | rejections (window) | resident evictions (window) | share of evictions that orphan | orphaned GB (window) | ratio to sampled LRU | reading |
|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | sampled LRU | 3,194 | 108,587 | 0.47 | 254 | 1.00 | mechanism |
| conversation | 0.25% × 1 | sampled LFU | 98,936 | 11,153 | 0.23 | 7 | 0.03 | mechanism |
| conversation | 0.25% × 1 | sampled 2-hit | 70,771 | 40,575 | 0.47 | 100 | 0.39 | mechanism |
| conversation | 0.25% × 1 | A_none | 90,734 | 18,745 | 0.25 | 15 | 0.06 | mechanism |
| conversation | 0.25% × 1 | B | 87,533 | 22,471 | 0.29 | 19 | 0.08 | mechanism |
| conversation | 0.25% × 1 | C_lru | 88,743 | 20,949 | 0.28 | 17 | 0.06 | mechanism |
| conversation | 0.25% × 1 | C_union | 93,167 | 17,330 | 0.26 | 13 | 0.05 | mechanism |
| conversation | 1% × 4 | sampled LRU | 0 | 93,115 | 0.48 | 236 | 1.00 | mechanism |
| conversation | 1% × 4 | sampled LFU | 57,408 | 40,630 | 0.29 | 37 | 0.16 | mechanism |
| conversation | 1% × 4 | sampled 2-hit | 70,451 | 21,870 | 0.48 | 54 | 0.23 | mechanism |
| conversation | 1% × 4 | A_none | 1,830 | 89,115 | 0.45 | 184 | 0.78 | mechanism |
| conversation | 1% × 4 | B | 31 | 90,889 | 0.45 | 191 | 0.81 | mechanism |
| conversation | 1% × 4 | C_lru | 40,964 | 51,701 | 0.36 | 62 | 0.26 | mechanism |
| conversation | 1% × 4 | C_union | 9,100 | 81,699 | 0.44 | 134 | 0.57 | mechanism |
| conversation | 2% × 4 | sampled LRU | 0 | 83,241 | 0.47 | 211 | 1.00 | mechanism |
| conversation | 2% × 4 | sampled LFU | 42,359 | 48,729 | 0.33 | 55 | 0.26 | mechanism |
| conversation | 2% × 4 | sampled 2-hit | 68,736 | 19,225 | 0.47 | 46 | 0.22 | mechanism |
| conversation | 2% × 4 | A_none | 31 | 81,206 | 0.44 | 162 | 0.77 | mechanism |
| conversation | 2% × 4 | B | 75,988 | 20,315 | 0.30 | 17 | 0.08 | mechanism |
| conversation | 2% × 4 | C_lru | 1,386 | 79,708 | 0.44 | 148 | 0.70 | mechanism |
| conversation | 2% × 4 | C_union | 307 | 80,903 | 0.44 | 156 | 0.74 | mechanism |
| toolagent | 0.25% × 1 | sampled LRU | 2,421 | 106,533 | 0.45 | 226 | 1.00 | mechanism |
| toolagent | 0.25% × 1 | sampled LFU | 94,037 | 13,372 | 0.21 | 8 | 0.03 | mechanism |
| toolagent | 0.25% × 1 | sampled 2-hit | 71,531 | 36,906 | 0.47 | 89 | 0.39 | mechanism |
| toolagent | 0.25% × 1 | A_none | 85,218 | 21,558 | 0.23 | 14 | 0.06 | mechanism |
| toolagent | 0.25% × 1 | B | 86,134 | 21,253 | 0.25 | 16 | 0.07 | mechanism |
| toolagent | 0.25% × 1 | C_lru | 85,515 | 21,630 | 0.25 | 16 | 0.07 | mechanism |
| toolagent | 0.25% × 1 | C_union | 88,778 | 18,922 | 0.24 | 13 | 0.06 | mechanism |
| toolagent | 1% × 4 | sampled LRU | 0 | 89,769 | 0.45 | 201 | 1.00 | mechanism |
| toolagent | 1% × 4 | sampled LFU | 49,457 | 45,774 | 0.28 | 37 | 0.18 | mechanism |
| toolagent | 1% × 4 | sampled 2-hit | 71,047 | 18,022 | 0.47 | 44 | 0.22 | mechanism |
| toolagent | 1% × 4 | A_none | 2,362 | 85,621 | 0.42 | 172 | 0.85 | mechanism |
| toolagent | 1% × 4 | B | 2 | 87,971 | 0.41 | 158 | 0.79 | mechanism |
| toolagent | 1% × 4 | C_lru | 36,010 | 53,884 | 0.34 | 63 | 0.32 | mechanism |
| toolagent | 1% × 4 | C_union | 7,832 | 80,068 | 0.43 | 145 | 0.72 | mechanism |
| toolagent | 2% × 4 | sampled LRU | 0 | 80,013 | 0.44 | 177 | 1.00 | mechanism |
| toolagent | 2% × 4 | sampled LFU | 34,172 | 53,684 | 0.31 | 54 | 0.30 | mechanism |
| toolagent | 2% × 4 | sampled 2-hit | 69,223 | 16,191 | 0.47 | 37 | 0.21 | mechanism |
| toolagent | 2% × 4 | A_none | 73 | 78,164 | 0.40 | 151 | 0.85 | mechanism |
| toolagent | 2% × 4 | B | 58,265 | 34,591 | 0.28 | 27 | 0.15 | mechanism |
| toolagent | 2% × 4 | C_lru | 1,570 | 76,268 | 0.39 | 130 | 0.73 | mechanism |
| toolagent | 2% × 4 | C_union | 1,049 | 76,797 | 0.40 | 145 | 0.82 | mechanism |

- **Every arm and target reads "mechanism" (60 of 60): no learned score
  orphans more than sampled LRU; the ratio is 0.03–1.05.** Sampled LRU
  orphans most: 44–48% of its resident evictions remove a state with resident
  descendants, 2.1–2.5 blocks per eviction, 177–254 GB per window. The heap
  LRU of Phase 0.95 orphaned nothing; the sampled draw of 16 residents can
  contain an ancestor without its descendants, and the LRU key then removes
  the ancestor (the older state) first. The learned arms orphan at 0.23–0.46
  of their evictions and 13–207 GB. The present-but-unusable tokens recorded
  in Phase 0.97 are therefore a cost of the sampled mechanism, not of the
  learned scores, and the readings refute the "learning-borne" alternative
  in every cell.
- The wasted-retention share (Table 9.1) tells the same story from the
  request side: sampled LRU 0.06–0.34, the learned arms 0.05–0.52 at
  0.25% × 1 and 0.05–0.20 at the larger budgets, B 0.28–0.39 where it
  collapses.

### 9.3 Decision-type regret and the ranking split

Window decisions with `t + H ≤ end`, binary label (reuse within H = 600 s),
seed rows aggregated; a "reused" rejection or eviction removed a state that
was requested again within H. Victim-vs-residents = pairwise AUC of the
arriving victim's label against each resident's under the store's own
scores; residents-only = within-decision AUC on the decision set minus the
arrival; victim rank fraction = the arrival's rank among the candidates
(0 = lowest score).

| trace | L1 × L2 | arm | rejections: n / reused within H | resident evictions: n / reused within H | victim-vs-residents AUC | residents-only AUC | whole-set AUC | victim rank fraction | reading |
|---|---|---|---|---|---|---|---|---|---|
| conversation | 0.25% × 1 | sampled LRU | 1,010 / 0.19 | 38,990 / 0.33 | 0.51 | 0.50 | 0.50 | 0.84 | other |
| conversation | 0.25% × 1 | sampled LFU | 35,763 / 0.29 | 4,237 / 0.48 | 0.90 | 0.32 | 0.48 | 0.03 | other |
| conversation | 0.25% × 1 | sampled 2-hit | 0 / 0.50 | 22,074 / 0.53 | 0.50 | 0.50 | 0.50 | 0.92 | other |
| conversation | 0.25% × 1 | A_none | 33,015 / 0.27 | 6,985 / 0.45 | 0.82 | 0.44 | 0.50 | 0.04 | other |
| conversation | 0.25% × 1 | B | 31,592 / 0.26 | 8,408 / 0.52 | 0.81 | 0.41 | 0.47 | 0.07 | other |
| conversation | 0.25% × 1 | C_lru | 32,293 / 0.26 | 7,707 / 0.50 | 0.81 | 0.45 | 0.51 | 0.06 | other |
| conversation | 0.25% × 1 | C_union | 33,391 / 0.28 | 6,609 / 0.47 | 0.69 | 0.35 | 0.39 | 0.04 | other |
| conversation | 1% × 4 | sampled LRU | 0 / – | 40,000 / 0.21 | 0.57 | 0.56 | 0.56 | 0.99 | other |
| conversation | 1% × 4 | sampled LFU | 22,171 / 0.24 | 17,829 / 0.24 | 0.55 | 0.68 | 0.66 | 0.14 | other |
| conversation | 1% × 4 | sampled 2-hit | 0 / – | 11,927 / 0.16 | 0.70 | 0.66 | 0.67 | 0.99 | other |
| conversation | 1% × 4 | A_none | 503 / 0.02 | 39,497 / 0.19 | 0.69 | 0.68 | 0.68 | 0.55 | other |
| conversation | 1% × 4 | B | 3 / 0.00 | 39,997 / 0.18 | 0.63 | 0.63 | 0.63 | 0.78 | other |
| conversation | 1% × 4 | C_lru | 16,448 / 0.19 | 23,552 / 0.19 | 0.66 | 0.63 | 0.63 | 0.23 | other |
| conversation | 1% × 4 | C_union | 2,390 / 0.07 | 37,610 / 0.19 | 0.71 | 0.68 | 0.68 | 0.45 | other |
| conversation | 2% × 4 | sampled LRU | 0 / – | 40,000 / 0.12 | 0.66 | 0.60 | 0.61 | 1.00 | other |
| conversation | 2% × 4 | sampled LFU | 16,799 / 0.21 | 23,201 / 0.17 | 0.52 | 0.66 | 0.63 | 0.20 | other |
| conversation | 2% × 4 | sampled 2-hit | 0 / – | 10,059 / 0.02 | 0.85 | 0.80 | 0.81 | 1.00 | other |
| conversation | 2% × 4 | A_none | 9 / 0.00 | 39,991 / 0.10 | 0.73 | 0.70 | 0.70 | 0.78 | other |
| conversation | 2% × 4 | B | 30,596 / 0.24 | 9,404 / 0.17 | 0.31 | 0.63 | 0.56 | 0.06 | arrival placement |
| conversation | 2% × 4 | C_lru | 614 / 0.01 | 39,386 / 0.09 | 0.73 | 0.70 | 0.70 | 0.68 | other |
| conversation | 2% × 4 | C_union | 65 / 0.00 | 39,935 / 0.10 | 0.74 | 0.71 | 0.71 | 0.72 | other |
| toolagent | 0.25% × 1 | sampled LRU | 942 / 0.18 | 39,058 / 0.32 | 0.51 | 0.48 | 0.49 | 0.84 | other |
| toolagent | 0.25% × 1 | sampled LFU | 34,806 / 0.29 | 5,194 / 0.46 | 0.90 | 0.34 | 0.49 | 0.03 | other |
| toolagent | 0.25% × 1 | sampled 2-hit | 1 / 0.50 | 20,386 / 0.56 | 0.51 | 0.50 | 0.50 | 0.92 | other |
| toolagent | 0.25% × 1 | A_none | 31,548 / 0.27 | 8,452 / 0.42 | 0.83 | 0.45 | 0.51 | 0.05 | other |
| toolagent | 0.25% × 1 | B | 31,396 / 0.27 | 8,604 / 0.46 | 0.76 | 0.37 | 0.42 | 0.06 | other |
| toolagent | 0.25% × 1 | C_lru | 31,477 / 0.27 | 8,523 / 0.46 | 0.77 | 0.40 | 0.45 | 0.06 | other |
| toolagent | 0.25% × 1 | C_union | 32,405 / 0.29 | 7,595 / 0.43 | 0.67 | 0.35 | 0.38 | 0.04 | other |
| toolagent | 1% × 4 | sampled LRU | 0 / – | 40,000 / 0.20 | 0.59 | 0.56 | 0.56 | 0.99 | other |
| toolagent | 1% × 4 | sampled LFU | 19,773 / 0.24 | 20,227 / 0.23 | 0.54 | 0.68 | 0.66 | 0.14 | other |
| toolagent | 1% × 4 | sampled 2-hit | 0 / – | 10,087 / 0.13 | 0.74 | 0.68 | 0.69 | 0.99 | other |
| toolagent | 1% × 4 | A_none | 849 / 0.02 | 39,151 / 0.18 | 0.70 | 0.68 | 0.68 | 0.55 | other |
| toolagent | 1% × 4 | B | 1 / 0.00 | 39,999 / 0.17 | 0.62 | 0.62 | 0.62 | 0.82 | other |
| toolagent | 1% × 4 | C_lru | 15,318 / 0.19 | 24,682 / 0.19 | 0.67 | 0.63 | 0.64 | 0.26 | other |
| toolagent | 1% × 4 | C_union | 3,158 / 0.02 | 36,842 / 0.19 | 0.71 | 0.68 | 0.68 | 0.45 | other |
| toolagent | 2% × 4 | sampled LRU | 0 / – | 40,000 / 0.10 | 0.69 | 0.61 | 0.62 | 1.00 | other |
| toolagent | 2% × 4 | sampled LFU | 14,121 / 0.21 | 25,879 / 0.15 | 0.52 | 0.66 | 0.63 | 0.21 | other |
| toolagent | 2% × 4 | sampled 2-hit | 0 / – | 8,794 / 0.00 | 0.87 | 0.84 | 0.84 | 1.00 | other |
| toolagent | 2% × 4 | A_none | 17 / 0.00 | 39,983 / 0.08 | 0.74 | 0.71 | 0.72 | 0.78 | other |
| toolagent | 2% × 4 | B | 23,980 / 0.23 | 16,020 / 0.20 | 0.36 | 0.58 | 0.54 | 0.10 | other |
| toolagent | 2% × 4 | C_lru | 684 / 0.00 | 39,316 / 0.08 | 0.75 | 0.71 | 0.72 | 0.69 | other |
| toolagent | 2% × 4 | C_union | 460 / 0.00 | 39,540 / 0.08 | 0.75 | 0.72 | 0.73 | 0.72 | other |

- **"Arrival placement" fires for B at conversation 2% × 4 only** (both
  targets: victim-vs-residents 0.31 / 0.39, residents-only 0.63 / 0.68). At
  toolagent 2% × 4 B shows the same pattern on the arrival criterion (0.36 /
  0.40) but its residents-only AUC (0.58 / 0.59) is below the 0.6 bar, so the
  rule reads "other". B places the arrival at the bottom in 90–94% of its
  first-round decisions; the victim-trained fit saw only L1 victims at their
  eviction, never a resident, and orders an arriving ancestor below the
  residents it would protect.
- **At 0.25% × 1 the arrival is placed better than the residents are
  ordered**: victim-vs-residents 0.67–0.83 for the learned arms against
  residents-only 0.35–0.45. The on-policy inversion that Phase 0.97 reported
  for this cell (whole-set AUC 0.24–0.27 at seed 0 for C_lfu / C_union;
  here 0.39–0.51 over the arms and seeds evaluated) is among the residents,
  and every arm except sampled LRU and 2-hit rejects 79–89% of arrivals
  there.
- A candidate not reused within H is available in 99–100% of the decisions
  of every arm (Phase 0.97's "avoidable" share), so the regret shares above
  are shares of avoidable removals.

### 9.4 Reading against the pre-registration and limits

- Orphaning: **mechanism-borne** in every cell, arm and target (Refuted:
  that the learned scores add ancestor loss).
- Dominant failure: **rejection of the arriving victim** for B at 2% × 4
  and C_lru at 1% × 4 on both traces, and for every non-LRU arm at 0.25% × 1
  where the sign-mixed components make the label a description of where
  the loss moved rather than of a net cost; **resident eviction** for B /
  next-use at 1% × 4 and C_lru / binary at 2% × 4; **not worse** than
  sampled LRU for A_none and C_union at 1% × 4 and 2% × 4.
- Ranking split: **arrival placement** for B at conversation 2% × 4; the
  same pattern at toolagent 2% × 4 misses the residents-only bar by 0.01–0.02;
  "other" everywhere else.
- Limits: the decision loss covers 22–41% of the L2-hit shortfall of the
  arms that fall behind sampled LRU, because the root-only attribution
  charges one block per broken chain; the compulsory class also shifts
  between arms (−0.4 to −0.9 points for the arms that fall behind) because
  the first missing block of a first-occurrence request moves with the
  prefix; five seeds, two real traces, three cells, H = 600 s, the sampled
  width-16 mechanism; on-policy regret and ranking are on the reservoir of
  40,000 window decisions per replay. Nothing here evaluates a new feature,
  policy, or mechanism; the pre-registration does not let any reading
  trigger one.
- What the results leave: the arrival's own first-round decision is where
  the non-LRU arms diverge from sampled LRU, and the one clear collapse (B)
  is the rejection of arriving ancestors, an interaction between a score
  that never saw residents and a mechanism that lets an ancestor leave while
  its descendants stay. A per-block attribution (charging every absent
  block beyond the prefix to its own last removal) would close the coverage
  gap of §9.1 and needs one more counter and a rerun of this grid; it is
  recorded in the plan as the candidate next measurement and is not
  scheduled.
