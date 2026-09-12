# Research Design

## Title

**Value-Aware Persistent KV Selection for Specialized and Agentic LLM Serving**

## Fixed research objective

Given a finite persistent KV cache, how should states with high future reuse
value be selected so that the limited capacity is used best. This objective does
not change with results. What changes is where the evidence says the answer is.

## Hypothesis 0 (superseded by evidence; kept for the record)

*Persistent KV states are not equally valuable. Structural and temporal signals
can identify states with higher future reuse value than generic
recency/frequency-only policies.*

Status after Phase 0 and Phase 0.5:

- **Structural signals** (fan-out, branch diversity) carry real but small
  incremental information (+0.004 to +0.006 AUC over frequency + recency +
  prefix length on the real traces), and branch diversity is nearly collinear
  with frequency. Not the main direction. (`docs/characterization-findings.md`)
- **Temporal-history signals** predict future reuse well (AUC 0.86–0.94 at
  300–600 s on the real traces), but a history-based learned scorer does not
  beat parameterless LRU/LFU under a byte budget: the best causal arm recovers
  at most 0.25 of the LRU-to-offline headroom, and the online learner is never
  the best arm. (`docs/temporal-prediction-findings.md`)

Hypothesis 0 is therefore not supported as stated. Reuse prediction is
possible; better retention did not follow from it.

## Current question

**Why does accurate future-reuse prediction fail to translate into effective KV
retention under a finite cache budget?**

The decomposition experiment (`docs/predictability-retention-gap.md`) separates
three candidate causes with the same eviction machinery for every arm:

- **Signal gap**: the causal predictor does not know the label well enough
  (`oracle_binary − learned_history`).
- **Objective gap**: the label itself is the wrong target
  (`oracle_next_use_sampled − oracle_binary`, `oracle_binary_per_byte −
  oracle_binary`, `oracle_count − oracle_binary`).
- **Candidate-search gap**: sampled leaf eviction cannot reach what the
  exact comparator reaches (`offline_next_use − oracle_next_use_sampled`).

Prefix dependency is a constraint shared by every arm, including the offline
comparator, and is not treated as a separate policy factor here.

## Research questions (as they now stand)

### RQ1 — Which KV states are valuable to retain?

Answered descriptively by Phase 0: reuse is heavy-tailed, most states are
never reused, and recency/frequency/window features carry most of the
predictable signal.

### RQ2 — Can we select future-useful KV states better than generic policies?

Phase 0.5 answer: not with single-state reuse history as the score. The gap
experiment says why (see the findings document for the decision).

### RQ3 — Does better selection reduce recomputation under the same budget?

Measured as avoided prefill tokens in trace replay. End-to-end GPU time,
TTFT, and energy are still out of scope.

## Important design constraint

Research 1 does **not** use semantic embeddings. Semantic-locality-aware
prediction (Research 2) is a separate experiment outside this repository's
plan; nothing here depends on it and nothing here gates it.

## Target change inside Research 1 (Phase 0.9, done)

The prediction target was changed with everything else held fixed. The gap
decomposition had shown that the fixed 600 s binary label is the wrong
objective at small budgets (its perfect oracle recovers 5–21% of the headroom)
and that the best label horizon grows with the budget. The same causal ranker
fitted to the reuse count or the next-use time recovers 0.07–0.11 more of the
headroom below 1% and is the best causal arm at 1%, but stays below LFU below
1% and reaches at most 42% of its own count oracle and 29% of its own
next-use oracle at 0.25–1%
(`docs/target-change-findings.md`). What remains is a signal gap on the new
target, on the eviction-candidate population. The next measurement inside
Research 1 is listed in `docs/experiment-plan.md`.

## Setting change after Phase 0.9: the persistent tier's population (Phase 0.95, done)

The objective is unchanged. What changed is the population the evaluation
speaks about: a persistent tier decides about the states an upper tier
evicts, not about every state ever seen. Phase 0.95 fixes the upper tier
(prefix-closed, LRU or LFU) and evaluates lower-tier retention on its victim
stream under a union-closure hit rule, with an independent-block control
that removes prefix dependency and a standalone-closure sensitivity. Result:
the room over generic policies on that population is 25–80% of the offline
gain, and prefix dependency costs generic policies nothing, because recency
and frequency are monotone along an ancestor chain and the upper tier
evicts leaves (`docs/two-tier-victim-findings.md`). Tree-aware allocation is
therefore not the direction; the open question on the victim population is
the same as before, ranking victims by future reuse. Phases 0–0.9 remain
valid as single-tier results.
