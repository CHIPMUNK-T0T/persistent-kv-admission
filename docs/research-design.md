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
  beat parameterless LRU/LFU under a byte budget (all arms through the same
  sampled eviction): the best causal arm recovers at most 0.25 of the
  sampled-LRU-to-offline headroom, and the online learner is never
  the best arm. (`docs/temporal-prediction-findings.md`)

Hypothesis 0 is therefore not supported as stated. Reuse prediction is
possible; better retention did not follow from it.

## Current question

**Which future-reuse information helps select states for a finite persistent
lower-tier KV cache, and how do its horizon, within-class selection rule and
candidate eligibility determine avoided prefill tokens?**

The objective remains selection under limited capacity. The current result is
a characterization of useful information and its use in retention decisions;
a deployable policy has not been established. The [research status](research-status.md)
and completed [matched-horizon checks](matched-horizon-checks-findings.md)
are the current entry points. The [2026-10-03 handoff](next-phase-handoff-20261003.md)
sets out the next planning step, without registering a new experiment.

The latest 840 replays check the evaluation window, the reuse class in which
the frozen order's losses occur, and leaf-only eligibility. Under `all16`,
bit plus recency reaches the local greedy label comparison in 12/12 when the
last 600 s are excluded; learned class recovery changes from 10/12 on the
full window to 9/12 on the head. Under `leaf16`, the learned class recovery
holds in the same 10/12, but the learned within-class order consistently beats
recency in 11/12, reversing the earlier comparison. Useful information and
the selection rule that uses it must therefore be evaluated together.

These future-aware comparisons do not give an optimal upper bound. `h*` is
selected after inspecting `all16` grids on the same traces, not before test
results. Class recovery divides by the gain from exact resident selection
with learned admission kept, not all headroom. Causal prediction of the
required information and independent-workload validation remain open.

## Earlier decomposition question (single-tier, Phase 0.5–0.75)

At that stage the question was why the measured global reuse predictability
did not translate into effective retention for the evaluated ranker. The
decomposition experiment (`docs/predictability-retention-gap.md`) separated
three candidate causes with the same eviction machinery for every arm:

- **Signal gap**: the causal predictor does not know the label well enough
  (`oracle_binary − learned_history`).
- **Objective gap**: the label itself is the wrong target
  (`oracle_next_use_sampled − oracle_binary`, `oracle_binary_per_byte −
  oracle_binary`, `oracle_count − oracle_binary`).
- **Candidate-search gap**: sampled leaf eviction cannot reach what the
  exact comparator reaches (`offline_next_use − oracle_next_use_sampled`).

Prefix dependency is a constraint shared by every arm, including the offline
comparator, and was not treated as a separate policy factor in that
decomposition. This section records that experiment's design, not an
exhaustive explanation of all later two-tier results.

## Research questions (as they now stand)

### RQ1 — Which KV states are valuable to retain?

Phase 0 answers descriptively: reuse is heavy-tailed, most observed states
are never reused within the trace, and recency/frequency/window features carry
most of the measured predictive signal. Later two-tier controls show
substantial avoided-token gains from exact future information on the L1
victim population. The value sought is reduced recomputation under capacity,
not reuse count considered separately from the retention decision.

### RQ2 — Can we select future-useful KV states better than generic policies?

Phase 0.5 did not obtain a general advantage with its tested single-tier
history scorers. This is a result for those features, targets, models and
mechanisms, not evidence that single-state history is inherently inadequate.
The latest controls show that exact future labels produce large utility gains
and that much of the frozen-ranker-to-label gap can be recovered by changing
resident selection. Appropriate horizons and within-class rules matter:
"one bit suffices" and "recency is universally best" are not established.
Whether past observations can provide the useful information well enough for
a causal policy remains unresolved.

### RQ3 — Does better selection reduce recomputation under the same budget?

Measured as avoided prefill tokens in trace replay, with a finite L2 receiving
evictions from a fixed upper cache. The latest evidence concerns two
approximately 59-minute Mooncake traces; it does not establish hours- or
days-scale persistent reuse. End-to-end GPU time, TTFT, energy and storage
throughput remain out of scope.

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
target, on the eviction-candidate population. This is the conclusion at
Phase 0.9; the later measurements are recorded in
[the experiment history](experiment-plan.md) and [research status](research-status.md).

## Setting change after Phase 0.9: the persistent tier's population (Phase 0.95, done)

The objective is unchanged. What changed is the population the evaluation
speaks about: a persistent tier decides about the states an upper tier
evicts, not about every state ever seen. Phase 0.95 fixes the upper tier
(prefix-closed, LRU or LFU) and evaluates lower-tier retention on its victim
stream under a union-closure hit rule, with an independent-block control
that removes prefix dependency and a standalone-closure sensitivity. Result:
the room over generic policies on that population is 25–80% of the greedy
offline gain, and the tested generic heap policies have no token loss from
prefix dependency in that control, consistent with recency and frequency
being monotone along an ancestor chain and the upper tier evicting leaves
(`docs/two-tier-victim-findings.md`). At that stage the result did not motivate
additional tree-aware allocation experiments. It does not show that candidate
eligibility or prefix effects are irrelevant in other mechanisms: the latest
`all16`/`leaf16` check directly changes the within-class ordering comparison,
with the reason still untested. Phases 0–0.9 remain valid as single-tier
results under their recorded conditions.
