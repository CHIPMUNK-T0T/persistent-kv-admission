# Phase 1 — Causal arrival-protection intervention

Status: pre-registered before implementation, smoke output, or confirmatory
output. This phase changes one eligibility rule in the sampled L2 mechanism;
it does not add a feature, refit a coefficient, change L1, change the cache
capacity, or use future information.

## 1. Question and scope

Phase 0.98b attributes much of the victim-trained B arm's missing reuse to
blocks removed by admission decisions, and shows present-but-unusable retained
KV behind absent ancestors. Phase 1 asks whether one causal intervention on B
improves the complete replay trajectory: while admitting an L1 victim that can
repair an existing parent hole, temporarily prevent that arrival from losing
the sampled L2 eviction rounds caused by its own admission.

This is an intervention experiment, not a one-decision causal oracle. Changing
one victim changes the subsequent resident set, samples, promotions, and
evictions. All token differences therefore describe the paired end-to-end
trajectories produced by the fixed mechanisms.

## 2. Fixed inputs

- Traces: `conversation_trace` and `toolagent_trace` only, using the existing
  local public-trace JSONL files identified by SHA-256.
- Cells, written as L1 working-set fraction × L2/L1 multiplier:
  `0.25% × 1`, `1% × 4`, and `2% × 4`.
- L1: the existing heap LRU implementation, unchanged.
- L2: union closure, tree hit rule, packed block sizes, sampled width 16.
- Targets: `next_use` and `binary`.
- Seeds: 0, 1, 2, 3, and 4.
- Fits: the Phase 0.97 victim-population B fits, with the same causal history
  features, target definitions, split and embargo, model family, L2 = 0.01,
  and coefficients. The runner must verify its fit inputs and coefficient rows
  against the published Phase 0.97 artifacts before replay. Because the fitted
  normalizer and intercept were not serialized in Phase 0.97, the runner
  deterministically reconstructs the same B fits from the existing victim-row
  artifacts; this is reproduction of the fixed model, not selection or fitting
  of a new alternative.
- Evaluation window: the last 40% of each real trace, exactly as in Phase 0.97
  and Phase 0.98b.

The confirmatory grid is 2 traces × 3 cells × 2 targets × 5 seeds × 3 B
variants = 180 replays. Phase 0.97's published generic heap, sampled generic,
and offline rows are references and are not rerun. A reference row is used only
after its trace, cell, requested-token, L1-token, policy, closure, hit-model,
size-model, and capacity identifiers match the current run.

## 3. Fixed arms and mechanism

All three arms use the original B score tuple, including its existing LRU
tiebreak. `none` is the unmodified Phase 0.97/0.98b B arm. `direct_child` is
the ancestry intervention. `all` is the general arrival-protection control.

An L1 victim is an *arrival*. At the start of one `admit` call, before inserting
the arrival, `direct_child` tests whether any direct child of the arrival is in
the live L2 resident dictionary. This boolean is snapshotted once. A deeper
descendant behind a missing direct child does not trigger protection. The test
uses only the trace's parent relation and current residency. `all` makes the
same snapshot true for every feasible arrival.

An arrival is feasible when its packed byte size is at most the L2 capacity.
If it is larger, protection is disabled from the start and the complete offer
uses the original `none` path. This is counted as `oversized_ineligible`; the
intervention does not evict residents in an attempt to protect an arrival that
cannot fit by itself. L2 disabled by zero capacity follows the existing replay
path. Sample width zero is rejected only when a protection mode is active, so
the default API retains its existing validation behavior.

If protection is inactive, every draw, candidate order, score, tie break,
removal classification, and RNG call follows the original path. If protection
is active:

1. The arrival is inserted exactly as before.
2. In the first overflow round, the resident list and resident random draw are
   made exactly as in the original mechanism. The arrival is omitted from the
   eligible candidate list passed to scoring and the decision hook. The
   lowest original B score tuple among the sampled residents is removed.
3. In later overflow rounds of the same `admit` call, the arrival remains
   ineligible. Up to 16 eligible live residents are drawn without replacement,
   using the store's existing RNG, and the lowest original B score tuple is
   removed. The arrival becomes eligible again after this `admit` call ends;
   no protection state survives to another offer.
4. A direct child that triggered protection has no special status and may be
   evicted. No resident ancestor, other descendant, leaf, or subtree is
   protected as a side effect.

For a feasible protected arrival with positive sample width, an overflow round
must have at least one eligible resident: if only the arrival remains, its size
is at most capacity and there is no overflow. An empty eligible set during an
overflow is therefore an internal invariant violation, not a silent fallback.
The decision hook receives only eligible candidates and their unchanged score
tuples, so its reported victim remains the argmin of what it records.

## 4. Required implementation checks

Before the confirmatory run:

- Unit regressions cover no child, a direct child, a deep descendant behind a
  missing child, `all`, eviction of the triggering child, multiple overflow
  rounds, an oversized arrival, no resident on the first round, and active
  sample-width validation.
- With protection mode omitted or `none`, generic sampled, generic heap,
  learned, and offline paths retain their existing results. The original B
  rows in the confirmatory runner must reproduce Phase 0.98b
  `avoided_prefill_tokens` exactly for all 60 matching B replays before any
  confirmatory files are published.
- Attribution's request partition and per-block identities remain asserted.
- A smoke run uses a separate output directory and is excluded from all
  confirmatory tables. No threshold or interpretation is changed after seeing
  smoke output.

The code/tests commit is made before the smoke and full runs. The full-run
configuration records the pre-registration commit, code commit, trace SHA-256
checksums, fit/coefficient verification, Phase 0.97 and 0.98b reference paths,
and a hash of each pre-existing uncommitted documentation diff that was present
before this phase.

## 5. Outcomes and paired comparisons

The primary cell is `2% × 4`, target `next_use`, on both real traces. The
primary comparison is `direct_child − none` in avoided prefill tokens, paired
by trace, cell, target, seed, and request order. The other two cells are fixed
capacity controls. `binary` is a fixed target-robustness check.

For every variant pair, the runner records:

- net avoided-token difference and its share of requested input tokens;
- per-request positive differences as saved tokens, negative differences in
  absolute value as lost tokens, counts of requests in each direction, and the
  identity `net = saved − lost`;
- `direct_child − all`, to separate an ancestry-specific intervention from
  general arrival protection;
- mean, seed range, and 95% t interval over the five sampling seeds. These
  intervals describe seed variability; they are not population-confidence
  intervals over workloads;
- the unchanged Phase 0.98b per-block removal attribution and present-unusable
  diagnostic for every replay; and
- whole-trace and evaluation-window counts when applicable for protected
  offers, protected offers that overflow, first-round overrides where the
  original full candidate argmin was the arrival, and oversized-ineligible
  offers.

Request vectors are paired only after asserting identical measured request
order, request identity/chain, requested tokens, and L1 prefix tokens across
the three variants. Saved/lost request totals describe differing replay
outcomes; they are not attributed to one particular earlier decision.

The best matching Phase 0.97 generic heap arm is reported beside each cell.
Improving a weak B trajectory alone does not establish a new best policy.

## 6. Pre-registered interpretation

For one tested target, an intervention supports a directional benefit only if
its mean token difference is positive and every one of the five seed-paired
differences is positive on both traces in the primary cell. Mixed signs across
seeds or traces are reported as heterogeneous/unresolved. The symmetric rule
is used for consistent harm. No statistical generalization is made beyond the
two traces.

An ancestry-specific explanation additionally requires `direct_child − all`
to satisfy the same positive directional rule in the primary cell. If both
protection arms improve over `none` but ancestry does not improve over `all`,
the benefit is achievable by general arrival protection; an additional
benefit from selecting arrivals by ancestry is not established. If the direct-child predicate rarely fires or the original arrival
argmin is rarely overridden, a null result is qualified as a weak intervention
test.

The next-use target carries the primary verdict. Binary agreement strengthens
robustness; binary disagreement is reported without replacing the next-use
result. The capacity-control cells show whether the primary behavior is local
to B's known `2% × 4` collapse. Phase 1 does not authorize another two-hit run,
a new feature, a new fit, a closure algorithm, or a post-hoc policy variant.
