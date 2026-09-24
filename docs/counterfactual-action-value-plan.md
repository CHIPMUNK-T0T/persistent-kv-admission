# One-step counterfactual retention action values

Status: reviewed pre-registration; no counterfactual outcomes inspected. This plan
must be committed before real-trace branch outcomes are generated. Code and
tests must then be reviewed and committed before the smoke and full runs.

## Question and scope

Do exact future state-reuse labels select the same eviction actions as
realized avoided-prefill-token value under a fixed continuation policy?
This follows the completed on-policy experiment and its post hoc
[saved-log diagnostics](retention-decision-diagnostics.md). Those results
motivated this design; the reused traces/test windows are not independent
confirmation. No new fitted model, feature, admission heuristic, or deployed
policy is introduced. No semantic or throughput experiment is included.

For the complete simulator state immediately before one sampled L2 victim
is removed, define Q(pi,H,S,a;omega) as future avoided prefill tokens when
candidate a is removed once and the frozen policy pi resumes thereafter.
Omega fixes the continuation randomness. Compare actions in the exact same
already-drawn candidate set C. Delta Q is relative to the policy's original
action, and candidate-relative regret is max(Q over C) minus Q of a selected
action. This is a conditional finite-trace action value, not an optimal-policy
oracle or an estimate of deployment-wide expected value. Q already includes
downstream trajectory effects; this experiment does not separate action
value from trajectory effects as independent causal components.

## Fixed population and selection

- Traces: pinned conversation_trace and toolagent_trace from the completed
  results/onpolicy_full_feb30eb_001 run.
- Cells: L1 unique packed-byte fraction 0.0025 with L2/L1=1, and 0.02 with
  L2/L1=4. Other settings follow that run: heap L1 LRU, exclusive union L2,
  tree hit accounting, packed sizes, 2048 bytes/token, sample width 16,
  no arrival protection, and the original timestamp/request order.
- Models: frozen next_use pi0 and pi3, lineage seeds 0,1,2,3,4. No fitting.
  Use canonical model and population paths/hashes from the published CSVs.
- Each of the 40 trace/cell/policy/seed lineages contributes eight decisions
  from its published uniform reservoir of 40,000 test decisions. Eligibility
  remains split <= t and t+600 seconds <= trace end.
- Rank complete decision identities by SHA256 of a versioned fixed namespace,
  lineage identity, timestamp-group index, and within-group decision ordinal.
  Take the first eight, independently of labels, scores, victims, or branch
  outcomes. Freeze the exact serialization and namespace in code/config.
  Preserve this hash rank separately from chronological processing order.
- Write and hash-freeze the 320 selected identities and source hashes in a
  new run directory before smoke or branch replay; publish after validation.
  Do not replace decisions because effects are zero, action values tie, or
  actions fail to change downstream cache contents. An integrity failure is
  an experiment failure, not permission to resample.
- pi0 and pi3 induce different states. Their samples are separate populations,
  not matched counterfactual states. Retain original arrival-rejection versus
  resident-eviction categories and denominators as descriptive strata.

## Intervention and continuation

Snapshot after the candidate draw and score evaluation but before removal.
Preserve the complete live state and execution position: L1/L2 residents and
ordering, sizes/counters, histories, scorer, RNG, pending admissions and
timestamp-group processing. Change only the index of the candidate removed
in that round. Every candidate exposed by the original union-store mechanism
is a legal action for this experiment; do not impose a new leaf constraint.

If a partial-block removal leaves the cache over capacity, subsequent rounds
run the original policy normally. Arrival rejection uses the existing
first-round semantics. Resume the remainder of the same timestamp group's
insertion work, but never re-serve that group's requests: their hits were
already computed before the decision. Parent trajectories never take a
counterfactual action. Branches cannot spawn further interventions.

The preferred exact snapshot is a POSIX process fork at this boundary, in a
single-threaded replay worker. This preserves suspended stack frames as well
as heap objects, avoiding reconstruction of a partially processed admission.
No shared mutable output/state is permitted. Record platform and execution
constraints; portability outside fork-capable systems is not claimed.

Run every candidate action, including the actual action, to trace end once.
Measure rewards separately for (t,t+600 seconds] (primary) and (t,trace end]
(secondary). Thus both endpoints are inclusive and the current timestamp
group is excluded. L1 behavior is unchanged, so total avoided-token and
L2-avoided-token action differences must agree. The secondary interval has
variable length; report it, and do not choose horizons after seeing results.

## Randomness

Primary: all actions inherit exactly the post-draw RNG state captured at
that decision. The actual-action branch must reproduce the parent's suffix.
The same initial random stream does not mean later draws address identical
residents: action-dependent populations and RNG call counts may diverge.

Sensitivity: on the first two hash-ranked decisions in every lineage (80
snapshots), repeat every action with two additional deterministically derived
continuation seeds. Seed derivation uses only the versioned namespace,
lineage/event identity, and replicate index. Reseed after the current candidate
draw; hold the current candidates fixed. Each extra stream includes its own
actual-action control. Comparisons across actions are paired within stream.
Do not compare a reseeded action with the captured-stream control. Report
per-stream effects and the three-stream mean as sensitivity, not as a
well-estimated expectation. This is at most 8,160 branch continuations.

## Action selectors and outcomes

At each captured state select victims using:

1. The current frozen learned score, including its existing tie-break.
2. Exact next-use label, clipped at 600 seconds as in training.
3. Exact reuse count within 600 seconds.
4. LRU using the live last_group value.
5. LFU using live frequency, then last_group.

For next-use and count, larger means more valuable to retain; evict the
minimum. Break ties by live last_group then first candidate draw position,
matching the existing scored mechanism. Next-use at exactly H is clipped
with no future reuse; count includes reuse exactly at H, as before. Exclude
current-timestamp occurrences from future labels. Preserve candidate values
and tie multiplicities so clipping/ties cannot masquerade as signal failure.

Primary reporting uses candidate-relative regret in tokens for each selector,
and paired regret differences (exact-label selector minus learned selector)
on identical states/streams. Report means, medians, 90th percentiles, maxima,
zero-regret rates, and fractions with regret >=512 tokens. 512 is a reporting
scale, not a practical-significance threshold. Also report Q range per
decision, actual-action advantage, candidate count, ties, and constant-Q
decisions. All candidate actions have equal status; do not choose a subset by
observed Q. Within-decision rank correlations are secondary; compare the
retention score with -Q(evict), using average ties and marking constant
rankings undefined.

Keep trace/cell/policy strata separate. Report all five lineage-seed summaries
and decision-level values; do not count candidates from one snapshot as
independent observations. Decisions share future requests and traces share a
deployment family. Do not assign workload-general significance from the five
seeds or extrapolate summed sampled regrets to recoverable trace-wide loss.
This is a preregistered estimation/diagnostic study, with no A/B/C pass rule.

Interpretation: substantial exact-label regret demonstrates surrogate error
on the evaluated states. Small exact-label regret and larger learned regret
locate a gap in predicting/using that target, without identifying whether
features or fitting are responsible. Small regrets for both leave the full
policy gap unexplained by these sampled single decisions. Positive or zero
results do not alone establish semantic necessity or trajectory dominance.

## Required validation and execution order

1. Review and commit this plan alone.
2. Implement and test on constructed traces. Required boundaries: no-op hook;
   actual-action fork vs uninterrupted parent; alternate removal including
   arrival rejection; multi-round overflow with partial blocks; same-time
   requests; exact H boundary; independent branch state/RNG; child failure
   propagation; candidate ties; source/model hash mismatch rejection.
3. Review and commit code/tests. Select/freeze all real decisions before
   any branch results. A designated first lineage/snapshot is the smoke:
   conversation, 0.0025x1, pi0, seed0, first hash rank. Its registered results
   may be reused only with identical code/config. Runtime-only adjustments
   may change concurrency, never population or outcome definitions.
4. Verify uninterrupted parent replays against the simulator fields of all
   40 published utility rows: exact identities, integer counters, deterministic
   floating-point fields, and label-window counts. Exclude elapsed runtime and
   derived comparisons against other policies.
   At every snapshot the captured-stream actual-action branch must match the
   parent request-level future reward vector/digest, both reward totals, and
   terminal simulator state/counters (excluding diagnostic hook bookkeeping).
   Verify the captured candidates, full score tuples, and actual victim
   against the saved decision population. Stop on any mismatch.
5. Run the fixed grid with bounded workers after smoke timing/memory review.
   Freeze source/reference hashes and protected pre-existing documentation
   diffs; verify unchanged through the run. No outcome-dependent early stop
   or selection. Interrupted runs resume only completed, hash-matched units.
6. Publish scripts/tests, run config and manifests, selected decisions,
   candidate/action outcomes, per-decision selector regrets, per-seed and
   stratified summaries, validation counts, and figures. Explain both what
   each figure shows and what it cannot establish. Retain detailed audit data
   outside lightweight paper artifacts. Report limitations and zero effects.

Existing dirty README.md, docs/decision-population-findings.md, and
docs/experiment-plan.md are unrelated pending edits and remain untouched.
