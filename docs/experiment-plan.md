# Experiment Plan

Status markers: **done** means the code, results, and findings document are in
the repository; **current** is the experiment in progress; the rest is planned.

## Phase 0 — Trace characterization (done)

- obtain public KV/prefix traces (Mooncake FAST'25: conversation, tool-agent, synthetic)
- reconstruct prefix chains
- derive reuse count distribution
- derive recency / inter-arrival distribution
- compute fan-out / branch diversity
- analyze prefix length vs reuse value
- sweep virtual cache budgets
- compare causal online policies and estimate LRU headroom with approximate offline-next-use

Decision gate outcome: the approximate offline-next-use comparator exceeds LRU
by well over 5% at every trace and budget, so the selection problem is not
already solved by generic policies. Structural signal is small and collinear
with frequency; structure-aware selection was not promoted.

Pipeline: `scripts/run_characterization.py`. Findings:
`docs/characterization-findings.md`.

## Phase 0.5 — Temporal-history prediction and cross-workload replay (done)

- 23 causal temporal features, feature-group ablation, three horizons
- one fitted model per workload, transferred across workloads; one online
  learner; one hyperparameter set for every trace
- fixed-budget replay through one sampled-eviction mechanism for every arm

Outcome: future reuse is predictable (AUC 0.86–0.94 on the real traces) and
the fitted ranking transfers between the two real traces; the same scores do
not beat LRU/LFU under a byte budget (max closure 0.248; online learner never
best). Pipeline: `scripts/run_cross_workload.py`. Findings:
`docs/temporal-prediction-findings.md`.

## Phase 0.75 — Predictability–retention gap decomposition (done)

Question: where does the unrecovered headroom live?

- **Oracle replay.** Arms that know the training label perfectly
  (`oracle_binary`), the label per byte (`oracle_binary_per_byte`), the reuse
  count within the horizon (`oracle_count`), and the exact next-use time
  (`oracle_next_use_sampled`) run through the same sampled-leaf eviction as the
  causal arms. The existing heap comparator stays as the ceiling so that the
  candidate-search loss is measured separately.
- **Candidate-set prediction.** Every sampled eviction candidate set is logged;
  ranking quality is measured on those sets and on the nested populations
  observed ⊃ cached ⊃ leaves at the shared test snapshots.
- **Standardisation robustness.** The transfer matrix is recomputed under
  per-decision, train-set, and no standardisation.
- Five seeds per sampled arm; mean, standard deviation, and 95% CI reported.

Decision rule (an engineering gate, not a statistical threshold), on
`oracle_binary` closure: ≥ 0.7 → signal problem, grounds for Research 2;
≤ 0.3 → objective problem, change the target inside Research 1; in between →
decompose further before deciding.

Outcome: budget-dependent. Below 1% the 600 s label is the wrong objective
(its perfect oracle recovers 5–21%); at 2–10% the label is right and the
history-based predictor fails on the decision population (AUC 0.57–0.63 on
eviction candidates, gradient boosting no better). Pipeline:
`scripts/run_predictability_gap.py`. Findings:
`docs/predictability-retention-gap.md`.

## Phase 0.9 — Target change inside Research 1 (done)

Keep the ranker, the features, the hyperparameters, and the eviction machinery;
change only what is predicted.

- **Decision-population predictability.** On the logged eviction candidate
  sets, fit single-feature, linear, and gradient-boosted models to the 60 s,
  300 s, and 600 s labels with a time split and a horizon embargo
  (`scripts/run_candidate_models.py`). This says which target a causal model
  can rank on the population that matters, and whether model capacity is the
  limit.
- **Replay with alternative targets.** Fit the same linear ranker to the
  binary label at 60 s / 300 s / 600 s, to the reuse count, and to the
  next-use time, plus a parameter-free Little's-law horizon-matched arm;
  replay each through the sampled-leaf eviction over five seeds and five
  budgets, against the oracle arms of Phase 0.75
  (`scripts/run_target_change.py`).

Gate: a causal target that recovers materially more headroom than the 600 s
binary target at the budgets where the oracle said the target was the problem.

Outcome: met in direction, not in magnitude. Count and next-use targets add
0.07–0.11 closure below 1% and next-use is the best causal target at 1%, but
the best causal arm is still below LFU below 1%, reaches at most 42% of its
own count oracle and 29% of its own next-use oracle at 0.25–1%, and its best
cell moves from 0.253 to 0.270. The matched arm adds nothing. Findings:
`docs/target-change-findings.md`.

## Next step inside Research 1 (to be decided)

The gap that remains after the target change is a signal gap on the new
target, on the decision population. Candidates that stay inside Research 1
and introduce no policy:

- the regression analogue of the Phase 0.9 capacity check on the candidate
  logs: fit the count / next-use targets on the eviction candidates, to
  separate "wrong fit population" from "features cannot rank live states";
- one sample-width setting above 16 to size the search gap (0.11–0.18 at
  1–5%);
- longer traces, which are the only way to test which budget regime a
  persistent tier sits in.

## Phase 1 — Policy simulator (planned; contents depend on the step above)

Policies to compare under the corrected objective:

- LRU, LFU, 2-hit
- the retention objective from Phase 0.9 (next-use time at 1%, count or next-use below 1%, 600 s binary at 5%)
- approximate offline-next-use comparator (headroom only)

Metrics: hit rate, admission precision / recall, reused tokens, avoided
recomputation, seed dispersion.

## Phase 2 — Online prototype (planned)

Target integration candidates: vLLM, LMCache. Hold cache capacity constant
across policies.

## Phase 3 — End-to-end validation (planned)

Optional system metrics: GPU power / energy, TTFT p50 / p95, storage overhead
as a control variable, not the research contribution.
