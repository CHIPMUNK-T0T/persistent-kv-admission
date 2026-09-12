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

## Phase 0.75 — Predictability–retention gap decomposition (current)

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

Pipeline: `scripts/run_predictability_gap.py`. Findings:
`docs/predictability-retention-gap.md`.

## Phase 1 — Policy simulator (planned; contents depend on Phase 0.75)

Policies to compare under the corrected objective:

- LRU, LFU, 2-hit
- the retention objective that the decomposition identifies
- approximate offline-next-use comparator (headroom only)

Metrics: hit rate, admission precision / recall, reused tokens, avoided
recomputation, seed dispersion.

## Phase 2 — Online prototype (planned)

Target integration candidates: vLLM, LMCache. Hold cache capacity constant
across policies.

## Phase 3 — End-to-end validation (planned)

Optional system metrics: GPU power / energy, TTFT p50 / p95, storage overhead
as a control variable, not the research contribution.
