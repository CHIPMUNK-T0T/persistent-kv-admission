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

## Phase 0.95 — L1 victim stream and tree allocation (done)

Setting change, not objective change: a persistent tier decides about the
states an upper tier evicts, so the evaluation population becomes the L1
victim stream. L1 is the existing prefix-closed cache with heap LRU or LFU
at 0.25 / 1 / 2% of the working set; every L1 eviction is logged as one
`state × eviction event` with causal and evaluation-only fields
(right-censored at the trace end); generic L2 policies (LRU, LFU, 2-hit)
and the offline comparator are replayed on the same stream at L2 = 1–16 × L1
under three fixed models: union closure with the tree hit rule (primary), an
independent-block hit rule (control that removes prefix dependency), and a
standalone prefix-closed L2 (sensitivity). Because every arriving block
enters L1, the victim stream is identical for every L2 arm and there is no
closed loop. Go / stop rule fixed before the run: absolute gain ≥ 5% of
input tokens, dependency cost ≥ 10% of the control's gain for some policy,
room over the best generic ≥ 20% of the offline gain, all three on both
real traces (`scripts/run_two_tier.py`).

Outcome: absolute and room pass in every cell (offline L2 adds 5–35 points
of input tokens, 2–5 × the best generic); dependency fails in 45 of 48
real-trace cells and the three passes are the offline comparator's
tie-break. Under LRU, LFU, and 2-hit the dependency cost is exactly zero
because recency and frequency are monotone along a chain and L1 evicts
leaves. No positive grounds for making tree allocation the centre were
found (the control rules out ancestor-loss for the policies tested, not
every allocation); the two-tier setting stays. The offline tie-break
diagnostic is part of the run (`two_tier_offline_tiebreak.csv`). Findings:
`docs/two-tier-victim-findings.md`.

## Capacity-normalized horizon sweep (drafted, not run)

A sweep of the oracle label horizon in units of cache-capacity worth of
incoming bytes (H = k · M) was specified after Phase 0.9 to test whether
the budget-dependent best horizon collapses to one k. It was superseded
before running by Phase 0.95: on these near-stationary traces the byte
clock and the wall clock coincide (Spearman 0.99), and the victim event log
records distinct competing bytes until reuse directly. It is kept here as a
possible appendix, not as a planned phase.

## Phase 0.97 — Decision-population-matched retention learning (current)

Question, fixed before the run: does training on the actual
retention-decision population convert global reuse predictability into
retention utility? Phases 0.5–0.95 showed that the same 23-feature causal
history representation loses predictive power as the population narrows
(observed → cached → eviction candidates → L1 victims) and that its
global-trained ranking does not beat LRU / LFU under a byte budget. This
phase separates two explanations, representation versus training
distribution, by changing only the training population.

Fixed: the two-tier setting of Phase 0.95 (heap LRU L1 at 0.25 / 1 / 2% of
the working set, every L1 eviction offered to an exclusive L2, L2 = 1 × and
4 × L1), the 23 causal features, the linear ranker and its standardisation,
the targets already evaluated in Phase 0.9 (600 s binary, reuse count,
next-use time), the train / test split and horizon embargo, sampled-leaf
candidate width 16, the state-size model, five seeds. Every L2 arm that
needs a time-varying score runs through one sampled-eviction mechanism: the
decision set is the arriving victim plus a uniform sample of 16 residents,
the lowest score leaves (the victim itself may be rejected). Generic L2
policies run through the same mechanism; their heap versions and the heap
offline comparator of Phase 0.95 are kept as references.

Varied: the training population only, with the same features, target, and
model:

- **A. global / observed** — the existing Phase 0.5 / 0.9 training set
  (control);
- **B. L1 victims** — victim events of the training split, features at
  eviction;
- **C. L2 decision candidates** — decision sets logged while a causal
  behaviour policy (sampled L2-LRU, sampled L2-LFU, and their union) runs on
  the training split; no future-aware policy generates training data, and
  the behaviour policy's identity is not a feature.

Metrics: primary is L2 replay utility, extra avoided prefill tokens over L1
alone and headroom closure = (arm − sampled L2-LRU) / (heap offline −
sampled L2-LRU), mean over seeds. Secondary are predictive metrics on the
test split of each population (observed, victims, candidates): AUC for the
binary target, Spearman for the graded targets, and inside each logged
decision set the within-decision AUC (binary) or Spearman (graded, decisions
with ≥ 2 distinct labels) and the share of decisions whose evicted state
has the lowest label. Two candidate evaluations are reported: off-policy
(every model scored on the LRU / LFU behaviour logs, a common condition) and
on-policy (each sampled arm's own decisions at seed 0, with the scores the
store actually used, since a learned policy changes the retained set).
Train-split candidate metrics and fit convergence are recorded so that a
null result can be told from a fitting failure. The central figure
follows one target from global → victim → candidate predictive quality to
global-trained → victim-trained → candidate-trained replay utility on the
same budget axis, against LRU, LFU, 2-hit, and the offline comparator.

Interpretation, fixed before the run and amended on 2026-09-13 before any
result was inspected (the amendment adds absolute-accuracy, convergence, and
on-policy conditions after an external review of the design). R is the
ranking metric on the matched candidate population (test-split decisions of
the sampled L2-LRU behaviour log of the same cell): within-decision AUC
(macro) for the binary target, mean within-decision Spearman over decisions
with ≥ 2 distinct labels for the graded targets; R_high = 0.70 (AUC) /
0.30 (Spearman). Closure differences are taken against A_none, the global
fit under the same standardisation as B and C; the best generic sampled L2
is the floor. Per target on the two real traces:

- **Case A (mismatch was the problem):** candidate-trained closure ≥
  A_none closure + 0.10 and ≥ the best generic L2's closure in ≥ 4 of the 6
  cells.
- **Case B (prediction adequate, retention not converted):** R(C) ≥ R_high,
  or R(C) − R(A_none) ≥ 0.05, while closure improves by < 0.05 over A_none in
  most cells.
- **Case C (representation is the limit):** R(C) < R_high and
  R(C) − R(A_none) < 0.05, the fit converged and its train-split R is also
  < R_high (so it is not an optimisation or overfitting failure), and
  closure(C) < best generic + 0.05. Only this case justifies adding
  non-history signal later; even then the claim is limited to "the
  evaluated causal per-state history representation does not provide
  sufficient ranking signal on the actual retention-decision population",
  not "history is useless".
- Otherwise **Unresolved**. In every case the on-policy R of the learned arm
  on the decisions it actually faced (seed 0) is reported next to the
  off-policy R; a difference above 0.05 flags the off-policy diagnosis.

Not done here: new temporal features, byte-clock or time studies, prefix
dependency, new candidate-search algorithms, semantic or embedding features,
neural rankers, result-driven heuristics. Pipeline:
`scripts/run_decision_population.py`. Findings:
`docs/decision-population-findings.md`.

## Phase 1 — Policy simulator (planned; contents depend on the step above)

Policies to compare under the corrected objective:

- LRU, LFU, 2-hit
- the retention objective from Phase 0.9 (next-use time at 1%, count or next-use below 1%, 600 s binary at 5%)
- approximate offline-next-use comparator (headroom only)

Metrics: hit rate, admission precision / recall, reused tokens, avoided
recomputation, seed dispersion; on the two-tier setting, extra avoided
tokens over L1 alone as a share of input tokens.

## Phase 2 — Online prototype (planned)

Target integration candidates: vLLM, LMCache. Hold cache capacity constant
across policies.

## Phase 3 — End-to-end validation (planned)

Optional system metrics: GPU power / energy, TTFT p50 / p95, storage overhead
as a control variable, not the research contribution.
