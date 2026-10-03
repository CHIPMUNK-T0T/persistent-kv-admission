# Experiment Plan

Status markers: **done** means the code, results, and findings document are in
the repository; **current** marks an experiment in progress when one exists;
the rest is planned.

Current navigation: [research status](research-status.md), the completed
[matched-horizon checks](matched-horizon-checks-findings.md) and the
[2026-10-03 next-phase handoff](next-phase-handoff-20261003.md). The fixed
objective is to increase avoided prefill tokens by selecting states for a
finite persistent lower-tier KV cache. The latest controls specify how useful
future information depends on horizon, resident selection and eligibility;
they do not establish a deployable policy. The handoff is a planning document,
not a new pre-registration or an experiment in progress.

Earlier evidence: [on-policy findings](onpolicy-learning-findings.md),
[post hoc retention diagnostics](retention-decision-diagnostics.md),
[one-step counterfactual findings](counterfactual-action-value-findings.md),
and the completed fresh-stream [pre-registration](counterfactual-randomness-plan.md),
[implementation guide](counterfactual-randomness-implementation.md), and
[findings](counterfactual-randomness-findings.md). The [old three-stream CSV
audit](counterfactual-randomness-reanalysis.md) is a separate post hoc calculation.

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

## Phase 0.97 — Decision-population-matched retention learning (done 2026-09-13)

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

**Outcome (commit `379aa20`, results in `results/paper/decision_population_*`).**
Case A refuted for every candidate population and target (0 of 6 cells in
16 of 18 combinations, 1 of 6 in the other two). Case B met by its letter for
C_lru on every target (off-policy R ≥ R_high in 6 of 6, closure gain over
A_none < 0.05 in 5–6 of 6) but flagged in every target: the on-policy R is
0.02–0.67 below the off-policy R (beyond 0.05 in 34 of 36 C_lru evaluations)
and below R_high in most cells. Case C not
established. Verdict: the primary question is refuted, the B / C diagnosis is
unresolved between the two decision populations; no non-history signal, new
heuristic or new policy follows from this phase. Best learned L2 closes
0.13–0.27 of the headroom (sampled L2-LRU floor) and differs from the best generic heap policy by
−0.8 to +0.7 input-token points. Its mean is above that policy in five of 12
real cells; the five-seed CIs measure sampling-seed variability, and the
post-hoc best-arm / target comparison is not a formal significance test. The
refuted statement is the pre-registered one (same features and model,
training set replaced by the LRU / LFU behaviour logs); mismatch with the
learned policy's own decisions is untreated. Candidate next step, not
scheduled: on the few cells where the learned or victim-trained L2 loses
most, separate rejections of the arriving victim from evictions of
residents, split the ranking evaluation into victim-versus-resident and
resident-versus-resident, and map each evicted state to its later reuse and
to present-but-unusable KV (`l2_present_unusable_tokens`, zero for the heap
policies, 0.05–3.9% of input under the sampled mechanism). No new features
or policies before that diagnostic.

## Phase 0.98 — Eviction-decision attribution (done 2026-09-13; diagnostic, no new policy)

Question, fixed before the run: on the cells where the learned or
victim-trained L2 of Phase 0.97 loses most, which eviction decisions cost the
reuse, and is the present-but-unusable KV a property of the sampled
mechanism or of the learned scores? Purely descriptive; nothing is fitted,
no feature or policy is added, and the Phase 0.97 arms are replayed as they
are (fits and scorers reloaded from the same procedure and seed).

Fixed: the Phase 0.97 setting and code; L1 = heap LRU; cells chosen by the
Phase 0.97 results before this phase runs: 0.25% × 1 (largest on-policy
inversion), 1% × 4 (C_lru below sampled LRU on binary / count, learned at
parity with 2-hit), 2% × 4 (B collapse, learned above heap LRU), on the two
real traces; arms: sampled LRU (mechanism control), sampled LFU, sampled
2-hit, and A_none, B, C_lru, C_union on the next-use and the binary target;
five seeds; evaluation window as before.

Measured, per trace × cell × arm × seed:

1. **Loss attribution.** For every measured request, blocks beyond the L1
   prefix are classified under the tree rule by the first block `m` that is
   not in L2: the tokens of `m` are a *root loss*, attributed to the last
   decision that removed `m` from L2 (rejection of the arriving victim,
   eviction as a resident, or compulsory: never offered); blocks after `m`
   that are in L2 are *present-unusable*, attributed to the same decision
   type as `m`; blocks after `m` not in L2 are *downstream-absent*, not
   attributed. Tokens per category, as a share of window input tokens, and
   the difference to sampled LRU of the same cell and seed.
2. **Orphaning.** At every resident eviction, the blocks and bytes of the
   evicted state's L2-resident descendants (which the tree rule makes
   unusable at that moment); share of evictions that orphan ≥ 1 block and
   total orphaned bytes, per arm, against sampled LRU (heap LRU orphans
   nothing, Phase 0.95).
3. **Decision-type regret.** Window decisions with `t + H ≤ end`, split into
   rejections of the arriving victim and evictions of a resident: count,
   share whose removed state is reused within H, share of those where a
   candidate not reused within H was available.
4. **Ranking split.** On the same decisions: the arriving victim against the
   residents (pairwise AUC of the victim's label versus each resident's,
   ordered by the store's scores) and residents-only within-decision AUC /
   Spearman, next to the whole-set metric of Phase 0.97.

Reading, fixed before the run, per arm and cell relative to sampled LRU in
the same cell and seed: the decision type (rejection / resident eviction)
whose attributed root + present-unusable loss accounts for ≥ 50% of the
arm's total loss difference to sampled LRU is named the dominant failure;
otherwise "mixed". Orphaned bytes within 1.2 × sampled LRU's read as
mechanism-borne, ≥ 2 × as learning-borne, between as unresolved. A
victim-versus-resident AUC below 0.5 with a residents-only AUC ≥ 0.6 reads
as a failure to place the arrival among the residents rather than a failure
to order residents. No threshold in this phase triggers a new feature or
policy; the output is the attribution table and figure. Pipeline:
`scripts/run_decision_attribution.py`; outputs
`results/paper/decision_attribution_*.csv`, `fig17_decision_attribution.png`;
findings appended to `docs/decision-population-findings.md` §9.

**Outcome (commit `bafce0a`, results in `results/paper/decision_attribution_*`).**
330 of 330 replays reproduce Phase 0.97 exactly; unexplained root losses 0.
Orphaning reads "mechanism" in 60 of 60 arm × target × cell readings
(0.03–1.05 × sampled LRU; sampled LRU itself orphans on 44–48% of its
evictions): the present-but-unusable KV is the sampled mechanism's cost,
not the learned scores'. Dominant failure: "rejected" for every non-LRU arm
at 0.25% × 1 (loss moved from resident evictions to rejections, net +0.06
to +0.96 points), for C_lru at 1% × 4 and for B at 2% × 4 (+2.3 to +3.0
points, 1.2–3.0 points present-unusable after rejections of arriving
ancestors); "evicted" for B / next-use at 1% × 4 and C_lru / binary at
2% × 4; "not worse" for A_none and C_union at 1% × 4 and 2% × 4. "Arrival
placement" fires for B at conversation 2% × 4 (toolagent misses the
residents-only bar by 0.01–0.02). Limit: the root-only decision loss covers
22–41% of the L2-hit shortfall of the arms that fall behind sampled LRU;
the rest is downstream-absent. Candidate next measurement, not scheduled: a
per-block attribution that charges every absent block beyond the prefix to
its own last removal (one more counter, one rerun of this grid). No new
feature, policy or mechanism follows from this phase.

### Phase 0.98b — Per-block attribution (done 2026-09-13; diagnostic, no new policy)

Question, fixed before the run: does the root-only dominant-failure label of
Phase 0.98 hold when every absent block beyond the L1 prefix is charged to its
own last removal, and how is the whole L2-hit shortfall of an arm split
between rejections of the arriving victim and evictions of a resident? Phase
0.98 charged one block per broken chain and covered 22–41% of the shortfall;
this measurement charges all of them.

Fixed: everything of Phase 0.98 (grid, arms, targets, seeds, fits, hooks,
readings, thresholds), which is rerun unchanged with one more counter family.
Nothing is fitted, no feature, policy, mechanism or threshold is added.

Measured, per trace × cell × arm × seed, in the same request hook: for every
measured request, every block beyond the L1 prefix that L2 does not hold (the
root block and the downstream-absent blocks) is charged to its own last
removal record — `absent_{rejected, evicted, compulsory, unexplained}` tokens
and blocks. The present-unusable blocks keep the Phase 0.98 charge (the root's
decision). Identities asserted per replay: `absent_* = root_* + downstream_*`
per category, `sum(absent_*) = root_loss + downstream_absent`, and the Phase
0.98 partition. The per-block decision loss is
`absent_rejected + absent_evicted + unusable_after_rejected +
unusable_after_evicted`; its difference to sampled LRU of the same cell and
seed, and the per-block dominant-failure label by the Phase 0.98 ≥ 50% rule on
that difference, are added next to the root-only columns.

Expected by construction, checked in the run and reported either way: L1 is
prefix-closed and independent of L2, so `beyond_prefix`, `l1_avoided` and
`root_compulsory + downstream_compulsory` should be identical across the 11
arms of each trace × cell × seed (Phase 0.98 already shows `beyond_prefix` and
`l1_avoided` identical, and `unusable_after_compulsory` = 0). If that holds,
the per-block decision loss difference to sampled LRU equals the L2-hit
shortfall to the token (unexplained is 0), the coverage of Phase 0.98 §9.1 is
closed by construction, and the informative output is the rejected / evicted
split of the full shortfall. If it does not hold, the discrepancy is reported
as a finding about the mechanism and not corrected.

Reading, fixed before the run, per arm × target × cell on the two real traces,
against sampled LRU in the same cell and seed:

1. Per-block dominant failure by the Phase 0.98 rule ("rejected", "evicted",
   "mixed", "not_worse"), and whether it agrees with the root-only label of
   the same row. Reported as the count of agreeing rows out of the 60 non-LRU
   readings; every disagreement listed.
2. For the arms that Phase 0.98 named ("rejected" for B at 2% × 4 and C_lru at
   1% × 4; "evicted" for B / next-use at 1% × 4 and C_lru / binary at 2% × 4):
   the share of the per-block decision-loss difference carried by rejections.
   ≥ 0.5 keeps the Phase 0.98 statement for that arm; < 0.5 replaces it with
   the per-block label in the findings.
3. Tokens per absent block by category (absent tokens / absent blocks), for
   the record.

Outputs: the same four CSVs with the new columns added (the pre-existing
columns must reproduce the committed Phase 0.98 values exactly — integers
equal, floats within 1e-9 — and every replay must again reproduce Phase 0.97;
otherwise nothing is written), `decision_attribution_config.json` with
`phase = "0.98b"`, and `fig18_perblock_attribution.png` (stack of the five
attributed classes, downstream no longer separate). Findings appended to
`docs/decision-population-findings.md` §9.5. No threshold in this phase
triggers a new feature, policy or mechanism.

**Outcome (pre-registration `4bd41be`, code `6db9fb3`, results in
`results/paper/decision_attribution_*`, findings §9.5).** 330 of 330 replays
reproduce Phase 0.97; every pre-existing Phase 0.98 column reproduces
exactly; unexplained 0; the arm-invariance holds in all 30 groups, so the
per-block decision loss equals the L2-hit shortfall to the token (coverage
1.000 in the 15 rows where an arm is below sampled LRU). Reading 1: the
per-block label agrees with the root-only label in 31 of 60 rows; the 29
disagreements are the 20 rows at 0.25% × 1 (every non-LRU arm: "rejected" →
not worse, they hold more L2 hits than sampled LRU), seven rows at 1% × 4
where a root-only loss is no loss under the full charge (B / next-use on
both traces, C_lru / next-use on both, A_none and C_union / binary on
toolagent) plus B / binary on toolagent at zero, and sampled 2-hit at 2% × 4
on both traces (not worse → "rejected": 2.3–2.7 points below sampled LRU,
all in first-arrival rejections). Reading 2: B at 2% × 4 keeps "rejected"
(rejected share 0.97–1.76), C_lru / binary keeps "rejected" at 1% × 4 and
"evicted" at 2% × 4; B / next-use at 1% × 4 and C_lru / next-use at 1% × 4
are replaced by "not worse". The compulsory share is the first-occurrence
floor of the window, 59.65% (conversation) and 40.80% (toolagent) of input,
identical across cells and arms. No new feature, policy or mechanism
follows from this phase.

## Phase 1 — Policy simulator (planned; contents depend on the user's decision after Phase 0.97)

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

## Current research status

The latest [matched-horizon checks](matched-horizon-checks-findings.md) are
complete (840 replays). On `all16`, removing the last 600 s preserves the
bit-plus-recency comparison in 12/12 and the sign of the within-class order
difference in 12/12; class recovery changes from 10/12 to 9/12. On `leaf16`,
the learned class-order recovery remains at least 0.9 in the same 10/12, but
the learned-versus-recency order comparison reverses. Useful reuse information
and the rule that uses it therefore need joint evaluation through avoided
prefill tokens. `h*` is selected after inspecting the same traces; causal
prediction, a horizon fixed before test results and independent-workload
validation remain open. Detailed registered outcomes follow below. The
[next-phase handoff](next-phase-handoff-20261003.md) proposes the next planning
step; no new run is registered by this summary.

The later [fixed three-update on-policy study](onpolicy-learning-findings.md)
reported complete held-out replay gains in some cells without passing its
registered common-terminal-population ranking threshold. The
[one-step counterfactual study](counterfactual-action-value-findings.md)
measured realized regret under captured continuation streams; its 80-state,
three-stream [post hoc CSV reanalysis](counterfactual-randomness-reanalysis.md)
checks arithmetic and cross-stream ranks. The registered
[16-fresh-stream experiment](counterfactual-randomness-findings.md) completed
40 fixed states and 10,888 branches. Unrestricted action selection averaged
+15,003.0 training-fold tokens versus the original action but −267.8 held-out;
exact-tie selection averaged +13,567.2 training and +90.3 held-out. These
descriptive results weaken the stable-gain reading of single-stream hindsight
maxima. They do not prove equal expected action values or establish a
trace-wide policy gain; further stages require separate designs.

The sampling-mechanism control left open by the handoff is pre-registered in
[mechanism-control-plan.md](mechanism-control-plan.md): a fixed four-rung score
ladder (LRU, frozen learned ranker, exact label, exact next use) replayed under
all-resident and leaf-only eligibility at widths 16 and 64.

Outcome ([findings](mechanism-control-findings.md), 960 replays, all required
checks passed): under the published mechanism the signal gap
`U(label) − U(learned)` holds 53.8–84.0% of the headroom `T` (heap offline
reference minus sampled LRU) and is the
dominant term in 12/12 real trace × cell and in every seed; it stays dominant
under leaf-only eligibility at width 64 (69.9–91.1%). The candidate-search gap
is 0–28.1% of `T` under the published mechanism and at most 5.5% under
`leaf64`, where it is negative at the three smallest cells because the leaf
mechanism exceeds the greedy heap reference. The ladder inequality fails only
at its top pair (`label > offline`, at most 1.6 points). The control fits
nothing and promotes no policy; it does not bound what other causal
information could do, and it did not run the `pi3` update, binary targets or
arrival protection.

The capacity reversal the handoff lists as candidate A is checked against an
existing explanation in
[working-set-ratio-plan.md](working-set-ratio-plan.md): the ratio of a
trace-derived reuse working set to the L2 capacity, with its definition and
the threshold of 1 fixed before it was computed.

Outcome ([findings](working-set-ratio-findings.md), arithmetic on the L1 offer
stream and published tables, all checks passed): the ratio places the sign
change of 2-hit admission against LRU in 12/12 real trace × cell for the heap
pair and 12/12 for the sampled pair. Arrival protection is published at 6 of
the 12 cells; the ratio agrees in 4, and both misses are mixed seed readings at
ratios 1.005 and 1.179. The reversal is therefore read as an instance of a
known working-set argument and not as a separate contribution. Only 2%×4 has a
ratio at or below 1 on either trace, so the crossing is bracketed between
0.553 and 1.005 and not located more finely; the check says nothing about
learned scores.

The gap the mechanism control left between the frozen ranker and its label is
located by decision type and error type in
[error-location-plan.md](error-location-plan.md): 18 arms under the published
mechanism — hybrids that take admission from one score and eviction from the
other, the label with controlled Gaussian error, victim swaps at a fixed rate,
and the `pi3` and `binary` rankers the ladder left out — with four decision
statistics per replay.

Outcome ([findings](error-location-findings.md), 1,080 replays, all required
checks passed): the gap is eviction-located in 12/12 real trace × cell — the
label's choice of victim among residents recovers 65–100% of it and the
label's rejection decision none — with a third or more of it needing both at
L1 0.25%. Utility is monotone in the noise level in every cell and seed, and
the frozen ranker lies between levels 1 and 2 in 10/12. At an equal
victim-agreement rate the runner-up swap is above the uniform swap by 4–27
points in 12/12. No statistic orders the arms by utility at ρ ≥ 0.9 in more
than 2/12 cells; for the two published ranker updates the victim's mean label
excess has the sign of the utility change in 24/24 and pairwise concordance
moves against it in 4. The exact `binary` label equals the `next_use` label at
1%×4 and 2%×4 only. Every constructed arm reads the future and none is a
policy; the control does not say why the ranker evicts the residents it does.

Two questions the error-location control left open are pre-registered
together: [ranker-error-diagnosis-plan.md](ranker-error-diagnosis-plan.md)
re-reads the saved held-out decision logs to separate selection from
population in the victim's label excess and to classify the frozen ranker's
errors, and [horizon-control-plan.md](horizon-control-plan.md) asks whether an
exact reuse label at a shorter horizon reaches the `next_use` label, whose
order inside a reuse class is needed, and whether the location holds under
leaf-only eligibility.

Outcome of the diagnosis ([findings](ranker-error-diagnosis-findings.md), 240
saved populations, no replay, all checks passed): on a policy's own decisions
the victim's mean label excess has the sign of label-window utility in 22/24
trace × cell × target, but with the candidate population fixed the sign
depends on the population in 5/12 (`next_use`) and 7/12 (`binary`). At least
99.3% of the frozen ranker's label excess is the eviction of a reusable state
while a non-reusable candidate was sampled; decisions among reusable
candidates of differing label are at most 0.3%. On its own candidate sets the
`next_use` ranker makes such evictions less often than recency in 9/12 cells
and more often in 3.

Outcome of the horizon and class-order control ([findings](horizon-control-findings.md),
540 replays, all checks passed): an exact one-bit reuse label with recency
reaches the `next_use` label, within 10% of its gain over sampled LRU, at some
horizon of {6, 15, 60, 300, 600} s in 8/12 trace × cell; the four that fail
share an L2 capacity of 1% and sit in the grid's gap between 60 and 300 s. A
[fill-in](horizon-fill-plan.md) pre-registered after that result, with five
horizons between 90 and 240 s and the prediction that one suffices in all
four, held 4/4 (minimum at 150 s in each); its count is reported beside the
8/12 and not merged with it. The horizon that works grows with L2 capacity
(60, 150, 300, 600 s) and neighbouring grid horizons lose 6–85% of the gain.
With the exact 600 s reuse class given on residents, the frozen ranker
recovers 94–104% of what the exact label recovers on eviction in the six
cells where that class is sufficient and 33–61% in the other six; within a
class its order beats recency in 9/12. Under leaf-only eligibility the gap is
eviction-located in 12/12, with the arrival a candidate in a minority of
decisions. Nothing predicts the horizon before the fact, and no learned
predictor of the bit has been measured.

Outcome of the matched-horizon follow-ups ([findings](matched-horizon-findings.md);
[diagnosis plan](matched-horizon-diagnosis-plan.md), [class-order plan](matched-class-order-plan.md)
with a pre-run addendum correcting its own count of eight new cells): on the
frozen ranker's own logs re-read at the capacity-matched horizon `h*`, the
label excess is avoidable reusable eviction in 24/24, and the `next_use`
ranker makes that eviction 0.64× as often as recency at 0.25%×1 (60 s) and
2.1–2.3× as often at the four 1% cells (150 s); its within-decision
separation of the `h*`-bit is below that of the 600-second bit in 6/8 (the
registered prediction), the two exceptions being 0.25%×1 where the 600-second
separation is at chance. In replay (120 replays, 40 reproducing the published
600-second arms exactly), the exact class at `h*` on residents with the
ranker's admission and order recovers at least 90% of the exact label's
eviction gain in 6/8 new cells — the prediction of 8/8 failed at 0.25%×1 (73%
and 81%) — and 4/4 reproduced; recency within the matched class recovers it in
8/8 new, the ranker's order within the class is a consistent loss against
recency in 8/12 (a gain in 9/12 at 600 s), and the label's admission is above
the ranker's by up to 4.1 points.

Outcome of the matched-horizon checks ([findings](matched-horizon-checks-findings.md);
[evaluation-window plan](tail-window-check-plan.md), [class-order mix plan](class-order-mix-plan.md),
[leaf-eligibility plan](leaf-matched-horizon-plan.md) with a pre-run addendum
replacing a reproduction check that the published rows could not support):
with the last 600 s of the trace excluded from the count, the bit at `h*`
with recency reaches the label in 12/12, class recovery changes from 10/12
on the full window to 9/12 on the head window in one threshold cell
(conversation 0.25%×4, 0.903 → 0.859) and the sign of the learned order's
loss within the class is the full window's in 12/12 and larger. Within the
matched class under `all16`, the learned order's loss sits in the
class not reused within `h*`: the ranker ordering only the positive class
gives six all-seed gains, two digest-identical matches and four mixed-sign
cells (three positive means and tool-agent 1%×4 at −0.002157 points),
with no consistent loss. The ranker ordering
only the states not reusable within `h*` is a consistent loss in 9/12 and in
every cell where the full order lost; the prediction placed the loss in the
reusable class and fails 0/8. A random order within both classes loses to
recency in 12/12. Under `leaf16` with `h*` carried over (240 replays, 60
reproducing the published `leaf16` `label` rows), the ranker given the class
reaches 90% of the label's eviction gain in the same 10/12, the bit at `h*`
with recency reaches the label in 9/12 (0.20–0.21 short at 0.25%×1; the
prediction of 12/12 fails), and the ranker's order within the class is a
consistent gain over recency in 11/12 (the prediction of a loss in 8/12 fails
in the opposite direction), with recency inside the class below 0.9 of the
label's eviction gain in 7/12. The within-class order reading of the
follow-ups is specific to the `all16` candidate set; why is not examined.
The 0.11–0.20 increase at every `h*` < 600 s is in normalized shortfall `S`,
not a decrease in absolute avoided tokens. `R` divides by the gain from
changing resident selection to `evict_label` while keeping learned admission;
it does not measure all headroom. States outside the positive class can be
reused after `h*`. Aggregate orphaning counters neither confirm nor rule out
an orphaning-mediated explanation. This conditional order reversal addresses
a different question from the earlier finding that sampling was not the
largest term of the frozen-ranker-to-label utility gap.
