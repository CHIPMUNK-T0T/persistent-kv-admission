# On-policy decision-population learning

Status: pre-registered before implementation, smoke output, or confirmatory
output.  The implementation, tests, smoke run, and full run start only after
this document is reviewed and committed.

## 1. Question and scope

Phase 0.97 left an off-policy/on-policy gap: a linear causal-history ranker can
rank the fixed behaviour-policy decisions reasonably well, but its quality can
fall on the decisions that the ranker itself creates.  This phase asks whether
repeatedly fitting on a policy's own training-window decision population
improves (a) ranking on its own held-out decisions and (b) retained-cache
utility.

This is a fixed-round on-policy retraining experiment, not model selection.  It
reports policies `pi0`, `pi1`, `pi2`, and `pi3`, and the primary contrast is
`pi3 - pi0`.  There is no best-iteration selection.  Three updates are not a
convergence guarantee, and failure to stabilize within three updates is not
evidence that the feature representation is insufficient.

Phase 1 arrival protection is closed.  Every replay here uses
`l2_arrival_protection="none"`.  This phase adds no feature, semantic signal,
tree heuristic, target, nonlinear model, or policy eligibility rule.

The two existing real traces have already been used for research decisions.
Their held-out windows are temporal test windows, not a fresh independent
validation set, and conclusions are limited accordingly.

## 2. Fixed experimental surface

- Traces: `conversation_trace` and `toolagent_trace`, with paths and SHA-256
  identities recorded before the run.
- Cells: the complete Phase 0.97 grid,
  `L1 in {0.0025, 0.01, 0.02} x L2/L1 in {1, 4}`.  All six cells are reported.
- Cache: heap LRU L1; union-closure L2; tree hit rule; packed block sizes;
  `2048` bytes/token; sampled eviction width `16`; no arrival protection.
- Features/model: the existing 23 causal history features and existing
  `base2` feature indices; logistic regression for `binary`, ridge regression
  for `next_use`; L2 penalty `0.01`; the existing implementations only.
- Targets: `next_use` is primary; `binary` is the fixed robustness target.
  `count` is excluded.
- Split: first 60% for training and last 40% for utility evaluation, with the
  existing horizon and embargo.  Both traces use `H=600 s`.  A training label
  is eligible only when `timestamp + H <= split`.
- Caps/sampling: at most 40,000 complete decisions per population, at most
  150,000 rows passed to a fit, and the existing 10 negative-to-positive
  case-control rule for `binary`.  A decision, rather than a row, is the
  reservoir unit so a retained group is always complete.
- Lineage seeds: `0, 1, 2, 3, 4`.  Each trace/cell/target/seed lineage is
  trained independently and never borrows rows, model state, or normalizer
  statistics from another lineage.

The ranking test window is narrower than the utility window.  Ranking labels
are observable only for `split <= timestamp` and `timestamp + H <= trace.end`;
the final 600 seconds therefore do not enter ranking metrics.  The primary
utility remains the complete last 40%, exactly as in Phase 0.97.  The same
replay also records utility restricted to the label-observable test subwindow
as a diagnostic; it does not replace the primary utility.

## 3. Policy sequence and seeds

`pi0` is the Phase 0.97 `A_none` policy: `fit_fixed_model` with
`standardization="ranker"`, target-specific logistic/ridge fitting, fit seed
`0`, and raw-feature replay through `RawFeatureScorer`.  It is not
`standardization="none"`; the ranker owns the training-population mean and
scale.  The four fits (2 traces x 2 targets) are reconstructed and their row
counts, positive counts, fit diagnostics, and standardized coefficients are
checked against the published Phase 0.97 fit CSV before use.  The same verified
`pi0` is the initial policy for every cell and lineage seed of its trace and
target.

For lineage seed `s`, collection seed and held-out replay seed are both `s`.
They control the sampled-L2 resident draws.  A held-out run is a cold replay
from the first request with fresh scorer, history, cache, and RNG instances;
no object is carried from collection.  `fit_population_ranker` uses learning
seed `0` for every update, preserving the Phase 0.97 fit rule.  Thus the five
lineages differ through their policy-generated data and replay sampling, not
through an additional hidden fit-seed factor.  No Python hash is used to derive
a seed.

For each `(trace, cell, target, s)` independently:

1. Cold-replay the training prefix with `pi0`, reservoir only eligible
   training decisions, save `D_train(pi0)`, and fit `pi1` from that dataset.
2. Repeat from an empty cache/history/RNG with `pi1`, save `D_train(pi1)`, and
   fit `pi2` from only that dataset.
3. Repeat cold with `pi2`, save `D_train(pi2)`, and fit `pi3` from only that
   dataset.
4. Repeat cold with terminal `pi3` and save `D_train(pi3)` for its training
   self-evaluation.  Do not fit or imply a `pi4`.

Training is not cumulative: `pi(k+1)` sees only `D_train(pi_k)`.  Each refit
computes its mean and scale from the post-case-control/post-row-cap matrix that
the existing `fit_population_ranker` actually fits.  No normalization statistic
is inherited, averaged, or refreshed from `pi_k`, another seed, or held-out
data.  All fitted fields needed to reproduce scoring (class, indices, penalty,
standardization flag, mean, scale, coefficients, intercept, and fit
diagnostics) are serialized in a canonical numeric format and SHA-256 hashed.

The new training collector must not reuse `CandidateLogger` unchanged.  That
logger reservoirs the whole trace and filters after sampling, which lets test
decisions compete for training reservoir slots.  The new collector increments
its eligible counter and performs reservoir sampling only for complete
decisions satisfying `timestamp + H <= split`; it never offers a test decision
to the training reservoir.  The training replay stops at the split and writes
no test population.  If implemented with a prefix trace or replay cutoff, its
horizon, split, capacities, working-set bytes, state-index map, occurrences,
and labels remain those of the full trace; none is recomputed from the shorter
replay.  Labels keep the existing `bisect_right` same-timestamp convention.
Per-decision standardization is forbidden: every learned score uses only the
ranker's current fitted mean and scale.

## 4. Frozen held-out evaluation

Only after every `pi0..pi3` model for the complete grid has been fitted,
serialized, and hashed may held-out replay begin.  Each model is deserialized
and cold-replayed from the trace's first request with fresh instances.
Its own actions warm it through the training prefix; metrics begin at the
fixed split.  A cache produced during training collection, by an older policy,
or by another iteration is never reused.

For every iteration and seed, the held-out logger reservoirs only decisions in
the label-observable test subwindow and saves `D_test(pi_i)`.  It records raw
features, stable state identifiers, timestamp/group/decision ordinals, the
arrival and actual victim, the full online score tuple `(model score,
last_group tie-break)`, next-use/count label primitives, and decision type.
The actual victim must equal the first lexicographic argmin of the recorded
full tuple for 100% of kept decisions.  Any mismatch is fatal and prevents
publication.

The collector preserves the exact float64 history features used at the live
decision for audit and 4 x 4 cross-scoring.  It separately casts the training
matrix through float32 before `fit_population_ranker`, preserving the Phase
0.97 fit-input precision.  Offline cross-scoring reproduces `score_row`: start
from the intercept and add the selected terms in `ranker.indices` order,
rather than using a BLAS matrix product with a different reduction order.  The
recomputed own-model primary score and full tuple must then agree with the
recorded online tuple and victim.  Any remaining diagonal mismatch is reported
with exact/tied score-pair counts and minimum top-two margins; if a ranking
verdict depends on such a numerical disagreement, it is unresolved.  The
recorded online tuple remains authoritative for the action that actually ran.

`pi0` held-out replay must reproduce the published Phase 0.97 `A_none` rows for
all 120 trace/cell/target/seed identities (2 x 6 x 2 x 5), including exact
capacity/request/L1 counters and exact integer replay outcomes.  A mismatch
stops the run.

## 5. Cross-policy ranking evaluation

For each lineage and split separately, every saved population
`D_split(pi_i)` is scored by every `pi_j`, producing the fixed 4 x 4 matrix
`i,j in {0,1,2,3}`.  This is done for both training and test datasets and needs
no additional replay.  Models score raw logged features with their own frozen
normalizer.  Their evaluation tuple appends the recorded candidate
`last_group` tie-break before dense lexicographic ranking.  Targets and
lineages are never crossed.

The authoritative own-policy diagonal uses the recorded online tuple, while
the 4 x 4 matrix supplies like-for-like comparisons on one fixed candidate
population.  For each matrix entry and each of five seeds, record:

- within-decision mean Spearman for `next_use` and within-decision macro AUC
  for `binary`;
- micro AUC for binary as a diagnostic;
- row count, complete decision count, scorable-decision count, constant-label
  decision count and share, and label prevalence;
- the logging policy's actual-victim lowest-label rate and actual victim's
  average label-rank fraction (`0` is the least valuable candidate, `1` the
  most valuable), distinguished from the rescoring model's proposed-victim
  versions of the same diagnostics; a proposed victim is a counterfactual on
  that one saved decision set, not a replay-trajectory outcome;
- rejection and resident-eviction counts, reused-within-H shares, and the
  existing avoidable rejection/eviction rates; and
- recorded-tuple argmin integrity plus recomputed-versus-recorded victim
  agreement.

The primary ranking change holds the decision population fixed: for each seed,
it is `R(pi3; D_test(pi3)) - R(pi0; D_test(pi3))`, using each model's full
score tuple.  This avoids treating a change in candidate-set difficulty as a
model improvement.  The recorded own-policy contrast
`R(pi3; D_test(pi3)) - R(pi0; D_test(pi0))`, every own-policy absolute R, and
every iteration remain visible as trajectory-level evidence.  Training
self-ranking is diagnostic only and is used to distinguish low fitted signal
from a train-to-test or policy-distribution shift; it is not a formal
convergence test.

All four models are also scored on the existing Phase 0.97 sampled LRU and LFU
candidate logs for the matching trace and cell.  Only their embargoed training
and label-observable test partitions are used.  The existing logs are first
checked for trace, cell, horizon, split, width, policy, row/group counts, and
artifact hash.  They are never pooled across the split.  These older logs hold
float32 features, so this common-policy comparison is auxiliary and carries
the same recomputed-score/near-tie precision diagnostics; it cannot override
the float64 terminal-population primary ranking result.

## 6. Utility and attribution

For every held-out replay, report all four iterations and all five seeds:

- avoided prefill tokens and the paired `pi_i - pi0` token difference;
- input-token points, defined as
  `100 * (avoided(pi_i) - avoided(pi0)) / requested_input_tokens`;
- extra L2 avoided tokens and existing `HeadroomClosure`, using that seed's
  published sampled LRU floor and published heap offline-next-use ceiling;
- difference from the best generic heap policy and, as context, the best
  generic sampled policy; generic winners are selected by their published
  five-seed cell mean, not per seed;
- L2 admissions, rejections, evictions, hits, and present-unusable tokens; and
- Phase 0.98b loss/orphaning outputs, including the complete per-block charge,
  `perblock_decision_loss_tokens`, rejected/evicted/compulsory partitions, and
  the request/block partition identities.

The Phase 0.97 generic, other learned, and offline rows and Phase 0.98b
attribution conventions are references, not newly fitted or replayed arms.
They are admitted only after exact identity checks on trace hash, split,
horizon, cell capacities, requested/L1 tokens, policy, closure, hit model,
size model, sample width/seed where applicable, and public CSV hash.  Missing
logs may be generated into a new run directory, but existing published
artifacts are never overwritten.

The primary utility window is the complete last 40%.  The label-observable
test-subwindow counters are a fixed diagnostic produced by the same replay and
reported beside it.  No result chooses between the windows.

## 7. Stability diagnostics

Diagnostics are reported but cannot select an iteration or alter the verdict:

- pairwise cosine similarity of the 23-dimensional raw-coordinate slopes,
  where fitted coefficients are divided by their fitted scales;
- prediction/victim agreement of model pairs on exactly matched decisions;
- exact decision-set match rate and candidate-set Jaccard overlap; and
- population sizes, cap-binding rates, feature/label summaries, fit iterations,
  logistic convergence flags, and ridge normal-matrix condition numbers.

Policies can change the number and sequence of decisions.  Pairwise matching
uses `(timestamp, trace group index, ordinal among L2 decisions at that
timestamp)` as the provisional event key.  A prediction-agreement comparison
is made only when the key and the full ordered candidate-state list match.
For every pair, report keys present only on either side, keys shared with a
different candidate list, exact matches, and Jaccard summaries.  No unmatched
decision is forced into a one-to-one pair.

## 8. Pre-registered interpretation

The primary comparison is `pi3 - pi0`; adjacent changes and `pi1/pi2` are
descriptive.  The fixed material-improvement thresholds are:

- utility: mean improvement of at least `0.10` input-token points and a
  strictly positive paired difference for every one of five seeds; and
- ranking: mean improvement of at least `0.05` in the common-terminal-
  population metric `R(pi3; D_test(pi3)) - R(pi0; D_test(pi3))` and a strictly
  positive paired difference for every seed.

Thresholds are evaluated per trace/cell/target, never after pooling cells or
traces.  A cross-workload directional claim additionally requires the same
cell to pass on both real traces.  Consistency across capacity is reported as
the number and identities of same cells passing on both traces; a broader
claim requires at least two such cells.  Every cell is shown even when this
rule is not met.

For the primary `next_use` target, the cases are:

- **A, ranking and utility improve:** both material thresholds pass.  Strong
  support additionally requires the own-policy `pi3 - pi0` ranking direction
  to agree across all seeds; otherwise the fixed-population result is reported
  with a candidate-population caveat.
- **B, ranking improves but utility does not:** only the ranking threshold
  passes.
- **C, the tested linear/target procedure remains low-ranking:** neither
  threshold passes; for all five seeds, `pi3` is below the existing
  `R_high=0.30` on the population that fitted it (`D_train(pi2)`), its terminal
  self-population (`D_train(pi3)`), and held-out `D_test(pi3)`; the
  fixed-population ranking gain is below `0.05`; every logistic fit converged or
  every ridge solve has finite parameters and condition diagnostics; and
  utility does not materially improve.  This supports only the claim that the fixed linear model, target
  procedure, and evaluated causal-history features did not yield adequate
  ranking in this experiment.  It does not establish that causal history is
  intrinsically insufficient or that semantic features are necessary.  A high
  fit-population R followed by a low terminal-self R instead indicates
  remaining population shift and cannot satisfy C.
- **D, training gain does not hold out:** the registered ranking improvement
  passes on the actual `pi3` fit population `D_train(pi2)` but fails on the
  fixed held-out population `D_test(pi3)`.  This is a train-to-held-out
  generalization failure; its utility result is reported separately.
- **Unresolved:** mixed seed signs, inconsistent traces/cells, cap- or
  constant-label-dominated populations, numerical diagonal disagreement, fit
  pathologies, a material utility gain without the registered ranking gain, or
  any pattern not satisfying A-D cleanly.

For binary, replace Spearman/R_high with macro AUC/`R_high=0.70`, retaining
the all-five-seed requirements.  Binary is a robustness result and cannot
replace the primary next-use verdict.  Ending at `pi3` or failing to improve
after three updates alone does not support C.  Observed instability, or high
ranking on fit population `D_train(pi2)` followed by low ranking on terminal
self-population `D_train(pi3)`, is remaining shift/unresolved and takes priority
over C.

Multiple diagnostic flags may coexist, but each trace/cell/target receives one
primary label in this fixed order: invalid integrity/leakage/numerical checks or
unresolved dynamics first; then A or B; then D; then C; otherwise unresolved.
All subordinate flags and every cell result remain in the output.

## 9. Implementation checks and run order

Before smoke/full execution, unit tests must cover: prefix-only eligibility
before reservoir sampling; complete-decision reservoir behavior and cap;
training/test embargo boundaries; equivalence of prefix and full replay before
the cutoff with full-trace horizon/split/capacity/state-index inputs; no
accumulation between iterations; normalizer replacement from the current fit
only; deterministic seed mapping; model serialization round-trip and stable
hash; actual full-tuple argmin; float64 sequential-score equality and near-tie
diagnostics; 4 x 4 matrix completeness; exact-match decision alignment with
unmatched accounting; and attribution partition identities.

The required order is:

1. review and commit this pre-registration alone;
2. implement and test, then commit code/tests before seeing smoke results;
3. run smoke in a separate directory, using one trace/cell/target/seed through
   all four iterations, and do not copy smoke artifacts into paper outputs;
4. use smoke wall time, peak worker RSS, serialized/log sizes, and available
   memory to fix a safe worker count; then run the full grid in a new directory;
5. publish CSV/config/model manifests and figures only after every identity and
   invariant check passes.

The full design contains 480 training-prefix collection replays
(`2*6*2*5*4`) and 480 frozen held-out replays, for 960 total.  The 4 x 4
cross-scoring matrices add no replay.  Existing compressed float32 candidate
logs are roughly 11--25 MiB each and the Phase 0.97 decision-population
directory is about 637 MiB; float64 logs and live logger copies will be larger.
Each worker therefore spools one iteration population to an ignored run
directory, releases it before advancing, and later loads one population at a
time to score all four models.  Workers return metrics/manifests rather than
row matrices to the parent.  Logs are retained for audit rather than deleted.
Worker count is chosen from measured smoke peak RSS and disk use rather than
fixed at 20 on the current 31 GiB host.

The run config records pre-registration/code commits, start/end source
manifests, trace and reference hashes, model hashes, seed roles, all fixed
constants, output identities, replay counts/timings, worker/RSS measurements,
and hashes of the three pre-existing dirty documentation diffs.  It refuses to
publish if those diffs change during the run.

Required paper artifacts are per-model details/hashes, training/test population
metadata, 4 x 4 train/test cross-score tables, own-policy per-seed ranking,
per-seed and aggregate utility, Phase 0.98b attribution, generic/reference
identity checks, stability/decision-overlap diagnostics, run config, and figures
showing all iterations without selecting a best one.  Figures are fixed before
the run: (1) iteration 0--3 utility for all six cells and both traces, with
`next_use` and `binary` separated; (2) terminal common-`D_test(pi3)` ranking
change against utility change for every cell, with registered thresholds; and
(3) fixed train/test 4 x 4 cross-score heatmaps for primary next-use across all
cells.  Binary cross-score remains complete in CSV and is shown in a separate
supplementary figure if space permits; test results cannot select which cell or
iteration is plotted.
