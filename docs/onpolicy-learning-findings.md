# On-policy decision-population learning: findings

This is the completed, fixed three-update experiment registered in
[the plan](onpolicy-learning-plan.md). The primary target is `next_use`;
`binary` is a robustness target. For each of two real traces, six L1/L2
capacity cells, two targets, and five lineage seeds, the runner collected
four training populations and cold-replayed `pi0` through `pi3`. Its scope is
the existing 23 causal-history features, fixed linear fitting and targets,
and three updates. These temporal held-out windows were used in earlier
research decisions, so the results are not independent confirmation.

**Main finding.** On `next_use`, `pi3` improves complete held-out utility by
at least the registered 0.10 input-token points on five-seed mean, with a
positive paired difference in every seed, in the 2%×1 and 2%×4 cells of
**both** traces. No cell passes the registered common-
terminal-population ranking improvement of 0.05. Thus there is no case A or B
for the primary target, and no cross-trace or cross-capacity ranking-plus-
utility claim. Utility improvement and ranking improvement are distinct
outcomes here: the former comes from a full policy replay; the latter
rescored the same saved `D_test(pi3)` decisions with `pi0` and `pi3`.

## Registered outcomes and all iterations

The tables show means across the fixed five seeds. `U1/U2/U3` are input-token
points relative to the paired `pi0` on the **complete last 40%** of requests;
`U3 range` and `ΔR range` show the minimum and maximum of the five paired
seed values. `ΔR` is the registered `R(pi3; D_test(pi3)) -
R(pi0; D_test(pi3))` (within-decision Spearman for `next_use`, macro AUC for
`binary`). `pi0` has zero paired utility by definition. These are fixed
iterations, not a selected best iteration. All 480 per-seed replay rows and
120 terminal ranking rows are in the linked raw tables below.

| Trace | L1×L2 | U1 / U2 / U3 (points) | U3 range | ΔR mean [range] | Automatic label |
|---|---:|---:|---:|---:|---|
| conversation | 0.25%×1 | −0.527 / −0.689 / −0.803 | [−0.863, −0.683] | −0.158 [−0.179, −0.133] | unresolved |
| conversation | 0.25%×4 | −0.435 / −0.661 / −0.603 | [−0.714, −0.451] | −0.128 [−0.143, −0.114] | C |
| conversation | 1%×1 | −0.701 / −0.741 / −0.681 | [−0.727, −0.617] | −0.059 [−0.067, −0.049] | C |
| conversation | 1%×4 | +0.346 / +0.520 / +0.253 | [+0.174, +0.329] | +0.018 [+0.010, +0.022] | unresolved |
| conversation | 2%×1 | +0.500 / +0.610 / +0.578 | [+0.377, +0.914] | +0.011 [+0.007, +0.014] | unresolved |
| conversation | 2%×4 | +0.430 / +0.528 / +0.538 | [+0.298, +0.633] | +0.005 [+0.004, +0.006] | unresolved |
| toolagent | 0.25%×1 | −0.345 / −0.501 / −0.458 | [−0.541, −0.421] | −0.256 [−0.268, −0.247] | unresolved |
| toolagent | 0.25%×4 | −0.459 / −0.625 / −0.611 | [−0.776, −0.461] | −0.170 [−0.183, −0.163] | C |
| toolagent | 1%×1 | −0.611 / −0.852 / −0.871 | [−0.979, −0.740] | −0.107 [−0.114, −0.089] | C |
| toolagent | 1%×4 | −0.039 / +0.229 / +0.121 | [−0.171, +0.339] | +0.016 [+0.012, +0.020] | unresolved |
| toolagent | 2%×1 | +0.137 / +0.307 / +0.229 | [+0.064, +0.388] | +0.002 [−0.008, +0.008] | unresolved |
| toolagent | 2%×4 | +0.445 / +0.454 / +0.483 | [+0.446, +0.539] | +0.014 [+0.013, +0.018] | unresolved |

The three smaller cells lose utility in all five seeds of both traces. The
2%×1 and 2%×4 cells gain utility in all seeds of both traces, but their
ranking changes are below 0.05; toolagent 2%×1 also mixes ranking signs.
Conversation 1%×4 passes utility, whereas toolagent 1%×4 mixes utility signs.
The registered automatic `next_use` count is C=4 and unresolved=8. The four
C rows are local to 0.25%×4 and 1%×1, each on both traces. The 0.25%×1
rows cannot be called C: `pi3` ranks about 0.34 on its fit and terminal
training populations, but its held-out own-population rank falls to −0.024
and −0.087 respectively. The held-out failure after a high training score
leaves population/generalization dynamics unresolved.

`binary` is a separate robustness reading. The same table conventions apply;
its `ΔR` uses macro AUC.

| Trace | L1×L2 | U1 / U2 / U3 (points) | U3 range | ΔR mean [range] | Automatic label |
|---|---:|---:|---:|---:|---|
| conversation | 0.25%×1 | −0.507 / −0.954 / −0.926 | [−0.963, −0.876] | −0.191 [−0.208, −0.161] | unresolved |
| conversation | 0.25%×4 | −1.562 / −1.682 / −1.941 | [−2.061, −1.865] | −0.095 [−0.104, −0.089] | D |
| conversation | 1%×1 | −1.372 / −1.980 / −1.846 | [−1.976, −1.775] | −0.024 [−0.037, −0.015] | D |
| conversation | 1%×4 | −0.439 / −0.896 / −0.683 | [−0.769, −0.574] | +0.047 [+0.044, +0.049] | C |
| conversation | 2%×1 | −1.338 / −1.304 / −1.584 | [−1.696, −1.464] | +0.082 [+0.080, +0.084] | B |
| conversation | 2%×4 | −0.039 / −0.091 / −0.047 | [−0.142, +0.103] | +0.020 [+0.019, +0.022] | unresolved |
| toolagent | 0.25%×1 | −0.377 / −0.663 / −0.626 | [−0.680, −0.595] | −0.231 [−0.311, −0.188] | unresolved |
| toolagent | 0.25%×4 | −1.121 / −1.238 / −1.203 | [−1.262, −1.102] | −0.110 [−0.126, −0.096] | D |
| toolagent | 1%×1 | −1.056 / −1.487 / −1.198 | [−1.286, −1.130] | −0.041 [−0.048, −0.034] | D |
| toolagent | 1%×4 | −0.265 / −0.466 / −0.138 | [−0.256, −0.068] | +0.055 [+0.053, +0.059] | B |
| toolagent | 2%×1 | −0.741 / −0.532 / −0.672 | [−0.810, −0.316] | +0.090 [+0.088, +0.095] | B |
| toolagent | 2%×4 | +0.183 / −0.007 / +0.144 | [+0.047, +0.212] | +0.023 [+0.022, +0.024] | unresolved |

The binary automatic labels are B=3, C=1, D=4, unresolved=4. In particular,
2%×1 has ranking gains on both traces but loses utility in all seeds; it
cannot establish a primary-target improvement. Binary D denotes a registered
gain on `D_train(pi2)` that did not hold on `D_test(pi3)`, with utility
reported separately. The 0.25%×1 binary terminal training population drops
well below its high fit-population AUC, so those rows remain unresolved.

## Utility against the fixed references

The five-seed mean `pi3` results below use the published per-seed sampled
LRU floor, heap offline-next-use ceiling, and best generic heap and sampled
arms selected by their published **cell means**. `Closure` is the existing
`HeadroomClosure` fraction, not the fraction of attainable optimal utility.
Generic columns give `pi3 - generic` in millions of avoided prefill tokens.

| Trace | L1×L2 | Closure | Best generic heap | Best generic sampled |
|---|---:|---:|---:|---:|
| conversation | 0.25%×1 | 0.072 | −0.588 | −0.216 |
| conversation | 0.25%×4 | 0.181 | −0.553 | −0.164 |
| conversation | 1%×1 | 0.159 | −0.577 | +0.074 |
| conversation | 1%×4 | 0.133 | −0.031 | +0.606 |
| conversation | 2%×1 | 0.140 | −0.454 | +0.284 |
| conversation | 2%×4 | 0.211 | +0.464 | +1.557 |
| toolagent | 0.25%×1 | 0.081 | −0.503 | −0.136 |
| toolagent | 0.25%×4 | 0.162 | −0.705 | −0.268 |
| toolagent | 1%×1 | 0.134 | −1.107 | −0.351 |
| toolagent | 1%×4 | 0.127 | −0.449 | +0.247 |
| toolagent | 2%×1 | 0.108 | −0.896 | −0.227 |
| toolagent | 2%×4 | 0.297 | +0.554 | +1.576 |

Only 2%×4 beats the best generic **heap** arm on both traces by five-seed
mean. Its margin corresponds to about +0.790 and +0.668 input-token points
on conversation and toolagent, respectively. Even there, closure is 0.211
and 0.297, and its common-population ranking gain misses the registered
threshold. The same replay's label-observable test subwindow is a diagnostic;
it does not replace the full last-40%-window utility above.

## Population, fitting, and attribution checks

The run completed 480 training-prefix and 480 held-out replays with exit
status 0. All four reconstructed `pi0` fits matched their published fits;
all 120 `pi0` replay rows matched the Phase 0.97 references. Sixty available
Phase 0.98b `pi0` loss rows matched; three of six cells have no published loss
reference. The 4×4 train/test matrices contain 3,840 entries and the
recorded-tuple table 960 entries. All recorded argmin checks passed, and the
float64 sequential rescoring agreed numerically with online decisions. The
run config records frozen source/reference hashes and unchanged pre-existing
documentation diffs. Older Phase 0.97 candidate logs had no historical
per-file hashes; their current hashes and published row/group identities
were checked, so auxiliary LRU/LFU rescoring remains secondary evidence.

Every one of the 480 training and 480 test population reservoirs reached its
40,000-decision cap. Eligible decisions ranged from 69,128–115,096 in
training and 44,620–62,474 in test. The cap therefore limits sampling
precision; cap binding alone does **not** prove that it dominates a verdict.
The maximum constant-label share was 1.725%/1.673% on `next_use` train/test
and 2.690%/1.758% on binary train/test. All 180 updated ridge models had
finite condition diagnostics (about 958–1,136); all 180 updated logistic
models converged. These measurements do not show a pervasive constant-label
or fit-pathology explanation, but no registered quantitative dominance cutoff
exists. Accordingly, the automatic C/D labels above are conditional on
manual review of capped populations, and are not unconditional final
classifications. The five seeds make the displayed directions reproducible
within this sampling scheme; they do not remove possible selection bias or
make the reused temporal test windows independent.

The saved policy populations also differ substantially. For `pi0` versus
`pi3` in the test reservoir, event keys overlap, but **zero** of the 120
trace/cell/target/seed comparisons has an exactly matching ordered candidate
set. Mean candidate-set Jaccard on shared keys is about 0.0093 for next-use
and 0.0051 for binary. Thus cross-trajectory prediction agreement has no
exactly matched test decisions to assess. The 4×4 rescoring matrix still
compares models on each *same saved population* and supplies the registered
common-terminal result; it is not a counterfactual full replay. Train-to-test
and policy-created population changes remain plausible explanations, not
measured causal contributions.

All 480 attribution rows pass the request/block partition and have zero
unexplained states. For each paired replay,
`perblock_decision_loss(pi0) - perblock_decision_loss(pi_i)` exactly equals
the L2-hit token gain, because the L1 and compulsory partitions are fixed.
The complete per-block charge includes absent descendants and present but
unusable blocks after rejected or evicted removals. This is accounting, not
an intervention identifying which individual decision caused a gain. For
example, at next-use 2%×4 the mean `pi3` gain is +0.538/+0.483 input-token
points on conversation/toolagent; the evicted-charge reduction supplies
about +0.538/+0.484 points while the rejected-charge change is near zero.
At 2%×1, larger evicted-charge reductions (+2.475/+2.106) offset increased
rejected charges (−1.897/−1.877), leaving the smaller net gains in the table.
Root-only loss or orphaning alone would miss that full split.

The figures show all iterations, registered terminal thresholds, and the
fixed 4×4 next-use train/test matrices. Their trends agree with the CSVs.
Some labels overlap in the terminal scatter and the original heatmaps are
dense; a [spaced heatmap](../results/paper/onpolicy_learning/onpolicy_cross_score_next_use_readable.png)
renders the same CSV means for reading. The tables and CSVs are the numerical
record. Neither the observed utility gain
nor the local automatic C/D patterns establish a need for semantic features,
eliminate distribution shift, or show convergence after three updates.

Primary data and provenance: [artifact guide](../results/paper/onpolicy_learning/README.md),
[seed utility](../results/paper/onpolicy_learning/onpolicy_seed_utility.csv),
[terminal ranking](../results/paper/onpolicy_learning/onpolicy_terminal_ranking.csv),
[registered verdicts](../results/paper/onpolicy_learning/onpolicy_registered_verdicts.csv),
and [run config](../results/paper/onpolicy_learning/onpolicy_learning_config.json).
