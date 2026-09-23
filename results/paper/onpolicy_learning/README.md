# On-policy learning artifacts

This directory is the published output of the fixed experiment in
[the pre-registration](../../../docs/onpolicy-learning-plan.md); see
[findings](../../../docs/onpolicy-learning-findings.md) for interpretation.
The full run used `results/onpolicy_full_feb30eb_001/` and finished with exit
status 0. Its config records pre-registration commit `c10256a`, code commit
`feb30eb8`, trace/reference/source hashes, seed roles, replay counts, worker
measurements, and SHA-256 hashes for every original run output. The later
report-only README and spaced figure are outside that frozen manifest. The model JSON
files and complete training/test population NPZ files stay in that run
directory. Relative paths in this guide assume the repository root.

The registered primary comparison is `pi3 - pi0` after three fixed updates,
with `next_use` primary and `binary` a robustness target. For each trace,
capacity cell, target, and seed, the complete held-out utility window is the
last 40%; ranking uses only test decisions whose 600-second label horizon is
observable. There are five lineage seeds `0..4`, and every `pi0..pi3`
iteration is retained. The `cell` field encodes L1 working-set fraction and
L2/L1 multiplier, for example `l1=0.02,l2x4`.

| Artifact | Contents |
|---|---|
| [config](onpolicy_learning_config.json) | Frozen setup, provenance, integrity counts, file hashes, runtime and memory. |
| [pi0 fit checks](onpolicy_pi0_fit_checks.csv), [model links](onpolicy_model_links.csv), [canonical models](onpolicy_canonical_models.csv) | Published-fit reproduction; 480 policy/seed model links; 364 distinct serialized models, including four shared `pi0` models. |
| [training populations](onpolicy_training_populations.csv), [test populations](onpolicy_test_populations.csv) | 480 rows each; population NPZ paths/hashes, eligible/kept decisions, cap status, and replay checks. |
| [cross-score matrix](onpolicy_cross_score_matrix.csv), [own-policy ranking](onpolicy_own_policy_ranking.csv) | 3,840 entries for both train/test 4×4 matrices; 960 recorded-tuple diagonal evaluations. |
| [terminal ranking](onpolicy_terminal_ranking.csv), [aggregate ranking](onpolicy_aggregate_ranking.csv) | Five-seed registered common-terminal contrast, fit-population contrast, own-policy trajectory comparison and 24 cell/target summaries. |
| [seed utility](onpolicy_seed_utility.csv), [aggregate utility](onpolicy_aggregate_utility.csv) | All four iterations and five seeds, paired token/point differences, generic references, headroom closure, and label-window diagnostic. |
| [attribution loss](onpolicy_attribution_loss.csv), [orphaning](onpolicy_attribution_orphaning.csv), [complete attribution](onpolicy_attribution_complete.csv) | Phase 0.98b request/block partitions, full per-block decision loss, rejection/eviction/orphaning counters, and replay counters. |
| [coefficient similarity](onpolicy_coefficient_similarity.csv), [decision alignment](onpolicy_decision_alignment.csv) | Raw-coordinate model similarity and exact-event/candidate overlap; agreement is defined only on exact candidate lists. |
| [auxiliary LRU/LFU ranking](onpolicy_auxiliary_lru_lfu_ranking.csv), [candidate-reference checks](onpolicy_candidate_reference_checks.csv) | Rescoring older float32 generic logs and their identity/current-hash checks. Historical individual-log hashes were not published. |
| [registered verdicts](onpolicy_registered_verdicts.csv) | Mechanical A/B/C/D/unresolved labels and diagnostic/manual-review flags; interpret with the findings, especially the bound population caps. |

The three fixed figures were preregistered before the full run:

| Figure | What it shows | What it does not establish |
|---|---|---|
| [All-iteration utility](onpolicy_all_iterations_utility.png) | Five-seed mean paired utility at `pi0..pi3` for every cell and both targets. | A capacity-independent gain or permission to choose the best iteration after seeing results. |
| [Terminal ranking versus utility](onpolicy_terminal_rank_utility.png) | Five-seed mean registered ranking and utility contrasts with fixed threshold lines. | Five-seed sign consistency or a verdict from a mean point alone; inspect seed CSVs. Some labels overlap. |
| [Next-use cross-score heatmaps](onpolicy_cross_score_next_use.png) | The fixed train/test 4×4 means on each saved decision population. | A counterfactual full replay or convergence after three updates. Panel labels overlap in this original figure. |

A [spaced heatmap](onpolicy_cross_score_next_use_readable.png) renders the
same 24 panels and five-seed means from the published cross-score CSV. It is
an additional report-only visualization produced by
[`scripts/plot_onpolicy_readable.py`](../../../scripts/plot_onpolicy_readable.py);
the original figure, data, models, and verdicts are unchanged. The CSVs are
the numerical record.

All 480 training and 480 test decision samples reached the 40,000-decision
reservoir cap. `population_dominance_manual_review=True` is a review flag,
not a measured dominance verdict. The same two trace test windows were used
in earlier research decisions. These constraints limit claims from the
automatic labels and the auxiliary old-log comparison.
