# Reviewable Characterization Artifacts

These lightweight artifacts were generated from the pinned Mooncake FAST'25 traces with:

```bash
python3 scripts/run_characterization.py \
  data/raw/conversation_trace.jsonl \
  data/raw/toolagent_trace.jsonl \
  data/raw/synthetic_trace.jsonl
```

The run uses 24 causal snapshots, ridge `0.001`, precision@100, seven cache-budget fractions from 0.1% to 10% of packed unique-state bytes, and both packed and fixed-block charging at the same absolute capacity. Raw traces and large per-state/generated outputs remain ignored.

## Files

- `online_policy_comparison.csv` and `online_policy_comparison.png`: causal online policies only, including 2-hit + LRU. The offline comparator is excluded.
- `longest_online_comparison.csv`: explicit Longest-first differences versus each causal comparator.
- `offline_headroom.csv` and `offline_headroom_sensitivity.png`: LRU versus approximate offline-next-use, separately for packed and fixed-block capacity.
- `structural_incremental.csv` and `structural_incremental_ap.png`: held-out Base versus Extended linear-ranking metrics.
- `structural_stratified.csv`: fan-out and branch-diversity ranking within approximate frequency and recency strata.
- `feature_correlations.csv` and `structural_coefficients.csv`: redundancy and fitted standardized weights.
- `capacity_sensitivity.csv`: policy-level packed versus fixed-block changes.
- `trace_summary.csv`: trace dimensions and idealized 2-hit accounting.

## Evidence-bounded conclusion

The approximate offline-next-use headroom remains above 5% in every trace, budget, and capacity model. Capacity-aware 2-hit helps or hurts depending on trace and budget. Longest-first is not a sufficient general selector when compared only with causal policies. Structural features provide small held-out gains in both real traces and separate reuse within several controlled strata, but the gains are not consistent in the synthetic trace and branch diversity is highly correlated with frequency. Structure-aware selection is therefore not promoted to the main research direction.

The real traces are about 59 minutes long and timestamp-bucketed. Structural signal can predict reuse, but whether its relative advantage increases at persistent-cache timescales remains unresolved.

## Phase 0.5 artifacts (`scripts/run_cross_workload.py`)

- `prediction_ablation.csv`, `fig1_prediction_ablation.png`, `fig4_feature_group_ablation.png`: temporal-feature ablation by horizon.
- `cross_workload_transfer.csv`, `fig5_cross_workload_transfer.png`: rankers fitted on one workload scored on another.
- `policy_replay.csv`, `fig2_budget_avoided_tokens.png`, `fig3_headroom_closure.png`: single-seed fixed-budget replay (superseded for closure numbers by the multi-seed Phase 0.75 run below).
- `model_coefficients.csv`, `run_config.json`: fitted standardised coefficients and the single hyperparameter set.

## Phase 0.75 artifacts (`scripts/run_predictability_gap.py`)

- `oracle_replay.csv`: seed-averaged avoided tokens and HeadroomClosure (mean, std, 95% CI) for every arm, including the oracle arms and the deterministic heap references.
- `oracle_decomposition.csv`: seed-paired signal / objective / candidate-search gaps per trace and budget, plus per-arm closure means.
- `candidate_set_prediction.csv`: ranking quality on the exact eviction candidate sets (pooled and within-decision AUC, prevalence, victim regret rates) per arm, budget, and horizon.
- `population_ladder.csv`: the same scorer's AUC on observed ⊃ cached ⊃ leaves at the shared test snapshots.
- `standardization_transfer.csv`, `coefficient_similarity.csv`: the transfer matrix and coefficient cosines under per-decision, train-set, and no standardisation.
- `fig6_oracle_closure.png`, `fig7_gap_decomposition.png`, `fig8_population_ladder.png`.
- `gap_run_config.json`: seeds, budgets, horizons, fit metadata.

The full per-seed dumps, the raw JSONL of every replay, and a sample of
candidate-level eviction decisions are written to `results/predictability_gap/`
and are not tracked.
- `candidate_model_check.csv`, `fig9_candidate_model_check.png`, `candidate_model_run_config.json`
  (`scripts/run_candidate_models.py`): within-decision ranking quality of single-feature,
  linear, and gradient-boosted models fitted on the same 23 history features over the
  logged eviction candidate sets, time-split with a horizon embargo.

## Phase 0.9 artifacts (`scripts/run_target_change.py`)

- `target_change.csv`: seed-averaged avoided tokens and HeadroomClosure (mean, std, 95% CI) for the same causal ranker fitted to each target (binary 60 / 300 / 600 s, count 60 / 600 s, next-use 600 s, Little's-law horizon-matched), with LRU, LFU, and the heap comparator re-run on the same seeds and budgets.
- `target_change_config.json`: budgets, seeds, fit metadata per target (rows, positives, horizon actually used) and the standardised coefficient vectors.
- `fig10_target_change.png`: closure by budget per target, with the same-target oracles of Phase 0.75 dotted.

Per-seed rows and the raw JSONL are written to `results/target_change/` and
are not tracked.

## Phase 0.95 artifacts (`scripts/run_two_tier.py`)

- `victim_events_summary.csv`: per trace × L1 policy × L1 budget, the L1 victim stream in the evaluation window: event counts, repeat-eviction share, share requested again before the trace end (the rest is right-censored), share whose ancestors are all still in L1 at reuse, median distances to reuse (seconds, requests, arriving bytes, distinct bytes, distinct bytes ÷ L1 capacity), the infinite-L2 rescue ceiling with and without ancestors as a share of input tokens, and the share of reused victims within k × L1 distinct bytes.
- `victim_events_by_depth.csv`: the same stream by block-depth bin.
- `two_tier_replay.csv`: one row per trace × L1 policy × L1 budget × L2 multiplier × (closure, hit model, L2 policy): avoided tokens by tier, extra avoided tokens over L1 alone and as a share of input tokens, L2 closure against the offline L2, dependency cost (independent − tree), standalone cost, byte-seconds and avoided tokens by depth (both restricted to the evaluation window), `l2_already_held` (standalone only: victims already resident in L2), and the single-tier references at L1 + L2 bytes.
- `two_tier_gate.csv`: the per-cell go / stop table (absolute, dependency, room) with the thresholds fixed before the run.
- `two_tier_single_tier_reference.csv`: LRU / LFU / offline in one prefix-closed cache of exactly L1 + L2 bytes (the two tier capacities summed).
- `two_tier_offline_tiebreak.csv`: the offline-L2 tie-break diagnostic on the cells 0.25% × 1 / 2 and 1% × 1 (every trace, both L1 policies): tree / independent / L1-only avoided tokens and the dependency cost under `prefix_first` (the grid's comparator) and `deeper_first`.
- `two_tier_config.json`: grid, arms, thresholds, evaluation-window start, working-set bytes.
- `fig11_victim_stream.png`, `fig12_two_tier_gain.png`, `fig13_depth_allocation.png`.

## Phase 0.97 artifacts (`scripts/run_decision_population.py`)

- `decision_population_fits.csv`: one row per fitted ranker (trace × target × training population × cell where applicable): rows, positives, horizon, split, standardisation convention, Newton iterations / convergence or ridge condition number, standardised coefficients, and the deviation of the A_pd control from the Phase 0.9 coefficients.
- `decision_population_predictive.csv`: off-policy predictive metrics of every ranker on the test-split (and train-split) observed, victim, and candidate populations: pooled AUC / Spearman, within-decision AUC (micro / macro) or Spearman over decisions with ≥ 2 distinct labels, constant-label counts, evicted-lowest-label share; `lru_key` / `lfu_key` rows give the behaviour policies' own orderings.
- `decision_population_replay.csv`, `decision_population_replay_seeds.csv`: L2 replay utility per trace × cell × arm, aggregated over five seeds (mean, std, CI95) and per seed: extra avoided tokens over L1 alone, share of evaluation-window input tokens, headroom closure = (arm − lru_s) / (heap offline − lru_s).
- `decision_population_onpolicy.csv`: on-policy within-decision metrics of every sampled arm at seed 0 on the decisions it actually faced, scored with the store's own score tuples; includes `evicted_positive_rate_when_avoidable` and `victim_matches_argmin_rate`.
- `decision_population_config.json`: grid, arms, seeds, hyperparameters, split, horizons.
- `fig14_decision_population.png` (closure by cell, targets × real traces), `fig15_population_ladder.png` (train population × evaluation population), `fig16_onpolicy_decisions.png`.

## Phase 0.98 artifacts (`scripts/run_decision_attribution.py`)

- `decision_attribution_losses.csv`, `decision_attribution_losses_seeds.csv`: per trace × cell × arm × target (aggregated over five seeds, and per seed) the partition of every window block beyond the L1 prefix into L2 hits, root loss by the removing decision (rejected / evicted / compulsory / unexplained), present-unusable by the same decision, and downstream-absent; tokens and shares of window input; differences to sampled LRU of the same cell and seed; the pre-registered readings (`dominant_failure`, `orphaning_reading`, `ranking_reading`).
- `decision_attribution_orphaning.csv`: per arm the rejections, resident evictions, share of evictions that orphan ≥ 1 L2-resident descendant, orphaned blocks and bytes (whole trace and window).
- `decision_attribution_decisions.csv`: window decisions split into rejections and resident evictions with reuse-within-H and avoidable shares; victim-vs-residents pairwise AUC / Spearman, residents-only and whole-set within-decision metrics, victim rank fraction; binary and next-use labels.
- `decision_attribution_config.json`: grid, arms, seeds, thresholds, git HEAD.
- `fig17_decision_attribution.png`: stacked attributed-loss classes per arm and cell (next-use target), downstream-absent share annotated.
- Phase 0.98b (same script, rerun; `phase = "0.98b"` in the config): the losses CSVs also carry the per-block charge — `absent_{rejected,evicted,compulsory,unexplained}_tokens/_blocks` (every block beyond the L1 prefix that L2 does not hold, charged to its own last removal), `downstream_*` (the part of it the root-only charge leaves unnamed), `perblock_decision_loss_tokens`, their shares and differences to sampled LRU, and the readings `perblock_dominant_failure`, `dominant_agrees`, `perblock_rejected_share`, `perblock_shortfall_coverage`, `tokens_per_absent_block_*`. The config records the Phase 0.98 column reproduction (`phase098_*`) and the arm-invariance check (`compulsory_arm_invariant`).
- `fig18_perblock_attribution.png`: the per-block charge stacked per arm and cell (next-use target); no leftover class.

The full event logs and raw replay rows are written to `results/two_tier/`
and are not tracked.
