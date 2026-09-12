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
