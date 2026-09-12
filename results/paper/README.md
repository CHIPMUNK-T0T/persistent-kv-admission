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
