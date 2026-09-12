# Mooncake Characterization Findings

These results use the pinned Mooncake FAST'25 conversation, tool-agent, and synthetic traces with 24 causal snapshots and cache budgets of 0.1%, 0.25%, 0.5%, 1%, 2%, 5%, and 10% of packed unique-state bytes. Packed and fixed-block models use the same absolute byte capacity. Detailed generated outputs are under `results/characterization/`; lightweight review artifacts are under `results/paper/`.

## Fixed-budget replay

| Trace | Unbounded 2-hit loss | Capacity-aware 2-hit vs LRU, packed | Offline-next-use headroom over LRU, packed | Maximum fixed-vs-packed avoided-token change |
|---|---:|---:|---:|---:|
| conversation | 41.74% | -23.73% to +76.37% | +31.61% to +357.67% | 2.13% |
| tool-agent | 17.42% | -10.02% to +37.60% | +8.99% to +61.32% | 2.07% |
| synthetic | 23.32% | +3.58% to +26.54% | +95.91% to +911.07% | 2.14% |

The capacity-aware 2-hit result includes avoided cache pollution. It improves every tested synthetic budget, but it changes sign across budgets in the two real traces. The unbounded first-reuse loss therefore does not justify accepting or rejecting 2-hit by itself.

Longest-first is compared only with causal online policies. Under the packed model, it is 18.78–55.54% below LFU on conversation, 74.84–91.89% below LFU on tool-agent, and 27.65–62.99% below LFU on synthetic. It is also below frequency × prefix length and the simple structural baseline at every conversation and tool-agent budget. It can beat LRU at isolated conversation or synthetic budgets, so the result is not that Longest-first always loses; the result is that prefix length alone is not a sufficient general selector.

The approximate offline-next-use comparator is used only for headroom. It exceeds LRU by more than 5% at every trace, budget, and size model, so the provisional stop criterion is not triggered. It is not included in causal online rankings.

The winning causal policy is unchanged between packed and fixed-block charging in all 21 trace-budget pairs. The complete online ordering is unchanged in 19/21 pairs. Holding byte capacity constant, the maximum absolute avoided-token change is 2.07–2.14% across the three traces. Capacity modeling changes some middle ranks, while the winner and the headroom decision are stable here.

## Incremental structural prediction

Base is a deterministic ridge linear ranker using transformed frequency, recency, and prefix length. Extended adds causal observed fan-out and branch diversity. Training uses the first 60% of sampled snapshots, testing uses the final 40%, and a full horizon embargo excludes training labels unavailable at the first test snapshot.

| Trace | Horizon | Extended − Base AUC | Extended − Base AP | Extended − Base P@100 |
|---|---:|---:|---:|---:|
| conversation | 10 s | +0.0028 | +0.0009 | +0.0044 |
| conversation | 1 min | +0.0022 | +0.0005 | +0.0156 |
| conversation | 10 min | +0.0048 | +0.0009 | +0.0222 |
| tool-agent | 10 s | +0.0020 | +0.0013 | +0.0033 |
| tool-agent | 1 min | +0.0023 | +0.0014 | +0.0189 |
| tool-agent | 10 min | +0.0045 | +0.0057 | +0.0767 |
| synthetic | 1 s | -0.0759 | -0.0244 | +0.0044 |
| synthetic | 10 s | -0.0004 | -0.0215 | +0.0167 |
| synthetic | 1 min | +0.0020 | +0.0089 | +0.0767 |

Branch diversity retains ranking signal inside approximate frequency strata: its AUC ranges from 0.5995 to 0.6671 for the observable 1-minute horizons, and reaches 0.6392–0.6552 at 10 minutes in the real traces. Similar separation often remains inside recency strata. However, transformed branch diversity and frequency have Pearson correlation 0.9901–0.9960. Extended gains are consistently positive but small in the real traces, while synthetic short-horizon AP declines.

The evidence supports the narrower statement that structural topology can contain predictive information after coarse frequency or recency control in these traces. It does not support a workload-independent incremental advantage beyond the Base model. Because gains are not consistent across traces and horizons, structure-aware selection should not be the main research direction at this stage.

## Time-scale limit

The two real traces cover about 59 minutes and have only 1,180 distinct timestamp buckets. Their 1-second windows have no future positives under simultaneous-batch semantics. One-hour, 6-hour, and 1-day held-out evaluations are unavailable; the 17-minute synthetic trace also cannot support an embargoed 10-minute train/test split. Structural signal can predict reuse, but whether its relative advantage increases at persistent-cache timescales remains unresolved.

## Method limits

The approximate offline-next-use comparator has future knowledge and greedy leaf eviction, so it is not a proven optimum for weighted prefix-dependent caching. Quantile stratification reduces variation in one control at a time but does not create matched causal pairs. Snapshot-state observations repeat states over time and are weighted equally. Avoided tokens are exact-prefix trace estimates rather than measured GPU time, FLOPs, restoration cost, or physical KV bytes. The traces have no session IDs, so fan-out and branch diversity are request-level terms.
