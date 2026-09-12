# Initial Mooncake Characterization Findings

These are results from the pinned Mooncake FAST'25 conversation, tool-agent, and synthetic traces using the default run configuration in `scripts/run_characterization.py`. Detailed CSVs, plots, and per-trace limitations are under `results/characterization/`.

| Trace | Requests | Duration | Unique cumulative-prefix states | 2-hit lost avoided-token opportunity | LRU → offline-next-use headroom across budgets | Longest-first gap to best |
|---|---:|---:|---:|---:|---:|---:|
| conversation | 12,031 | 3,537 s | 182,790 | 41.74% | 31.61–357.67% | 50.55–81.92% |
| tool-agent | 23,608 | 3,537 s | 183,300 | 17.42% | 8.99–61.32% | 78.31–93.32% |
| synthetic | 3,993 | 1,022 s | 43,924 | 23.32% | 95.91–911.07% | 70.83–92.42% |

## Current decisions

1. **2-hit admission is not free.** In an unbounded-cache idealization it discards the first reusable occurrence, losing 17–42% of incremental reusable-block token opportunities depending on the trace. Capacity-aware 2-hit replay is still needed before deciding whether pollution avoidance compensates for this loss.
2. **Prefix length alone does not explain value in this replay.** Longest-first is far behind the best comparator at every tested budget (0.1%, 0.25%, 0.5%, 1%, 2%, 5%, and 10% of unique-state bytes).
3. **The time-scale hypothesis is not established.** The synthetic trace shows recency leading at 1 s–1 min and branch diversity nearly matching frequency at 10 min. The two timestamp-bucketed real traces instead show frequency and branch diversity ahead at 10 s, while recency leads at 10 min. Six-hour and one-day horizons are unobservable, and one-hour windows have no usable history in these traces.
4. **The provisional stop criterion is not triggered.** The approximate offline-next-use comparator exceeds LRU by more than 5% at every tested budget in every trace.

## Limits on these decisions

The offline-next-use comparator uses future knowledge and greedy leaf eviction. It is a useful headroom indicator, but it is not a mathematical upper bound for weighted prefix-dependent caching. The real traces have only 1,180 distinct timestamps across roughly one hour; equal timestamps are treated as simultaneous, which makes the 1-second horizon unmeasurable and prevents within-bucket reuse. The results measure exact request-level cumulative-prefix reuse and do not establish cross-session reuse or measured GPU savings.
