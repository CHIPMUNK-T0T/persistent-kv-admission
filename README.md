# Persistent KV Admission

Research repository for **Value-Aware Persistent KV Selection for Specialized and Agentic LLM Serving**.

## Research question

Under a fixed persistent KV cache budget, can we select KV states with higher future reuse value than generic cache policies such as LRU and LFU?

The core objective is not I/O throughput optimization. It is **retention / eviction under finite capacity**.

## Core idea

We treat each reusable KV state as a materialized inference state with heterogeneous future value.

A tentative value model is:

\[
V(s) = E[N_{future}(s)] \times C_{recompute}(s)
\]

subject to:

\[
\sum_{s \in cache} Size(s) \le M
\]

where:

- `E[N_future(s)]`: expected number of future reuses
- `C_recompute(s)`: recomputation cost if the state is evicted
- `M`: persistent cache capacity

## Scope of Research 1

Research 1 uses only **observed structural and temporal locality**. Semantic embeddings are explicitly excluded from the policy signal in this phase.

Candidate signals:

- exact reuse count
- recency
- inter-arrival interval
- prefix length
- state size
- request-level fan-out
- branch diversity
- shared ancestry in the prefix tree
- avoided prefill tokens / FLOPs

## Baselines

- LRU
- LFU
- 2-hit / delayed admission
- frequency + recency
- structural-aware policy
- value-aware policy
- approximate offline-next-use comparator (headroom only)

## Evaluation

Primary metrics:

- cache hit rate under a fixed budget
- reused prefix tokens
- avoided prefill tokens
- avoided prefill FLOPs / GPU compute
- avoided recomputation per unit cache capacity
- admission precision / recall
- future-reuse prediction precision / recall

Secondary metrics:

- TTFT p50 / p95
- GPU energy per request, if measurable

## Correctness guardrail

Reuse is only allowed when exact-prefix and compatibility requirements are satisfied. Eligibility is conceptually:

\[
Eligible(s,t) = IdentityOK \land Compatible(t) \land IntegrityOK
\]

Potential invalidation factors include model weights, tokenizer, RoPE configuration, KV layout, attention backend, quantization / parallelism setup, and persistent-format version.

## Experimental plan

1. **Trace characterization**
   - Reconstruct prefix chains from public traces such as Mooncake.
   - Measure reuse count, recency, inter-arrival, fan-out, branch diversity, and prefix length.
   - Compare causal online policies separately, then measure LRU headroom against an approximate offline-next-use comparator.

2. **Policy simulator**
   - Replay traces under a fixed cache budget.
   - Compare LRU, LFU, 2-hit, Longest-first, frequency × prefix length, and a simple structural baseline.

3. **Online prototype**
   - Integrate the policy into vLLM / LMCache or an equivalent KV persistence layer.
   - Evaluate with real KV states under identical cache capacity.

4. **End-to-end validation**
   - Validate avoided compute, TTFT, and optionally GPU energy.

## Non-goals

- SSD / GDS / PCIe bandwidth optimization as the main contribution
- restore-vs-recompute crossover as the main result
- storage-engine redesign
- semantic embedding prediction
- approximate KV reuse
- improving model answer quality via caching

## Repository status

Early-stage research. Problem formulation, policy design, datasets, and evaluation methodology are expected to evolve as characterization results and prior-work review progress.

## Research 1: trace characterization

The repository now contains a reproducible characterization pipeline. It deliberately stops before implementing a new online policy.

```bash
./scripts/download_mooncake_traces.sh
python3 scripts/run_characterization.py data/raw/*_trace.jsonl
```

Raw generated outputs are written below `results/characterization/<trace>/`: state and reuse CSVs, causal horizon-ranking metrics, fixed-budget replay results, required plots, the exact run configuration, and a short `INTERPRETATION.md`. Lightweight review artifacts are tracked under `results/paper/`.

### Input schema and validated assumptions

The loader expects the official Mooncake FAST'25 JSONL fields `timestamp`, `input_length`, `output_length`, and ordered `hash_ids`.

Data source: [Mooncake FAST'25 trace release](https://github.com/kvcache-ai/Mooncake/tree/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces), pinned by the download script to commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`. Schema semantics are also described in the [FAST'25 paper](https://www.usenix.org/system/files/fast25-qin.pdf), Appendix A.

- `timestamp` is relative arrival time in milliseconds and must be nondecreasing.
- One `hash_id` is a cumulative prefix hash through one 512-token block. Equality means exact prefix reuse through that depth; it is not a hash of an independent block.
- `len(hash_ids)` must equal `ceil(input_length / 512)`. The final node can therefore contain fewer than 512 tokens.
- A repeated hash must always have the same depth, parent hash, prefix length, and incremental block length. The loader rejects inconsistent traces.
- File order is preserved, but requests with an equal timestamp are treated as simultaneous. They observe the cache before that timestamp's batch and cannot create within-batch hits.
- No session ID is present. The analysis therefore reports `request-level fan-out` and `branch diversity`, never cross-session reuse.

The state-size proxy is `incremental block tokens × bytes_per_token` (default 2048). It is only a capacity-normalization proxy, not a measured model-specific KV layout. A state's `prefix_tokens` is cumulative, while storage charges only its incremental final block. `potential_reuses × prefix_tokens` is reported as a cumulative-prefix value proxy; `potential_reuses × block_tokens` avoids double-counting ancestors and is reported as estimated avoided prefill tokens in the unbounded idealization.

### Prediction protocol

At sampled timestamp boundaries, every previously seen state is ranked using one causal signal at a time: recency, frequency, prefix length, observed direct fan-out, or observed terminal-branch diversity. The binary label is whether the exact state appears strictly after the boundary and within the horizon. AUC, threshold-based average precision, and tie-aware expected precision@100 are macro-averaged across snapshots. Windows extending past trace end are excluded; 6-hour and 1-day claims cannot be made from an approximately one-hour trace.

Structural incrementality is tested with two simple ridge linear rankers implemented with NumPy: `Base = log1p(frequency) + -log1p(age_seconds) + log1p(prefix_length)` and `Extended = Base + log1p(fan_out) + log1p(branch_diversity)`. The first 60% of sampled snapshots are training candidates, the final 40% are held out, and a full future-horizon embargo removes training labels that would not be known at the first test snapshot. This avoids a scikit-learn dependency and keeps fitting deterministic. Snapshot-state observations are weighted equally. The pipeline also reports transformed-feature Pearson correlations and structural-signal ranking within separate frequency and recency quantile strata.

### Replay protocol

All retained states form a prefix-closed forest. A child is useful only while its full ancestor chain is retained, and eviction considers leaves so it cannot strand a child. Causal online replay compares LRU, 2-hit admission with LRU eviction, LFU, Longest-first, frequency × prefix length, and the deliberately simple `fan-out + branch diversity` score. Equal-timestamp observations can satisfy the 2-hit gate only after the batch and cannot hit within that batch. The approximate offline-next-use comparator is excluded from online rankings and used only to estimate headroom. It has future knowledge and greedy leaf eviction, so it is not a proven optimum for weighted prefix caching.

Cache budgets are fixed byte capacities expressed as fractions of the packed unique-state working set. The same absolute capacity is used for both models: `packed` charges actual incremental block tokens, while `fixed_block` charges all states as a full 512-token block, including partial final blocks. Both models run every trace, budget, and policy; `effective_capacity_fraction` records the resulting fraction of each model's working set. Exact-prefix hit tokens, avoided prefill tokens, block and request hit rates, and avoided tokens per capacity byte are reported. This trace-only estimate does not measure FLOPs, GPU time, restore cost, SSD/GDS/PCIe throughput, or a restore/recompute crossover.

The current evidence shows that structural signals can rank reuse within some frequency/recency strata, but Extended does not improve Base consistently across all traces and horizons. Structure-aware selection is therefore not promoted to the main research direction. Structural signal can predict reuse, but whether its relative advantage increases at persistent-cache timescales remains unresolved.
