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
- offline oracle

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
   - Compare LRU/LFU against an oracle under synthetic cache budgets.

2. **Policy simulator**
   - Replay traces under a fixed cache budget.
   - Compare LRU, LFU, 2-hit, structural-aware, value-aware, and oracle policies.

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
