# Research Design

## Title

**Value-Aware Persistent KV Selection for Specialized and Agentic LLM Serving**

## Hypothesis

Persistent KV states are not equally valuable. Structural and temporal signals can identify states with higher future reuse value than generic recency/frequency-only policies.

## Research questions

### RQ1 — Which KV states are valuable to retain?

Characterize reuse distributions before designing the policy.

Key dimensions:

- reuse count
- recency / inter-arrival
- prefix length
- fan-out / branch diversity
- prefix-tree ancestry
- state size
- lifetime / stability
- avoided recomputation

### RQ2 — Can we select future-useful KV states better than generic policies?

Compare proposed policies against LRU, LFU, 2-hit, and an offline oracle under the same cache budget.

### RQ3 — Does better selection reduce recomputation under the same budget?

Measure avoided prefill work, reused tokens, compute savings, and end-to-end effects.

## Important design constraint

Research 1 does **not** use semantic embeddings. Semantic-locality-aware prediction is deferred to Research 2.
