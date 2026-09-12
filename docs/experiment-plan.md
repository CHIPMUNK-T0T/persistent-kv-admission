# Experiment Plan

## Phase 0 — Trace characterization

- obtain public KV/prefix traces
- reconstruct prefix chains
- derive reuse count distribution
- derive recency / inter-arrival distribution
- compute fan-out / branch diversity
- analyze prefix length vs reuse value
- sweep virtual cache budgets
- compare LRU/LFU/oracle headroom

Decision gate: if generic policies are already near oracle, revise the hypothesis before implementing a new policy.

## Phase 1 — Policy simulator

Policies:

- LRU
- LFU
- 2-hit
- structural-aware
- value-aware
- oracle

Metrics:

- hit rate
- admission precision / recall
- reused tokens
- avoided recomputation

## Phase 2 — Online prototype

Target integration candidates:

- vLLM
- LMCache

Hold cache capacity constant across policies.

## Phase 3 — End-to-end validation

Optional system metrics:

- GPU power / energy
- TTFT p50 / p95
- storage overhead as a control variable, not the research contribution
