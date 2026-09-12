# Persistent KV Admission

Research repository on **retention under finite capacity for persistent LLM KV
caches**: which materialised prefix states to keep when the cache cannot hold
them all.

Everything here is trace replay over the pinned Mooncake FAST'25 traces
(conversation, tool-agent, synthetic). No serving system is modified. Avoided
prefill tokens are exact-prefix trace estimates, not measured GPU time.

## 1. Original question

Given a finite persistent KV cache, how should states with high future reuse
value be selected so that the limited capacity is used best?

The objective is retention / eviction under a byte budget, not I/O throughput.
The value of a state was first written as

\[
V(s) = E[N_{future}(s)] \times C_{recompute}(s), \qquad \sum_{s \in cache} Size(s) \le M
\]

and the initial hypothesis (now Hypothesis 0, see
`docs/research-design.md`) was that structural and temporal locality signals
identify higher-value states than LRU or LFU do. Research 1 uses only observed
structural and temporal locality; semantic signals are reserved for Research 2.

## 2. Phase 0 finding: structural signal exists but is small

`scripts/run_characterization.py`, `docs/characterization-findings.md`.

- Reuse is heavy-tailed; most cumulative-prefix states are never reused.
- The approximate offline-next-use comparator beats LRU by far more than the
  5% stop criterion at every trace and budget, so generic policies do leave
  headroom.
- Fan-out and branch diversity add +0.002 to +0.008 AUC over frequency +
  recency + prefix length on the real traces and are nearly collinear with
  frequency. Longest-first is not a viable general selector.

Structure-aware selection was not promoted to the main direction.

## 3. Phase 0.5 finding: reuse is predictable, retention does not follow

`scripts/run_cross_workload.py`, `docs/temporal-prediction-findings.md`.

- 23 causal temporal features; an L2 logistic ranker fitted on an earlier
  window with a horizon embargo. **AUC 0.86–0.94** for the 300–600 s reuse
  label on the real traces. One hyperparameter set for every workload.
- A simple temporal ranking structure (mostly frequency + recency) fitted on
  one real workload scores the other real workload as well as its own model
  does. Two traces from one deployment family are not enough to call this
  general transfer.
- In fixed-budget replay through one shared sampled-eviction mechanism, the
  learned scorer does not beat parameterless LRU/LFU. **The best causal arm at
  any trace and budget recovers 0.248 of the LRU-to-offline headroom**; the
  online learner is never the best arm; the policy ranking inverts with budget.

So the claim "predict future reuse well and retention improves" is refuted as
stated on these traces.

## 4. Current question

**Why does accurate future-reuse prediction fail to translate into effective
KV retention under a finite cache budget?**

Three candidate causes, each measurable with the existing replay engine and
without inventing a policy:

| gap | measured as | meaning |
|---|---|---|
| signal gap | `oracle_binary − learned_history` | the causal predictor does not know the label well enough |
| objective gap | `oracle_next_use_sampled − oracle_binary` (also per-byte and reuse-count oracles) | the binary "reused within H" label is the wrong target |
| candidate-search gap | `offline_next_use (heap) − oracle_next_use_sampled` | sampled 16-leaf eviction cannot reach what exact search reaches |

Every arm above runs through the same prefix-closed, sampled-leaf eviction as
the causal arms. Prefix dependency is a shared constraint, not a separate arm.

## 5. Oracle decomposition results (Phase 0.75)

`scripts/run_predictability_gap.py`, `docs/predictability-retention-gap.md`.
Five seeds per arm, seven budgets, same hyperparameters as Phase 0.5.

**The cause of the gap depends on the cache budget.** HeadroomClosure of an
oracle that knows the exact training label (reused within 600 s on the real
traces, 300 s on synthetic):

| budget | conversation | tool-agent | synthetic |
|---|---:|---:|---:|
| 0.25% | 0.13 | 0.22 | 0.06 |
| 1% | 0.43 | 0.44 | 0.21 |
| 5% | 0.86 | 0.96 | 0.29 |

- **Below 1%: objective problem.** The perfect label recovers 5–21%. Among the
  leaves a small cache holds, 88–97% will be reused within 600 s, so the label
  cannot separate them. The best label horizon grows with the budget (60 s
  below 1%, 300 s at 1–2%, 600 s at 5–10%); a next-use-time oracle recovers
  0.82–1.00 everywhere. Value per byte accounts for ≤ 0.01 on the real traces
  at every budget ≥ 0.5%; sampled eviction accounts for 0.00–0.18.
- **At 2–10% on the real traces: signal problem.** The label is right (oracle
  0.72–0.96) and the history-based predictor fails on the decision population:
  the same score that reaches AUC 0.86–0.94 on observed states reaches
  0.57–0.63 on the eviction candidates, and its regret rate (evicting a leaf
  that will be reused when one that will not was available) equals LRU's.
- The cross-workload transfer is unchanged under train-set and no
  standardisation (real-trace cells identical to three decimals; coefficient
  cosine ≥ 0.98), so it is not an artefact of the preprocessing.

## 6. Decision after Phase 0.75: Research 1 continues with a changed target

Under the gate fixed before the run (`oracle_binary` closure ≥ 0.7 → signal
problem, ≤ 0.3 → objective problem, between → decompose further), the answer is
regime-dependent. What follows from it for this repository is one thing:

**Research 1 continues, with the prediction target changed.** The fixed
600 s binary label is replaced by targets the oracle arms showed to be the
right objective: a reuse label whose horizon matches the cache's residence
time, the reuse count within the horizon, or the next-use time. This changes
what is predicted, not the policy, and it is measured first on the decision
population using the candidate logs this phase produced, then in replay with
the same causal model and the same eviction machinery.

Semantic or embedding-based signals (Research 2) are a separate experiment
outside this repository's plan and are not a dependency of anything here.

Details, tables, and the Confirmed / Refuted / Unresolved lists are in
`docs/predictability-retention-gap.md`.

## 7. Phase 0.9 result: the target change helps, and does not close the gap

`scripts/run_target_change.py`, `docs/target-change-findings.md`. Same
features, ranker, penalty, training window, deployment path, eviction, seeds;
only the target changes: binary at 60 / 300 / 600 s, log reuse count within
60 / 600 s, negative log next-use time, and a parameter-free arm that picks the
binary horizon nearest the cache's Little's-law residence time.

HeadroomClosure of the best causal target against the Phase 0.5 target and
LFU, five seeds:

| budget | conversation: 600 s → best target | LFU | tool-agent: 600 s → best target | LFU |
|---|---|---:|---|---:|
| 0.25% | 0.06 → 0.15 (count 600 s) | 0.19 | 0.14 → 0.25 (next use) | 0.27 |
| 1% | 0.19 → 0.25 (next use) | 0.22 | 0.20 → 0.24 (next use) | 0.21 |
| 2% | 0.25 → 0.27 (binary 60 s) | 0.16 | 0.25 → 0.26 (binary 60 s) | 0.15 |
| 5% | 0.08 → 0.09 (binary 60 s, within CI) | −0.21 | 0.13 (600 s stays best) | −0.27 |

- **Confirmed:** below 1% the count and next-use targets recover 0.07–0.11
  more headroom than the 600 s label (outside the seed CIs); next-use time is
  the best causal target at 1% on both real traces; the horizon ordering of
  the oracle sweep carries over to the causal ranker.
- **Refuted:** that the target change closes the gap. Below 1% the best causal
  arm is still under LFU; it reaches 16–37% of its own count oracle and
  8–29% of its own next-use oracle at ≤ 1% on conversation; the best cell
  over all budgets from 0.25% up moves from 0.253 to 0.270. The Little's-law
  matched arm selects 60 s at every budget ≤ 2% and adds nothing.
- **What remains** is a signal gap on the new target: once the target is
  right, the history features cannot rank it on the eviction candidates. The
  synthetic trace stays unlearnable (≤ 0.09) under every target.

## Repository layout

- `src/persistent_kv_admission/` — trace loader, prefix-closed replay engine,
  temporal features, rankers, online scorer, cross-workload and gap modules
- `scripts/` — one runner per phase (`scripts/README.md`)
- `docs/` — design, plan, and one findings document per phase
- `results/paper/` — tracked review artifacts (CSV summaries, figures, run configs)
- `results/<phase>/` — full untracked dumps
- `tests/` — `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -t tests`

## Reproduction

```bash
./scripts/download_mooncake_traces.sh
python3 scripts/run_characterization.py data/raw/*_trace.jsonl
python3 scripts/run_cross_workload.py data/raw/*_trace.jsonl
python3 scripts/run_predictability_gap.py data/raw/*_trace.jsonl --seeds 5
# needs scikit-learn; see scripts/README.md for the virtualenv
.venv/bin/python scripts/run_candidate_models.py data/raw/*_trace.jsonl
python3 scripts/run_target_change.py data/raw/*_trace.jsonl --seeds 5
```

Dependencies: NumPy and Matplotlib; scikit-learn only for the candidate-model check.

## Reference: data and protocol

### Input schema and validated assumptions

The loader expects the official Mooncake FAST'25 JSONL fields `timestamp`,
`input_length`, `output_length`, and ordered `hash_ids`. Data source:
[Mooncake FAST'25 trace release](https://github.com/kvcache-ai/Mooncake/tree/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces),
pinned by the download script to commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`.
Schema semantics: [FAST'25 paper](https://www.usenix.org/system/files/fast25-qin.pdf), Appendix A.

- `timestamp` is relative arrival time in milliseconds and must be nondecreasing.
- One `hash_id` is a cumulative prefix hash through one 512-token block.
  Equality means exact prefix reuse through that depth.
- `len(hash_ids)` must equal `ceil(input_length / 512)`; the final node can hold
  fewer than 512 tokens.
- A repeated hash must always have the same depth, parent, prefix length, and
  incremental block length. The loader rejects inconsistent traces.
- Requests with an equal timestamp are simultaneous: they observe the cache
  before that timestamp's batch and cannot hit within the batch.
- No session ID is present, so fan-out and branch diversity are request-level
  terms.

The state-size proxy is `incremental block tokens × bytes_per_token` (default
2048). Storage charges only a state's own block; hits count the reused prefix.

### Prediction protocol

At sampled timestamp boundaries every previously seen state is ranked by a
causal score. The label is whether the exact state appears strictly after the
boundary and within the horizon. AUC, threshold-integrated average precision,
and tie-aware expected precision@100 are macro-averaged across snapshots.
Training uses the first 60% of the trace with a full horizon embargo; test
snapshots are shared across horizons. The real traces span about 59 minutes,
so horizons beyond 600 s are unavailable.

### Replay protocol

Retained states form a prefix-closed forest: a child is usable only while its
full ancestor chain is retained, and eviction removes leaves. Cache budgets are
fixed byte capacities expressed as fractions of the packed unique-state working
set. Every scored policy, baselines included, runs through the same sampled
eviction (16 random retained leaves re-scored at the decision); the exact heap
path is kept for the offline comparator and for deterministic reference arms.
`HeadroomClosure = (Policy − LRU) / (OfflineNextUse − LRU)` in avoided prefill
tokens over the evaluation window. The offline comparator has future knowledge
and greedy leaf eviction; it is a headroom estimate, not a proven optimum.

## Non-goals

- SSD / GDS / PCIe bandwidth optimisation as the main contribution
- restore-vs-recompute crossover as the main result
- storage-engine redesign
- semantic embedding prediction (Research 2, a separate experiment)
- approximate KV reuse
- improving model answer quality via caching
- vLLM / LMCache integration before the retention objective is settled

## Citation

If you use or reference this research, design, or implementation in your work, please cite it as:

```bibtex
@software{chipmunk2026persistent_kv,
  author = {CHIPMUNK},
  title = {Persistent KV Admission: Retention under Finite Capacity for Persistent LLM KV Caches},
  year = {2026},
  url = {https://github.com/CHIPMUNK-T0T/persistent-kv-admission}
}
```

