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

## 8. Phase 0.95 result: on the victim stream, the tree costs nothing and the room stays

`scripts/run_two_tier.py`, `docs/two-tier-victim-findings.md`. A persistent
tier decides about the states an upper tier evicts, not about every state
ever seen. The upper tier (L1) is the same prefix-closed cache with heap LRU
or LFU at 0.25 / 1 / 2% of the working set; every eviction is one logged
event; generic lower-tier (L2) policies and the offline comparator run on the
same victim stream at L2 = 1–16 × L1, under a union-closure hit rule, an
independent-block control that removes prefix dependency, and a standalone
prefix-closed sensitivity. Because every arriving block enters L1, the stream
is identical for every L2 arm. No policy, no fitting, one seed.

Extra avoided prefill over L1 alone, share of evaluation-window input tokens,
L1 = LRU:

| L1, L2 | conversation: offline L2 / best generic L2 / infinite L2 | tool-agent: same |
|---|---|---|
| 0.25%, 4 × | 21.2 / 5.7 (LFU) / 35.8 | 15.6 / 4.7 (LFU) / 23.8 |
| 1%, 1 × | 22.1 / 6.2 (2-hit) / 34.9 | 15.4 / 4.6 (2-hit) / 22.4 |
| 1%, 4 × | 34.4 / 16.1 (2-hit) / 34.9 | 22.4 / 11.7 (2-hit) / 22.4 |
| 2%, 2 × | 32.2 / 15.5 (LRU) / 32.2 | 20.4 / 11.0 (LRU) / 20.4 |

- **Confirmed:** about 30% of victim events are requested again within the
  window; an infinite L2 would add 20–36 points of input tokens; the offline
  L2 adds 5–35 points and 2–5 × what the best generic L2 adds, in every cell.
  A victim's value is coupled to its ancestors (only 5–7% of reuses find them
  in L1), and LRU / LFU / 2-hit hold the chain for free: recency and
  frequency are monotone along a chain and L1 evicts leaves, so ancestors
  outlive descendants. The dependency cost is exactly zero tokens for all
  three policies in all 60 cells.
- **Refuted:** that the policies tested lose reuse to missing ancestors. The
  pre-fixed dependency gate fails in 45 of 48 real-trace cells; the three
  passes are the offline comparator's tie-break (0 with a depth-consistent
  tie-break). The control keeps each policy's contents and changes only the
  hit accounting, so this rules out ancestor-loss for these policies, not
  every tree-aware allocation. Once L1 + L2 reaches about 4% of the working
  set the offline L2 is at 0.98–1.00 of the infinite-L2 ceiling, so
  set-level allocation has ≤ 2% to add there; at 0.25% × 4 it is at
  0.59–0.66 and the question is open.
- **Decision (provisional):** no positive grounds were found to make
  tree-aware allocation the centre. The two-tier setting stays as the
  evaluation setting; the open question is the lower tier's own decision, an
  arriving victim against the residents it would displace, where generic
  policies leave 25–80% of the offline gain.

## 9. Phase 0.97 result: matching the training population does not convert prediction into retention

`scripts/run_decision_population.py`, `docs/decision-population-findings.md`.
On the two-tier setting of Phase 0.95, the L2 ranker's training population is
the only varied factor: A, the global observed set (Phase 0.9 control, in the
Phase 0.9 standardisation and in raw features); B, the L1 victim events; C,
the decision sets logged while sampled L2-LRU / L2-LFU run on the training
split. Same 23 features, same linear ranker, same three targets, same split
and embargo. Every scored L2 runs through one sampled-eviction mechanism
(arriving victim + 16 sampled residents, lowest score leaves); the generic
policies run through it too, and their heap versions and the heap offline
comparator are references. Headroom closure = (arm − sampled L2-LRU) /
(heap offline − sampled L2-LRU), five seeds. Design and interpretation cases
were pre-registered and amended before any result was inspected.

Best learned L2 versus best generic heap policy, extra avoided prefill over
L1 alone, share of evaluation-window input tokens, L1 = LRU:

| L1, L2 | conversation: best learned / best generic / offline | tool-agent: same |
|---|---|---|
| 0.25%, 1 × | 1.6 / 1.7 (LFU) / 8.4 | 1.8 / 1.9 (LFU) / 6.5 |
| 0.25%, 4 × | 5.7 / 5.7 (LFU) / 21.2 | 4.6 / 4.7 (LFU) / 15.6 |
| 1%, 4 × | 16.2 / 16.1 (2-hit) / 34.4 | 11.3 / 11.7 (2-hit) / 22.4 |
| 2%, 4 × | 22.3 / 21.6 (LRU) / 32.2 | 15.8 / 15.3 (LRU) / 20.4 |

- **Refuted:** that, with the same features and model, training on the
  decision sets that sampled L2-LRU / L2-LFU generate converts the history
  representation's predictive quality into retention (pre-registered Case
  A). No candidate-trained ranker reaches closure(A) + 0.10 together with
  the best generic sampled arm in more than 1 of 6 cells, for any target.
  This does not refute population mismatch in general: the mismatch with
  the decisions the learned policy creates for itself is untreated. The
  best learned L2 on any population closes 0.13–0.27 of the headroom,
  leaves 73–87% of it in every real cell, and is within −0.8 to +0.7
  input-token points of the best generic heap policy (above it in five
  cells, two of them inside the seed CI). The victim-trained ranker
  collapses at L1 = 2% (closure −0.27 to −1.87).
- **Confirmed, with a flag:** off-policy, on the LRU behaviour log, the
  global and the LRU-log-trained rankers clear the pre-registered accuracy
  bar (within-decision AUC ≥ 0.70, Spearman ≥ 0.30) in every cell, and
  candidate training adds ≤ 0.06; on-policy, on the decisions each learned
  arm creates for itself, the same rankers are 0.02–0.67 lower, beyond the
  0.05 tolerance in 28–36 of 36 evaluations, and below the bar in most cells,
  inverting to AUC 0.24–0.27 at 0.25% × 1. The behaviour keys show no such
  gap (≤ 0.007). The learned arms evict an avoidable reused state as often
  as sampled LRU does.
- **Unresolved:** whether the limit is "prediction adequate but not
  converted" (Case B, met by its letter) or "representation insufficient"
  (Case C, not established): the two decision populations give opposite
  answers, a policy's own decisions differ in difficulty from policy to
  policy, and a whole-set ranking metric does not weight the one state
  actually dropped. Per the pre-registration this does not license
  non-history signal, new heuristics, or a new policy. One recorded lead:
  under the sampled mechanism every arm, generic or learned, leaves
  0.05–3.9% of input tokens present in L2 but unusable for a missing
  ancestor, a cost that was exactly zero for the heap policies of Phase
  0.95; the victim-trained ranker is highest where it collapses. It is an
  upper bound on recoverable reuse, not an estimate.

## 10. Phase 0.98 result: the lost reuse is in the arrival's own decision, and the unusable KV is the mechanism's

`scripts/run_decision_attribution.py`, `docs/decision-population-findings.md`
§9. A descriptive replay of the Phase 0.97 arms (reproduced exactly, 330 of
330) on the cells 0.25% × 1, 1% × 4 and 2% × 4 of both real traces, with
read-only hooks that charge every lost block beyond the L1 prefix to the
decision that removed its first missing ancestor (rejection of the arriving
victim, eviction of a resident, or compulsory), count the L2-resident
descendants orphaned by each eviction, and split the ranking into
arrival-versus-residents and residents-only. Pre-registered readings, five
seeds, sampled LRU as the mechanism control.

- **Refuted:** that the learned scores add ancestor loss. Orphaned bytes
  are 0.03–1.05 × sampled LRU's for every arm and target (60 of 60 read
  "mechanism"); sampled LRU itself orphans on 44–48% of its evictions,
  177–254 GB per window, because a sample of 16 residents can hold an
  ancestor without its descendants. The present-but-unusable KV of Phase
  0.97 is the sampled mechanism's cost.
- **Confirmed:** the divergence from sampled LRU sits in the arriving
  victim's own first-round decision. Every non-LRU arm at 0.25% × 1 (71–99
  thousand rejections per window against 2–3 thousand), C_lru at 1% × 4 and
  B at 2% × 4 read "rejected"; B's collapse is the rejection of arriving
  ancestors of blocks it holds (1.2–3.0 input-token points present but
  unusable after a rejection, 0 for sampled LRU) and its arrival is ranked
  below the residents (victim-vs-residents AUC 0.31–0.40), the one cell pair
  where "arrival placement" fires. A_none and C_union are not worse than
  sampled LRU at 1% × 4 and 2% × 4.
- **Limit:** the pre-registered root-only attribution charges one block per
  broken chain and covers 22–41% of the L2-hit shortfall of the arms that
  fall behind; the rest is downstream-absent. A per-block attribution is the
  candidate next measurement, not scheduled. No new feature, policy or
  mechanism follows from this phase.

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
python3 scripts/run_two_tier.py data/raw/*_trace.jsonl
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

