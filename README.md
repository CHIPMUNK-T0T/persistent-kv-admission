# Persistent KV Admission

Research on **which materialised prefix states to retain in a finite persistent
LLM KV cache**. This repository studies avoided prefill tokens with trace
replay on the pinned Mooncake FAST'25 conversation, tool-agent, and synthetic
traces. It does not measure GPU latency or modify a serving system.

## Research question and current position

A persistent tier receives states evicted by an upper cache. Its decision is
whether an arriving victim is worth more than the states it would displace.
The fixed objective is to increase avoided prefill tokens by choosing which
materialised states to retain under finite lower-tier capacity. The current
question is **which future-reuse information helps resident selection, and how
its horizon, within-class selection rule and candidate eligibility affect
that information's utility**. The
[research status](docs/research-status.md) separates supported findings,
unresolved mechanisms, and the current measurement boundary.

The latest [matched-horizon checks](docs/matched-horizon-checks-findings.md)
complete 840 replays. Exact reuse classes help in most tested conditions, but
the learned-versus-recency order within those classes reverses when eligibility
changes from all residents to leaves. This specifies conditions for useful
selection information; a deployable policy, a horizon selected before test
results and independent-workload validation remain open. Toward that
validation, the [Bailian input audit](docs/bailian-input-audit.md) checks the
four Qwen-Bailian traces and their 512-token conversion (138/138 checks, no
replay), and the [external-workload check plan](docs/bailian-external-check-plan.md)
is a pre-registration draft under review; nothing has been replayed on those
traces. The Japanese
[2026-10-03 next-phase handoff](docs/next-phase-handoff-20261003.md) sets out the
next planning step. The earlier [2026-09-27 research handoff](docs/strategy-handoff-20260927.md)
records the strategy at that stage. Neither handoff is a new preregistered experiment.

The observations so far are:

- Exact-prefix reuse has substantial skew and a greedy future-aware comparator
  leaves measurable room above generic policies. That comparator is not a
  proved optimum; its denominator and baseline change across phases.
- With the frozen `next_use` ranker and target held fixed, exact labels produce
  substantially more avoided tokens through the same sampled mechanism.
  Replacing resident selection recovers 65–100% of the learned-to-label gap in
  the error-location control; this does not make admission universally irrelevant.
- An exact reuse bit and its within-class selection rule must be evaluated
  together. The matched-horizon class recovery passes in 10/12 full-window
  cells and 9/12 with the last 600 s excluded. Under `all16`, the learned
  within-class order consistently loses to recency in 8/12; under `leaf16` it
  consistently wins in 11/12. This reversal does not identify its mechanism
  and does not contradict sampling being a secondary term in the earlier
  frozen-ranker-to-label gap decomposition.
- Changing the prediction target and matching training to L2 decision logs
  have not generally converted reuse prediction into better retention. The
  off-policy and policy-created decision populations give different ranking
  readings. This does not show that causal history is inherently inadequate.
- The fixed three-update on-policy experiment improved **complete held-out
  replay utility** at some capacities but did not pass its registered
  common-terminal-population ranking threshold. A later one-step
  counterfactual experiment found substantial *realized* action regret for
  both learned and exact-label choices. It used a single captured
  continuation for its main 320-state comparison; expected action values
  remain unresolved.
- The registered 16-fresh-stream cross-fit on 40 fixed states found a
  descriptive unrestricted held-out mean gain of −267.8 tokens versus the
  original action, after +15,003.0 on the selection folds. Exact-label-tie
  restriction gave +90.3 held-out after +13,567.2 on selection folds.
  Stratum effects and state signs vary; this does not establish equal
  expected action values.
- A follow-up replayed two fixed zero-own-reuse actions on those same 40 states
  and 16 streams. Same-size focal blocks showed mixed reward differences;
  later L2 victim sequences usually diverged before the first hit difference,
  and differences often continued after both focal blocks left L2. This is a
  trajectory diagnostic, not a new expected-`Q` estimate.

No result here establishes that a semantic signal, a new policy, or a
particular tree-aware allocation is necessary.

## Representative results and how to read them

| Experiment | Observation | Scope |
|---|---|---|
| [Two-tier victim stream](docs/two-tier-victim-findings.md) | About 30% of L1 victim events on the two real traces return before trace end. At L1=1%, L2=4×L1, the greedy offline L2 adds 34.4 / 22.4 input-token points over L1 alone on conversation / tool-agent; the best tested generic L2 adds 16.1 / 11.7. | Fixed heap-LRU L1, same victim stream, union-hit accounting; offline L2 is greedy, not optimal. |
| [Decision-population learning](docs/decision-population-findings.md) | The post hoc best learned L2 exceeds the best generic **heap** mean in five of 12 real-trace cells, by at most 0.7 input-token points, while closing 0.13–0.27 of that phase's headroom, whose floor is sampled L2-LRU; on the same scale heap LRU sits at 0.00–0.19 across cells, 0.15–0.19 at 2%×4. | Best model/target selected after inspection; five sampling-seed intervals do not establish formal superiority. This is separate from on-policy `pi3`. |
| [On-policy updates](docs/onpolicy-learning-findings.md) | Primary `next_use` `pi3 − pi0` complete-last-40% utility at 2%×4 is +0.538 / +0.483 input-token points, with 5/5 positive paired seeds on conversation / tool-agent. The common `D_test(pi3)` ranking gains are +0.005 / +0.014, below the registered +0.05 threshold. | Fixed three updates, reused trace windows, capped decision reservoirs. In the shorter label-observable window the corresponding utility gains are +0.143 (3/5) / +0.198 (5/5), an exploratory time-window diagnostic. |
| [One-step counterfactual Q](docs/counterfactual-action-value-findings.md) | 320 fixed decisions and 8,143 action/stream continuations. The exact 600-second next-use selector has lower mean realized regret than the learned selector in five of eight source-policy strata, higher in three. Its minimum-label tie averages 9.8875 actions; the hindsight best tied action has lower realized regret on these states. | Each branch forces one action, then resumes the source policy. The maximum over actions is chosen after observing a continuation. It is not expected-Q superiority or recoverable trace-wide gain. |
| [Fresh-stream cross-fit](docs/counterfactual-randomness-findings.md) | On 40 fixed states and 16 new streams, unrestricted training/held-out gain is +15,003.0 / −267.8 tokens; exact-tie gain is +13,567.2 / +90.3. Fixed exact next use gains +2,350.1 tokens on the same paired streams. | Eight streams select and the other eight evaluate, then folds swap. Pooled state means are descriptive, the future-informed exact comparator is not deployable, and no trace-wide gain follows. |
| [Sampling-mechanism control](docs/mechanism-control-findings.md) | With the L2 score held fixed, the exact training label of the frozen `next_use` ranker reaches 76.5–98.2% of the heap offline reference through the published sampled mechanism; the frozen ranker obtains 11.1–27.9% of that rung's gain over sampled LRU. The ranker-to-label gap is the largest term of the headroom (heap offline reference minus sampled LRU) in 12/12 real trace × cell under all four mechanisms. | 960 replays, five sampling seeds. The label rung reads the future; one linear ranker and one target. Leaf-only eligibility is a control, not a policy, and exceeds the greedy heap reference in three small cells. Does not bound other causal information. |
| [Error-location control](docs/error-location-findings.md) | Under the published sampled mechanism, giving only the choice of victim among residents to the exact label recovers 65–100% of the gap between the frozen `next_use` ranker and that label in 12/12 real trace × cell; giving it only the decision to reject the arrival recovers none. At L1 0.25% a third or more of the gap needs both. At an equal victim-agreement rate, swapping the victim to the runner-up costs at most 1.4 points while a uniform swap at rate 0.5 falls below sampled LRU in 12/12. No decision statistic orders the 18 arms by utility (ρ ≥ 0.9 in at most 2/12); the victim's mean label excess follows the sign of the `pi3 − pi0` utility change in 24/24, pairwise concordance moves against it in 4. The exact `binary` label equals the `next_use` label at 1%×4 and 2%×4 only. Every constructed arm reads the future; none is a policy. | |
| [Ranker-error diagnosis](docs/ranker-error-diagnosis-findings.md) | On the saved held-out decision logs of `pi0` and `pi3` (no replay), the victim's mean label excess on a policy's own decisions has the sign of label-window utility in 22/24 trace × cell × target, but with the candidate population held fixed the sign depends on the population in 5/12 (`next_use`) and 7/12 (`binary`). At least 99.3% of the frozen ranker's label excess is the eviction of a state requested within 600 s while a sampled candidate is not; in 69–87% of decisions it differs from the reference only by tie-break. On its own candidate sets the `next_use` ranker makes such evictions less often than recency in 9/12 cells and more often in 3. | |
| [Horizon and class-order control](docs/horizon-control-findings.md) | An exact one-bit reuse label with recency reaches the exact `next_use` label (within 10% of its gain over sampled LRU) at some horizon of {6, 15, 60, 300, 600} s in 8/12 real trace × cell; a follow-up fill-in, pre-registered after that result and counted separately, finds a horizon between 90 and 180 s in the remaining 4/4. The horizon that works grows with L2 capacity (60, 150, 300, 600 s) and the neighbouring grid horizons lose 6–85% of the gain. Given the exact 600 s reuse class on residents, the frozen ranker recovers 94–104% of what the exact label recovers on eviction in the six cells where that class is sufficient and 33–61% elsewhere; its order within a class beats recency in 9/12. Under leaf-only eligibility the gap is eviction-located in 12/12. | 540 + 160 replays, five seeds. Every horizon is the best of a grid read after the run; nothing predicts it before the fact. A one-bit label is above the `next_use` label in six cells. Every arm reads the future; none is a policy. |
| [Matched-horizon follow-ups](docs/matched-horizon-findings.md) | On the frozen ranker's own logs, its label excess at the capacity-matched horizon `h*` (60/150/300/600 s) is avoidable reusable eviction in 24/24; against recency on the same candidate sets the `next_use` ranker makes that eviction 0.64× as often at 0.25%×1 (60 s) and 2.1–2.3× as often at the four 1% cells (150 s). In replay, the exact reuse class at `h*` on residents, with the ranker's admission and order kept, recovers ≥ 90% of the exact label's eviction gain in 6/8 new cells (prediction 8/8 failed at 0.25%×1: 73% and 81%) and reproduces 4/4 at 600 s; recency within the matched class recovers it in 8/8, and the ranker's order within the class is a consistent loss against recency in 8/12. The label's admission beats the ranker's by up to 4.1 points at `h*`. | 120 replays plus a re-read of 120 saved populations; `h*` is the best of grids run on the same traces, read after the fact. Nothing is fitted; every arm reads the future. |
| [Matched-horizon checks](docs/matched-horizon-checks-findings.md) | Removing the last 600 s preserves the bit-plus-recency comparison in 12/12 and the sign of the learned-versus-recency order difference in 12/12; class recovery changes from 10/12 to 9/12 (conversation 0.25%×4, 0.903 → 0.859). Under `all16`, ordering only states reused within `h*` gives six all-seed gains, two digest-identical matches and four mixed-sign cells (one mean −0.002157 points); ordering only states not reused within `h*` loses consistently in 9/12, including all eight original consistent-loss cells. A random within-class order loses to recency in 12/12. Under `leaf16`, the class recovery remains ≥ 0.9 in 10/12, bit plus recency reaches the local label comparison in 9/12, and the learned within-class order wins consistently in 11/12. | 840 completed replays, five seeds; four of nine predictions hold, five fail, two in the opposite direction. `h*` is selected after inspecting `all16` grids and carried over unchanged. Class recovery is relative to `evict_label`, which keeps learned admission, not all headroom. The larger relative shortfall under `leaf16` is not an absolute utility decline. Causes of the order reversal are untested; every constructed arm reads the future. |
| [Bailian input audit](docs/bailian-input-audit.md) | The four Qwen-Bailian traces match the upstream checkout and their 512-token conversion reproduces byte for byte (138/138 checks). Children almost never carry the parent's last 16-token block (93–94%), so at 512 tokens a child reuses only up to the previous 512-token boundary; partial final 512-token blocks hold 10.3 / 25.8 / 4.9 / 4.3% of input tokens (To-C, To-B, thinking, coder) against 2.2–2.6% on Mooncake, and the coarsening removes 7.8 / 19.0 / 2.8 / 3.2 points of repeatable input. | No cache policy was run and nothing was fitted; the audit reads no horizon-dependent reuse statistic. The traces are a second workload family of one provider, not four independent deployments. |
| [Working-set ratio check](docs/working-set-ratio-findings.md) | A trace-derived reuse working set over the L2 capacity (the ratio of EfficientAgent, arXiv 2609.33762, fixed before it was computed) places the sign change of 2-hit admission against LRU in 12/12 real trace × cell for the heap pair and 12/12 for the sampled pair; arrival protection agrees in 4 of its 6 published cells, with both misses mixed readings at ratios 1.005 and 1.179. The capacity reversal is an instance of a known working-set argument, not a separate contribution; the grid brackets the crossing between ratios 0.553 and 1.005. | |
| [Zero-reuse residence diagnostic](docs/counterfactual-residence-findings.md) | Across 640 fixed E/Z paired streams, 629 subsequent victim-sequence differences precede the first hit difference; in 491/497 pairs where both focal blocks leave before 600 seconds, per-request reward differences continue afterward. | The blocks are equally sized and have no own reuse within 600 seconds. Residence correlation and downstream event order do not establish causal mediation or an expected-action advantage. |

Here `2%×4` means L1 capacity is 2% of the packed unique-state working set
and L2 capacity is four times L1. “Input-token points” means percentage points
of evaluation-window requested input tokens saved through exact-prefix hits;
it is neither GPU time nor a fraction of cache requests.

The original counterfactual [80-state, three-stream sensitivity](docs/counterfactual-action-value-findings.md)
reports mean 600-second realized regret of 34,419 tokens for exact next use
and 35,107 for learned. A **post hoc, independently CSV-checked reanalysis** of these same 80
states and three streams puts the in-stream hindsight-best regret within
the exact-label tie at 4,114 tokens, versus held-out regret of 35,613 for a
tie-restricted action chosen on another stream and 37,360 for an action chosen
from all candidates on another stream; it also reports mean cross-stream
candidate-`Q` rank Spearman 0.071. The check verifies published-CSV arithmetic
and the minimum-action-index tie rule; see the [reanalysis audit](docs/counterfactual-randomness-reanalysis.md)
and its [machine-readable summary](results/paper/counterfactual_stream_reanalysis/summary.json).
Those figures caution against treating an
in-stream maximum as a stable action choice. Three streams cannot settle the expected-Q question or assign a percentage of regret to
randomness.

## Evidence trail

The historical sequence and its exact thresholds remain in the detailed
reports; this page states the current interpretation.

| Stage | Detailed record |
|---|---|
| Structural characterization and temporal prediction | [characterization](docs/characterization-findings.md), [temporal prediction](docs/temporal-prediction-findings.md), [research design](docs/research-design.md) |
| Oracle decomposition and target change | [predictability–retention gap](docs/predictability-retention-gap.md), [target change](docs/target-change-findings.md) |
| Two-tier population and decisions | [victim stream](docs/two-tier-victim-findings.md), [decision-population results](docs/decision-population-findings.md), [per-block attribution](docs/decision-population-findings.md) |
| Local arrival intervention | [Phase 1 findings](docs/phase1-intervention-findings.md): capacity-local gain for the weak B arm; general arrival protection gave similar gain, so ancestry-specific benefit was not established. |
| Policy-created populations and action values | [on-policy findings](docs/onpolicy-learning-findings.md), [post hoc diagnostics](docs/retention-decision-diagnostics.md), [counterfactual findings](docs/counterfactual-action-value-findings.md) |
| Fresh-stream measurement | [pre-registration](docs/counterfactual-randomness-plan.md), [implementation guide](docs/counterfactual-randomness-implementation.md), [findings](docs/counterfactual-randomness-findings.md), and [paper output](results/paper/counterfactual_randomness_001/README.md). The [old three-stream CSV audit](docs/counterfactual-randomness-reanalysis.md) is a separate post hoc calculation. |
| Residence mechanism diagnostic | [pre-registration](docs/counterfactual-residence-plan.md), [findings](docs/counterfactual-residence-findings.md), and [paper output](results/paper/counterfactual_residence_001/README.md). It instruments the same states/streams; full event logs are ignored. |
| Sampling-mechanism control | [pre-registration](docs/mechanism-control-plan.md), [findings](docs/mechanism-control-findings.md), and [paper output](results/paper/mechanism_control_001/README.md). A fixed four-rung score ladder under all-resident / leaf-only eligibility at widths 16 / 64; nothing is fitted. |
| Matched-horizon checks | [evaluation-window pre-registration](docs/tail-window-check-plan.md), [class-order mix pre-registration](docs/class-order-mix-plan.md), [leaf-eligibility pre-registration](docs/leaf-matched-horizon-plan.md) with its pre-run addendum, [findings](docs/matched-horizon-checks-findings.md), and paper output for the [evaluation window](results/paper/tail_window_check_001/README.md), the [class-order mix](results/paper/class_order_mix_001/README.md) and the [leaf-eligibility check](results/paper/leaf_matched_horizon_001/README.md). The seven arms counted on three windows, three within-class arms, and the matched-horizon arms under `leaf16`; 840 replays; nothing is fitted. |
| External-workload check (Bailian) | [input audit](docs/bailian-input-audit.md) with its [paper output](results/paper/bailian_input_audit_001/README.md) (`scripts/audit_bailian_inputs.py`), and the [pre-registration draft](docs/bailian-external-check-plan.md) under review. No replay, no implementation, nothing fitted. |
| Working-set ratio check | [pre-registration](docs/working-set-ratio-plan.md) with its pre-computation addendum, [findings](docs/working-set-ratio-findings.md), and [paper output](results/paper/working_set_ratio_001/README.md). Arithmetic on the L1 offer stream and on published sign outcomes; no L2 policy is run and nothing is fitted. |
| Error-location control | [pre-registration](docs/error-location-plan.md) with its pre-smoke addendum, [findings](docs/error-location-findings.md), and [paper output](results/paper/error_location_001/README.md). Eighteen arms under the published mechanism (hybrids by decision type, Gaussian label noise, victim swaps, the `pi3` and `binary` rankers); 1,080 replays; nothing is fitted. |
| Ranker-error diagnosis | [pre-registration](docs/ranker-error-diagnosis-plan.md) with its pre-computation addendum, [findings](docs/ranker-error-diagnosis-findings.md), and [paper output](results/paper/ranker_error_diagnosis_001/README.md). A read-only re-scoring of 240 saved decision populations; no replay, nothing fitted. |
| Horizon and class-order control | [pre-registration](docs/horizon-control-plan.md) with its pre-smoke addendum, the [fill-in pre-registration](docs/horizon-fill-plan.md) written after the first result, [findings](docs/horizon-control-findings.md), and paper output for the [grid](results/paper/horizon_control_001/README.md) and the [fill-in](results/paper/horizon_fill_001/README.md). Exact reuse labels at eight horizons, two class-order arms and the leaf16 hybrids; 700 replays; nothing is fitted. |
| Matched-horizon follow-ups | [diagnosis pre-registration](docs/matched-horizon-diagnosis-plan.md), [class-order pre-registration](docs/matched-class-order-plan.md) with its pre-run addendum, [findings](docs/matched-horizon-findings.md), and paper output for the [log diagnosis](results/paper/matched_horizon_diagnosis_001/README.md) and the [class-order control](results/paper/matched_class_order_001/README.md). The saved `pi0` logs re-read at 60/150/300/600 s, and the class-order arms at the capacity-matched horizon; 120 replays; nothing is fitted. |

The fresh-stream run completed all 40 lineages and 10,888 branches without
recorded failures. Its primary statistic evaluates an action selected on one
eight-stream fold against the original action on the other and swaps folds.
The near-zero pooled held-out means and mixed state signs weaken the
interpretation of in-stream hindsight maxima as selectable gains. They do not
estimate the best action for arbitrary future workloads.

## Model and measurement

The [data and code layout](data/README.md) and [script guide](scripts/README.md)
cover the original phases. `src/persistent_kv_admission/` contains the trace
loader, simulators, features, and rankers. `docs/` contains plans and findings;
`results/paper/` contains tracked tables, figures, and run configs. Large raw
traces, model files, saved decision-population NPZ files, and full branch dumps
are ignored artifacts. A fresh clone with only the paper CSVs cannot directly
run the later counterfactual runner.

**Input.** The downloader pins the [Mooncake FAST'25 trace release](https://github.com/kvcache-ai/Mooncake/tree/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces)
to commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`; the
[paper](https://www.usenix.org/system/files/fast25-qin.pdf), Appendix A,
describes the schema. Each JSONL record has `timestamp`, `input_length`,
`output_length`, and ordered `hash_ids`. Timestamp is relative arrival time
in milliseconds. Each hash is the cumulative prefix through a 512-token
block, so equal hashes at the same depth mean exact-prefix reuse.
`len(hash_ids) = ceil(input_length / 512)`; the final block can be partial.
A repeated hash must retain its depth, parent, prefix length, and block
length or loading fails. Equal-timestamp requests observe the same pre-batch
cache and cannot hit one another. There is no session ID; fan-out and branch
diversity are request-level signals.

**Units and split.** A state's storage proxy is incremental block tokens ×
`bytes_per_token` (default 2048); budgets are fixed fractions of packed
unique-state bytes. Hit reward is avoided input prefill tokens for the
contiguous reused prefix. The first 60% of the trace trains or warms caches;
the last 40% of requests is the held-out replay window. Prediction labels look strictly
after a snapshot, within their specified horizon, with a full-horizon
training embargo. Macro AUC, threshold-integrated average precision, and
tie-aware expected precision@100 describe the original prediction protocol.
The real traces span about 59 minutes, so horizons beyond 600 seconds were
not used for that protocol.

**Single-tier replay.** Retained states form a prefix-closed forest. A child
is usable only with its ancestors, and eviction removes leaves. Scored
online arms, including their sampled baselines, compare 16 randomly sampled
retained leaves at a decision; the exact heap paths provide deterministic
references and the greedy future-aware comparator. The exact mechanics and
phase-specific arms are in the [script guide](scripts/README.md) and
[gap findings](docs/predictability-retention-gap.md).

**Two-tier replay.** L1 uses exact heap LRU/LFU and admits every arriving
block. Its leaf-eviction victim stream is consequently independent of L2.
The primary L2 is an *exclusive union store*: it receives L1 victims and is
not required to be prefix-closed by itself. A block in L2 counts as a hit only
when its entire prefix chain is available across `L1 ∪ L2`; on request it is
materialised in L1 and leaves L2. The independent-block control changes hit
accounting; the standalone prefix-closed L2 is an inclusive sensitivity
model, with ancestor copying and leaf eviction. These models are not
interchangeable. In Phase 0.97 and later learned-policy replays, an overflowing
L2 decision compares the arriving victim with 16 sampled L2 residents and
removes the lowest-scored member under the arm's key. Sampling can leave
present-but-unusable descendants. The zero missing-ancestor cost reported
for heap LRU/LFU/2-hit in Phase 0.95 does not apply to every sampled arm.

**Comparators.** In single-tier phases,
`HeadroomClosure = (policy − LRU) / (greedy offline next use − LRU)` on avoided
tokens, where LRU is the same seed's *sampled* LRU, not the heap reference.
In the Phase 0.97 two-tier comparison the floor is likewise *sampled L2-LRU*
and the denominator uses the *heap offline L2* under the same L1 stream.
Phase-specific closure values must not be mixed as an optimality ratio. The
offline next-use arms have future information but use greedy eviction and are
not proved globally optimal. The counterfactual `Q` study instead compares
legal actions in a fixed already-drawn set, with one forced victim followed by
a frozen source policy; `Q600(S,a,ω)` is avoided prefill in `(t,t+600 s]` for
that continuation stream. Its hindsight maximum and exact 600-second
next-use label have different estimands.

## Reproduction and provenance

For early phases, use [scripts/README.md](scripts/README.md) and the pinned
trace downloader. For later phases, follow their dedicated guides:
[on-policy](docs/onpolicy-implementation.md),
[one-step counterfactual](docs/counterfactual-implementation.md), and
[fresh-stream cross-fit](docs/counterfactual-randomness-implementation.md).
The later runners require the ignored traces, model JSON, saved decision NPZ,
and published references with matching hashes. Rebuild prerequisites as
described in those guides, use fresh run/paper directories, and inspect each
frozen manifest before execution; the old run directories are immutable
audits. The 235-test result in the fresh-stream implementation guide is a
recorded checkpoint at commit `3362c7b`, not a test run performed for this
README review. Its guide gives the exact unittest command and prerequisite
environment variables.

## Scope

This is a fixed-trace simulator of exact-prefix reuse, with a byte proxy and
the last 40% of requests as its held-out replay window. Request traces
and held-out windows informed earlier research choices; five lineage seeds
share the same future requests. Counterfactual states are sampled from
policy-created populations, and `pi0` and `pi3` states are not paired across
policies. A finite-stream, per-state result does not imply workload-general
significance or trace-wide recoverable loss. SSD/GDS/PCIe bandwidth, restore
costs, storage-engine design, semantic embedding prediction (Research 2),
approximate KV reuse, answer quality, and vLLM/LMCache integration are outside
this experiment.

## Citation

If you use or reference this research, design, or implementation, cite:

```bibtex
@software{chipmunk2026persistent_kv,
  author = {CHIPMUNK},
  title = {Persistent KV Admission: Retention under Finite Capacity for Persistent LLM KV Caches},
  year = {2026},
  url = {https://github.com/CHIPMUNK-T0T/persistent-kv-admission}
}
```
