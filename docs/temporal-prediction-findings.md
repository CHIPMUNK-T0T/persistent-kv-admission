# Temporal-History Selection: Cross-Workload and Online-Adaptation Findings

Phase 0.5 asked whether persistent KV retention can be driven by a value function
learned from observed reuse history, and whether that function generalises across
heterogeneous reuse regimes **without workload-specific tuning**.

All numbers below come from one run of `scripts/run_cross_workload.py` over the
pinned Mooncake FAST'25 conversation, tool-agent, and synthetic traces. **A single
hyperparameter set is used for every workload** (`results/paper/run_config.json`);
no per-trace tuning of learning rate, regularisation, horizon, sample width, or
feature set was performed at any point. The review artifacts — the five required
figures `fig1`–`fig5`, the ablation, transfer, replay, and coefficient CSVs, and
the run config — are in `results/paper/`. The full dump, including the Spearman
feature-correlation matrix, is written to `results/crossworkload/` and is not
tracked.

## 1. Setup

23 causal features in six groups (`core`, `rate`, `interarrival`, `window`,
`dynamics`, `structural`). Base-0 = `core` (3 features: log frequency, negative
log recency, log prefix tokens). Base-1 = all non-structural (21). Base-2 = all (23).
Features are standardised **within each decision point**, not globally, so that
trace-position drift cannot leak across the train/test split and so that a model
fitted on one workload is applied on a comparable scale on another.

The ranker is L2-regularised logistic regression fitted with Newton steps on
case-control-sampled rows. It is deliberately weak; no novelty is claimed for the
model. Train is the first 60% of snapshots with a full horizon embargo, test is
the remainder. Real traces support horizons up to 600 s, synthetic up to 300 s.

Fixed-budget replay uses the prefix-closed cache with **sampled eviction**
(16 random retained leaves re-scored at the moment of the decision). Every arm —
including LRU and LFU — runs through the same scorer interface and the same
eviction mechanism, so policy differences are not confounded with eviction
mechanics. Unit tests assert that the recency scorer reproduces LRU exactly and
the frequency scorer reproduces LFU exactly under this path.

`HeadroomClosure = (Policy − LRU) / (OfflineNextUse − LRU)` in avoided prefill
tokens over the evaluation window. Offline-next-use is an **approximate headroom
comparator, not an oracle or an upper bound**: offline caching with variable item
sizes and prefix dependency is NP-hard, and this comparator additionally uses
greedy leaf eviction.

## 2. A simple temporal ranking structure transfers between the two real workloads

*(Wording revised after Phase 0.75. The fitted coefficient vectors at 300–600 s
are dominated by log frequency and negative log recency with near-equal weight,
so what transfers between conversation and tool-agent is close to an LFU/LRU
blend, not a rich value function. The standardisation-robustness check is in
`docs/predictability-retention-gap.md`.)*

Cross-workload transfer AUC at the 300 s horizon (`fig5`, rows = fitted on,
columns = evaluated on):

| fitted on \ evaluated on | conversation | synthetic | tool-agent |
|---|---:|---:|---:|
| conversation | **0.881** | 0.837 | **0.940** |
| synthetic | 0.708 | 0.673 | 0.749 |
| tool-agent | 0.879 | **0.841** | 0.939 |

Two facts stand out.

**The diagonal is not the best cell in any column.** On conversation, the
tool-agent model loses 0.002 AUC against the self-fitted model. On tool-agent,
the conversation model *wins* by 0.001. On synthetic, the self-fitted model is the
worst of the three by a wide margin (0.673 vs 0.837/0.841).

**The two real workloads learn nearly the same function.** Cosine similarity of
the standardised coefficient vectors is **+0.968** between conversation and
tool-agent. Against synthetic it is +0.05 and +0.18 — essentially orthogonal.

The same holds in replay: `fixed_cross` is within noise of `fixed_self` at almost
every trace and budget, and beats it on synthetic at every budget
(e.g. 1% budget: `fixed_cross_toolagent` +0.040 vs `fixed_self` −0.037).

So on the axis the pivot asked about: **nothing in these traces required a
workload-specific model**, and the premise that temporal-feature work leads to
per-workload tuning is not supported. The claim stops there. Two real traces
from one deployment family, sharing a frequency-plus-recency structure, do not
establish that a general value function transfers across workloads.

Caveat: synthetic is fitted at 300 s and the real traces at 600 s, so the
synthetic column is not a perfectly matched comparison. The conversation ↔
tool-agent comparison is matched.

## 3. Only one feature group carries incremental value

ΔAUC over Base-0 from adding one group (`fig4`); win rate is the fraction of test
snapshots where the variant beats Base-0.

| group | conversation 600 s | tool-agent 600 s | synthetic 300 s |
|---|---:|---:|---:|
| `window` (recent-window frequency) | **+0.0179** (win 1.00) | **+0.0192** (win 1.00) | −0.0016 (win 0.12) |
| `structural` (fan-out, branch diversity) | +0.0058 (win 1.00) | +0.0044 (win 1.00) | +0.0153 (win 1.00) |
| `rate` (age, reuse rate) | +0.0040 (win 0.79) | +0.0037 (win 0.79) | +0.0013 (win 0.79) |
| `dynamics` (trend, burstiness) | −0.0018 (win 0.00) | −0.0042 (win 0.00) | −0.0090 (win 0.08) |
| `interarrival` (mean/median/std/CV/lags) | −0.0048 (win 0.00) | −0.0078 (win 0.00) | −0.0703 (win 0.00) |

`window` accounts for essentially the entire Base-1 and Base-2 gain in the real
traces (Base-1 is +0.0177/+0.0165; `base0_plus_window` alone is +0.0179/+0.0192).
The ten `interarrival` features are actively harmful at every horizon in both real
traces, with a win rate of 0.00. `structural` reproduces the Phase-0 conclusion:
real but small, and not the main story.

On synthetic the ordering is different — Base-1 is **−0.138 AUC below Base-0** at
300 s — which is itself the clearest single piece of evidence that the synthetic
generator produces a reuse regime unlike either real trace.

## 4. Prediction quality does not convert into retention benefit

This is the central negative result. Best causal arm at each budget, by
HeadroomClosure (`fig3`):

| trace | 0.25% | 1% | 5% |
|---|---|---|---|
| conversation | lfu **0.187** | lfu **0.215** | fixed_cross_tool **0.086** |
| tool-agent | lfu **0.248** | lfu **0.212** | fixed_self **0.130** |
| synthetic | lfu **0.120** | lfu **0.080** | lru **0.000** |

**The maximum closure achieved by any causal policy at any trace and budget is
0.248.** At least 75% of the LRU-to-offline gap is unrecovered everywhere, and
that gap is large: offline-next-use beats LRU by 18% to 770% in avoided tokens
depending on trace and budget.

Meanwhile the same features rank future reuse well: AUC 0.86–0.94 and precision@100
of 0.66–0.74 at the 300–600 s horizons in the real traces. **On tool-agent, AUC 0.94
coexists with a best-over-all-budgets HeadroomClosure of 0.25.**

The mechanism proposed here was a population mismatch, not a modelling failure:
the prediction task ranks ~40,000 observed states of which 3–5% are positive, so
most of its measured skill is in separating dead one-time states from live ones,
whereas the eviction decision ranks only the **retained live leaves under a byte
budget**. At the time of writing this was a hypothesis. It is measured directly
in `docs/predictability-retention-gap.md` (candidate-set prediction and the
population ladder), which also separates it from the objective and
candidate-search explanations.

## 5. No policy wins across budgets, and the online learner never wins

The policy ranking **inverts with cache budget in every trace**. LFU leads at
0.25% on all three traces (+0.19 / +0.25 / +0.12) and is worst or near-worst at 5%
(−0.21 / −0.28 / −0.10). LRU wins outright at 5% on synthetic. Longest-first is
catastrophic on tool-agent (closure −5.01 at 0.25%, −90.9% avoided tokens).

The online adaptive scorer is positive at small budgets (+0.053 / +0.135 / +0.058)
and roughly neutral-to-negative at 5% (−0.028 / +0.011 / −0.057). **It is not the
best arm at any of the nine trace-budget pairs**, and is beaten by parameterless
LFU at five of them. Online learning therefore removes the need for per-workload
tuning, but it does not currently buy an advantage over the parameterless
baselines it replaces.

Two caveats on this arm. First, the budget-inversion means the best baseline is
budget-dependent; an adaptive policy that merely matched the per-budget best
baseline would already be worth something operationally, and the online scorer
does approximately that at 5% on conversation and tool-agent while LFU collapses.
Second, the online scorer is trained against a 600 s reuse label on every trace,
including synthetic where 600 s is longer than the usable horizon.

## 6. Required statements

**何が確認できたか (what was confirmed)**

- A simple temporal ranking structure learned from reuse history **transfers
  between the two real workloads at no measurable cost**. Coefficient cosine
  between conversation and tool-agent is +0.968; cross-fitted AUC is within 0.002
  of self-fitted in both directions, and on synthetic the cross-fitted models
  beat the self-fitted one by ~0.17 AUC. The transferred structure is close to
  frequency + recency, so this is not evidence for a general value function.
- The whole study ran on **one hyperparameter set across three workloads** with no
  per-trace tuning, and still produced AUC 0.86–0.94 on the real traces.
- Among 23 features, **`window` (recent-window frequency at 10 s/60 s/300 s/600 s)
  is the only group with consistent incremental value**; it accounts for nearly all
  of the Base-1/Base-2 gain.
- Structural signal (fan-out, branch diversity) is real but small (+0.004 to
  +0.006 AUC), confirming the Phase-0 conclusion independently under a different
  model, different features, and a different eviction mechanism.
- The LRU-to-offline headroom is large at every trace and budget (18%–770%), so
  the Phase-0 provisional stop criterion remains untriggered.

**何が否定されたか (what was refuted)**

- **"Temporal-feature work forces per-workload tuning."** Not supported. Nothing
  here needed a workload-specific model, and the self-fitted model was never the
  clear winner.
- **"A model fitted on a workload is the right model for that workload."** Refuted
  on synthetic, where the self-fitted model is the worst of the three.
- **"Better future-reuse ranking yields better retention."** Refuted as stated.
  Ranking quality and closure are decoupled: tool-agent has the best AUC of the three
  traces (0.94) and still leaves at least 75% of its headroom unrecovered.
- **"A learned history-based score beats parameterless baselines under a byte
  budget."** Not supported as a general claim. Parameterless LFU is the best causal
  arm at 6 of the 9 trace-budget pairs and LRU at 1; the fitted models win only the
  two 5%-budget cases on the real traces, and the online arm wins none of the 9.
- **Longest-first as a general selector.** Independently re-confirmed as
  non-viable (closure −5.01 on tool-agent).

**何が未解決か (what is unresolved)**

- **Where the remaining ≥75% of headroom lives.** The candidates are (a) value per
  byte rather than value, (b) prefix dependency — the cost of evicting a leaf whose
  subtree is valuable, and (c) the comparator's knowledge of which states are never
  reused again, which no causal signal can have at admission time. These are
  separable by measurement, and (c) in particular is cheap to bound.
- **Persistent timescales.** The real traces span ~59 minutes. Horizons beyond
  600 s are unavailable, so whether history-based selection improves or degrades
  at the hours-to-days scale that motivates *persistent* KV cache is untested.
- **Workload breadth.** "Cross-workload" here rests on two real traces from one
  deployment family plus one synthetic generator. That is enough to refute a
  necessity claim about per-workload tuning; it is not enough to establish general
  transfer.
- **Session structure.** The traces have no session IDs, so fan-out and branch
  diversity remain request-level terms and no cross-session claim is made.
- Whether an adaptive policy that tracks the *budget-dependent* best baseline —
  rather than a fixed learned score — is where the online-adaptation value actually is.

**Research 1 を続けるべきか (should Research 1 continue)**

Yes, but the measurement target must move, and the research question must not.
The fixed question remains: *given a finite persistent KV cache, how should state
with high future reuse value be selected so that limited capacity is used best.*
What these results change is where to look for the answer. Selection driven by
single-state reuse history is now measured: it predicts well, it generalises well,
and it does not pay off under a byte budget. Continuing to search for better
temporal features against these traces is not justified — `window` is the only
group that carried value, and it is already in.

The next measurement is not a new policy. It is to decompose the unrecovered
headroom into the three candidates above, because that decomposition determines
whether the problem is a *signal* problem (which Research 2 would address) or a
*decision-formulation* problem (value per byte, prefix dependency) that stays
inside Research 1.

**Research 2 へ進む根拠が生じたか (is there grounds to move to Research 2)**

Partial, and not yet sufficient. The Phase-0.5 trigger for Research 2 was
"history signal alone is not enough". The replay evidence is consistent with that:
no history-based arm beats LRU-plus-LFU-class baselines robustly, and closure
caps at 0.248. But the transfer evidence rules out the specific explanation the
pivot anticipated — heterogeneous regimes requiring workload-specific adaptation —
and the population-mismatch mechanism in §4 offers an alternative explanation that
does not require a richer signal at all.

Moving to Research 2 now would be acting on an unidentified cause. The
decomposition in §6 is the cheaper discriminating experiment and should run first.
If it shows the residual headroom is concentrated in "which states are never
reused again", that is a signal problem and is the grounds for Research 2. If it
is concentrated in value-per-byte or prefix dependency, Research 1 continues with
a corrected decision formulation.

## 7. Method limits

Avoided tokens are exact-prefix trace estimates, not measured GPU time, FLOPs,
restoration cost, or physical KV bytes. The offline comparator has future knowledge
and greedy leaf eviction and is neither an oracle nor a proven bound. Sampled
eviction is an approximation of full re-ranking; sample width 16 was fixed across
all workloads and not tuned. Snapshot-state observations repeat states over time
and are weighted equally. The online scorer's 600 s label horizon exceeds the
usable horizon on synthetic. Results are from a single seed per configuration;
the ΔAUC columns carry per-snapshot dispersion and win rates, but the replay
numbers do not carry a seed distribution.
