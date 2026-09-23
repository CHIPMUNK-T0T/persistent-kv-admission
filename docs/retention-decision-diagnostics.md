# Retention decision diagnostics (exploratory)

This is a post hoc analysis of the completed on-policy run, specified after
seeing its published results. It does not change the pre-registered A–D
thresholds or verdicts, fit a policy, or replay a trace. The 600-second
label-observable window and the complete last-40% utility window measure
different time supports. The same-window ranking and request utility still
have different estimands: ranking scores sampled candidate decisions using
future labels, while utility counts request-level avoided tokens. A decision
near the label-window end can also use future reuse beyond that window.

## Fixed input and outputs

The input is the published `onpolicy_seed_utility.csv` (all 480 rows),
`onpolicy_terminal_ranking.csv`, and the SHA-256 identified saved test
populations. Decision analysis is fixed in advance to `pi0` and `pi3` on
both traces, all six cells, both targets, and five seeds: 240 NPZ files,
40,000 complete sampled decisions each. It loads one NPZ at a time and uses
the stored manifest hash before reading. The 40,000-decision reservoir is a
uniform sample of eligible decisions; summing its proxy gaps is a sample
summary, not a trace-wide token-loss estimate.

For each utility window separately, subtract the paired `pi0` avoided tokens
from `pi_i` and normalize by that window's requested tokens:
`100 * delta_avoided_tokens / requested_tokens`. Report raw tokens, points,
and all five seed signs for every iteration, target, trace, and cell. Only the
full-window `pi3` contrast has the original registered utility threshold.
Compare its sign with the published common-`D_test(pi3)` ranking delta only
descriptively; no formal subwindow pass label is assigned.

For each saved complete decision, verify exactly one actual victim and its
equality to the first lexicographic minimum of the recorded `(arm_score,
arm_tiebreak)` tuple. Classify a victim matching the arriving candidate as
arrival rejection and every other victim as resident eviction. We retain
decisions without an arriving candidate as resident evictions. Candidate
values are `target_column(next_use_delta_ms, count_within_h, target, H)`,
with larger values meaning more valuable to retain. The target-native
diagnostic reports actual-victim value minus the candidate minimum, the
minimum-value tie-aware correctness rate, the uniform-random expected rate
`number_of_minima / candidate_count`, and the same rates on decisions with
nonconstant labels. We report candidate-minimum-above-zero rates for reuse
count and the token proxy; the native binary minimum also has this meaning.
The native next-use value is nonpositive by definition, so its positive-minimum
rate is omitted. The exact horizon boundary follows the existing target convention:
next-use clipping ties it with no future reuse, while binary/count include a
reuse at exactly H. We also report the actual-victim within-H reuse count
minus the candidate minimum count.

The exposure proxy for candidate `j` is
`count_within_h[j] * state.block_tokens[j]`, where `block_tokens` is the
incremental block size of the state, not its cumulative prefix size. The
nonnegative candidate-relative gap is the actual victim's proxy value minus
the smallest candidate proxy value. State indices map to trace states in
insertion order, as in the original `state_indices(trace)`. This proxy can
double count tree/ancestor effects, readmissions, and repeated decisions for
the same state. It is neither avoided tokens nor causal regret. Report the
positive-gap rate, gap distribution, and top 1%/10% share using
`ceil(q * all_kept_decisions)` largest gaps, with all-zero totals undefined.
Per-group sums are sampled potential exposure, not realized loss.

On `D_test(pi3)`, additionally score the saved candidate sets with the
canonical `pi0` and `pi3` models using the existing exact sequential
float64 score function and stored tie-break. Compare the resulting argmin
choices and their target-native and proxy gaps on that identical population.
These are proposed victims on fixed sets, not replay outcomes. The full
population of L2 residents was not saved, so candidate extraction quality
cannot be measured from these logs. No new replay is part of this analysis.

Outputs are generated under `results/paper/retention_diagnostics/` by a new
postprocessing script, with source hashes, assumptions, CSVs, figures, and
integrity counts. Small boundary tests cover tie-aware minima, constant
labels, horizon boundary, partial blocks, zero-total concentration, and
actual tuple argmin. The original run outputs and protected documentation
remain untouched.

## Observations from the saved run

The postprocessor verified all 240 selected NPZ hashes and all 9.6 million
sampled decisions. Every decision had one actual victim equal to the first
recorded full-tuple argmin. Exact sequential rescoring with each canonical
`pi3` model selected the same victim on `D_test(pi3)`. All six unit boundary
tests passed. The figures, per-seed tables, and provenance config are in the
[artifact directory](../results/paper/retention_diagnostics/README.md).

For the primary `next_use` target, the same `pi3-pi0` utility comparison
changes with the request window. Values below are five-seed means in input
token points, followed by the number of positive paired seeds. The full
column remains the registered result; the label window is exploratory.

| Trace | Cell (L1% × L2/L1) | Full | Label-observable |
|---|---:|---:|---:|
| conversation | 0.25 × 1 | −0.803 (0/5) | −1.067 (0/5) |
| conversation | 0.25 × 4 | −0.603 (0/5) | −0.653 (0/5) |
| conversation | 1 × 1 | −0.681 (0/5) | −0.348 (0/5) |
| conversation | 1 × 4 | +0.253 (5/5) | +0.219 (5/5) |
| conversation | 2 × 1 | +0.578 (5/5) | +0.435 (5/5) |
| conversation | 2 × 4 | +0.538 (5/5) | +0.143 (3/5) |
| toolagent | 0.25 × 1 | −0.458 (0/5) | −0.578 (0/5) |
| toolagent | 0.25 × 4 | −0.611 (0/5) | −0.639 (0/5) |
| toolagent | 1 × 1 | −0.871 (0/5) | −0.807 (0/5) |
| toolagent | 1 × 4 | +0.121 (4/5) | −0.082 (1/5) |
| toolagent | 2 × 1 | +0.229 (5/5) | +0.067 (4/5) |
| toolagent | 2 × 4 | +0.483 (5/5) | +0.198 (5/5) |

The label-window narrowing changes the sign at toolagent 1% × 4 and weakens
the 2% × 4 gain on conversation. This is time-window sensitivity. It does
not assign a new formal pass/fail result, and matching the decision timestamp
range to the ranking population does not make ranking and utility the same
estimand.

Actual-victim proxy positive-gap rates on `next_use` depend strongly on the
capacity cell. On `pi3`'s own saved populations they are about 31% at
0.25% × 1, 18% at 1% × 4, and 8–10% at 2% × 4. The five-seed mean is close
to `pi0` on each policy's *different* population, so that comparison does not
isolate a decision rule. In small 0.25% × 1 cells, sampled arrival rejections
dominate (about 33–34 thousand of 40 thousand decisions); in 2% × 4 cells,
almost all decisions are resident evictions. These categories should not be
pooled without showing their denominators.

Scoring both models on the **same** `D_test(pi3)` shows a cell-dependent
choice effect. On 1% × 4, `pi3` improves discriminative minimum-label
correctness over `pi0` by about +0.012/+0.017 on conversation/toolagent; on
2% × 1, by +0.015/+0.023. At 2% × 4 it is lower by about −0.005/−0.004,
and its mean token-count proxy gap is higher by +8.6/+6.0 tokens per sampled
decision. Yet the complete-window utility rises in that cell on both traces.
Thus choosing the candidate with the lowest 600-second label or this local
proxy does not by itself account for the request-level gain. The fixed-set
comparison does not measure either model's counterfactual replay trajectory.

On a five-seed mean, the `pi3` candidate-minimum proxy exceeds zero in at
most about 0.11% of sampled decisions in any `next_use` cell. Nearly every
saved candidate set therefore contains a state with zero reuse within H. This
says only that the sampled set usually has a local zero-count alternative; it
cannot establish candidate extraction quality because unsampled residents
were not saved.
At 2% × 4, fewer than 10% of sampled decisions have a positive proxy gap,
so the top 10% share equals 1 mechanically. The proxy totals and top shares
remain within-sample exposure summaries. Tree closure, ancestor effects,
readmission, and repeated decisions prevent interpreting them as realized
token loss or causal regret.
