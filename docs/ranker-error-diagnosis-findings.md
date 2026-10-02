# Ranker-error diagnosis: the excess is avoidable reusable eviction, and the statistic follows the policy's own population

This [pre-registered diagnosis](ranker-error-diagnosis-plan.md) re-reads the saved held-out decision logs of the published on-policy run; it runs no replay and fits nothing. Three things come out. The mean label excess of the victim (`m3`), measured on each policy's own logged decisions, still has the sign of the `pi3 − pi0` utility change on the label window in 22/24 trace × cell × target and never moves against a consistent change. That correspondence is not one of selection on fixed candidates: with the candidate population held fixed, `pi3`'s victims are better or worse than `pi0`'s in the direction of utility on both populations in 7/12 trace × cell for `next_use` and 4/12 for `binary`. And of the frozen ranker's label excess, at least 99.3% comes from evicting a state that is requested within 600 seconds while a sampled candidate is not; decisions whose candidates are all reusable and differ in label make up at most 0.3% of its decisions.

## Run and integrity

The plan was committed alone (`094bd9c`). Its addendum (`9bb162f`), written before any saved log was read, states that the four error classes overlap when a victim's next use is at exactly 600 seconds, takes them in the listed order, and records the readings the implementation fixed. The implementation and tests were committed next (`0511103`), and the diagnosis ran once from a clean tree at that commit into [`results/paper/ranker_error_diagnosis_001/`](../results/paper/ranker_error_diagnosis_001/README.md) (8 workers, 44 seconds).

All required checks passed and are recorded in the [run config](../results/paper/ranker_error_diagnosis_001/run_config.json):

- 240 populations (2 traces × 6 cells × 2 targets × 5 seeds × {`pi0`, `pi3`}), 124 models and 4 published tables have the published SHA-256.
- Each scorer applied to its own population reproduces the logged victim in every one of the 9,600,000 decisions (0 mismatches; the recomputed score equals the logged score exactly in all 240 populations), so the stop rule of 0.1% was not approached and the logged and recomputed own-policy statistics coincide.
- 120 utility pairs are matched on trace, cell, target, seed and iteration with none missing.

Each population is a reservoir of 40,000 whole decisions, 64.0–89.6% of the eligible held-out decisions of its replay (timestamp at or after the split and at least 600 seconds before the trace end). Every number below is a column of the published CSVs. To rerun, check out `0511103` with `results/onpolicy_full_feb30eb_001/` in place and run

```bash
.venv/bin/python scripts/run_ranker_error_diagnosis.py --output-dir <new directory> --workers 8
```

`m3` is the mean of `label(victim) − min label` over the decisions of a population, in label units; a rise is worse. "Consistent" means the same sign in all five seed-paired values.

## Reading 1 — selection on a fixed population

`Δsel(P) = m3(P, pi3) − m3(P, pi0)` on population `P`; `Δown = m3(pi3, pi3) − m3(pi0, pi0)`; `ΔU` is the published full-window utility change.

| target | carried by selection | population-dependent | not carried | `−Δsel` has the sign of `ΔU` on the `pi0` population | on the `pi3` population | `−Δown` has it |
|---|---:|---:|---:|---:|---:|---:|
| `next_use` | 7 | 5 | 0 | 10/12 | 9/12 | 11/12 |
| `binary` | 4 | 7 | 1 | 11/12 | 4/12 | 11/12 |

On `next_use`, the seven cells carried by selection are the three smallest cells on both traces (0.25%×1, 0.25%×4, 1%×1), where utility falls and `pi3` chooses worse victims than `pi0` on either population in every seed, and conversation 1%×4, where utility rises and it chooses better ones. In the five population-dependent cells — 2%×1 and 2%×4 on both traces and tool-agent 1%×4, all with a positive mean utility change — `pi3` is better than `pi0` on one population and worse on the other.

On `binary`, the four cells carried by selection are the two L1 0.25% cells on both traces. In six of the seven population-dependent cells `pi3` chooses worse victims than `pi0` on `pi0`'s population and better ones on its own (conversation and tool-agent 1%×1, 1%×4 and 2%×1), while utility falls. At conversation 2%×4, where the utility change is mixed, neither population agrees.

The same table for `m4` reads 7 / 5 / 0 on `next_use` and 4 / 6 / 2 on `binary`.

## Reading 2 — the label window

`ΔU_label` is the published change of label-window extra avoided tokens, in points of label-window input tokens.

| target | `−Δown` has the sign of `ΔU_label` | of `ΔU` | cells where `ΔU_label` is consistent and `−Δown` has the other sign |
|---|---:|---:|---|
| `next_use` | 10/12 | 11/12 | none |
| `binary` | 12/12 | 11/12 | none |

The two `next_use` disagreements with `ΔU_label` are conversation 2%×4 (`ΔU_label` +0.14, seeds `+++−−`; `m3` rises by 0.004 in every seed) and tool-agent 1%×4 (`ΔU_label` −0.08, seeds `+−−−−`; `m3` falls by 0.005). Both have a mixed `ΔU_label`. Conversation 2%×4 is also the one `next_use` disagreement with the full-window `ΔU` (+0.54, consistent): on the label-window sample the statistic worsens while full-window utility improves. The 24/24 of the error-location control was measured on every decision of the full window; on the label-window logs the count is 22/24 against label-window utility and 22/24 against full-window utility.

## Reading 3 — kinds of error of the frozen ranker

`pi0` populations, by the logged victim. Shares of decisions; the last column is the share of the population's label excess.

| class | `next_use`: share of decisions | share of excess | `binary`: share of decisions | share of excess |
|---|---:|---:|---:|---:|
| no excess, reference victim | 0.7–7.0% | 0 | 0.8–14.4% | 0 |
| no excess, other tie-break | 68.6–86.6% | 0 | 68.6–77.0% | 0 |
| avoidable reusable eviction | 8.3–30.3% | 99.3–100% | 8.6–30.5% | 99.7–100% |
| order error among reusable candidates | 0.0–0.2% | 0.0–0.7% | 0.0–0.1% | 0.0–0.3% |

In every trace × cell the ranker's victim has the minimum label in 70–92% of decisions, and in almost all of those it is a different candidate from the reference's, which breaks equal labels by recency. The excess is carried by one class. Decisions whose victim's next use is at exactly 600 seconds and that therefore meet two definitions number 462 (`next_use`) and 342 (`binary`) over all populations; they have no excess.

By victim, on `next_use`: the victim is the arrival (a rejection) in 70.0–82.5% of decisions at L1 0.25%, 54.9–60.6% at 1%×1, 36.3–40.9% at 2%×1 and at most 2.1% at 1%×4 and 2%×4. Among decisions that evict a resident, 8.3–45.0% are avoidable reusable evictions; among rejections, 0–27.2%.

## Reading 4 — conditional rates against recency and a uniform victim

The avoidable-eviction rate over decisions that contain both a reusable and a non-reusable candidate (39,380–39,997 of the 40,000 decisions of a population), on the same candidate sets.

| cell | conv `next_use` ranker | recency | uniform | reading | tool `next_use` ranker | recency | uniform | reading |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 0.25%×1 | 0.304 | 0.247 | 0.649 | worse | 0.302 | 0.329 | 0.674 | better |
| 0.25%×4 | 0.277 | 0.258 | 0.567 | worse | 0.277 | 0.305 | 0.582 | better |
| 1%×1 | 0.269 | 0.253 | 0.542 | worse | 0.268 | 0.295 | 0.558 | better |
| 1%×4 | 0.183 | 0.222 | 0.344 | better | 0.177 | 0.264 | 0.353 | better |
| 2%×1 | 0.215 | 0.251 | 0.414 | better | 0.211 | 0.276 | 0.419 | better |
| 2%×4 | 0.099 | 0.117 | 0.226 | better | 0.084 | 0.099 | 0.225 | better |

| target | better than recency | worse | mixed | better than uniform |
|---|---:|---:|---:|---:|
| `next_use` | 9/12 | 3/12 | 0 | 12/12 |
| `binary` | 12/12 | 0 | 0 | 12/12 |

On its own candidate sets the frozen `next_use` ranker makes an avoidable reusable eviction 0.67–0.92 times as often as recency would in nine cells and 1.07–1.23 times as often in the three smallest conversation cells; the `binary` ranker 0.56–0.98 times as often. Both are well below a uniformly drawn victim (0.22–0.67).

The order-error rate is not readable: decisions whose candidates are all reusable and not all of one label number 0–122 of 40,000 per population (`next_use`), and none at 1%×4, 2%×1 and 2%×4 on either trace. The registered counts are 2 better, 0 worse and 10 mixed for both targets.

## Reading 5 — concentration, descriptive

Over the avoidable reusable evictions of each `pi0` `next_use` population (3,304–12,130 per population on 3,163–8,646 distinct states): 8.3–50.6% fall on states that are such a victim more than once in the sample, and the 10% of states with the most hold 13.8–22.8%. For resident victims the two shares are 8.3–41.8% and 13.8–21.1%; for arrival victims 0–35.2% and 10.1–19.4%. The repeat share falls with capacity (49–51% at 0.25%×1, 8% at 2%×4). Sampling makes repeat counts lower bounds; no run of consecutive decisions is estimated.

## Observations outside the five readings

- **The own-policy change is small beside the selection difference.** On `next_use` the absolute `Δsel` is 0.004–0.172 on the `pi0` population and 0.002–0.072 on the `pi3` population, against an absolute `Δown` of 0.002–0.022. At conversation 0.25%×1, `pi3`'s victims carry 0.165 more excess than `pi0`'s on `pi0`'s population and 0.058 more on its own, while each policy on its own population differs by 0.017. The population a policy produces offsets much of the difference in how it chooses.
- **Counts of errors by victim do not locate recoverable utility.** At conversation 0.25%×1, 8,987 of the 12,130 avoidable reusable evictions in a population are rejections of a reusable arrival. The error-location control gave the rejection decision to the label there and recovered nothing, and gave resident eviction to the label and recovered 67%.
- **Candidate sets of reusable states are a property of the store.** Sets whose candidates are all reusable and differ in label are almost absent from the frozen ranker's populations, whose store holds many states that are not requested again. A store kept by the exact label would present different candidate sets; these logs do not contain them.

## What this establishes and what it does not

Established, on the held-out label-window decision logs of the published `pi0` and `pi3` replays:

- The victim's mean label excess on a policy's own decisions has the sign of label-window utility in 22/24 trace × cell × target and never opposes a consistent utility change.
- That correspondence is not reproduced by scoring a fixed population with both models: the sign depends on the population in 5/12 (`next_use`) and 7/12 (`binary`) trace × cell.
- The frozen ranker's label excess is, to within 0.7%, the excess of evicting a reusable state while a non-reusable candidate was sampled; in 69–87% of its decisions it differs from the reference only in which candidate of the minimum label it takes.
- On its own candidate sets the frozen `next_use` ranker avoids such evictions somewhat more often than recency in nine cells and less often in three; both rankers do so far more often than a uniform victim.
- A share of those evictions, 8–51%, falls on states evicted that way more than once.

Not established:

- **A causal split between selection and population.** Reading 1 is four means per cell; no share of a utility change is attributed to either.
- **Utility of a scorer on another policy's population.** A scorer applied to logged candidates does not produce a trajectory of its own.
- **That order among reusable states is unimportant.** It is rare on the frozen ranker's own decisions. Whether it matters on a store kept by a better score, and at which horizon, is the question of the [horizon and class-order control](horizon-control-plan.md).
- **Persistence of errors in time.** Repeat counts are lower bounds from a reservoir; run lengths are not measured.
- **Lost utility per error.** Shares are of decisions and of label excess. An avoidable eviction of one state and of another differ in what keeping the state would have saved.
- **Why the ranker fails to tell reusable from non-reusable candidates,** whether from the features or from the fit.
- **Generality.** Two traces from one deployment family, one ranker family, the published mechanism.
