# Error-location control: the ranker's cost is in which resident it evicts

This [pre-registered control](error-location-plan.md) keeps the published sampled mechanism (`all16`) and places 18 arms between the frozen `next_use` ranker and its exact label. Replacing only the ranker's choice of victim among residents by the label's recovers 65–100% of the gap between them in **12/12** real trace × cell; replacing only its decision to reject the arrival recovers none. Utility falls monotonically as Gaussian error is added to the label, and the frozen ranker sits between error levels 1 and 2 in 10/12 cells. At an equal rate of wrong victims, swapping the victim to the runner-up costs at most 1.4 points while swapping it to a uniformly drawn candidate at a rate of 0.5 falls below sampled LRU in every cell. No decision statistic orders utility across the arms by the registered threshold; the mean label excess of the victim follows the sign of the `pi3 − pi0` utility change in 24/24 comparisons, and pairwise concordance moves against it in 4. The exact `binary` label equals the `next_use` label at the two largest L2 capacities and falls short of it by up to 82% of that label's gain over LRU at the smallest.

Every constructed arm reads the trace's future on purpose. None is a policy, nothing is fitted, and the results describe two traces from one deployment family.

## Run and integrity

The plan was committed alone (`19901b5`). Its addendum (`b404b65`), written before any real replay of the new arms, replaces one required check that could not hold — the label rung's `m4` is not identically zero, because a next use at exactly 600 seconds ties with no reuse under the clipped label while the `binary` convention counts it as reuse — and records the readings the implementation fixed. The implementation and tests were committed next (`bc66138`). A smoke on conversation 1%×4, seed 0 (41 replays, excluded from every table) held all 23 identity and hook-off comparisons: "admission by X, eviction by X" equals rung X decision by decision for X = learned and X = label, noise at `s = 0` and swaps at `p = 0` equal the label, and every arm gives identical counters with and without the statistics hook. The full grid then ran once from a clean tree at `bc66138`: 2 traces × 6 cells × 5 seeds × 18 arms = 1,080 replays, 14 workers, 3,917 seconds, peak worker RSS 1,278 MiB.

All required checks passed and are recorded in the [run config](../results/paper/error_location_001/run_config.json):

- The four reference rungs reproduce the 240 `all16` rows of the mechanism control and the three real rankers their 180 published on-policy rows exactly (`avoided_prefill_tokens` and the cell identifiers; 60 matched, 0 missing, 0 mismatched per arm). Models are loaded by their published SHA-256.
- In every replay the statistics saw as many decisions, rejections and evictions as the replay counted, with the final victim.
- For `label`: `m1 = 1`, `m2 = 1`, `m3 = 0` in all 60 replays, and every `m4` victim has its next use at exactly the horizon (0–81 such victims per trace × cell over five seeds, among 360,955–506,197 window decisions).
- `l1_avoided_tokens`, `requested_tokens` and the compulsory-absent charge are identical across the 18 arms of each of the 60 trace × cell × seed groups; the per-block absent charge has 0 unexplained tokens.

Every number below is a column of the [published CSVs](../results/paper/error_location_001/README.md) or simple arithmetic on them, printed by

```bash
.venv/bin/python scripts/tabulate_error_location.py results/paper/error_location_001
```

The Spearman correlations of reading 4 and the terms of reading 1 were also recomputed from the per-seed table independently of the runner and agree exactly. To rerun the grid, check out `bc66138` with the ignored traces and the published models of `results/onpolicy_full_feb30eb_001/` in place and run `scripts/run_error_location.py` into a fresh output directory.

Units: `U` is extra avoided prefill tokens over L1 alone, in points of window input tokens. Means are over five sampling seeds; "consistent" means the same sign in all five seed-paired values. Cells are written L1 fraction × L2 multiplier.

## Reading 1 — location by decision type

`G = U(label) − U(learned)`, `A = U(adm_label) − U(learned)` (admission by the label, eviction by the ranker), `E = U(evict_label) − U(learned)` (admission by the ranker, eviction by the label).

| cell | conversation `G` | `A` | `E` | `E/G` | `(A+E−G)/G` | tool-agent `G` | `A` | `E` | `E/G` | `(A+E−G)/G` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25%×1 | 4.92 | −0.31 | 3.30 | 0.670 | −0.39 | 3.37 | −0.16 | 2.41 | 0.715 | −0.33 |
| 0.25%×4 | 10.96 | −0.76 | 7.12 | 0.650 | −0.42 | 7.56 | −0.51 | 5.15 | 0.680 | −0.39 |
| 1%×1 | 12.13 | −0.54 | 11.48 | 0.946 | −0.10 | 8.36 | −0.41 | 7.92 | 0.948 | −0.10 |
| 1%×4 | 15.93 | −0.03 | 15.49 | 0.972 | −0.03 | 10.75 | −0.04 | 10.58 | 0.984 | −0.02 |
| 2%×1 | 13.39 | −0.34 | 12.76 | 0.953 | −0.07 | 9.47 | −0.24 | 9.11 | 0.962 | −0.06 |
| 2%×4 | 7.91 | −0.04 | 7.90 | 1.000 | −0.00 | 4.59 | −0.06 | 4.59 | 1.000 | −0.01 |

Every trace × cell reads **eviction-located** (`E ≥ 0.5·G` and `A < 0.5·G`), on the five-seed means and in each of the 60 seeds, and the reading is the same on both traces in all six cells. `G` and `E` are consistent gains in every cell. `A` is never positive on the means: a consistent loss in 8 cells and mixed in the four ×4 cells at L1 1% and 2%, where it is within 0.06 points of zero.

The two replacements do not add. The interaction `A + E − G` is negative in every cell (consistent in 9/12): 33–42% of `G` at L1 0.25%, 6–10% at ×1 with L1 1% and 2%, and at most 3% in the remaining four cells. At L1 0.25% about a third of the gap is obtained only when both decisions are the label's.

## Reading 2 — dose–response of Gaussian label error

`U(noise_s)` for key `(label + s·z, last_group)`, `z` standard normal; the label spans [−6.40, 0].

| cell | conv `s`=0.5 | 1 | 2 | 4 | `U(learned)` | tool `s`=0.5 | 1 | 2 | 4 | `U(learned)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25%×1 | 3.80 | 2.29 | 0.72 | 0.21 | 1.52 | 3.32 | 2.25 | 1.14 | 0.52 | 1.71 |
| 0.25%×4 | 12.81 | 8.46 | 3.37 | 1.24 | 5.35 | 9.49 | 6.59 | 3.10 | 1.50 | 4.41 |
| 1%×1 | 14.90 | 10.44 | 4.61 | 1.87 | 5.90 | 10.35 | 7.44 | 3.28 | 1.32 | 4.16 |
| 1%×4 | 25.37 | 19.83 | 11.44 | 5.77 | 15.79 | 17.94 | 14.11 | 8.22 | 4.19 | 11.07 |
| 2%×1 | 21.19 | 16.01 | 8.98 | 4.36 | 9.97 | 14.98 | 11.49 | 6.50 | 3.21 | 6.95 |
| 2%×4 | 24.97 | 21.13 | 15.04 | 9.07 | 21.81 | 17.35 | 14.90 | 10.73 | 6.66 | 15.48 |

The means are non-increasing in `s` in 12/12 trace × cell and in 5/5 seeds of each. They cross `U(learned)` between `s` = 1 and 2 in ten cells and between 0.5 and 1 at 2%×4 on both traces. At `s` = 4 the noisy label is below sampled LRU in 10/12 cells, and at `s` = 2 at 1%×4 and 2%×4 on both traces (`replay.csv`).

## Reading 3 — placement at an equal error rate

With probability `p` the label rung's victim is replaced, by a uniformly drawn other candidate or by the candidate ranked second. Both families swap at the same decisions; their realised victim agreement `m2` is 0.749–0.750 at `p` = 0.25 and 0.499–0.501 at `p` = 0.5.

| cell | conv `label` | runner-up 0.25 / 0.5 | uniform 0.25 / 0.5 | `lru` | tool `label` | runner-up 0.25 / 0.5 | uniform 0.25 / 0.5 | `lru` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25%×1 | 6.44 | 5.83 / 5.61 | 0.25 / 0.09 | 0.12 | 5.07 | 4.70 / 4.58 | 0.69 / 0.49 | 0.78 |
| 0.25%×4 | 16.31 | 15.38 / 14.92 | 1.48 / 0.65 | 1.11 | 11.98 | 11.32 / 11.11 | 1.67 / 0.98 | 1.52 |
| 1%×1 | 18.03 | 17.32 / 16.81 | 1.97 / 0.94 | 2.04 | 12.51 | 11.91 / 11.66 | 1.45 / 0.68 | 1.40 |
| 1%×4 | 31.72 | 30.83 / 30.43 | 6.44 / 3.51 | 13.21 | 21.82 | 21.48 / 21.31 | 4.50 / 2.56 | 9.56 |
| 2%×1 | 23.36 | 22.65 / 22.49 | 4.12 / 2.27 | 7.82 | 16.42 | 15.87 / 15.77 | 3.02 / 1.65 | 5.77 |
| 2%×4 | 29.71 | 29.67 / 29.64 | 9.69 / 6.03 | 19.70 | 20.07 | 20.09 / 20.07 | 6.74 / 4.36 | 14.07 |

`U(swap_runnerup_p) − U(swap_uniform_p)` is a consistent gain in **12/12** trace × cell at both rates (+4.0 to +24.4 points at 0.25, +4.1 to +26.9 at 0.5). Evicting the runner-up in half of all decisions costs 0.00–1.39 points against the label. Evicting a uniformly drawn candidate in a quarter of them falls below sampled LRU in 8/12 cells, and in half of them in 12/12. The share of decisions whose victim is the reference victim is the same in both families and does not determine utility.

## Reading 4 — which statistic orders utility across arms

Spearman correlation over arms between the five-seed mean of each oriented statistic and the five-seed mean `U`, per trace × cell.

| statistic | primary set (12 single-key arms): range of ρ | cells with ρ ≥ 0.9 | secondary set (18 arms): range of ρ | cells with ρ ≥ 0.9 |
|---|---|---:|---|---:|
| `m1` pairwise concordance | 0.46–0.72 | 0/12 | 0.24–0.43 | 0/12 |
| `m2` victim agreement | 0.13–0.62 | 0/12 | 0.24–0.48 | 0/12 |
| `−m3` victim label excess | 0.62–0.93 | 2/12 | 0.73–0.92 | 1/12 |
| `−m4` avoidable reusable eviction | 0.74–0.85 | 0/12 | 0.74–0.85 | 0/12 |

No statistic *orders utility* in more than 2 of 12 cells. `−m3` reaches the threshold only at 0.25%×1 (0.928 and 0.914 on the primary set). Of the two statistics that weigh how bad the victim was, `−m4` is above both agreement counts (`m1`, `m2`) in every cell and both sets, and `−m3` is too except on the primary set at conversation 2%×4 (0.620 against 0.720 for `m1`).

## Reading 5 — real ranker pairs

`U(pi3) − U(pi0)` and the seed-paired change of each oriented statistic.

| target | utility change over 12 cells | `m1` agrees | `m2` agrees | `−m3` agrees | `−m4` agrees |
|---|---|---:|---:|---:|---:|
| `next_use` | 5 consistent gain, 6 consistent loss, 1 mixed | 12 | 12 | 12 | 12 |
| `binary` | 1 consistent gain, 10 consistent loss, 1 mixed | 8 | 11 | 12 | 11 |

On `next_use` the update loses consistently at the three smallest cells (0.25%×1, 0.25%×4, 1%×1; −0.46 to −0.87) and gains at the three largest (+0.12 to +0.58; mixed at tool-agent 1%×4), and all four statistics move with it in every cell. On `binary` it loses in ten cells (−0.14 to −1.94).

Cells where utility changes consistently while a statistic's mean moves the other way, all on `binary`:

| trace | cell | statistic | utility change | change of the statistic |
|---|---|---|---:|---:|
| conversation | 1%×4 | `m1` | −0.68 (loss) | +0.014 |
| conversation | 2%×1 | `m1` | −1.58 (loss) | +0.005 |
| tool-agent | 2%×1 | `m1` | −0.67 (loss) | +0.001 |
| tool-agent | 2%×4 | `m1` | +0.14 (gain) | −0.019 |
| tool-agent | 2%×4 | `m2` | +0.14 (gain) | −0.041 |

`−m3` has the sign of the utility change in 24/24 comparisons and `−m4` in 23/24; its one disagreement is a mixed utility reading (conversation 2%×4, −0.05). Pairwise concordance rises while utility falls consistently in three cells.

## Reading 6 — the binary target's ceiling, descriptive

| cell | conv `U(label) − U(label_binary)` | seeds | ratio | tool `U(label) − U(label_binary)` | seeds | ratio |
|---|---:|---|---:|---:|---|---:|
| 0.25%×1 | +5.20 | `+++++` | 0.798 | +3.54 | `+++++` | 0.775 |
| 0.25%×4 | +7.01 | `+++++` | 0.505 | +4.72 | `+++++` | 0.480 |
| 1%×1 | +6.16 | `+++++` | 0.366 | +4.08 | `+++++` | 0.346 |
| 1%×4 | −0.02 | mixed | 0.134 | 0.00 | mixed | 0.120 |
| 2%×1 | +1.50 | `+++++` | 0.176 | +0.66 | `+++++` | 0.138 |
| 2%×4 | +0.01 | mixed | 0.241 | −0.02 | mixed | 0.244 |

ratio = `(U(pi0_binary) − U(lru)) / (U(label_binary) − U(lru))`.

The exact `binary` label (requested within 600 seconds or not, ties by recency) is below the `next_use` label in the eight cells with multiplier 1 or L1 0.25% and equal to it within 0.03 points at 1%×4 and 2%×4. The frozen `binary` ranker obtains 12–80% of its own label's gain over sampled LRU; the 78–80% is at 0.25%×1, where that gain is itself 0.8–1.1 points.

## Observations outside the six readings

These were not registered and are descriptions of the published tables.

- **How often each hybrid rejects.** The label rung rejects 172,079–181,382 arrivals at 0.25%×1 on its own trajectory. Under `adm_label` the same rule rejects 9,691 and 13,568 there and at most 104 in the other ten cells: a store filled by the ranker's evictions almost always holds a sampled resident that the label ranks below the arrival, so the arm is close to admitting everything and evicting by the ranker. Under `evict_label` the ranker rejects 44,277–69,934 arrivals at L1 0.25% and at most 592 at L1 1% and 2%, so in the eight cells where `E/G` ≥ 0.946 the arm is close to admitting everything and evicting by the label. The cells where a third of the gap needs both decisions are the cells where the ranker still rejects tens of thousands of arrivals against a store kept by the label. The two L1 fractions that differ here share an L2 capacity (0.25%×4 and 1%×1 are both 1% of the base), so this pattern follows the L1 fraction in this grid and not the L2 capacity alone.
- **`m2` rewards the reference's tie-break.** Sampled LRU has `m2` 0.71–0.91 and the frozen ranker 0.007–0.057. `K*` breaks equal labels by recency, so among candidates that are all not requested within the horizon the reference victim is the oldest, which is LRU's choice; a ranker that evicts a different such candidate loses agreement and no label. `m3` and `m4` do not have this property.
- **Avoidable reusable evictions, as counts.** The frozen ranker evicts a state requested within the horizon while a sampled candidate is not in 7.2–27.4% of window decisions, against 8.8–29.0% for sampled LRU; per million window input tokens that is 67–510 such evictions against 85–551 (80–93% of LRU's count in every cell). The ranker's decisions number 78,237–109,479 per replay window and LRU's 80,013–111,780. `noise_2` makes fewer such evictions than the ranker in every cell (43–477 per million tokens) and has lower utility in every cell. These counts are not losses: an eviction counted here differs in what it would have saved, and the arms' decision populations differ.
- **The binary label's shortfall follows the working-set ratio.** `(U(label) − U(label_binary)) / (U(label) − U(lru))` is 0.82 at 0.25%×1, 0.45–0.46 at 0.25%×4, 0.37–0.39 at 1%×1, 0.06–0.10 at 2%×1 and 0.00 at 1%×4 and 2%×4 on both traces. Against the ratio `γ` of the [working-set check](working-set-ratio-findings.md) (18–21, 4.5–5.2, 4.0–4.7, 1.7–2.2, 1.0–1.2, 0.4–0.6) it is monotone on both traces. `γ` was computed in a separate check and this comparison was made after both results were known.
- **A ranker above the binary label.** At 0.25%×1 the frozen `next_use` ranker (1.52, 1.71) is above the exact `binary` label (1.25, 1.54); in the other ten cells it obtains 12–52% of that label's gain over LRU.
- **The sampled offline key and the statistics.** `offline` has `m3 = 0` and `m4 = 0` in every cell and `m2` between 0.004 and 0.77: it never evicts a candidate above the minimum label, and it resolves the label's ties by next use and not by recency.

## Figures

- [`dose_response.png`](../results/paper/error_location_001/dose_response.png): `U` against the noise level per trace × cell with the `lru`, `learned` and `label` levels, and both swap families.
- [`statistic_utility.png`](../results/paper/error_location_001/statistic_utility.png): the 18 arms per trace × statistic × cell, oriented statistic against `U`.

## What this establishes and what it does not

Established, on the two Mooncake traces, the six cells and the published `all16` mechanism:

- With every decision of the frozen `next_use` ranker kept except the choice of victim among residents, which is given to its exact label, 65–100% of the gap to the label is recovered; with every decision kept except whether to reject the arrival, none is (12/12 eviction-located, every seed).
- The two replacements interact: at L1 0.25% a third or more of the gap is recovered only when both are replaced, and at most 10% elsewhere.
- Under independent Gaussian error in label units, utility is monotone in the error level in every cell and seed, and the frozen ranker's utility lies between levels 1 and 2 (between 0.5 and 1 at 2%×4).
- At an equal victim-agreement rate, which wrong victim is taken changes utility by 4–27 points on the five-seed means, with the same sign in every cell and seed.
- None of the four decision statistics, computed on each arm's own decisions, orders the 12 or the 18 arms by utility at ρ ≥ 0.9 in more than 2 cells.
- For the two published ranker updates, the victim's mean label excess moves with utility in 24/24 trace × cell × target, and pairwise concordance moves against a consistent utility change in 4.
- The exact `binary` label reaches the `next_use` label at 1%×4 and 2%×4 and not in the other eight cells.

Not established:

- **That a causal predictor can supply what the hybrids take from the label.** They read the future; the location says where a perfect score would be worth most, on these traces.
- **That admission cannot matter.** `adm_label` applies the label's sampled comparison — reject the arrival when it ranks lowest among the candidates — to a store filled by the ranker's evictions, where it rejects at most 104 arrivals in ten cells. It is a test of that rule on that store, not of an absolute admission filter. The [working-set check](working-set-ratio-findings.md) counts 77–99.9% of offered bytes as declinable without losing a hit in an LRU tier; an arm that declines those offers outright was not run.
- **Why the ranker picks the residents it picks.** The control locates the cost on a decision type. It does not separate missing information in the features from a learning procedure that fails to use it, and it does not show whether the ranker's errors repeat on the same states.
- **The structure of the ranker's actual errors.** The equivalent noise level is a description under one error model. The ranker makes more avoidable reusable evictions than `noise_2` and has higher utility, so its errors are not exchangeable with independent noise at an equal count.
- **A validated evaluation statistic.** Reading 5 is a sign count over two ranker pairs of one family and reading 4 fails its threshold; the statistics are of each arm on its own decision population, with the `next_use` label as reference, and are not the registered on-policy ranking statistic.
- **Capacity as the cause of the L1 0.25% pattern.** Three L1 fractions and two multipliers do not separate L1 size, L2 size and their ratio; the comparison of the binary shortfall with `γ` is post hoc.
- **Independent causes.** `A`, `E` and their interaction are differences between whole replays whose trajectories differ, not additive shares of a loss.
- **Generality.** Two traces from one deployment family, one mechanism, one ranker family.

Relation to existing work. Injecting error into exact next-use predictions and into binary Belady labels is standard in learning-augmented caching (for instance [GUARD](https://arxiv.org/abs/2507.16242)), and a decision-level statistic related to miss ratio is the *good decision ratio* of [LRB](https://www.usenix.org/conference/nsdi20/presentation/song), which samples eviction candidates and predicts log time-to-next-request as this mechanism does. The method here is therefore not new. What the tables add is specific to an exclusive second tier that receives L1's victims and may reject them: the location of the ranker's cost on resident eviction, the non-additivity at the smallest L1, the gap between the two swap families at one agreement rate, and the cells where a binary reuse label is and is not enough.
