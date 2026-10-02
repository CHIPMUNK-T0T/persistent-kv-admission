# Horizon and class-order control: an exact reuse bit reaches the next-use label when its horizon fits the capacity

This [pre-registered control](horizon-control-plan.md) and its [follow-up](horizon-fill-plan.md) replace the exact `next_use` label by one exact bit — requested within `h` seconds or not, ties by recency — and vary `h`. On the registered grid of five horizons the bit comes within 10% of the label's gain over sampled LRU in 8/12 trace × cell. In the other four, a follow-up written after that result and judged separately finds such a horizon in 4/4. The horizon that works differs by L2 capacity (60, 150, 300 and 600 seconds from the smallest capacity to the largest), and a neighbouring horizon of the grid loses 6–85% of the gain. Given the exact 600-second reuse class on residents, the frozen ranker recovers 94–104% of what the exact label recovers on eviction in the six trace × cell where 600 seconds is a horizon that works, and 33–61% in the other six. Under leaf-only eligibility the gap between the frozen ranker and the label is again located on eviction, in 12/12.

Every arm reads the trace's future on purpose. None is a policy, and no horizon here was chosen before its result was seen.

## Run and integrity

The plan was committed before any implementation, in one commit with the plan of the ranker-error diagnosis (`094bd9c`), and an addendum was written before the smoke (`c84ef53`); the implementation and tests were committed next (`4ad7040`). The smoke (conversation 1%×4, seed 0, 23 replays) passed its identities and is not interpreted. The grid ran once from a clean tree at `9333d5c` into [`results/paper/horizon_control_001/`](../results/paper/horizon_control_001/README.md): 540 replays (2 traces × 6 cells × 9 arms × 5 seeds) on 10 workers in 2,537 seconds.

All required checks passed and are recorded in the [run config](../results/paper/horizon_control_001/run_config.json):

- `label_binary_600` reproduces the 60 published `label_binary` rows exactly.
- Capacities, requested tokens, L1-avoided tokens and the compulsory-absent charge of the 540 replays equal those of the published reference rows of their trace × cell × seed (3,780 comparisons).
- The 120 `leaf16` replays have no present-but-unusable token.
- Every decision is seen with its final victim; no arm without an override overrides one; the Phase 0.98b identities hold in every replay with 0 unexplained tokens.

References are published rows, not reruns: `lru`, `learned`, `label` and `evict_label` under `all16` from the error-location control, `learned` and `label` under `leaf16` from the mechanism control. Every number below is a column of the published CSVs; `scripts/tabulate_horizon_control.py results/paper/horizon_control_001` prints the tables. To rerun, check out `9333d5c` with the traces and `results/onpolicy_full_feb30eb_001/` in place and run

```bash
.venv/bin/python scripts/run_horizon_control.py data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl \
  --output-dir <new directory> --paper-dir <new directory> --workers 10
```

`U(a)` is the extra avoided prefill of arm `a` over L1 alone, in points of window input tokens. "Consistent" means the same sign in all five seed-paired values. Values are given as conversation / tool-agent.

## Reading 1 — horizon of an exact reuse label

`label_binary_h` keeps a state by `(next use within h seconds or not, recency)`. `S_h = (U(label) − U(label_binary_h)) / (U(label) − U(lru))` is the share of the exact `next_use` label's gain over sampled LRU that the arm does not reach.

| cell | `S_6` | `S_15` | `S_60` | `S_300` | `S_600` | smallest, at | reading |
|---|---:|---:|---:|---:|---:|---|---|
| 0.25%×1 | 0.952 / 0.947 | 0.850 / 0.846 | 0.028 / 0.001 | 0.774 / 0.772 | 0.822 / 0.824 | 60 s | a reuse label suffices |
| 0.25%×4 | 0.974 / 0.974 | 0.911 / 0.904 | 0.408 / 0.377 | 0.296 / 0.312 | 0.461 / 0.452 | 300 s | order needed |
| 1%×1 | 0.954 / 0.945 | 0.857 / 0.846 | 0.367 / 0.348 | 0.215 / 0.226 | 0.385 / 0.367 | 300 s | order needed |
| 1%×4 | 0.961 / 0.962 | 0.905 / 0.907 | 0.701 / 0.687 | 0.238 / 0.214 | −0.001 / 0.000 | 600 s | a reuse label suffices |
| 2%×1 | 0.952 / 0.954 | 0.852 / 0.852 | 0.536 / 0.535 | −0.038 / −0.031 | 0.097 / 0.062 | 300 s | a reuse label suffices |
| 2%×4 | 0.985 / 0.960 | 0.951 / 0.921 | 0.771 / 0.744 | 0.260 / 0.225 | 0.001 / −0.003 | 600 s | a reuse label suffices |

A reuse label suffices in 8/12 trace × cell and order is needed in 4/12. Over the eight with a published `S_600` above 0.05, the counts are 4/8 and 4/8.

What the count contains:

- Six trace × cell had a published `S_600` above 0.10. A shorter horizon of the grid reaches the label in two of them, 0.25%×1 on both traces, where the 600-second label loses 82% of the label's gain and the 60-second label 3% and 0%. In the other four — 0.25%×4 and 1%×1 on both traces — no horizon of the grid comes below 0.21.
- The two 2%×1 cells were within 0.10 at 600 seconds already (0.097 and 0.062). There the 300-second label is above the exact `next_use` label in every seed, by 0.59 and 0.33 points.
- At 1%×4 and 2%×4 the 600-second label equals the `next_use` label, as published, and every shorter horizon is below it in every seed.
- At 6 and 15 seconds the arm loses 85–99% of the gain in every cell.

A horizon next to the best one on the grid is costly. At 0.25%×1, 15 seconds loses 85% and 300 seconds 77%, on either side of a horizon that loses at most 3%. At 1%×4 and 2%×4, 300 seconds loses 21–26% where 600 loses nothing. At 2%×1, 60 seconds loses 54% and 600 seconds 6–10%.

## Follow-up — filling the grid between 60 and 300 seconds

The four "order needed" trace × cell share one L2 capacity (1% of the capacity base), and the registered grid has no point between 60 and 300 seconds. The [fill-in plan](horizon-fill-plan.md) was written after reading 1 was seen and because of it. It fixed five horizons (90, 120, 150, 180, 240 seconds), a prediction (a reuse label suffices at one of them in all four), and the rule that its count is reported beside the count above and never merged with it: these four are judged on ten horizons where the other eight were judged on five.

The plan was committed alone (`dfcf5b3`) and the runner and tests next (`78d64af`). The run executed once at that commit from a clean detached worktree, because the tables of the first run were not committed yet, into [`results/paper/horizon_fill_001/`](../results/paper/horizon_fill_001/README.md): 160 replays (2 traces × 2 cells × 8 horizons × 5 seeds) on 10 workers in 338 seconds. Its required checks passed: `label_binary_600` reproduces the 20 published rows; the 60 replays at 60, 300 and 600 seconds equal the first run's rows in tokens, counter digest and decision digest; identifiers, statistics and invariants hold with 0 unexplained tokens. `scripts/tabulate_horizon_fill.py results/paper/horizon_fill_001` prints the tables.

| trace × cell | `S_60` | `S_90` | `S_120` | `S_150` | `S_180` | `S_240` | `S_300` | reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| conversation 0.25%×4 | 0.408 | 0.120 | −0.011 | −0.035 | 0.025 | 0.194 | 0.296 | suffices at a filled horizon |
| conversation 1%×1 | 0.367 | 0.158 | 0.021 | −0.036 | −0.016 | 0.114 | 0.215 | suffices at a filled horizon |
| tool-agent 0.25%×4 | 0.377 | 0.099 | −0.008 | −0.028 | 0.049 | 0.220 | 0.312 | suffices at a filled horizon |
| tool-agent 1%×1 | 0.348 | 0.138 | −0.003 | −0.036 | −0.007 | 0.134 | 0.226 | suffices at a filled horizon |

1. **Fill.** A reuse label suffices at a filled horizon in 4/4, as predicted. The smallest `S_h` is at 150 seconds in all four, where the arm is above the exact `next_use` label in every seed (by 0.53, 0.58, 0.29 and 0.40 points).
2. **Shape.** `S_h` over the seven horizons from 60 to 300 seconds is unimodal in 4/4, and the two cells of each trace, which share an L2 capacity and differ in L1 by a factor of four, have the same minimising horizon.
3. **Sensitivity.** The horizons with `S_h ≤ 0.10` are 120, 150 and 180 seconds in three trace × cell and 90 to 180 seconds in tool-agent 0.25%×4 (`S_90` = 0.099): a width of 60 to 90 seconds. At 90 and 240 seconds the arm loses 10–22% of the gain.

The two counts, side by side: 8/12 on the registered grid; 4/4 of the remaining trace × cell at a horizon added afterwards.

## Reading 2 — class order on resident eviction

Both arms keep the published admission rule on the frozen `next_use` ranker and evict residents by the exact 600-second reuse class first; within a class `evict_binary_learned` uses the ranker's score and `evict_binary_recency` recency. `R(a) = (U(a) − U(learned)) / (U(evict_label) − U(learned))`.

| cell | `R(evict_binary_learned)` | `R(evict_binary_recency)` | ranker's order − recency, points | reading |
|---|---:|---:|---|---|
| 0.25%×1 | 0.350 / 0.330 | −0.156 / −0.133 | +1.67 / +1.11, consistent gain | the ranker's order costs |
| 0.25%×4 | 0.605 / 0.558 | 0.321 / 0.291 | +2.03 / +1.37, consistent gain | the ranker's order costs |
| 1%×1 | 0.470 / 0.450 | 0.439 / 0.456 | +0.36 consistent gain / −0.04 mixed | the ranker's order costs |
| 1%×4 | 0.998 / 0.987 | 1.006 / 1.001 | −0.12 mixed / −0.15 consistent loss | reuse identification suffices |
| 2%×1 | 0.938 / 0.956 | 0.872 / 0.917 | +0.84 / +0.35, consistent gain | reuse identification suffices |
| 2%×4 | 1.043 / 1.042 | 0.999 / 1.003 | +0.35 / +0.18, consistent gain | reuse identification suffices |

Reuse identification suffices with the ranker's order in 6/12 trace × cell; the ranker's order costs in 6/12. The ranker's order is a consistent gain over recency in 9/12, a consistent loss in 1/12 and mixed in 2/12.

The six trace × cell that read "suffices" are the six whose `S_600` is at most 0.10, and the six that read "costs" are the six where the 600-second label itself falls short. Where the 600-second class is one that works, a ranker given it on residents recovers 94–104% of what the exact label recovers on eviction; where it is not, 33–61%. The registered name of the second reading attributes that shortfall to the ranker's order, and the table does not support the attribution: in those six, recency within the class recovers no more (−16% to 46%; level with the ranker at tool-agent 1%×1), and reading 1 and the fill-in show that 600 seconds is the wrong boundary there. An arm with the reuse class at the horizon that works for the cell was not run.

In every replay of both arms, the number of resident evictions that discard a reusable state while a sampled resident is not reusable is zero, as constructed.

## Reading 3 — location under leaf eligibility

`G = U(label) − U(learned)`, `A` and `E` the gains of `adm_label` and `evict_label` over `learned`, all under `leaf16`.

| cell | `G` | `A` | `E` | `A + E − G` | `E / G`, `leaf16` | `E / G`, `all16` (published) | arrival is a candidate, share of decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25%×1 | 7.33 / 5.01 | −0.08 / −0.03 | 7.14 / 4.95 | −0.27 / −0.09 | 0.974 / 0.988 | 0.670 / 0.715 | 49% / 51% |
| 0.25%×4 | 13.64 / 9.67 | −0.11 / −0.08 | 13.55 / 9.62 | −0.20 / −0.13 | 0.994 / 0.995 | 0.650 / 0.680 | 13% / 18% |
| 1%×1 | 14.18 / 10.11 | −0.06 / −0.05 | 14.16 / 10.11 | −0.09 / −0.05 | 0.998 / 1.000 | 0.946 / 0.948 | 12% / 16% |
| 1%×4 | 15.04 / 9.76 | +0.02 / +0.00 | 15.04 / 9.76 | +0.02 / +0.00 | 1.000 / 1.000 | 0.972 / 0.984 | 3% / 6% |
| 2%×1 | 14.92 / 10.14 | −0.10 / +0.00 | 14.92 / 10.13 | −0.09 / −0.01 | 1.000 / 0.998 | 0.953 / 0.962 | 5% / 9% |
| 2%×4 | 7.71 / 4.31 | +0.04 / +0.02 | 7.71 / 4.31 | +0.04 / +0.02 | 1.000 / 1.000 | 1.000 / 1.000 | 3% / 4% |

The gap is eviction-located in 12/12 trace × cell under `leaf16`, the published `all16` label in each. `G` and `E` are consistent gains everywhere; `A` is a consistent loss in 6/12 and mixed in 6/12, never above 0.04 points.

The last column is the share of the evaluation-window decisions of the `adm_label` replays in which the arrival is among the candidates. Under `leaf16` an arrival with a cached child is not a candidate ([addendum](horizon-control-plan.md)), so admission is decided for a minority of decisions: half at 0.25%×1 and 3–18% elsewhere, against 92–99% in the class-order arms under `all16`. `adm_label` rejects 0–583 arrivals per replay on the five-seed mean. The reading therefore says that the label's gain under `leaf16` is reached through eviction alone; it cannot separate "admission does not matter" from "this mechanism leaves admission little to decide".

## Observations outside the registered readings

- **The horizon that works grows with L2 capacity.** By L2 capacity as a share of the capacity base: 60 seconds at 0.25%, 150 seconds at 1% (from the fill-in), 300 seconds at 2%, 600 seconds at 4% and 8%. The first run suggested the pattern after the fact; the fill-in's prediction was built on it and held. It is five capacities on two traces of one family, each horizon the best of a grid, and 600 seconds is the longest horizon tried.
- **A one-bit label is above the exact `next_use` label in six trace × cell.** At 2%×1 with 300 seconds and at the four 1% trace × cell with 150 seconds, in every seed, by 0.29–0.59 points. `evict_binary_learned` is above it at 2%×4 (+0.34 / +0.19). The sampled label rung is not a ceiling, as the published offline rung already showed; why recency among reusable states does better than farthest next use there is not examined.
- **Exact information at the wrong horizon can be worse than the frozen ranker.** At 0.25%×1 the exact 600-second label is below `learned` in every seed (by 0.28 and 0.17 points), and so is `evict_binary_recency` (0.51 and 0.32).
- **Admission by the ranker costs once eviction is exact.** `evict_binary_recency` and `label_binary_600` evict residents by the same key and differ in who decides rejection. The ranker's admission is a consistent loss in 9/12 trace × cell, by 0.15–1.67 points, and mixed in the other three. The error-location control found the mirror image: with the ranker evicting, the label's admission recovered nothing.
- **Rejections do not mark the horizon that works.** `label_binary_h` rejects an arrival only when every sampled resident is reusable within `h` and the arrival is not. At the best horizon it rejects 72–74 thousand arrivals per replay at 0.25%×1, 5–13 thousand at the 1% trace × cell and 171–227 at 2%×1, where the `next_use` label rejects 172–181 thousand, 68–88 thousand and 14–20 thousand.

## What this establishes and what it does not

Established, on these two traces under the published mechanism:

- An exact one-bit reuse label with recency reaches the exact `next_use` label, to within 10% of its gain over sampled LRU, at some horizon of {6, 15, 60, 300, 600} seconds in 8/12 trace × cell; in the remaining 4/4 it does so at a horizon between 90 and 180 seconds added afterwards.
- The horizon that does so differs by capacity, and missing it is costly: the neighbouring horizons of the registered grid lose 6–85% of the gain, and the fill-in's sufficing range is 60 to 90 seconds wide.
- Given the exact 600-second reuse class on residents, the frozen ranker recovers 94–104% of what the exact label recovers on eviction in the six trace × cell where that class is itself sufficient, and 33–61% in the other six; within a class its score orders residents better than recency in 9/12.
- Under leaf-only eligibility the gap between the frozen ranker and the label is recovered by exact eviction alone (97–100%) in every trace × cell.

Not established:

- **That a horizon can be chosen before the fact.** Every horizon reported is the best of a grid, read after the run; the four 1% trace × cell were given a second grid because they failed the first. Nothing here predicts the horizon from capacity or from observable state.
- **That a learned predictor can supply the bit.** Every arm reads the future. The published ranker fitted to the 600-second `binary` target was no better than the `next_use` one, and no ranker has been fitted to another horizon.
- **That the ranker's order within a class is as good as the label's at small capacity.** The class given to the ranker in reading 2 is the 600-second one, which is the wrong one in the six trace × cell where the ranker's arm falls short.
- **Why a reuse bit can exceed the exact `next_use` label,** or whether a horizon beyond 600 seconds would do better at the two largest capacities.
- **That admission is unimportant.** Under `leaf16` few arrivals are candidates; under `all16` admission by the ranker costs up to 1.7 points once eviction is exact.
- **Generality.** Two traces from one deployment family, one frozen linear ranker, two eligibility rules of one sampled mechanism.
