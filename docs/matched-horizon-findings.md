# Matched-horizon follow-ups: the frozen ranker's errors at the boundary that works for the cell, on saved logs and in replay

Two pre-registered follow-ups of the [horizon and class-order control](horizon-control-findings.md), planned together ([diagnosis plan](matched-horizon-diagnosis-plan.md), [class-order plan](matched-class-order-plan.md) with its pre-run addendum; commits `a928501`, `e61aa18`). The horizon control had found that an exact reuse bit with recency reaches the exact `next_use` label at a horizon that grows with L2 capacity — 60 s at 0.25%, 150 s at 1%, 300 s at 2%, 600 s at 4% and 8% of the capacity base — while the frozen ranker's errors had only been counted at 600 s. Part A re-reads the saved `pi0` decision logs with the boundary at each cell's matched horizon `h*` and at every horizon of {60, 150, 300, 600} s; it replays nothing. Part B gives the frozen ranker the exact reuse class at `h*` on residents, in replay.

What comes out. On the ranker's own logs, every bit of its label excess at `h*` is avoidable reusable eviction (24/24), and against recency on the same candidate sets the `next_use` ranker is better at 60 s (0.64× recency's rate at 0.25%×1) and worse at 150 s (2.1–2.3× at the four 1% trace × cell). Its score's within-decision separation of the `h*`-bit is below its separation of the 600-second bit in 6/8 trace × cell, as predicted, and above it at 0.25%×1, where its separation of the 600-second bit is at chance. In replay, the exact class at `h*` with the ranker's admission and order reaches at least 90% of the exact label's eviction gain in 6/8 new trace × cell (the prediction of 8/8 fails at 0.25%×1, 73% and 81%) and reproduces the published 4/4 at 600 s. Within the matched class the ranker's order is a consistent loss against recency in 8/12 (at 600 s it was a gain in 9/12), and recency within the class reaches the label's eviction gain in every new cell (`R` 1.01–1.07). Admission by the ranker costs up to 4.1 points against the label's at `h*`, consistently in 8/12.

## Run and integrity

- **Part A** ran once from a clean tree at code `8a34136` (10 workers, 27 s) into [`results/paper/matched_horizon_diagnosis_001/`](../results/paper/matched_horizon_diagnosis_001/README.md). The 120 `pi0` populations and 4 models carry their published hashes; the scorer reproduces the logged victim in all 4,800,000 decisions; at `h` = 600 s the class shares, excess shares and conditional rates equal the published diagnosis tables in every one of 8,640 values. `scripts/tabulate_matched_horizon_diagnosis.py` prints every table.
- **Part B** ran once from a clean tree at `aca5964` (implementation `dff88da`; 10 workers, 988 s, 120 replays) into [`results/paper/matched_class_order_001/`](../results/paper/matched_class_order_001/README.md). The 40 replays at `h*` = 600 s reproduce the published `evict_binary_learned` / `evict_binary_recency` rows exactly in `avoided_prefill_tokens` and in the counter and decision digests; 840 identifier pairs equal the published reference rows; in every replay no resident eviction discards a state reusable within `h*` while a sampled resident is not, and no decision is overridden outside a first round; the Phase 0.98b identities hold with 0 unexplained tokens. `scripts/tabulate_matched_class_order.py` prints every table.
- An independent recomputation from the raw population files (Part A, two populations, all four horizons) and from the per-seed replay rows and published references (Part B, every reading) agrees with both runs' tables.
- The class-order plan miscounted its own table: it said six trace × cell have `h*` < 600 s where the table gives eight (2%×1 is at 300 s), and 60 reproduction replays where there are 40. The implementation review found it; the [addendum](matched-class-order-plan.md) was committed before any replay. Because 2%×1 already read "suffices" at 600 s, the prediction is easier there; the count over the six trace × cell that read "costs" at 600 s is reported beside the eight.

## Part A — the frozen ranker's errors at the matched boundary, on saved logs

Plan: [matched-horizon diagnosis](matched-horizon-diagnosis-plan.md), committed `a928501`; code `8a34136`; run [`results/paper/matched_horizon_diagnosis_001/`](../results/paper/matched_horizon_diagnosis_001/README.md) (10 workers, 27 s). The 120 `pi0` populations and 4 models carry their published hashes; the scorer reproduces the logged victim in every one of 4,800,000 decisions; and at `h` = 600 s every class share, excess share and conditional rate equals the published diagnosis tables (8,640 values, none differing). `scripts/tabulate_matched_horizon_diagnosis.py` prints every table below.

Two unregistered looks before the run, disclosed: one population (conversation 0.25%×1, `next_use`, seed 0) was read twice before the run, once for an independent validation of the 600-second numbers and once to time the concordance code, and both looks printed its concordance at the four horizons (ranker 0.555 / 0.564 / 0.548 / 0.498). That population's cell is one of the two where the registered prediction of reading 3 fails. The prediction was not changed.

`h*` is 60 s at 0.25%×1, 150 s at 0.25%×4 and 1%×1, 300 s at 2%×1 and 600 s at 1%×4 and 2%×4, the same on both traces. Values are conversation / tool-agent, five-seed means.

### Reading 1 — composition at the matched horizon

At `h*`, avoidable reusable eviction (the victim is reusable within `h*` while a sampled candidate is not) carries 100.0% of the `label_h*` excess in all 24 trace × cell × target, in every seed; the prediction of at least 0.99 holds 24/24. The reason is in the candidate sets: no decision of any population has every candidate reusable within 60 s and labels that differ, at most 4 per population do at 150 s, 24 at 300 s and 131 at 600 s. Order among reusable candidates is not a kind of error the frozen ranker can make at the shorter horizons on its own store.

The share of the `next_use` ranker's decisions that are such an eviction at `h*` is 0.07–0.18, against 0.08–0.30 at 600 s; over the resident-victim decisions alone the share is higher at every `h*` < 600 s (0.11–0.20 against 0.07–0.17 over all decisions).

### Reading 2 — avoidable-eviction rate at the matched horizon against recency and a uniform victim

Over the decisions with both a reusable-at-`h*` and a non-reusable candidate, on the same candidate sets. "Better / worse" is lower / higher than recency in all five seeds.

| cell | `h*` | `next_use` ranker | recency | uniform | ranker / recency | reading |
|---|---:|---:|---:|---:|---:|---|
| 0.25%×1 | 60 s | 0.076 / 0.078 | 0.119 / 0.120 | 0.258 / 0.259 | 0.64 / 0.65 | better / better |
| 0.25%×4 | 150 s | 0.159 / 0.166 | 0.072 / 0.078 | 0.327 / 0.333 | 2.22 / 2.12 | worse / worse |
| 1%×1 | 150 s | 0.152 / 0.157 | 0.067 / 0.072 | 0.313 / 0.317 | 2.27 / 2.17 | worse / worse |
| 1%×4 | 600 s | 0.183 / 0.177 | 0.222 / 0.264 | 0.344 / 0.353 | 0.83 / 0.67 | better / better |
| 2%×1 | 300 s | 0.164 / 0.165 | 0.157 / 0.185 | 0.334 / 0.341 | 1.04 / 0.90 | worse / better |
| 2%×4 | 600 s | 0.099 / 0.084 | 0.117 / 0.099 | 0.226 / 0.225 | 0.85 / 0.85 | better / better |

| horizon | `next_use` ranker better than recency | worse | `binary` ranker better | worse | mixed |
|---|---:|---:|---:|---:|---:|
| 60 s | 5 | 7 | 8 | 4 | 0 |
| 150 s | 3 | 9 | 10 | 2 | 0 |
| 300 s | 5 | 7 | 11 | 0 | 1 |
| 600 s (published) | 9 | 3 | 12 | 0 | 0 |
| `h*` | 7 | 5 | 11 | 0 | 1 |

At the matched horizon the `next_use` ranker is better than recency in 7/12 and worse in 5/12. The five are the four 1% trace × cell, where its matched horizon is 150 s and it makes the error 2.1–2.3 times as often as recency, in every seed, and conversation 2%×1 at 300 s. At 0.25%×1, where the matched horizon is 60 s, it makes the error 0.64–0.65 times as often as recency. Every ranker at every horizon is far below a uniform victim (0.29–0.53 of its rate). The `binary` ranker, fitted to the 600-second bit, is better than recency at `h*` in 11/12 (0.48–0.98 of recency's rate) and mixed at tool-agent 2%×1.

Recency as the diagnosis defines it never takes the arrival, whose tie-break is the newest; the ranker's victim is the arrival in 55–82% of decisions at L1 0.25% and 1%×1. Part of its avoidable evictions at `h*` are therefore rejections of arrivals that return within `h*`, a choice recency does not have; the resident-victim subset, where both choose among residents, is reported in `statistics_seeds.csv` and shows the same direction.

### Reading 3 — within-decision separation of the reuse bit by the ranker's key

Concordance: over the decisions with both classes present, the share of (reusable, non-reusable) candidate pairs whose key would evict the non-reusable one first, ties one half, averaged over decisions; a uniform victim has 0.5.

| cell | `h*` | `next_use` ranker at 60 / 150 / 300 / 600 s | recency at 60 / 150 / 300 / 600 s |
|---|---:|---|---|
| 0.25%×1 | 60 s | 0.552 / 0.559 / 0.543 / **0.495** ; 0.564 / 0.584 / 0.578 / **0.510** | 0.513 / 0.553 / 0.584 / 0.594 ; 0.522 / 0.562 / 0.587 / 0.554 |
| 0.25%×4 | 150 s | 0.585 / 0.593 / 0.586 / 0.609 ; 0.585 / 0.593 / 0.592 / 0.627 | 0.544 / 0.573 / 0.571 / 0.556 ; 0.541 / 0.569 / 0.557 / 0.533 |
| 1%×1 | 150 s | 0.565 / 0.577 / 0.578 / 0.604 ; 0.568 / 0.579 / 0.584 / 0.622 | 0.557 / 0.567 / 0.561 / 0.539 ; 0.552 / 0.561 / 0.545 / 0.515 |
| 1%×4 | 600 s | 0.635 / 0.655 / 0.671 / 0.679 ; 0.628 / 0.649 / 0.667 / 0.680 | 0.542 / 0.532 / 0.512 / 0.508 ; 0.541 / 0.528 / 0.508 / 0.505 |
| 2%×1 | 300 s | 0.602 / 0.604 / 0.621 / 0.643 ; 0.593 / 0.600 / 0.625 / 0.654 | 0.591 / 0.573 / 0.551 / 0.547 ; 0.589 / 0.566 / 0.546 / 0.541 |
| 2%×4 | 600 s | 0.698 / 0.703 / 0.707 / 0.703 ; 0.700 / 0.704 / 0.715 / 0.718 | 0.603 / 0.581 / 0.568 / 0.575 ; 0.610 / 0.587 / 0.575 / 0.584 |

The registered prediction — the `next_use` ranker separates the `h*`-bit worse than the 600-second bit where `h*` < 600 s — holds in 6/8 and fails in 2/8. It holds at the 150- and 300-second cells, by 0.016–0.043 in every seed. It fails at 0.25%×1 on both traces, where the ranker's concordance for the 600-second bit is at chance (0.495 / 0.510) and for the 60-second bit above it (0.552 / 0.564), by 0.054–0.058 in every seed. The ranker's concordance for the 600-second bit rises with L2 capacity on both traces, from chance at 0.25%×1 to 0.70–0.72 at 2%×4; recency's is 0.51–0.59 with no pattern in capacity.

Concordance and the conditional rate of reading 2 disagree at the 1% cells: there the `next_use` ranker's pairwise order for the 150-second bit is above recency's (0.577–0.593 against 0.561–0.573) while its chosen victim is an avoidable eviction twice as often. The key's first minimum is not its average pair; the chooser's error is in which candidate it puts last, and at those cells that candidate is often the arrival.

The `binary` ranker's concordance over the four horizons peaks at 150 or 300 s in 7/12 trace × cell, at its 600-second fit horizon in 3/12 (0.25%×1 on both traces, tool-agent 1%×4) and at 60 s in 2/12 (2%×4); not registered.

### Reading 4 — bridge, descriptive

Per trace × cell: the ranker's `m4_h*` beside the published utility gaps of its own replay (`learned` for `next_use`, `pi0_binary` for `binary`, both `all16`).

| cell | `h*` | `m4_h*`, all decisions, `next_use` | resident-victim decisions | `U(label) − U(learned)`, points | `U(label_binary_h*) − U(learned)` |
|---|---:|---:|---:|---:|---:|
| 0.25%×1 | 60 s | 0.073 / 0.074 | 0.122 / 0.111 | 4.92 / 3.37 | 4.74 / 3.36 |
| 0.25%×4 | 150 s | 0.159 / 0.165 | 0.194 / 0.200 | 10.96 / 7.56 | 11.49 / 7.86 |
| 1%×1 | 150 s | 0.151 / 0.156 | 0.197 / 0.196 | 12.13 / 8.36 | 12.72 / 8.76 |
| 1%×4 | 600 s | 0.183 / 0.177 | 0.185 / 0.181 | 15.93 / 10.75 | 15.96 / 10.75 |
| 2%×1 | 300 s | 0.163 / 0.165 | 0.175 / 0.175 | 13.39 / 9.47 | 13.97 / 9.80 |
| 2%×4 | 600 s | 0.097 / 0.083 | 0.097 / 0.083 | 7.91 / 4.59 | 7.90 / 4.61 |

Spearman over the twelve trace × cell, `m4_h*` over all decisions against `U(label) − U(ranker)`: 0.67 (`next_use`) and 0.71 (`binary`); over resident-victim decisions 0.50 and 0.63. The same statistic at a fixed horizon is negative or near zero (−0.22 at 600 s, −0.32 at 150 and 300 s, −0.06 at 60 s against the label gap; −0.73 at 600 s against the `label_binary_600` gap). Twelve points of two traces, both quantities ordered largely by capacity; a description, not a test.

## Part B — the frozen ranker given the exact class at the matched horizon, in replay

Both arms keep the published admission rule on the frozen `next_use` ranker and evict residents by the exact reuse class at `h*` first: `evict_binary_h*_learned` orders within the class by the ranker's score, `evict_binary_h*_recency` by recency. `R_h*(a) = (U(a) − U(learned)) / (U(evict_label) − U(learned))`, in five-seed means; `U` in points of window input tokens.

### Reading 1 — class order at the matched horizon

| cell | `h*` | role | `U(learned)` | `U(evict_label)` | `U(h*, ranker order)` | `U(h*, recency)` | `R_h*(ranker order)` | `R_h*(recency)` | reading | `R_600(ranker order)`, published |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0.25%×1 | 60 s | new | 1.52 / 1.71 | 4.82 / 4.11 | 3.94 / 3.65 | 5.03 / 4.18 | 0.734 / 0.806 | 1.063 / 1.029 | the ranker's order costs | 0.350 / 0.330 |
| 0.25%×4 | 150 s | new | 5.35 / 4.41 | 12.47 / 9.56 | 11.78 / 9.38 | 12.71 / 9.70 | 0.903 / 0.965 | 1.033 / 1.027 | reuse identification suffices | 0.605 / 0.558 |
| 1%×1 | 150 s | new | 5.90 / 4.16 | 17.37 / 12.08 | 16.98 / 11.90 | 17.53 / 12.22 | 0.966 / 0.977 | 1.013 / 1.017 | reuse identification suffices | 0.470 / 0.450 |
| 1%×4 | 600 s | reproduced | 15.79 / 11.07 | 31.27 / 21.65 | 31.24 / 21.51 | 31.36 / 21.67 | 0.998 / 0.987 | 1.006 / 1.001 | reuse identification suffices | 0.998 / 0.987 |
| 2%×1 | 300 s | new | 9.97 / 6.95 | 22.73 / 16.06 | 23.11 / 16.31 | 23.59 / 16.61 | 1.029 / 1.027 | 1.067 / 1.061 | reuse identification suffices | 0.938 / 0.956 |
| 2%×4 | 600 s | reproduced | 21.81 / 15.48 | 29.71 / 20.07 | 30.06 / 20.26 | 29.71 / 20.09 | 1.043 / 1.042 | 0.999 / 1.003 | reuse identification suffices | 1.043 / 1.042 |

Reuse identification at the matched horizon suffices with the ranker's order in **6/8** new trace × cell; the prediction of 8/8 **fails** at 0.25%×1 on both traces, where the arm recovers 73% and 81% of the exact label's eviction gain. The reproduced count is 4/4, as published. Over the six trace × cell that read "the ranker's order costs" at 600 s (0.25%×1, 0.25%×4, 1%×1), the matched class turns four into "suffices" (0.25%×4 and 1%×1 on both traces, `R` 0.90–0.98 where the 600-second class gave 0.45–0.61) and leaves two (0.25%×1). The counts are never merged.

### Reading 2 — order within the matched class

`U(evict_binary_h*_learned) − U(evict_binary_h*_recency)`, seed-paired.

| cell | `h*` | ranker's order − recency, points | at 600 s, published |
|---|---:|---|---|
| 0.25%×1 | 60 s | −1.08 / −0.54, consistent loss / loss | +1.67 / +1.11, gain / gain |
| 0.25%×4 | 150 s | −0.93 consistent loss / −0.32 mixed (one seed +) | +2.03 / +1.37, gain / gain |
| 1%×1 | 150 s | −0.55 / −0.32, loss / loss | +0.36 gain / −0.04 mixed |
| 1%×4 | 600 s | −0.12 mixed / −0.15 loss (published) | the same |
| 2%×1 | 300 s | −0.48 / −0.31, loss / loss | +0.84 / +0.35, gain / gain |
| 2%×4 | 600 s | +0.35 / +0.18, gain / gain (published) | the same |

Within the class at `h*`, the ranker's order is a consistent loss against recency in 8/12 trace × cell, a consistent gain in 2/12 (2%×4, the reproduced rows) and mixed in 2/12; at 600 s it had been a consistent gain in 9/12. Recency within the matched class reaches or exceeds the exact label's eviction gain in every new trace × cell (`R_h*` 1.01–1.07) and in the reproduced ones at 1%×4 (1.00–1.01); the exact bit at `h*` with recency inside is at or above `evict_label` everywhere except conversation 2%×4 (0.999).

### Reading 3 — admission at the matched horizon, descriptive

`U(label_binary_h*) − U(evict_binary_h*_recency)`: the two arms evict residents by the same key and differ in who decides rejection.

| cell | `h*` | label's admission − ranker's admission, points | seed signs |
|---|---:|---|---|
| 0.25%×1 | 60 s | +1.24 / +0.89 | consistent / consistent |
| 0.25%×4 | 150 s | +4.13 / +2.57 | consistent / consistent |
| 1%×1 | 150 s | +1.08 / +0.70 | consistent / consistent |
| 1%×4 | 600 s | +0.38 / +0.16 | consistent / mixed |
| 2%×1 | 300 s | +0.35 / +0.13 | consistent / mixed |
| 2%×4 | 600 s | +0.00 / −0.00 | mixed / mixed |

Admission by the ranker is a consistent loss against the label's in 8/12 and mixed in the four trace × cell where the difference is at most 0.16 points. At the matched horizon the loss is larger than the 0.15–1.67 points the horizon control found at 600 s: 4.1 and 2.6 points at 0.25%×4, where the recency arm rejects 52 and 43 thousand arrivals per replay.

### Reading 4 — matched versus 600 s, descriptive

In the eight new trace × cell, the arm at `h*` is above the published arm at 600 s in every seed for both orders: by 0.65–5.69 points with the ranker's order (largest at 1%×1) and by 1.31–6.60 with recency.

## Observations outside the registered readings

- **Where the frozen ranker's shortfall sits, by capacity.** At the four 1% trace × cell the exact bit at 150 s given on residents, with the ranker's admission and order kept, recovers 90–98% of the exact label's eviction gain, where the 600-second bit recovered 45–61%: there the shortfall was the bit at the matched boundary. At 0.25%×1 the exact 60-second bit recovers 73–81% with the ranker's order and 103–106% with recency: there the remainder is the ranker's order within the class. At 2%×1 and above, the bit at `h*` recovers everything with either order.
- **The ranker's order within a class helps at 600 s and costs at `h*`.** At 600 s the score ordered residents within the reusable class better than recency in 9/12; once the class is at `h*`, its order is worse than recency in 8/12. One reading: within a 600-second class the score's residual order separates the states reused soon from the rest, which is the shorter-horizon bit; once that bit is given exactly, what remains of the score orders worse than recency. This is a reading of two tables, not a tested mechanism.
- **A statistic on the ranker's own store does not predict its order on the class-kept store.** On its own logs at 0.25%×1 the `next_use` ranker makes an avoidable 60-second eviction 0.64× as often as recency (Part A, reading 2), yet within the exact 60-second class in replay its order loses 1.08 and 0.54 points to recency (Part B, reading 2). At the 1% cells the two agree in direction (2.1–2.3× recency's rate; a loss within the class). The error-location control had already shown that decision statistics are not utility proxies; here the store differs as well.
- **Separation at chance for the fitted bit at the smallest capacity.** At 0.25%×1 the `next_use` ranker's within-decision concordance for the 600-second bit, its fit horizon, is 0.495 / 0.510, and its concordance for the 60-second bit 0.552 / 0.564; the arm with the 600-second class there was below `learned` in the horizon control. The score carries some of the 60-second bit at that capacity and none of the 600-second one on those candidate sets.
- **Admission.** The label's admission beats the ranker's by up to 4.1 points at the matched horizon; the horizon control's observation at 600 s (0.15–1.67) was the smaller end of the same effect. `R` is measured against `evict_label`, which keeps the ranker's admission, so this cost is outside every `R` above.

## What this establishes and what it does not

Established, on these two traces under the published mechanism, with `h*` the best horizon of grids run on the same traces:

- On the frozen ranker's own logged decisions, the label excess at `h*` is avoidable reusable eviction to within 0.1% in every trace × cell × target, and decisions whose candidates are all reusable within `h*` are absent at 60 s and at most 4 per 40,000 at 150 s.
- Against recency on the same candidate sets, the `next_use` ranker makes the avoidable eviction at `h*` less often at 0.25%×1 (0.64–0.65×) and more often at the four 1% trace × cell (2.1–2.3×), in every seed; the `binary` ranker less often in 11/12.
- The `next_use` ranker's within-decision separation of the `h*`-bit is below its separation of the 600-second bit in 6/8 trace × cell with `h*` < 600 s (the registered prediction) and above it at 0.25%×1, where the 600-second separation is at chance.
- Given the exact reuse class at `h*` on residents, with its admission and order kept, the frozen ranker reaches at least 90% of the exact label's eviction gain in 6/8 new trace × cell (prediction 8/8 failed at 0.25%×1: 73% and 81%) and in 4/4 reproduced; recency within the matched class reaches it in 8/8 new.
- Within the matched class the ranker's order is a consistent loss against recency in 8/12; the label's admission is above the ranker's by 0.1–4.1 points, consistently in 8/12.

Not established:

- **That the horizon can be chosen before the fact.** `h*` is read from replays of the same traces; nothing here predicts it from capacity or observable state.
- **That a learned predictor can supply the `h*`-bit.** Nothing is fitted. The published rankers were fitted to 600-second targets; their within-decision separation of the shorter bits is 0.55–0.64, above chance and far from exact. Phase 0.9 fitted single-tier rankers to 60- and 300-second binary targets with small gains ([target change](target-change-findings.md)).
- **Why the ranker's order within the matched class loses to recency,** beyond the reading above; no feature or coefficient was examined.
- **A causal share for admission.** Reading 3 compares two arms that differ in who rejects; `R` does not contain it.
- **Generality.** Two traces of one deployment family, one frozen linear ranker family, the published sampled mechanism with `all16` eligibility; reservoirs of 40,000 decisions on the ranker's own store for Part A.
