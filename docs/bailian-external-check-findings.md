# External-workload check on the Qwen-Bailian traces: the reuse bit reaches the label at a horizon that does not transplant, and the gap between mechanisms corresponds to requested tokens that are present but unusable

The [pre-registered](bailian-external-check-plan.md) check of the Mooncake readings on a second workload family: four Qwen-Bailian traces converted to 512-token blocks ([input audit](bailian-input-audit.md)). Plan `553b026` with Addendum 1 (`2ac7c9d`, reading 7), Addendum 2 (`b4d6d47`) and Addendum 3 (`3ce935d`, control C not run); code `e6e702e`, reviewed and committed before any replay; Mooncake reproduction anchor `3f0986f`; smoke `a33b9c0`; main run `7873a0f`, in [`results/paper/bailian_external_check_001/`](../results/paper/bailian_external_check_001/README.md). Every arm reads the trace's future on purpose; none is a policy and nothing is fitted.

What comes out. Five of the seven registered predictions hold and two do not. Under `all16`, an exact reuse bit with recency reaches the capped next-use label at some grid horizon in 23 of 23 evaluable trace × cell (prediction 3; in 15 of them the label and the 600-second bit already give the same utility, so the comparison has content in 8), and recency inside the class beats a random order on every seed in 23 of 23 (prediction 5). The Mooncake horizon table does not transplant: `S_h*` ≤ 0.10 in 11 of 23 against a threshold of 12 (prediction 1 fails), seven of the eleven being 600-second cells where the bit and the label give the same utility to within 0.05 point; every failure is toward a longer horizon (12 to 0, prediction 2), and the best grid horizon is nondecreasing in L2 bytes in 4 of 4 traces (prediction 4). Under `leaf16` some grid horizon suffices in 17 of 22 (prediction 6). The value of exact order beyond the bit is larger under `leaf16` than under `all16` in 8 of the 16 cells with `h*` < 600 s against a threshold of 10 (prediction 7 fails), and the sign reversal seen on Mooncake (the bit ahead of the label under `all16`) appears in none. A post-hoc accounting from the published rows, on both workloads, relates the gap between mechanisms to one counter: the requested tokens that are present in L2 but unusable behind a missing ancestor under `all16`. It leaves a mean absolute residual of 0.13 point on Bailian and 0.17 on Mooncake, and it is a candidate explanation, not a tested cause.

## Objective, scope and comparison metrics

The fixed objective is to select materialised states that increase avoided prefill tokens in a finite persistent lower-tier KV cache receiving L1 evictions. These 2,448 replays ask whether three Mooncake readings hold on another workload family: that a one-bit reuse label with recency reaches the next-use label under `all16`, that recency inside the class matters, and that the value of order information beyond the bit depends on which residents are eligible. They do not establish a deployable policy or a way to predict the bit from history.

`U` is extra avoided prefill tokens over L1 alone, in points of the window's input tokens. `S_h = (U(label) − U(label_binary_h)) / (U(label) − U(lru))`; `S_h` ≤ 0.10 is "suffices". `label` is the published comparator, sampled-16 greedy on `−log1p(min(Δ, 600 s))`: it is not an optimum, and its cap makes the bit at 1,200 s something other than a coarsening of it, so `S_1200` can be negative. `H_off` is the heap reference without the cap; it is not an upper bound either (under `leaf16`, thinking 0.25%×1, `U(label)` = 10.754 exceeds `H_off` = 10.422). A trace × cell is evaluable under a mechanism when the five-seed mean of `U(label) − U(lru)` is at least 1.0 point. All readings are five-seed means on the primary window `W` = [split, end − 1,200 s] (1,677–1,680 s), with the full window as context. `h*` is the Mooncake table, unchanged: 60 s at 0.25%×1, 150 s at 0.25%×4 and 1%×1, 300 s at 2%×1, 600 s at 1%×4 and 2%×4. The four traces are one provider's and are not pooled with each other or with Mooncake.

## Run and integrity

- The **anchor** reran 70 Mooncake replays through the new runner before any Bailian replay: 70 of 70 reproduce the published rows (digests against `error_location_001`, `horizon_control_001` and `leaf_matched_horizon_001`; 87 counter columns against `mechanism_control_001`).
- The **smoke** ran 16 replays at 512 tokens (3.4–28.3 s, at most 567 MiB) and wrote no utility. The first 16-token replay exceeded the registered limits and control C was not run ([Addendum 3](bailian-external-check-plan.md)).
- The **main run** ran once from a clean tree at `a33b9c0` (10 workers, 9,555 s): 2,400 sampled replays and 48 heap references. All nine required checks pass: counts, completeness, identifiers equal across every arm of a trace × cell × seed on every window (120), window consistency (2,448), mechanism and arm identity (2,448), decision statistics and no override (2,400), no present-but-unusable token under `leaf16` (1,200), the attribution identities with no unexplained absent token (2,400), and the random arm's own stream (240).
- An independent recomputation of every reading and every prediction count from `replay_seeds.csv` agrees with the published tables (0 mismatches). `scripts/tabulate_bailian_external_check.py` prints every registered table; `scripts/tabulate_bailian_closure_cost.py` (with `--mooncake-root results/paper`) prints the post-hoc tables below, after checking its values against the published ones.
- Evaluable cells: 23 of 24 under `all16` (To-B 2%×4 has 0.48 point of headroom) and 22 of 24 under `leaf16` (To-B 1%×4 and 2%×4).

## The registered predictions

| # | Prediction (primary window) | Count | Threshold | Holds | On the full window |
|---|---|---|---|---|---|
| 1 | `S_h*` ≤ 0.10, `all16` | 11 of 23 | 12 | no | 11 of 23 |
| 2 | failing cells: best horizon longer than `h*`, `all16` | 12 longer, 0 shorter | longer > shorter | yes | 11 longer, 0 shorter, 1 equal |
| 3 | some grid horizon suffices, `all16` | 23 of 23 | 20 | yes | 22 of 23 |
| 4 | best horizon nondecreasing in L2 bytes, `all16` | 4 of 4 traces | 3 | yes | 4 of 4 |
| 5 | random order a loss on every seed, `all16` | 23 of 23 | 16 | yes | 23 of 23 |
| 6 | some grid horizon suffices, `leaf16` | 17 of 22 | 16 | yes | 18 of 22 |
| 7 | `Δ(h*)` separated, larger under `leaf16` | 8 of 16 | 10 | no | 8 of 16 |

The thresholds are absolute counts and were not rescaled to the evaluable cells. Prediction 7's threshold was set by judgement in Addendum 1, before any Bailian replay.

## Reading 1 — the Mooncake horizon table carried over

`S_h*` on `W` under `all16`; "n.h." is no headroom.

| Cell (`h*`) | To-C | To-B | Thinking | Coder |
|---|---|---|---|---|
| 0.25%×1 (60 s) | 0.097 | 0.410 | 0.245 | 0.076 |
| 0.25%×4 (150 s) | 0.053 | 0.390 | 0.490 | 0.152 |
| 1%×1 (150 s) | 0.085 | 0.484 | 0.534 | 0.233 |
| 2%×1 (300 s) | 0.161 | 0.305 | 0.202 | 0.274 |
| 1%×4 (600 s) | 0.000 | 0.000 | 0.000 | 0.000 |
| 2%×4 (600 s) | 0.000 | n.h. | 0.000 | −0.002 |

The transplant suffices in 11 of 23: To-C 5 of 6, coder 3 of 6, thinking 2 of 6, To-B 1 of 5. Seven of the eleven are 600-second cells, where the bit at 600 s and the label give the same utility on every seed (To-C, To-B, thinking) or differ by at most 0.045 point (coder 2%×4): at these capacities a victim outside the reusable class is available and both arms order that class by recency, so the comparison carries no information about the bit. Outside the 600-second cells the transplant suffices in 4 of 16 (To-C 3, coder 1, To-B 0, thinking 0); on Mooncake's own traces it sufficed in 8 of 8 (`S_h*` −0.038 to 0.028 on the full window). Under `leaf16` the transplant suffices in 6 of 22, all six being 600-second cells.

Absolute values on `W` under `all16`, range over the six cells (`T = H_off − U(lru)`; share is `(U(label) − U(lru)) / T`):

| Trace | `U(lru)` | `U(label)` | `T` | Label's share of `T` |
|---|---|---|---|---|
| To-C | 0.63–23.08 | 10.72–34.56 | 8.93–23.53 | 0.75–0.85 |
| To-B | 0.96–6.26 | 2.24–9.47 | 1.15–6.50 | 0.42–0.94 |
| Thinking | 1.52–7.05 | 5.27–13.49 | 3.22–7.41 | 0.61–0.95 |
| Coder | 3.91–13.46 | 13.49–22.50 | 7.96–16.33 | 0.50–0.84 |

## Reading 2 — the grid: which horizon, in which direction

Best grid horizon in seconds and `min_h S_h` on `W`, `all16`:

| Cell | To-C | To-B | Thinking | Coder |
|---|---|---|---|---|
| 0.25%×1 | 60 / 0.097 | 600 / 0.008 | 150 / 0.038 | 60 / 0.076 |
| 0.25%×4 | 150 / 0.053 | 1,200 / −0.114 | 1,200 / −0.023 | 300 / 0.003 |
| 1%×1 | 150 / 0.085 | 1,200 / −0.152 | 1,200 / −0.094 | 300 / −0.040 |
| 2%×1 | 600 / 0.060 | 1,200 / −0.230 | 1,200 / −0.189 | 600 / −0.001 |
| 1%×4 | 1,200 / −0.072 | 1,200 / −0.253 | 1,200 / −0.199 | 1,200 / −0.255 |
| 2%×4 | 1,200 / −0.225 | n.h. | 1,200 / −0.360 | 1,200 / −0.404 |

- **Some grid horizon suffices in 23 of 23** (prediction 3). Thirteen of the 23 have their best point at 1,200 s and are flagged as resting on it; the count does not depend on that point, because in each of the thirteen the 600-second bit already suffices (`S_600` ≤ 0.006). With the grid cut at 600 s the count is still 23 of 23. The count overstates what was tested: in 15 of the 23 cells the label and the 600-second bit give the same utility to within 0.05 point, so the bit "suffices" there whatever the order inside the class is worth. The comparison has content in the other 8 (To-C 0.25%×1, 0.25%×4, 1%×1 and 2%×1; coder 0.25%×1, 0.25%×4 and 1%×1; thinking 0.25%×1), and there a bit at 60–600 s with recency is within 10% of the label (`min S` −0.040 to 0.097).
- **A negative `S_1200`** means the bit at 1,200 s with recency beats the capped label, by up to 40% of the label's gain over `lru` (coder 2%×4, 1.6 points). The bit at 1,200 s also uses future information beyond the label's 600-second cap, so its lead cannot be read as a benefit of coarsening the information. The best horizon sits on the grid's largest point in those thirteen cells, so the grid gives a direction there, not an optimum.
- **Direction** (prediction 2): in the 12 `all16` cells where the transplant fails, the best horizon is longer than `h*` in 12 and shorter in none. Under `leaf16`, of 16 failing cells, 13 are longer, none shorter and 3 equal.
- **Monotonicity** (prediction 4): the best horizon is nondecreasing in L2 bytes in 4 of 4 traces under both mechanisms: To-C 60, 150, 600, 1,200, 1,200 s over the 0.25%, 1%, 2%, 4% and 8% levels; coder 60, 300, 600, 1,200, 1,200; thinking 150, then 1,200 four times; To-B 600, then 1,200 three times. Every sequence ends on the grid's ceiling and is flagged; what is observed below the ceiling is a rise with capacity.
- **Classification** under `all16`: transplant suffices 11, re-tuned grid point suffices 12, none in the grid 0, no headroom 1. Under `leaf16`: 6, 11, 5 and 2. The five `leaf16` cells with no sufficing horizon are small cells: To-C 0.25%×1 (0.202), 0.25%×4 (0.144) and 1%×1 (0.189), thinking 0.25%×1 (0.113), coder 0.25%×1 (0.192).
- **Calibration half** (reading 4, descriptive): the horizon that is best on the first half of `W` suffices on the second half in 22 of 23 under `all16` (To-B 1%×4 has no computable first-half value) and in 17 of 22 under `leaf16`, against 23 and 18 for the second half's own best.

## Reading 3 — recency beyond the bit

`D_rand = U(label_binary_random_h*) − U(label_binary_h*)` in points on `W`. Under `all16` it is negative on all five seeds in every evaluable cell:

| Cell | To-C | To-B | Thinking | Coder |
|---|---|---|---|---|
| 0.25%×1 | −0.39 | −0.08 | −0.22 | −0.42 |
| 0.25%×4 | −0.81 | −0.17 | −0.35 | −0.65 |
| 1%×1 | −0.88 | −0.09 | −0.40 | −0.78 |
| 2%×1 | −0.86 | −0.12 | −0.42 | −0.72 |
| 1%×4 | −0.81 | −0.05 | −0.14 | −0.63 |
| 2%×4 | −2.01 | n.h. | −0.33 | −1.19 |

Under `leaf16` the same difference is a loss on every seed in 5 of 22, a gain on every seed in 8 (To-B 4, To-C 2, thinking 1, coder 1) and mixed in 9, between −0.48 and +0.21 point; outside To-C 2%×4 its magnitude is at most 0.21. With the bit fixed, recency is worth something under `all16` on every trace and close to nothing under `leaf16`.

## Reading 7 — exact order beyond the bit, by mechanism

`Δ_m(h) = U_m(label) − U_m(label_binary_h)`, seed-paired. At `h*`, for the 16 cells with `h*` < 600 s, `all16` / `leaf16` in points; **bold** is separated (the smallest `leaf16` seed above the largest `all16` seed):

| Cell (`h*`) | To-C | To-B | Thinking | Coder |
|---|---|---|---|---|
| 0.25%×1 (60 s) | **0.98 / 2.33** | 2.50 / 2.47 | **1.69 / 2.40** | **0.73 / 2.06** |
| 0.25%×4 (150 s) | **1.02 / 3.10** | 1.25 / 1.21 | 3.16 / 3.04 | **2.08 / 3.50** |
| 1%×1 (150 s) | **1.45 / 3.72** | 1.07 / 1.05 | 2.77 / 2.75 | **2.64 / 3.74** |
| 2%×1 (300 s) | **2.66 / 3.27** | 0.39 / 0.33 | 0.76 / 0.73 | 2.55 / 2.50 |

- **Separated in 8 of 16** (To-C 4, coder 3, thinking 1, To-B 0); prediction 7 asked for 10. All 32 values are positive on every seed: at the Mooncake `h*`, the label is ahead of the bit under both mechanisms in every cell.
- **Reversed in 0.** On Mooncake's eight such cells `Δ_all16(h*)` was negative on every seed in six (−0.29 to −0.59), positive in one and mixed in one, while `Δ_leaf16(h*)` was positive in all eight (+0.9 to +1.9): separated in 8 of 8 and reversed in 6. On Bailian the bit never beats the label at `h*` under `all16`.
- **At 600 s**, where the bit and the label share one class boundary: separated in 8 of 22 (To-C 3, coder 3, thinking 1, To-B 1), reversed in 0. In 12 cells under `all16` and 13 under `leaf16` the difference is zero or mixed in sign: the order inside the reusable class does nothing there. Where it is not zero it is large at small capacity (To-C 0.25%×1: 8.86 / 10.26; coder 0.25%×1: 6.52 / 7.71), because 600 s is too long a horizon for those cells. In one cell the larger value is under `all16` (To-C 2%×1: 0.98 / 0.46). No prediction was registered at 600 s, and Mooncake has a `leaf16` value at 600 s only in its four 600-second cells.
- In the eight cells that are not separated (To-B 4, thinking 3, coder 1), `Δ(h*)` is the same under both mechanisms to within 0.12 point and `Δ(600)` is at most 0.10 point: what `Δ(h*)` measures there is the cost of a horizon that is too short, which does not depend on eligibility.

## Observations outside the registered readings

These are post hoc, descriptive and computed from the published rows; none was predicted.

1. **Where the question has content.** The eight cells where the label and the 600-second bit differ by more than 0.5 point under `all16` (reading 2) are exactly the eight cells separated at `h*` (reading 7).
2. **At each mechanism's own best grid horizon**, the shortfall is larger under `leaf16` in 7 of those 8 cells (To-C 0.097 → 0.202, 0.053 → 0.144, 0.085 → 0.189; thinking 0.038 → 0.113; coder 0.076 → 0.192, 0.003 → 0.038, −0.040 → 0.059) and smaller in one (To-C 2%×1, 0.060 → 0.028). The larger value of exact order under `leaf16` is therefore not an artefact of comparing at the Mooncake `h*`.
3. **The gap between mechanisms corresponds to the requested tokens that are present but unusable.** For an arm `a`, let `pu(a)` be the requested tokens found in L2 at request time but unusable because an ancestor is absent, in points of the window's input tokens. It is a count at request time; it is not stranded capacity and not bytes × residence time. Under `leaf16` it is zero in all 1,200 replays, by construction. In every replay the requested tokens found in L2 split into the usable and the unusable, `P = U + pu`, so the two sides below are tied by accounting and differ only by how `P` moves:

   `Δ_leaf16(h*) − Δ_all16(h*) = [pu_all16(label) − pu_all16(bit)] + residual`, with `residual = [P_leaf16(label) − P_all16(label)] − [P_leaf16(bit) − P_all16(bit)]` (exact because `pu` is zero under `leaf16`; the script checks it with `P` read from the L2-avoided column).

   The correlation between the first two terms over the 22 cells evaluable under both mechanisms is 0.994, but the accounting makes a high correlation unsurprising; the content is the size of the residual, in points:

   | Trace | Cells | `pu(label) − pu(bit)`, range | Correlation | Mean residual | Mean absolute | Largest absolute |
   |---|---|---|---|---|---|---|
   | To-C | 6 | 0.000 to 2.741 | 0.994 | −0.280 | 0.287 | 0.656 |
   | To-B | 4 | −0.016 to 0.082 | 0.263 | −0.051 | 0.051 | 0.112 |
   | Thinking | 6 | −0.022 to 0.640 | 0.994 | −0.006 | 0.029 | 0.091 |
   | Coder | 6 | −0.045 to 1.665 | 0.999 | −0.127 | 0.128 | 0.295 |
   | All four | 22 | −0.045 to 2.741 | 0.994 | −0.122 | 0.130 | 0.656 |

   In the eight cells where the comparison has content, `pu(label)` is 0.65–3.21 points and `pu(bit)` 0.01–0.48, and the separation is 0.70–1.11 of their difference. The difference in unusable tokens does not fully account for the gap between mechanisms, and the residual leans negative (six of these eight cells; To-C 0.25%×4: a difference of 2.73 against a separation of 2.08). The residual is not a tabulation error: it is the difference in how `P` moves between mechanisms, which includes their different resident sets and trajectories. In the other cells both counters are at most 0.26 and within 0.08 of each other, and the separation is within 0.12 of zero. To-B has no cell with a difference above 0.1, so its own correlation says nothing; the relation is carried by To-C, coder and one thinking cell.
4. **The same accounting on Mooncake**, from the published rows on their full window: in all eight cells with `h*` < 600 s, `pu(label)` is 2.03–4.64 points and `pu(bit)` 0.40–2.27, and the separation is 0.80–0.97 of their difference. Over the 12 cells the correlation is 0.994 (conversation 0.993, tool-agent 0.995), the mean residual −0.17, the mean absolute residual 0.17 and the largest 0.52. Put beside each other, the two workloads differ in a balance of two magnitudes: in Mooncake's six reversed cells the label's extra unusable tokens (1.54–3.01 points) are more than its order is worth when no token can be unusable (`Δ_leaf16`, 0.91–1.90), and in the seven separated cells of To-C and coder they are less (0.60–2.74 against 2.06–3.74). This is a description of the two signs, not a test of why they differ.
5. **The random order's loss and the same counter.** By the same accounting, `D_rand = −[pu_all16(rand) − pu_all16(bit)] + residual`, the residual being `P_all16(rand) − P_all16(bit)`. Under `all16` the random order leaves more requested tokens unusable than recency in every cell (0.03–1.21 points more), and over the 23 evaluable cells the residual has mean −0.11, mean absolute value 0.12 and a largest value of 0.80 (To-C 2%×4, where the loss is −2.01 against −1.21); the correlation is 0.968 (0.91–0.98 within each trace). Under `leaf16`, where no token can be unusable, the random order's loss is absent (reading 3).
6. **A score that is monotone along a prefix does not keep the cache closed.** A request that uses a state uses every ancestor, so an ancestor's next use is never later than a descendant's and the label never scores an ancestor below its descendant; the bit and recency share the property. The label still leaves up to 3.21 points of requested tokens unusable on Bailian and 4.64 on Mooncake: with 16 sampled candidates an ancestor can be the worst candidate while its descendants are not in the sample. Why exact next-use order leaves more unusable than recency was not tested.

## To-B and the granularity control

To-B is the trace the 16→512 coarsening changes most (25.8% of its input tokens in partial final blocks). Control C, To-B at 16-token blocks under the same bytes, was not run: the registered stop condition applied ([Addendum 3](bailian-external-check-plan.md)). A single 16-token `lru` replay, timed outside the runner while the main run was in progress and writing no utility, took 4,909 s with a peak of 9,481 MiB, which puts the reduced control (54 replays, two workers) at 37 hours or more. Every To-B statement here is therefore about the 512-token representation.

During the main run, before any utility had been read, a rule was fixed for whether to propose a graded control (256- and 128-token blocks) as a separate plan: only if To-B's classification or reading 7 differed from the other three traces. It could not be committed at the time, because the runner refuses to publish when `HEAD` moves during a run; it is recorded here. Applied: To-B (transplant 1, re-tuned 4, no headroom 1; separated 0 of 4) is at one end of an ordering To-C, coder, thinking, To-B, and shares its pattern with thinking (transplant 2, re-tuned 4; separated 1 of 4), a trace the coarsening changes little (4.9%). To-B does not stand apart from all three, so the rule does not trigger. That To-B resembles thinking is not a test of granularity: it does not show that To-B would read the same at 16 tokens, and the limit to the 512-token representation stands. To-B is also the extreme case, with the smallest headroom (two cells not evaluable under `leaf16`), and the judgement that it "shares its pattern with thinking" was the reviewer's to overrule. The review of this analysis (2026-10-04) left the control unrun and kept the limit.

## What this establishes and what it does not

On four Qwen-Bailian traces at 512-token blocks, with exact-information arms and the published two-tier replay:

- **Reproduced.** Under `all16`, a one-bit reuse label with recency reaches the capped next-use label at some horizon in every evaluable cell. The comparison has content in eight of them, where it does so with a horizon of 60–600 s; in the other fifteen the label and the 600-second bit coincide. Recency inside the class beats a random order in every cell under `all16`.
- **Not reproduced.** The Mooncake horizons are not portable: outside the cells where the comparison is empty they suffice in 4 of 16, and they are too short in every failing cell. The best horizon rises with L2 bytes in every trace and reaches the grid's ceiling in 13 cells, where the grid does not locate it. The bit never beats the label under `all16` at `h*`: the Mooncake reversal is absent.
- **Reproduced in part.** Exact order beyond the bit is worth more under `leaf16` in 8 of 16 cells, short of the registered 10. The eight are the cells where the label and the bit differ at all; in the other eight, exact order inside the 600-second class is worth at most 0.10 point under either mechanism.
- **New here.** Under `leaf16` recency inside the class is worth close to nothing, on all four traces. Post hoc, on both workloads, the difference between mechanisms in what the exact order is worth beyond the bit equals the difference in requested tokens left present but unusable under `all16`, up to a residual of 0.13 point (Bailian) and 0.17 (Mooncake) in mean absolute value and at most 0.66; the random order's loss under `all16` matches the same counter to 0.12 on average and at most 0.80.

It does not establish a policy, a way to predict the bit or its horizon from history, or anything about a learned order: no model was run. It does not locate the best horizon above 600 s, since the comparator is capped there and the grid ends at 1,200 s. It says nothing at 16-token granularity. The post-hoc accounting is not a cause established by a control: the two quantities are tied by `P = U + pu`, the residual is not zero (the unusable tokens overshoot the gap by up to 0.66 point), `leaf16` also changes which residents are sampled and what is held, and no intervention removed the unusable tokens while holding the rest fixed. The relation is carried by To-C, coder, one thinking cell and the Mooncake traces; To-B has no cell where the counters differ.

## Discussion

The three Mooncake readings come out of this check in different states. That a coarse reuse bit can stand in for exact next-use order under `all16` holds on all four traces. That its horizon is a property of the cache size carries over only as a direction: the horizon grows with L2 bytes, but the seconds differ by trace, and for To-B and thinking they lie at or beyond the label's own cap at all but the smallest capacity. A horizon picked on the first half of the window held on the second in 22 of 23 cells, which suggests the horizon is stable within a trace over tens of minutes, though nothing here chose it from history.

The third reading, that the value of order information depends on eligibility, survives in a narrower form than it was registered in. The registered form, a count of cells where exact order is worth more under `leaf16`, fell short, and the reversal did not appear. What the two workloads share, post hoc, is a quantity that corresponds to the gap by accounting and leaves a residual of 0.13–0.17 point in mean absolute value: the requested tokens an order leaves present but unusable under `all16`. Recency leaves few and a random order more, and the random order's loss is of that size under `all16` and absent under `leaf16`; exact next-use order leaves the most and gains the most from `leaf16`, and on Mooncake the bit is ahead of it under `all16`. This makes the counter a candidate explanation for when the sign changes. It is not shown to be the cause: nothing here varied it alone.

For the objective, the distinction it draws is between a token being retained and a token being usable. Every arm here is given exact future-reuse information, and one thing that differs measurably between arms and mechanisms is how much of what they retain can be read back through an unbroken prefix at request time.

On prefix monotonicity, the check supports one statement and no more: a score that never ranks an ancestor below its descendant does not keep the cache prefix-closed under sampled eviction, since the exact label has that property and leaves the most tokens unusable. It does not identify the cause of the difference between orders.

Left open by this check: whether the correspondence survives an intervention that removes the unusable tokens without changing the sampled population; whether a learned order's loss and gain correspond to the same counter (the frozen ranker's published rows carry it); where the best horizon lies above 600 s with an uncapped comparator; how the bit or its horizon would be obtained from history; and whether To-B reads the same at 16 tokens. The experimental phase ends with this check (decided 2026-10-04): no further replay, intervention or fit is planned, and these points are carried as limits of the evidence.
