# Matched-horizon checks: the evaluation window, the class that carries the learned order's loss, and leaf eligibility

Three pre-registered checks of the [matched-horizon follow-ups](matched-horizon-findings.md), planned together and committed alone (`b41514d`) before any implementation: the [evaluation-window check](tail-window-check-plan.md), the [class-order mix](class-order-mix-plan.md) and the [leaf-matched horizon check](leaf-matched-horizon-plan.md) with its pre-run addendum (`8cf8d65`). Implementations `ec7c802`, `ed7171f` and `f4bbdbe` were reviewed and committed before any replay; the runs are committed as `d865332`, `0735733` and `6ba2884`. The matched-horizon follow-ups had found, under the published `all16` mechanism, that an exact one-bit reuse label at the capacity-matched horizon `h*` with recency reaches the exact `next_use` label, that the frozen ranker given that class on residents reaches 90% of the label's eviction gain in 10/12 trace × cell, and that within the class the ranker's order loses to recency in 8/12. The three checks ask whether those readings rest on the last 600 s of the trace, in which class the learned order's loss sits and whether recency within a class carries information beyond the bit, and whether the readings hold under leaf-only eligibility.

What comes out. The readings do not rest on the last 600 s: on the window with the last 600 s excluded the bit at `h*` reaches the label in 12/12, the class-order reading moves in one threshold cell (conversation 0.25%×4, 0.903 → 0.859), and the sign of the learned order's loss within the class is unchanged in 12/12 and larger. The learned order's loss within the matched class sits in the non-reusable class, not in the reusable one as predicted (0/8): with the ranker ordering only the states reusable within `h*` it is at or above recency in every cell, and with the ranker ordering only the states not reusable within `h*` it is a consistent loss in all eight cells where the full order lost; a random order within the class loses to recency in 12/12. Under leaf eligibility the horizon reading weakens (`S_h*` ≤ 0.10 in 9/12, 0.20–0.21 at 0.25%×1; the prediction of 12/12 fails), the class reading holds (`R_h*` ≥ 0.9 in the same 10/12), and the within-class order reading reverses: the ranker's order is a consistent gain over recency in 11/12 (the prediction of a loss in at least 8/12 fails in the opposite direction), and recency within the class falls below the label's eviction gain in 7/12. Of the nine registered predictions four hold and five fail, two in the opposite direction.

## Run and integrity

- **Part 1** ran once from a clean tree at `ec7c802` (10 workers, 2,218 s, 420 replays) into [`results/paper/tail_window_check_001/`](../results/paper/tail_window_check_001/README.md). All 420 replays reproduce their published rows exactly (`avoided_prefill_tokens`, counter and decision digests); 2,940 identifier pairs equal the published reference rows; the window counters are consistent in every replay; the full-window readings recomputed from the reproduced rows equal the published tables (120 values). The runner carries one check the plan did not name — that recomputation of the published tables — and it is reported as such.
- **Part 2** ran once from a clean tree at `d865332` (10 workers, 1,437 s, 180 replays) into [`results/paper/class_order_mix_001/`](../results/paper/class_order_mix_001/README.md). Identifiers equal the 900 published reference rows; the class statistic is zero in every replay; the random arm's stream is its own; the eight cells of prediction 1 are exactly the published consistent losses.
- **Part 3** ran once from a clean tree at `0735733` (10 workers, 1,149 s, 240 replays) into [`results/paper/leaf_matched_horizon_001/`](../results/paper/leaf_matched_horizon_001/README.md). The plan required the 60 `label` replays to reproduce the published `leaf16` rows "exactly (`avoided_prefill_tokens` and the counter digest)"; the implementation review found that the published rows (`mechanism_control_001`, which predates the digests) carry no digest, and the [addendum](leaf-matched-horizon-plan.md) committed before the implementation and before any replay replaced the digest with every published counter column (87 per row). All 60 reproduce in `avoided_prefill_tokens` and all 87 columns; identifiers equal the 2,640 published reference rows (eleven per trace × cell × seed, `leaf16` and `all16` alike); no replay has a present-but-unusable token; the class statistic is zero in the 180 `h*` replays; the published `all16` values beside every reading were recomputed from the published per-seed rows and equal the published tables before any replay.
- An independent recomputation from the per-seed rows and the published references agrees with every table of the three runs. `scripts/tabulate_tail_window_check.py`, `scripts/tabulate_class_order_mix.py` and `scripts/tabulate_leaf_matched_horizon.py` print every cited table.
- Every arm reads the trace's future; none is a policy, nothing is fitted, and no existing replay path was changed (the existing tests pass with the new ones, 810 in all).

## Part 1 — the evaluation window: the last 600 seconds excluded

Plan: [tail-window check](tail-window-check-plan.md), committed `b41514d`; code `ec7c802`; run [`results/paper/tail_window_check_001/`](../results/paper/tail_window_check_001/README.md) (10 workers, 2,218 s, 420 replays). Every exact-label arm treats a state with no later occurrence in the trace as "not reused", which at the end of a finite trace is knowledge of the end. The seven arms behind the matched-horizon readings — `lru`, `learned`, `label`, `evict_label`, `label_binary_h*`, `evict_binary_h*_learned`, `evict_binary_h*_recency` — were rerun with everything unchanged, each built by the module that built its published rows, and each replay was counted on three windows: `full` (the published window, split to trace end, 1,414.8 s), `head` (split to trace end − 600 s, 814.8 s, 56–57% of the window's input tokens) and `tail` (the last 600 s, 43–44%). `h*` was kept, not re-chosen. The predictions are on `head`.

Integrity. All 420 replays reproduce their published rows exactly (`avoided_prefill_tokens`, counter digest and decision digest, 420 of each); 2,940 identifier pairs equal the published reference rows; in every replay the head counters are at most the full ones, the tail is full minus head, and the head's requests, input tokens and L1-avoided tokens are the same across the seven arms of a trace × cell × seed; the Phase 0.98b identities hold; the class statistic is zero in the 120 class-order replays; and the full-window readings recomputed from the reproduced rows equal the published `horizon.csv`, `fill.csv`, `class_order.csv`, `order.csv` and `admission.csv` values (120 values, none differing). An independent recomputation from the per-seed rows agrees with every table. `scripts/tabulate_tail_window_check.py` prints them.

Values are conversation / tool-agent, five-seed means; `h*` is 60 s at 0.25%×1, 150 s at 0.25%×4 and 1%×1, 300 s at 2%×1 and 600 s at 1%×4 and 2%×4.

### Reading 1 — the matched-horizon bit on the head window

`S_h*` ≤ 0.10 on the head window in 12/12 trace × cell, as predicted (**holds**). The head values lie between −0.076 and 0.015, the tail values between −0.060 and 0.048, the full values between −0.038 and 0.028; every reading is "reuse_label_suffices" on every window. The bit at `h*` with recency reaches the exact label on both halves of the window.

### Reading 2 — the exact class at the matched horizon on the head window

`R_h*(evict_binary_h*_learned)` ≥ 0.9 on the head window in 9/12, against a prediction of the same 10/12 as on the full window with 0.25%×1 below on both traces (**fails**). The head reading equals the full reading in 11/12. The cell that changes is conversation 0.25%×4 (`h*` = 150 s): `R` 0.903 on the full window, 0.859 on the head, 0.959 on the tail — a threshold reading on every window. At 0.25%×1 the learned order is below 0.9 on all three windows (full 0.734 / 0.806, head 0.620 / 0.741, tail 0.881 / 0.891), and the nine other trace × cell are at or above 0.93 on the head. Recency within the class recovers at least 0.9 on every window in 12/12 (head 0.98–1.14; 1.140 / 1.121 at 0.25%×1, where it exceeds the exact label by more than on the full window).

### Reading 3 — order within the matched class on the head window

The sign of the five-seed mean of `U(evict_binary_h*_learned) − U(evict_binary_h*_recency)` on the head window is the full window's in 12/12, as predicted (**holds**), and the seed-sign categories are the full window's cell for cell: a consistent loss in the same 8/12, a consistent gain at 2%×4 on both traces, mixed at conversation 1%×4 and tool-agent 0.25%×4. The head losses are larger than the full-window losses in every one of the eight (conversation 0.25%×1 −1.72 against −1.08 points; tool-agent 0.25%×4 −0.79 against −0.32), and the tail differences are smaller and less consistent: on the tail the sign agrees with the full window in 9/12 (conversation 1%×4 +0.04, tool-agent 0.25%×4 +0.30 and 1%×1 +0.03 turn positive), with 5 consistent losses, 5 mixed and the same 2 gains. The learned order's loss within the matched class is not a product of the last 600 s; it is larger where the exact bit does not know the end.

### Reading 4 — admission at the matched horizon on the head window

The sign of the five-seed mean of `U(label_binary_h*) − U(evict_binary_h*_recency)` on the head window is the full window's in 10/12, against a prediction of 12/12 (**fails**). The two disagreements are the 2%×4 cells (`h*` = 600 s) on both traces, where the full-window means are +0.001 and −0.000 points with mixed seed signs (`-+++0`, `-0000`) and the head means are exactly zero in every seed (`00000`); the registered rule counts a zero against a non-zero as a disagreement. In those two cells the ranker's admission costs nothing on either window. In the ten others the sign is the full window's; the admission cost is a consistent gain of the label's rule in the same 8/12 on the head as on the full window (head 0.26–4.39 points) and in 9/12 on the tail.

### Reading 5 — the tail's share, descriptive

The tail holds 43–44% of the window's input tokens and 42–47% of the gap `G = U(label) − U(learned)` in every trace × cell (0.416–0.468), 39–47% of `U(label) − U(lru)`, 44–52% of `U(evict_binary_h*_learned) − U(learned)`, 39–46% of `U(evict_binary_h*_recency) − U(learned)` and 44–47% of `U(evict_label) − U(learned)`. Neither the gap nor its recovery is concentrated in the last 600 s. The shares of the two small differences (reading 3's order difference, −0.41 to +0.74; reading 1's `U(label) − U(label_binary_h*)`, −102 to +12) are ratios of small numbers and say nothing.

### What Part 1 establishes

The matched-horizon readings do not rest on the last 600 s: the bit at `h*` reaches the label on the head window in every cell, the class-order reading moves in one threshold cell only (conversation 0.25%×4, 0.903 → 0.859), the sign of the learned order's loss is the same in every cell, and the admission cost's sign is the same wherever it is not zero. A difference between head and tail mixes the exact labels' knowledge of the end with any change of the workload over time; nothing here attributes it to either.

## Part 2 — within-class order at the matched horizon: which class, and recency beyond the bit

Plan: [class-order mix](class-order-mix-plan.md), committed `b41514d`; code `ed7171f`; run [`results/paper/class_order_mix_001/`](../results/paper/class_order_mix_001/README.md) (10 workers, 1,437 s, 180 replays). The matched class-order control had found the frozen ranker's order within the exact class at `h*` a consistent loss against recency in 8/12 trace × cell. Three arms split that order: `mix_in_learned` (the reusable class at `h*` ordered by the ranker, the non-reusable class by recency), `mix_out_learned` (the reverse) and `random_within_class` (a uniform draw per candidate per decision inside both classes, from a stream separate from the store's sampling). Admission by the ranker, the exact bit at `h*` on residents, sampled-16 eviction and `h*` are the matched control's; the ranker inside the mixed arms is the one object the admission override consults, observed once. The store's key is `((bit, within-class key), last_group)`, so a non-reusable candidate is always evicted before a reusable one and the within-class key decides only among candidates of the same class; the order of the reusable class decides only in a decision whose every candidate is reusable within `h*`. Differences are seed-paired against the published matched recency arm: `D_in`, `D_out`, `D_rand`, and `D_learned` (the published matched learned arm's).

Integrity. The identifiers of the 180 replays equal those of the 900 published reference rows of their trace × cell × seed; every decision was seen with its final victim; the class statistic is zero in every replay (no resident eviction discards a state reusable within `h*` while a sampled resident is not; no override outside a first round); the random arm drew 207,742,418 uniforms over its 60 replays, each from its own stream; the Phase 0.98b identities hold; no unexplained absent tokens. The eight cells named by prediction 1 are exactly the published consistent losses of `D_learned`. At 2%×4 on both traces the mixed arms coincide with the matched arms decision for decision: `mix_in_learned` carries the matched recency arm's counter and decision digests and `mix_out_learned` the matched learned arm's, in every seed (20 of 20), so there the reusable class's order never decided a victim. An independent recomputation from the per-seed rows agrees with every published table. `scripts/tabulate_class_order_mix.py` prints them.

### Reading 1 — which class carries the learned order's loss

Prediction 1 — in each of the eight named cells `D_in < 0` and `D_out > D_in` — **fails, 0/8**, and the readings point the other way. Five-seed means, conversation / tool-agent, points against the matched recency arm; seed signs in parentheses, "named" marks the cells of prediction 1:

| cell | `h*` | `D_in` (ranker in the reusable class) | `D_out` (ranker in the non-reusable class) | `D_learned` (published) | `D_in + D_out` within `D_learned`'s seed interval |
|---|---|---|---|---|---|
| 0.25%×1, named / named | 60 s | +0.86 (+++++) / +0.68 (+++++) | −1.63 (−−−−−) / −0.97 (−−−−−) | −1.08 / −0.54 | no / no |
| 0.25%×4, named / — | 150 s | +1.06 (+++++) / +0.82 (+++++) | −1.60 (−−−−−) / −0.89 (−−−−−) | −0.93 / −0.32 | no / yes |
| 1%×1, named / named | 150 s | +0.62 (+++++) / +0.44 (+++++) | −0.76 (−−−−−) / −0.42 (−−−−−) | −0.55 / −0.32 | no / no |
| 1%×4, — / named | 600 s | +0.10 (+−+++) / −0.00 (−+++−) | −0.17 (−−−−+) / −0.28 (−−−−−) | −0.12 / −0.15 | yes / no |
| 2%×1, named / named | 300 s | +0.16 (+++−+) / +0.05 (+−+++) | −0.47 (−−−−−) / −0.27 (−−−−−) | −0.48 / −0.31 | yes / no |
| 2%×4, — / — | 600 s | 0.00 (00000) / 0.00 (00000) | +0.35 (+++++) / +0.18 (+++++) | +0.35 / +0.18 | yes / yes |

The ranker's order inside the reusable class is never a loss: a consistent gain over recency in 6/12 (5 of the 8 named cells), mixed with a mean within ±0.16 points in 4/12, and exactly the recency arm in the two 2%×4 cells. The ranker's order inside the non-reusable class is a consistent loss in 9/12 — in all eight named cells, by −0.27 to −1.63 points — mixed at conversation 1%×4, and a consistent gain only at 2%×4, where it is the whole learned order. `D_out < D_in` in 10/12 and in every named cell. In `R_h*` terms (share of the exact label's eviction gain over the ranker's own), the ranker in the reusable class is at or above the matched recency arm in 11/12 and 0.002 points below it at tool-agent 1%×4 (1.32 / 1.31 at 0.25%×1, 1.18 / 1.19 at 0.25%×4, 1.07 / 1.07 at 1%×1, 1.08 / 1.07 at 2%×1), and the ranker in the non-reusable class is below it everywhere it differs (0.57 / 0.63 at 0.25%×1, 0.81 / 0.85 at 0.25%×4, 0.95 / 0.96 at 1%×1), the published matched learned arm (0.73 / 0.81, 0.90 / 0.97, 0.97 / 0.98) lying between the two.

Additivity, descriptive: `D_in + D_out` is within the seed interval of `D_learned` in 5/12. In six of the seven other cells the sum is above `D_learned` (conversation 0.25%×1: −0.77 against −1.08; tool-agent 1%×1: +0.02 against −0.32): the full learned order loses more than its two class orders' differences add to. At tool-agent 1%×4 the sum is below (−0.28 against −0.15).

### Reading 2 — recency beyond the bit

Prediction 2 — the random order within both classes a consistent loss against the matched recency arm in at least 8/12 — **holds, 12/12**. `D_rand` is −0.90 / −0.67 at 0.25%×1, −1.95 / −1.61 at 0.25%×4, −2.90 / −2.20 at 1%×1, −0.39 / −0.15 at 1%×4, −0.80 / −0.60 at 2%×1 and −0.24 / −0.11 at 2%×4, every seed negative. Recency within the class carries selection information beyond the bit in every trace × cell; it is largest at 1%×1 (`h*` = 150 s), where the random order gives back 2.9 / 2.2 points of the 11.5 / 7.9 points between the ranker's own eviction and the exact label's. Against the learned order, descriptive: the random order is a consistent loss in 8/12 (−0.28 to −2.35 points), mixed in 3 (conversation 1%×4, tool-agent 0.25%×1 and 1%×4) and a consistent gain in 1, conversation 0.25%×1 (+0.19), the cell in which the ranker's order within the class is furthest below recency. The learned order within the class is above a random order in most cells and below recency in the same eight; what it adds over a random order it adds in the reusable class, as reading 1 locates.

### What Part 2 establishes

The learned order's loss within the matched class sits in the non-reusable class, not where prediction 1 placed it: with the ranker ordering only the states not reusable within `h*`, the loss against recency is consistent in every one of the eight cells where the full learned order lost, and with the ranker ordering only the states reusable within `h*`, the ranker is at or above recency in every cell, consistently in six. Recency inside the class is not the bit alone: a random order within the class loses to it in 12/12. Nothing here says why the ranker's order among the non-reusable states is below recency; "not reusable within `h*`" includes states reused after `h*` and states never reused, and nothing here separates the two. Every arm reads the trace's future and none is a policy.

## Part 3 — the matched-horizon bit and the class order under leaf eligibility

Plan: [leaf-matched horizon](leaf-matched-horizon-plan.md), committed `b41514d`, addendum `8cf8d65`; code `f4bbdbe`; run [`results/paper/leaf_matched_horizon_001/`](../results/paper/leaf_matched_horizon_001/README.md) (10 workers, 1,149 s, 240 replays). Under `leaf16` (the arrival if it is a leaf plus up to 16 uniformly sampled leaf residents; a resident with a cached child is never a candidate) the horizon control had found the ranker's gap eviction-located in 12/12, but had run neither the bit at any horizon nor the class-order arms. Four arms per trace × cell: `label_binary_h*` (the bit at `h*` with recency, built as the horizon control and its fill-in build it), `evict_binary_h*_learned` and `evict_binary_h*_recency` (built as the matched class-order control builds them) and the `label` rung as a reproduction anchor; the arms differ from their `all16` rows in `l2_eligibility` only, and `h*` is carried over from `all16` unchanged. References are the published `leaf16` rows of `lru`, `learned`, `label` (mechanism control) and `evict_label` (horizon control); every value is reported beside the published `all16` value of the same arm.

### Reading 1 — horizon under leaf eligibility

`S_h*` ≤ 0.10 in 9/12 trace × cell against a prediction of 12/12 (**fails**). Five-seed means, conversation / tool-agent:

| cell | `h*` | `S_h*` under `leaf16` | under `all16` (published) |
|---|---|---|---|
| 0.25%×1 | 60 s | 0.211 / 0.196 | 0.028 / 0.001 |
| 0.25%×4 | 150 s | 0.090 / 0.080 | −0.035 / −0.028 |
| 1%×1 | 150 s | 0.084 / 0.071 | −0.036 / −0.036 |
| 1%×4 | 600 s | −0.000 / 0.000 | −0.001 / 0.000 |
| 2%×1 | 300 s | 0.112 / 0.098 | −0.038 / −0.031 |
| 2%×4 | 600 s | −0.001 / −0.000 | 0.001 / −0.003 |

The bit at `h*` with recency falls short of the `leaf16` label at 0.25%×1 on both traces and at conversation 2%×1, and is within 0.002 of the threshold at tool-agent 2%×1. In every cell with `h*` < 600 s the value is higher under `leaf16` than under `all16`, by 0.11 to 0.20; at the two 600-second cells it is zero under both, as the 600-second bit equals the `next_use` label there. The `leaf16` label rung itself is above the `all16` one (9.2 against 6.4 points at conversation 0.25%×1, 20.0 against 16.3 at 0.25%×4), and the bit at `h*` keeps 79–80% of its gain over `lru` at 0.25%×1 and 89–93% in the other cells with `h*` < 600 s. `h*` is the `all16` best; whether another horizon does better under `leaf16` is not examined.

### Reading 2 — class order under leaf eligibility

`R_h*(evict_binary_h*_learned)` ≥ 0.9 in 10/12 against a prediction of at least 10/12 (**holds**), in the same ten cells as under `all16`: 0.763 / 0.799 at 0.25%×1 (`all16` 0.734 / 0.806), 0.90–0.92 in the six cells with `h*` of 150 or 300 s (`all16` 0.90–1.03) and 1.01–1.04 at the 600-second cells. Recency within the class, `R_h*(evict_binary_h*_recency)`, is 0.752 / 0.763 at 0.25%×1, 0.87–0.91 in the six 150- and 300-second cells and 1.00 at the 600-second cells: below 0.9 in 7/12, where under `all16` it was 0.98–1.14 and at or above 0.9 in 12/12. Under leaf eligibility the bit at `h*` with recency inside the class does not reach the exact label's eviction gain where `h*` < 600 s, and the ranker's order within the class is above recency's.

### Reading 3 — order within the matched class

`U(evict_binary_h*_learned) − U(evict_binary_h*_recency)` is a consistent loss in 0/12 against a prediction of at least 8/12 (**fails**), and the readings point the other way: a consistent gain in 11/12 and mixed in 1. Five-seed means, points, conversation / tool-agent, seed signs in parentheses:

| cell | `h*` | under `leaf16` | under `all16` (published) |
|---|---|---|---|
| 0.25%×1 | 60 s | +0.08 (+++++) / +0.18 (+++++) | −1.08 (−−−−−) / −0.54 (−−−−−) |
| 0.25%×4 | 150 s | +0.39 (+++++) / +0.14 (+++++) | −0.93 (−−−−−) / −0.32 (−+−−−) |
| 1%×1 | 150 s | +0.41 (+++++) / −0.00 (−−−++) | −0.55 (−−−−−) / −0.32 (−−−−−) |
| 1%×4 | 600 s | +0.16 (+++++) / +0.15 (+++++) | −0.12 (−−−+−) / −0.15 (−−−−−) |
| 2%×1 | 300 s | +0.47 (+++++) / +0.29 (+++++) | −0.48 (−−−−−) / −0.31 (−−−−−) |
| 2%×4 | 600 s | +0.18 (+++++) / +0.16 (+++++) | +0.35 (+++++) / +0.18 (+++++) |

The seed-sign category changes in 10/12 (every cell but the two 2%×4 gains): eight consistent losses under `all16` are consistent gains under `leaf16` in seven and mixed in one (tool-agent 1%×1), and the two mixed cells are consistent gains. The within-class order comparison of the matched control is specific to the `all16` candidate set.

### Reading 4 — admission under leaf eligibility, descriptive

`U(label_binary_h*) − U(evict_binary_h*_recency)` is at most 0.054 points under `leaf16` (a consistent gain at conversation 0.25%×1, +0.037, and tool-agent 0.25%×4, +0.049; mixed in 10; zero in every seed at 2%×4), against 0.13–4.13 points under `all16` (a consistent gain in 8/12). The arrival is a candidate in 2.5–10.1% of the recency arm's decisions under `leaf16` and in 92.5–98.7% under `all16`. Under leaf eligibility the admission rule rarely decides, and the label's rule and the ranker's differ by nothing that the seeds separate.

### What Part 3 establishes

With `h*` carried over from `all16` and the `leaf16` references, the class reading holds (the ranker given the class reaches 90% of the label's eviction gain in the same 10/12) and the horizon reading weakens (the bit at `h*` with recency reaches the label in 9/12, and is 0.11–0.20 further from it at every `h*` < 600 s). The within-class order reading reverses: under `leaf16` the ranker's order within the matched class is a consistent gain over recency in 11/12, and recency inside the class no longer reaches the label's eviction gain where `h*` < 600 s. Nothing here says which horizon is best under `leaf16`, and `leaf16` remains a control on eligibility, not a mechanism proposed.

## Observations outside the registered readings

- **The within-class order reading depends on the candidate set.** Under `all16` the ranker's order within the matched class loses to recency in 8/12, and Part 2 locates the loss in the non-reusable class; under `leaf16`, where only leaves are candidates, the same two orders compare the other way in 11/12 and recency within the class falls below the label. Under `all16` a candidate may be a state with cached descendants, whose eviction leaves them present but unusable; under `leaf16` none is. Whether the non-reusable class's loss under `all16` is the eviction of such states is not tested here. Two descriptive counters do not settle it: with the ranker ordering the non-reusable class under `all16`, the share of evictions that leave a descendant present is lower than with recency in 12/12 (0.34–0.45 against 0.42–0.47), and the tokens requested while present but unusable are higher in 9/12 and lower in 3. Neither is a registered reading.
- **The learned order's loss is more than the sum of its class orders'.** `D_in + D_out` is above `D_learned` in six of the seven cells where it is outside `D_learned`'s seed interval (Part 2, reading 1): the ranker ordering both classes loses more than its two class orders' differences add to. This is a reading of one table.
- **The head window sharpens the order reading.** The learned order's loss within the class is larger on the head window than on the full window in all eight consistent-loss cells (Part 1, reading 3), and the tail differences are smaller and mixed; where the exact bit does not know the end of the trace, the learned order loses more, not less.
- **The horizon under `leaf16`.** `S_h*` is 0.11–0.20 higher under `leaf16` at every `h*` < 600 s. `h*` was chosen on `all16` grids; a `leaf16` grid might choose otherwise, and a horizon that works under one candidate set need not under another. Not examined.

## What this establishes and what it does not

Established, on these two traces, with `h*` the best horizon of `all16` grids run on the same traces:

- The matched-horizon readings do not rest on the last 600 s of the trace: on the head window the bit at `h*` reaches the label in 12/12, the class reading changes in one threshold cell, the sign of the learned order's loss is the same in 12/12, and the admission cost's sign is the same wherever it is not zero (two of the nine predictions fail, both by one or two threshold or zero-valued cells).
- Within the matched class under `all16`, the learned order's loss sits in the non-reusable class: the ranker ordering only the reusable class is at or above recency in 12/12 (a consistent gain in 6/12) and the ranker ordering only the non-reusable class is a consistent loss in 9/12 and in every cell where the full order lost; prediction 1 placed it in the reusable class and fails 0/8. A uniformly random order within both classes loses to recency in 12/12.
- Under `leaf16` with `h*` carried over, the ranker given the class reaches 90% of the label's eviction gain in the same 10/12; the bit at `h*` with recency reaches the label in 9/12 (not 12/12); the ranker's order within the class is a consistent gain over recency in 11/12 (not a loss in 8/12); recency within the class is below 0.9 of the label's eviction gain in 7/12.

Not established:

- **Why the ranker's order in the non-reusable class loses to recency under `all16`, or why the within-class comparison reverses under `leaf16`.** The orphaning and unusable-token counters above do not explain it; no feature, coefficient or state property was examined.
- **The best horizon under `leaf16`,** or whether any horizon works there as well as `h*` does under `all16`.
- **Anything about a policy or a predictor.** Nothing is fitted; every arm reads the trace's future; `h*` is read after the fact from the same traces, and no other trace has been run.
