# Working-set ratio check: the capacity reversals follow a known working-set argument

This [pre-registered check](working-set-ratio-plan.md) asks whether the ratio of EfficientAgent ([arXiv 2609.33762](https://arxiv.org/abs/2609.33762)) — a reuse working set over the tier's capacity — places the published sign changes with capacity where they occur. It does, for the two outcomes the published tables cover at every cell: 2-hit admission against LRU changes sign where the ratio crosses 1, in **12/12** trace × cell for the heap pair and **12/12** for the sampled pair. For arrival protection the tables cover 6 of the 12 cells; the ratio predicts 4 of the 6, and both misses are mixed seed readings at ratios of 1.005 and 1.179. By the rule fixed in the plan, the capacity dependence of admission filtering on these two traces is an instance of that explanation and is not a separate contribution of this repository.

## Run and integrity

The plan was committed alone (`19901b5`). Its addendum (`1e48bab`), written before any real trace was processed, corrects one factual error — arrival protection `all − none` is published at three cells per trace, not six — and fixes the readings the implementation had to choose. The implementation and tests were committed next (`f8a1fcf`), and the check ran once from a clean checkout of that commit into [`results/paper/working_set_ratio_001/`](../results/paper/working_set_ratio_001/README.md) (7 seconds, 6 workers, peak 671 MiB per worker).

The run is arithmetic on the L1 offer stream and on published tables. It runs no L2 policy and fits nothing. Before deriving anything it checked, and the [run config](../results/paper/working_set_ratio_001/run_config.json) records:

- The recorded offer stream of each trace × L1 fraction has as many offers as every published replay of that fraction has L1 evictions (`decision_population_replay_seeds.csv`) and as the `all16` rows of the mechanism control have admissions plus rejections; L1 and L2 capacities, requested tokens and L1-avoided tokens equal the published values (0 problems).
- `D + size ≤ C` agrees with a direct simulation of the reference tier for every returning offer at both L2 capacities (0 violations over 76,296–93,496 returning offers per stream).

As a separate check outside the run, a plain FIFO admit-all tier written independently of the module holds 49.998% of the window-returning offer tokens at `C = W − 1` byte and 50.001% at `C = W` (conversation 1%; 49.999% and 50.002% for tool-agent 2%), which is what a token-weighted median of `D + size` means.

To rerun: check out `f8a1fcf` with the ignored traces in place and run

```bash
.venv/bin/python scripts/run_working_set_ratio.py data/raw/conversation_trace.jsonl \
  data/raw/toolagent_trace.jsonl --output-dir <new directory> --workers 6
```

Every number below is a column of the published CSVs. Effects are in points of window input tokens; "consistent" means the same sign in all five seed-paired values.

## The working set and the ratio

`W(f)` is the token-weighted median of `D + size` over the offers whose return falls in the evaluation window. As a fraction of the capacity base (the trace's packed working set, of which the L1 and L2 fractions are taken):

| trace | L1 fraction | offers | returning | returning in window | q25 | `W` (median) | q75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conversation | 0.25% | 275,832 | 93,496 | 41,122 | 2.82% | 5.17% | 11.18% |
| conversation | 1% | 271,289 | 90,300 | 40,034 | 2.21% | 4.72% | 11.01% |
| conversation | 2% | 261,582 | 82,422 | 36,996 | 1.69% | 4.42% | 10.95% |
| tool-agent | 0.25% | 271,080 | 88,232 | 37,813 | 2.55% | 4.54% | 8.78% |
| tool-agent | 1% | 265,705 | 84,184 | 36,343 | 1.97% | 4.02% | 8.38% |
| tool-agent | 2% | 255,969 | 76,296 | 33,210 | 1.44% | 3.39% | 8.04% |

`W` is 3.4–5.2% of the base and moves little with the L1 fraction, while `C_L2` spans 0.25–8%. `γ = W / C_L2` therefore falls with L2 capacity:

| cell | conversation `γ` | tool-agent `γ` |
|---|---:|---:|
| 0.25%×1 | 20.69 | 18.16 |
| 0.25%×4 | 5.17 | 4.54 |
| 1%×1 | 4.72 | 4.02 |
| 1%×4 | 1.179 | 1.005 |
| 2%×1 | 2.21 | 1.69 |
| 2%×4 | 0.553 | 0.424 |

Only 2%×4 has `γ ≤ 1`, on both traces. Tool-agent 1%×4 sits 0.5% above the threshold.

## Readings 1 and 2 — agreement and verdicts

| outcome | cells compared | agreement | misses | verdict |
|---|---:|---:|---|---|
| 1. heap `lru_2hit − lru` | 12 | **12/12** | none | located |
| 2. sampled `lru_2hit_s − lru_s` | 12 | **12/12** | none | located |
| 3. protection `all − none`, `next_use` (primary) | 6 | 4/6 | conversation 1%×4 (`γ` = 1.179, seeds `+−−−+`), tool-agent 1%×4 (`γ` = 1.005, seeds `+++−−`) | 12-cell rule not computable; on the 6 published cells, located outside the transition |
| 3. protection `all − none`, `binary` (secondary) | 6 | 6/6 | none | 12-cell rule not computable; on the 6 published cells, located |

Outcomes 1 and 2. 2-hit admission gains over LRU in the ten cells with `γ > 1` (heap +0.32 to +3.55, sampled +0.25 to +2.89, the sampled pair consistent in every cell) and loses at 2%×4 where `γ ≤ 1` (heap −4.21 and −3.37; sampled −2.68 and −2.26, consistent). The sign follows the rule in every cell.

Outcome 3. Arrival protection forbids declining a write, so the rule predicts a loss where `γ > 1` and a gain where `γ ≤ 1`. On the `next_use` target it loses consistently at 0.25%×1 (−0.19, −0.12; `γ` ≈ 18–21), gains consistently at 2%×4 (+1.52, +0.60; `γ` ≈ 0.4–0.6), and is mixed at 1%×4 (means −0.07 and +0.02), where `γ` is 1.18 and 1.005. A mixed reading counts as a miss under the plan; both misses lie inside the transition band `0.5 ≤ γ ≤ 2`. On the `binary` target the 1%×4 cells are consistent losses (−0.17, −0.15) and all six cells agree. The plan's verdict needs twelve cells and cannot be computed for this outcome; the verdicts on six cells are given with that denominator and carry correspondingly less weight.

## Reading 3 — thresholds that maximise agreement

| outcome | maximum agreement | agreement at 1 | maximising interval on `γ` |
|---|---:|---:|---|
| 1 | 12 | 12 | [0.553, 1.005) |
| 2 | 12 | 12 | [0.553, 1.005) |
| 3, `next_use` | 4 of 6 | 4 | [0.553, 18.16) |
| 3, `binary` | 6 of 6 | 6 | [0.553, 1.005) |

For outcomes 1 and 2 every threshold between 0.553 and 1.005 gives 12/12, and no threshold outside it does. The pre-registered threshold of 1 lies inside that interval, 0.5% below its upper end. The sign pattern is monotone in `γ`. For outcome 3 on `next_use` no threshold exceeds 4 of 6, because the two 1%×4 cells are mixed and match neither sign.

## Reading 4 — declinable share, descriptive

The share of bytes offered in the evaluation window whose offer never returns or has `D + size > C_L2` — the writes Proposition 1 of the source says an LRU tier can decline without losing a hit — is 76.6–99.9%. The never-returning part is 68.8–70.8% in every cell; the beyond-capacity part falls from 30.5–30.8% at 0.25%×1 to 6.1–7.3% at 2%×4.

| cell | declinable (conv / tool) | `label` rejection share | `learned` rejection share |
|---|---:|---:|---:|
| 0.25%×1 | 99.9% / 99.6% | 65.8% / 63.5% | 78.6% / 74.9% |
| 0.25%×4 | 98.6% / 98.3% | 31.9% / 29.3% | 64.5% / 60.8% |
| 1%×1 | 97.2% / 97.0% | 28.1% / 25.4% | 52.1% / 47.4% |
| 1%×4 | 84.3% / 83.0% | 0.1% / 0.0% | 0.9% / 1.2% |
| 2%×1 | 89.9% / 88.8% | 7.6% / 5.6% | 25.9% / 23.3% |
| 2%×4 | 78.1% / 76.6% | 0.0% / 0.0% | 0.0% / 0.0% |

The rejection shares are of offers over the whole trace under `all16`; the count-based declinable shares on the same footing (`declinable.csv`) differ from the byte shares by at most 1.3 points. Both rungs reject far fewer offers than are declinable, and reject almost none at 1%×4 and 2%×4. Under the sampled mechanism an arrival is rejected only when it loses its own first round, so a rung that admits an offer and evicts it later disposes of it without a rejection; the rejection share is not an estimate of the declinable share. Offers made late in the trace that would return after its end count as never returning, so the never-returning part is an upper bound.

## Observations outside the four readings

- **The grid has one sign change per trace.** `γ` falls with `C_L2` and only 2%×4 lies at or below 1. The test is whether the sign changes between the cells the ratio places on either side of 1, and it does; the grid has no second crossing against which to test the ratio, and three L1 fractions give three values of `W` per trace.
- **The scale is located to within the grid's resolution.** The effect is still clearly positive just above 1 (tool-agent 1%×4, `γ` = 1.005: heap +0.57, sampled +1.33) and clearly negative at 0.42–0.55. The zero crossing lies somewhere between `γ` = 0.553 and 1.005, so a scale off by up to a factor of 1.8 would give the same 12/12.
- **The verdict depends on the pre-registered quantile.** With the 25th percentile in place of the median, `γ ≤ 1` also at 1%×4 and 2%×1 and outcomes 1 and 2 would agree in 8/12; with the 75th percentile `γ > 1` everywhere (tool-agent 2%×4 at 1.005) and they would agree in 10/12 (`gamma_q25`, `gamma_q75` in `ratio.csv`). The median was fixed before the computation and is the only one read as a verdict.
- **The size of the effect is not ordered by `γ`.** At `γ` ≈ 18–21 (0.25%×1) the gain is +0.25 to +0.38; at `γ` ≈ 4–5 it is +1.8 to +3.6. The rule predicts a sign, and only the sign is tested.

## What this establishes and what it does not

Established, on the two Mooncake traces and the six published cells:

- The capacity at which 2-hit admission turns from a gain into a loss, for the heap and for the sampled pair, is where a trace-derived reuse working set equals the L2 capacity, by a definition and threshold fixed before the ratio was computed (12/12 each).
- Arrival protection changes sign in the direction the same rule predicts at the two ends of the published capacity range, and is mixed where the ratio is within 18% of 1.
- Under the plan's rule, the capacity reversal listed as candidate A in the [handoff](strategy-handoff-20260927.md) is accounted for by a working-set argument that existing work already makes (EfficientAgent Sections 4 and 5.5; the concurrency analysis in Appendix B of [arXiv 2609.28870](https://arxiv.org/abs/2609.28870) gives the same account in terms of stack distance). It is not claimed here as a new explanation.

Not established:

- **EfficientAgent's system or its closed form.** Its tier, workload and write path differ; the ratio here uses a trace-based stack distance in place of `(A − 1)·N̄·β`, an offer-ordered reference tier and no tree hit rule.
- **A threshold of exactly 1.** The grid brackets the crossing between 0.553 and 1.005.
- **Arrival protection at the six unpublished cells,** or a twelve-cell verdict for it.
- **Anything about learned scores.** The outcomes compared are LRU, 2-hit admission and arrival protection; whether the error that matters for a learned ranker changes with `γ` is the question of the [error-location control](error-location-plan.md), not of this check.
- **Generality.** Two traces from one deployment family, agreement counts over twelve cells, one crossing per trace.
