# Horizon fill-in: is there a reuse horizon between 60 and 300 seconds where none of five sufficed

Status: pre-registration of a follow-up, written **after** the results of the
[horizon and class-order control](horizon-control-plan.md) were inspected and
because of them. No replay at a horizon named below has been run. This plan is
committed alone; the implementation is reviewed and committed before the run.

## Why a follow-up, and what was already seen

The horizon control ran `label_binary_h` for `h` in {6, 15, 60, 300, 600}
seconds (code `9333d5c`, tables in `results/paper/horizon_control_001/`, not
yet committed when this plan was written). Its registered reading 1 is not
revised by anything here: a reuse label suffices at 8/12 trace × cell and
order is needed at 4/12.

The four "order needed" trace × cell are 0.25%×4 and 1%×1 on both traces.
They share one L2 capacity (1% of the capacity base). Their shortfall `S_h` at
the two horizons that bracket a gap in the grid:

| trace × cell | `S_60` | `S_300` | `S_600` |
|---|---:|---:|---:|
| conversation 0.25%×4 | 0.408 | 0.296 | 0.461 |
| conversation 1%×1 | 0.367 | 0.215 | 0.385 |
| tool-agent 0.25%×4 | 0.377 | 0.312 | 0.452 |
| tool-agent 1%×1 | 0.348 | 0.226 | 0.367 |

A post hoc observation, not a registered reading: the horizon that minimises
`S_h` on that grid grows with L2 capacity — 60 seconds at 0.25% of the base
(`S` 0.03 and 0.00), 300 seconds at 2% (−0.04 and −0.03), 600 seconds at 4%
and 8% (0.00) — and the grid has no point between 60 and 300 seconds, where
that pattern would put the 1% cells.

## Question and prediction

At the four trace × cell above, with the mechanism and the recency tie-break
unchanged, is there a horizon strictly between 60 and 300 seconds at which
the exact reuse label reaches the exact `next_use` label?

Prediction, fixed before the run: **yes at all four** — in each, at least one
of the five horizons below has `S_h ≤ 0.10`. If a trace × cell fails it,
"order needed" stands there on a ten-point grid and the capacity pattern is
not supported at that capacity.

## Fixed surface

- Traces, capacities, seeds, evaluation window, L1, L2 store, hit rule, sizes
  and mechanism (`all16`): as Part 1 of the horizon control.
- Arms `label_binary_h`: key `(1 if the state's next use is at most h seconds
  away else 0, last_group)`, built exactly as in the horizon control.
- Fill horizons: **90, 120, 150, 180, 240 seconds**, fixed here as multiples
  of 30 seconds on the 3-second timestamp grid; none is added, dropped or
  moved after the run.
- Rerun for a self-contained curve and as checks: `h` = 60, 300 and 600.
- Trace × cell: {conversation, tool-agent} × {0.25%×4, 1%×1}; seeds 0–4.
- References, published rows and not rerun: `label` and `lru` under `all16`
  from `results/paper/error_location_001/replay_seeds.csv`.

Replays: 8 horizons × 4 trace × cell × 5 seeds = 160, each with the attribution
hooks and decision statistics of the horizon control.

## Required checks

- `label_binary_600` reproduces the 20 published `label_binary` rows of
  `results/paper/error_location_001/replay_seeds.csv` exactly
  (`avoided_prefill_tokens`). Nothing is published otherwise.
- When the first run's directory is given, the 60 rows at `h` = 60, 300 and
  600 equal its rows of the same trace × cell × seed in
  `avoided_prefill_tokens` and in the replay's counter digest; the directory's
  table hash is recorded. A difference stops publication.
- `l1_avoided_tokens`, `requested_tokens`, the capacities and the
  compulsory-absent charge of every replay equal those of the published
  reference rows of its trace × cell × seed.
- The request partition and per-block identities of Phase 0.98b hold in every
  replay; no arm has an override, so no decision is overridden.
- No existing replay path or runner is changed; the existing tests pass.

## Readings, fixed before the run

`S_h = (U(label) − U(label_binary_h)) / (U(label) − U(lru))` on the five-seed
means, with the seed signs of `U(label) − U(label_binary_h)`, as in the
horizon control. Every horizon's value is reported.

1. **Fill.** A trace × cell reads *a reuse label suffices at a filled horizon*
   when the smallest `S_h` over the five fill horizons is at most 0.10
   (exactly 0.10 suffices), and *order needed stands* otherwise; every horizon
   attaining the minimum is listed. Count out of 4, against the prediction of
   4/4.
2. **Shape, descriptive.** Over the seven horizons from 60 to 300 seconds:
   whether `S_h` is unimodal (non-increasing up to a minimiser and
   non-decreasing after it), and whether the two cells of a trace, which share
   an L2 capacity and differ in L1, have the same minimising horizon.
3. **Sensitivity, descriptive.** The width of the set of the seven horizons
   with `S_h ≤ 0.10`, in seconds from its smallest to its largest member,
   reported as empty when no horizon reaches 0.10.

## Interpretation boundaries

- This is a second look at four trace × cell chosen because they failed the
  first one. They are judged on ten horizons where the other eight were judged
  on five, so passing is easier here than under the first plan. The two counts
  are reported side by side and never merged: "8/12 on the registered grid",
  and "n/4 of the remaining cells at a horizon added afterwards".
- A horizon found by filling a grid after the run describes that grid. It does
  not show that the horizon can be chosen before the fact, predicted from
  capacity, or learned from features; the capacity pattern stays a post hoc
  observation on five capacities of two traces.
- Seed intervals describe sampling-seed variability only.
- Every arm reads the trace's future on purpose and none is a policy. Nothing
  here triggers a feature, a target, a fit or a deployed policy.

## Execution order

1. Commit this plan alone.
2. Implement a separate runner on the horizon control's arm construction and
   replay call, with tests on constructed traces; no change to an existing
   replay path or runner. Review and commit before any replay.
3. No separate smoke: the arm is the one the horizon control ran, with another
   parameter, and the run carries its own reproduction checks, without which
   nothing is published.
4. Run once with 10 workers from a tree whose execution source is the code
   commit, into `results/paper/horizon_fill_001/`, recording plan and code
   commits and the hashes of the traces and of every table read.
5. Report the three readings next to the horizon control's, including a
   failed prediction.
