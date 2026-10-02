# Horizon and class-order control: is a reuse label enough, at which horizon, and with whose order

Status: pre-registration. No replay of the new arms has been run and no outcome
has been inspected. This plan is committed alone; implementation and tests are
reviewed and committed before the smoke and the full run.

## Question and scope

The [error-location control](error-location-findings.md) found that the exact
`binary` label (requested within 600 seconds or not, ties by recency) equals
the exact `next_use` label at 1%×4 and 2%×4 and falls short of it elsewhere,
by up to 82% of the label's gain over sampled LRU. One reading is that small
capacities need the order among reusable states. A rival reading is that 600
seconds is the wrong boundary there: the horizon was fixed for fitting, and a
tier that turns over in seconds may need only a reuse label at a shorter
horizon. [LRB](https://www.usenix.org/conference/nsdi20/presentation/song)
treats every candidate whose next use lies beyond a cache-dependent boundary
as an equally good victim, so the rival reading has precedent.

The same control located the frozen ranker's cost on resident eviction under
`all16` only, and did not separate two things a ranker must do there: tell
reusable residents from the rest, and order residents within each class.

Questions, fixed before the run:

1. With the mechanism and the recency tie-break held fixed, does an exact
   reuse label at some shorter horizon reach the `next_use` label where the
   600-second one does not?
2. With the arrival handled by the frozen ranker as published, and resident
   eviction given the exact 600-second reuse label, does ordering residents
   within a class by the ranker's score, or by recency, recover what the
   exact `next_use` label recovers on eviction?
3. Is the location of the gap on resident eviction the same under leaf-only
   eligibility?

Nothing is fitted and no arm is a proposed policy: every constructed arm reads
the trace's future on purpose.

## Fixed surface

- Traces, cells, seeds, evaluation window, L1, L2 store, hit rule, sizes: as
  the error-location control. 2 traces × 6 cells × 5 seeds.
- Part 1, mechanism `all16`. Arms `label_binary_h` for `h` in {6, 15, 60, 300,
  600} seconds: key `(1 if the state's next use is at most h seconds away else
  0, last_group)`, the exact `binary` target at horizon `h` as
  `decisionpop.target_column` defines it. `h = 600` is the published
  `label_binary` arm, rerun through the new parameter. The grid spans the
  time an admit-all tier of these capacities takes to turn over (about 6 to
  200 seconds at the observed offer rate) and the 3-second timestamp grid.
- Part 2, mechanism `all16`. Two arms that keep the published admission rule
  on the frozen `next_use` ranker's keys — the arrival is rejected if and only
  if it is the first minimum of the ranker's keys over the candidates — and
  otherwise evict the first minimum, over the candidates other than the
  arrival, of a composite key; every later round uses the composite key:
  - `evict_binary_learned`: `(reusable within 600 s, ranker score, last_group)`;
  - `evict_binary_recency`: `(reusable within 600 s, last_group)`.
- Part 3, mechanism `leaf16` (the arrival plus up to 16 sampled leaf
  residents; a candidate with a cached child may not leave, as in the
  mechanism control). Arms `adm_label` and `evict_label` as the error-location
  control defined them.
- References are published rows, matched on trace, cell, seed and the
  arm-independent identifiers, not rerun: `lru`, `learned`, `label`,
  `evict_label` under `all16` from `results/paper/error_location_001/`, and
  `learned`, `label` under `leaf16` from `results/paper/mechanism_control_001/`.

New replays: (5 + 2 + 2) arms × 60 = 540, each with the Phase 0.98b
attribution hooks and the decision statistics of the error-location control.

## Required checks

- `label_binary_600` reproduces the 60 published `label_binary` rows of
  `results/paper/error_location_001/replay_seeds.csv` exactly
  (`avoided_prefill_tokens`). Nothing is published otherwise.
- `l1_avoided_tokens`, `requested_tokens`, the capacities and the
  compulsory-absent charge of every new replay equal those of the published
  reference rows of its trace × cell × seed; under `leaf16` every replay has
  zero present-but-unusable tokens.
- Identities, in unit tests on constructed traces and in the smoke on the real
  cell, decision by decision: with admission by the exact 600-second reuse
  label, `evict_binary_recency` is the `label_binary_600` rung; under `leaf16`,
  "admission by X, eviction by X" is rung X for X = learned and X = label.
- The statistics hook is read-only (an arm replayed with and without it gives
  identical counters), and the request partition and per-block identities of
  Phase 0.98b hold in every replay.
- Existing replay paths are byte-identical and the existing tests pass
  unchanged.

## Readings, fixed before the run

`U(a)` is extra avoided prefill tokens of arm `a` over L1 alone, in points of
window input tokens. "Consistent" means the same sign in all five seed-paired
values. Every value of every registered arm is reported, not only the best.

1. **Horizon.** Per trace × cell and horizon,
   `S_h = (U(label) − U(label_binary_h)) / (U(label) − U(lru))` on the
   five-seed means, with the seed signs of `U(label) − U(label_binary_h)`. A
   trace × cell reads *a reuse label suffices* when the smallest `S_h` over
   the five horizons is at most 0.10, and *order needed* otherwise; the
   horizon attaining the minimum is reported beside it. Counts out of 12, and
   separately over the eight trace × cell where the published `S_600` exceeds
   0.05.
2. **Class order.** With `R(a) = (U(a) − U(learned)) / (U(evict_label) −
   U(learned))` on the five-seed means: a trace × cell reads *reuse
   identification suffices with the ranker's order* when
   `R(evict_binary_learned) ≥ 0.9`, and *the ranker's order costs* otherwise.
   `U(evict_binary_learned) − U(evict_binary_recency)` is read seed-paired as
   a consistent gain, consistent loss or mixed. `R(evict_binary_recency)` is
   reported beside both. Counts out of 12.
3. **Location under leaf eligibility.** `G`, `A`, `E` and `A + E − G` under
   `leaf16` with the rule of the error-location control's reading 1
   (admission-located, eviction-located, both, neither), next to the
   published `all16` label of the same trace × cell; the count of trace × cell
   with the same label, out of 12.

Interpretation boundaries. A reuse label that suffices at a horizon chosen
from five after the run is a description of that grid; it does not show that
the horizon can be chosen before the fact, or predicted. `R` compares whole
replays whose trajectories differ. Part 3 changes eligibility only; it is not
SGLang's eviction policy. Nothing in this phase triggers a feature, a target,
a fit or a deployed policy.

## Execution order

1. Commit this plan alone.
2. Implement in a new module and runner on the existing hooks, with tests; no
   change to an existing replay path. Review and commit before any real
   replay of the new arms.
3. Smoke: conversation, 1%×4, seed 0, all nine arms plus the identity arms,
   in a separate output directory, excluded from the confirmatory tables.
4. Full grid with at most 12 workers into a new run directory; the runner
   refuses an existing directory and uncommitted execution source and records
   plan and code commits, trace, model and reference hashes.
5. Publish per-seed and aggregated tables, the three readings, the run config
   and a figure under `results/paper/horizon_control_001/`, with a read-only
   tabulation script and a findings document that states what each table can
   and cannot establish, including null and adverse results.

## Addendum, recorded before the smoke (2026-10-02)

Written after the implementation was reviewed and before any real replay of
the new arms; no outcome has been inspected. The text above is unchanged. This
addendum records what the implementation had to fix.

- **Leaf eligibility and the arrival.** Under `leaf16` the arrival is a
  candidate of its first round only when it is a leaf. An arrival with a
  cached child is not among the candidates, cannot be rejected, and the round
  evicts by the eviction score; "admission by X" therefore judges leaf
  arrivals only. This is the mechanism as the mechanism control ran it, and
  Part 3 is read with that restriction stated. No counter is added for it.
- **Leaf identities.** The X/X hybrids under `leaf16` are compared with the
  `leaf16` rungs decision by decision in the smoke, where the two rungs are
  replayed and must also equal their published rows; the grid stays at 540
  replays and uses the published rows as references.
- **Composite key.** `evict_binary_learned` gives the store the score pair
  `(reusable, ranker score)`; with the store's `last_group` appended it orders
  as `(reusable, ranker score, last_group)`. The ranker is one object in both
  roles and observes each timestamp group once. `m1` of the two class-order
  arms is computed on the store's composite key, as for the hybrids of the
  error-location control.
- **Thresholds and ties.** `S_h ≤ 0.10` suffices, `R ≥ 0.9` suffices, and the
  separately counted subset is published `S_600 > 0.05`, computed by the
  runner from the published rows. Horizons that attain the minimum exactly
  are all listed. A zero denominator gives no value and falls on the "order
  needed" or "the ranker's order costs" side.
- **Further checks.** An arm without an override, and every identity arm,
  must show zero overridden decisions; the smoke's `leaf16` label replays must
  satisfy the label identities of the error-location addendum; the published
  `all16` location labels are checked against the rule applied to their own
  published `G`, `A`, `E`.
- **Smoke.** The nine arms with and without the statistics hook, the three
  identity arms, and the `leaf16` `learned` and `label` rungs: 23 replays.
