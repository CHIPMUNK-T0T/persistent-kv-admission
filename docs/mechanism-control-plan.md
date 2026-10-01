# Mechanism control: a fixed score ladder under four L2 eviction mechanisms

Status: pre-registration. No replay of the new arms has been run and no outcome
has been inspected. This plan is committed alone; implementation and tests are
reviewed and committed before the smoke and the full run.

## Question and scope

The learned L2 arms of Phase 0.97 and later run through one mechanism: an
arriving L1 victim and up to 16 uniformly sampled L2 residents are scored, and
the lowest leaves, whatever its position in the prefix tree. The published
headroom denominator compares that mechanism's LRU with a heap offline-next-use
comparator that scores every resident exactly. The two-tier setting has never
measured what a *perfect* score achieves through the sampled mechanism, so it
cannot say whether the remaining headroom is lost by the predictor, by the
target, or by the mechanism that applies the score.

Question, fixed before the run: with the score held fixed, how much of the
published headroom is lost at each rung of a score ladder (LRU, the frozen
learned ranker, the exact training label, the exact next use), and does that
split change when the mechanism is made tree-native (leaf-only eligibility) or
wider (64 candidates)?

This is the two-tier counterpart of the Phase 0.75 single-tier decomposition
(`docs/predictability-retention-gap.md`: signal, objective and candidate-search
gaps), with eligibility added as a factor. It is the sampling-mechanism control
left unexecuted by `docs/strategy-handoff-20260927.md` (candidate B). It fits
nothing, adds no feature, and promotes no policy: leaf-only eligibility is a
control that corresponds to tree-native engines, not a proposal.

## Fixed surface

- Traces: `conversation_trace`, `toolagent_trace` (SHA-256 recorded before the
  run). Cells: the complete Phase 0.97 grid, L1 fraction in {0.0025, 0.01,
  0.02} x L2/L1 in {1, 4}. Seeds 0..4. Evaluation window: the last 40%.
- L1 heap LRU; exclusive union L2; tree hit rule; packed sizes; 2048
  bytes/token; no arrival protection; everything else as Phase 0.97.
- Score ladder, four rungs, identical under every mechanism:
  1. `lru`: the existing sampled LRU key.
  2. `learned`: the frozen `next_use` pi0 ranker of each trace from
     `results/onpolicy_full_feb30eb_001/models/pi0/`, loaded by its published
     SHA-256 (the Phase 0.97 A_none fit); key `(score, last_group)` as before.
  3. `label`: the exact training target at the decision instant,
     `-log1p(min(next_use_delta_s, 600))` with occurrences at the current
     timestamp excluded, as `decisionpop.target_column` defines it; key
     `(label, last_group)`. A perfect predictor of what `learned` was fitted to.
  4. `offline`: the heap offline comparator's own key
     `(-next_occurrence_group, prefix_tokens)` ("prefix_first"), evaluated at
     the decision on the sampled path.
- Mechanisms, 2 x 2:
  - eligibility `all` (published): the arrival plus a uniform sample of the
    other residents; any candidate may leave.
  - eligibility `leaf`: a resident is a leaf when none of its direct children
    is in L2. Only leaves are candidates. In the first round the arrival is a
    candidate if and only if it is a leaf; the other candidates are a uniform
    sample without replacement of up to K of the other leaf residents, in L2
    insertion order. Later rounds sample up to K leaves, the arrival included
    if it is one. Leaf status is re-evaluated every round. The lowest key
    leaves; the first minimum in draw order wins; an arrival that loses its own
    first round is a rejection, as before. An overflow with no leaf is an
    invariant violation, not a fallback.
  - width K in {16, 64}.
  The base mechanism `(all, 16)` must follow the published code path, RNG
  consumption included.
- References, not rerun: the published per-seed heap `offline_next_use` and
  heap `lru` rows of `decision_population_replay_seeds.csv`, used only after
  trace, cell, capacity, requested-token and L1-token identifiers match.

Grid: 2 traces x 6 cells x 5 seeds x 4 mechanisms x 4 rungs = 960 replays. Each
carries the Phase 0.98b attribution hooks (read-only).

## Required checks

- `(all, 16)` `lru` reproduces the 60 published `lru_s` rows and `(all, 16)`
  `learned` the 60 published `A_none` / `next_use` rows exactly
  (`avoided_prefill_tokens`), before anything is published.
- Every `leaf` replay has zero present-but-unusable tokens and blocks; a
  non-zero count aborts the run. (L1 is prefix-closed and evicts leaves first,
  so a child reaches L2 before its parent; under leaf eligibility a parent
  cannot leave L2 while a child is resident.)
- The request partition and per-block identities of Phase 0.98b hold in every
  replay; `l1_avoided_tokens` and `requested_tokens` are identical across the
  16 arms of each trace x cell x seed.
- With eligibility and the new rungs absent, every existing replay path is
  byte-identical; the existing 244 tests pass unchanged.
- Unit tests on constructed traces: leaf set maintenance under admit, evict,
  reject and promote; an arrival with a resident child cannot be rejected under
  `leaf`; a parent becomes eligible after its last resident child leaves; no
  orphan under `leaf` on a real-shaped replay; `label` equals
  `target_column` at the horizon boundary and excludes same-timestamp
  occurrences; `offline` on the sampled path uses the heap comparator's key;
  `(all, 16)` unchanged.

## Outcomes

`U_m(s)` is extra avoided prefill tokens over L1 alone as a share of window
input tokens, for mechanism `m` and rung `s`, per trace x cell x seed. `H_off`
and `H_lru` are the heap references. For every mechanism the published
headroom scale is `T_m = H_off - U_m(lru)` and the identity

`T_m = [H_off - U_m(offline)] + [U_m(offline) - U_m(label)] + [U_m(label) - U_m(learned)] + [U_m(learned) - U_m(lru)]`

splits it into the candidate-search gap, the objective gap, the signal gap and
the achieved part, the Phase 0.75 names. All four terms are reported in
input-token points and as shares of `T_m`, per seed, with five-seed mean, range
and 95% t interval (seed variability only), next to rejections, resident
evictions, present-but-unusable tokens and the per-block rejected / evicted
charge of every arm.

## Readings, fixed before the run

"Consistent" means the same sign in all five seed-paired values on a trace; a
reading that names a cell needs it on both real traces.

1. **Dominant gap under the published mechanism `(all, 16)`**, per cell: the
   gap holding at least half of `T`, else "mixed". A cell reads
   mechanism-bound, objective-bound or signal-bound accordingly.
2. **Mechanism effect on each rung**: `U_(leaf,16)(s) - U_(all,16)(s)` and
   `U_(e,64)(s) - U_(e,16)(s)` for every rung `s`, read as consistent gain,
   consistent loss or mixed, per cell.
3. **Ladder order**: a trace x cell x mechanism is "ordered" when
   `lru <= learned <= label <= offline` holds in all five seeds. Reported as
   the number of ordered cells out of 12 per mechanism, with every inversion
   listed (which adjacent pair, how many seeds). If an inversion present under
   `(all, 16)` disappears under a leaf mechanism in the same cell on both
   traces, that inversion is read as mechanism-borne; if it persists under all
   four mechanisms, as a property of the score in that cell.
4. **Where the learned arm stands once the mechanism is controlled**: the
   signal gap and the achieved part under `(leaf, 64)`, per cell, with the
   same dominant-gap rule as reading 1.
5. **Capacity pattern, descriptive**: the sign of `U(learned) - U(lru)` and
   `U(label) - U(lru)` by cell under each mechanism, and whether any sign
   change across capacity present under `(all, 16)` persists under `leaf`.
   No verdict is attached; it informs candidate A of the handoff.

Interpretation boundaries. The candidate-search gap is measured against a
greedy heap comparator that is not a proved optimum and is not tree-native, so
it can be negative. A reading of "mechanism-bound" says a perfect next-use
score loses that share through the mechanism; it does not say a different
mechanism recovers it unless reading 2 shows the gain. A small signal gap does
not show the features are sufficient in general, and a large one does not show
history is inadequate: one linear ranker, one target, two traces from one
deployment family, 59 minutes, 512-token blocks. `K = 64` is a width control;
a monotone width effect alone is not evidence about prediction. Nothing in
this phase triggers a new feature, target, fit or deployed policy; a further
experiment needs its own recorded design.

## Execution order

1. Commit this plan alone.
2. Implement in the existing simulator behind defaults that leave every
   published path unchanged (`l2_eligibility="all"`), plus a new runner and
   tests. Review and commit before any real replay of the new arms.
3. Smoke: conversation, 1% x 4, seed 0, all 16 arms, in a separate output
   directory, excluded from the confirmatory tables. It checks integrity,
   runtime and memory only; no reading or threshold changes after it.
4. Full grid with bounded workers into a new run directory; the runner refuses
   an existing directory and uncommitted execution source, and records plan
   and code commits, trace and model hashes, and the reference reproduction
   counts. Nothing is published if a required check fails.
5. Publish per-seed and aggregated tables, the decomposition, the ladder
   table, the run config and two figures under
   `results/paper/mechanism_control_001/`, and a findings document that states
   what each figure can and cannot establish, including null and adverse
   results.
