# Error-location control: which decisions and which errors separate a ranker from its label

Status: pre-registration. No replay of the new arms has been run and no outcome
has been inspected. This plan is committed alone; implementation and tests are
reviewed and committed before the smoke and the full run.

## Question and scope

The [mechanism control](mechanism-control-findings.md) held the score fixed and
found that, through the published sampled mechanism, the exact training label
recovers 74.3–95.7% of the headroom above sampled LRU while the frozen
`next_use` ranker recovers 9.0–22.2%. It has two points between LRU and the
label, so it cannot say which of the ranker's errors cost the utility, and the
[on-policy study](onpolicy-learning-findings.md) left a case where a ranking
statistic improved and utility fell.

Question, fixed before the run: with the mechanism held at the published one,
(a) on which kind of decision — rejecting an arrival or evicting a resident —
does replacing the ranker by its label recover the gap, (b) how does utility
fall as a controlled error is added to the label, and does the placement of an
error matter at an equal error rate, and (c) which decision-level statistic of
a score moves with its utility, for the constructed arms and for the real
rankers the ladder did not include (`pi3`, and the `binary` target)?

Nothing is fitted, no feature or target is added, and no arm is a proposed
policy: every constructed arm reads the trace's future on purpose.

## Fixed surface

- Traces, cells, seeds, evaluation window, L1, L2 store, hit rule, sizes: as
  the mechanism control. Mechanism: `all16` only (the arrival plus up to 16
  uniformly sampled residents; any candidate may leave), the one every learned
  arm of earlier phases ran through. 2 traces × 6 cells × 5 seeds.
- The reference key `K*` of a candidate at a decision is the label rung's key
  `(-log1p(min(next_use_delta_s, 600)), last_group)`; the reference victim of a
  candidate set is the first minimum of `K*` in draw order — what the label
  rung would remove from that set.
- Arms (18), all scored at the decision instant:
  1. Reference rungs: `lru`, `learned` (frozen pi0 `next_use`), `label`,
     `offline`, as in the mechanism control.
  2. Real rankers: `pi3_next_use` (the published third-update model of the
     same trace × cell × seed), `pi0_binary`, `pi3_binary`.
  3. `label_binary`: the exact `binary` target (1 if the state is requested
     within 600 seconds, else 0) as `decisionpop.target_column` defines it, key
     `(label, last_group)`.
  4. Hybrids of the label and the frozen `next_use` ranker by decision type.
     A first-round decision in which the arrival is a candidate is an
     *admission* decision. Under "admission by X, eviction by Y" the arrival is
     rejected if and only if it is the first minimum of X's keys over the
     candidates; otherwise the victim is the first minimum of Y's keys over the
     candidates other than the arrival. Every later round uses Y. With X = Y
     this is the published rule. Arms: `adm_label` (X = label, Y = learned) and
     `evict_label` (X = learned, Y = label).
  5. Noise on the label: `noise_s` for `s` in {0.5, 1, 2, 4}; key
     `(label + s·z, last_group)` with `z` standard normal, a deterministic
     function of (seed, state, decision timestamp): fresh at every timestamp,
     identical when the same state is scored again at the same timestamp. The
     label ranges over [−6.40, 0].
  6. Victim swaps on the label rung: `swap_uniform_p` and `swap_runnerup_p`
     for `p` in {0.25, 0.5}. At each decision with at least two candidates,
     with probability `p` (a deterministic function of seed and decision
     index) the label rung's victim is replaced, by a uniformly drawn other
     candidate, or by the candidate ranked second by `K*` in draw order.
  Noise and swap randomness must not consume the store's sampling RNG.

Grid: 2 × 6 × 5 × 18 = 1,080 replays, each with the Phase 0.98b attribution
hooks (read-only) and the decision statistics below.

## Decision statistics

Computed on every sampled decision whose timestamp lies in the evaluation
window, for the arm's own decisions, against `K*`:

- `m1` pairwise concordance: over candidate pairs with different `K*`, the
  share the arm's key orders the same way (a tie in the arm's key counts one
  half). For a hybrid the arm's key is Y's; for a swap arm it is the label's.
- `m2` victim agreement: the share of decisions whose victim is the reference
  victim.
- `m3` victim label excess: the mean of `label(victim) − min label` over the
  candidates, in label units; zero when the victim has the minimum label.
- `m4` avoidable reusable eviction: the share of decisions whose victim is
  requested within 600 seconds while some candidate is not.

`m2`–`m4` are also reported separately for admission decisions and for
resident-only decisions. These are statistics of each arm on its own decision
population with the `next_use` label as reference. They are not the registered
on-policy ranking statistic, which used a common terminal population and each
target's own label, and they do not re-test it.

## Required checks

- The four reference rungs reproduce the 240 `all16` rows of
  `results/paper/mechanism_control_001/replay_seeds.csv` exactly
  (`avoided_prefill_tokens`); `pi3_next_use`, `pi0_binary` and `pi3_binary`
  reproduce their 180 rows of
  `results/paper/onpolicy_learning/onpolicy_seed_utility.csv` exactly, with
  models loaded by their published SHA-256. Nothing is published otherwise.
- The statistics hook is read-only: an arm replayed with and without it gives
  identical counters (unit test and smoke).
- "Admission by X, eviction by X" is identical to rung X, decision by
  decision, for X = learned and X = label; a swap arm with `p = 0` and a noise
  arm with `s = 0` are identical to `label` (unit tests on constructed traces,
  and the smoke on the real cell).
- For `label`: `m1 = 1`, `m2 = 1`, `m3 = 0`, `m4 = 0` in every replay.
- The request partition and per-block identities of Phase 0.98b hold in every
  replay; `l1_avoided_tokens` and `requested_tokens` are identical across the
  18 arms of each trace × cell × seed.
- Existing replay paths are byte-identical and the existing tests pass
  unchanged.

## Readings, fixed before the run

`U(a)` is extra avoided prefill tokens of arm `a` over L1 alone, in points of
window input tokens. "Consistent" means the same sign in all five seed-paired
values on a trace; a reading that names a cell needs it on both real traces.

1. **Location by decision type.** `G = U(label) − U(learned)`,
   `A = U(adm_label) − U(learned)`, `E = U(evict_label) − U(learned)`, on the
   five-seed means. A trace × cell reads *admission-located* if `A ≥ 0.5·G` and
   `E < 0.5·G`, *eviction-located* if `E ≥ 0.5·G` and `A < 0.5·G`, *both* if
   both reach half, *neither* if neither does. `A + E − G` is reported as the
   interaction, with seed signs of every term.
2. **Dose–response.** `U(noise_s)` per trace × cell: whether the five-seed
   means are non-increasing in `s` (and in how many seeds), and the pair of
   adjacent levels between which the mean crosses `U(learned)`, or "above
   all" / "below all". Descriptive.
3. **Placement at an equal error rate.**
   `U(swap_runnerup_p) − U(swap_uniform_p)`, seed-paired, per `p` and trace ×
   cell, read as consistent gain, consistent loss or mixed, with each arm's
   realised `m2`. Counted out of 12 per `p`.
4. **Which statistic orders utility.** Per trace × cell, the Spearman rank
   correlation over arms between the five-seed mean of each statistic
   (oriented so that larger is better: `m1`, `m2`, `−m3`, `−m4`) and the
   five-seed mean `U`. Primary set: the twelve arms whose victim is the
   minimum of a single key (the four reference rungs, the three real rankers,
   `label_binary`, the four noise arms). Secondary set: all 18. A statistic
   *orders utility* in a cell when the correlation is at least 0.9; the count
   out of 12 is reported per statistic and set.
5. **Real ranker pairs.** For `next_use` and for `binary`, per trace × cell:
   `U(pi3) − U(pi0)` seed-paired with its consistency reading, and the
   seed-paired mean change of each oriented statistic. A statistic *agrees*
   in a cell when its mean change has the sign of the mean utility change;
   the count out of 12 is reported per statistic and target, and every cell
   where utility changes consistently while a statistic moves the other way
   is listed.
6. **The binary target's ceiling, descriptive.** `U(label) − U(label_binary)`
   with seed signs, and `(U(pi0_binary) − U(lru)) / (U(label_binary) − U(lru))`
   per trace × cell.

Interpretation boundaries. The hybrids replace the ranker's decisions by a
future-reading label; a share located on a decision type says where a perfect
score would be worth most, not that a causal predictor can supply it. The
noise arms are one error model (independent Gaussian error in label units);
the equivalent noise level of the learned ranker is a description under that
model and says nothing about the structure of its actual errors. Readings 4
and 5 are rank correlations and sign counts over a small, heterogeneous set of
arms on two traces from one deployment family; a statistic that orders utility
here is a candidate evaluation statistic, not a validated one. Nothing in this
phase triggers a new feature, target, fit or deployed policy.

## Execution order

1. Commit this plan alone.
2. Implement behind the existing hooks (`l2_decision_hook`,
   `l2_override_hook`, a scorer wrapper) with a new runner and tests; no
   change to an existing replay path. Review and commit before any real replay
   of the new arms.
3. Smoke: conversation, 1%×4, seed 0, all 18 arms, in a separate output
   directory, excluded from the confirmatory tables. It checks integrity, the
   identities above, runtime and memory only.
4. Full grid with bounded workers into a new run directory; the runner refuses
   an existing directory and uncommitted execution source and records plan and
   code commits, trace and model hashes, and the reproduction counts.
5. Publish per-seed and aggregated tables, the six readings, the run config
   and figures under `results/paper/error_location_001/`, with a read-only
   tabulation script, and a findings document that states what each table can
   and cannot establish, including null and adverse results.
