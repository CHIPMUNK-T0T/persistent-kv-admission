# Ranker-error diagnosis on saved decision logs: selection or population, and which resident errors

Status: pre-registration. Nothing below has been computed. The diagnosis reads
saved held-out decision logs and published tables; it runs no replay and fits
nothing.

## Question and scope

The [error-location control](error-location-findings.md) found that the mean
label excess of the victim (`m3`) has the sign of the `pi3 − pi0` utility
change in 24/24 trace × cell × target, and that the frozen ranker's cost lies
in which resident it evicts. Two things in those readings are confounded or
unresolved:

1. Each ranker's `m3` was measured on the decisions that ranker itself
   produced, over a window that includes the last 600 seconds of the trace.
   A change of `m3` between `pi0` and `pi3` mixes a change in how the ranker
   chooses with a change in the candidate sets it is shown, and the window
   differs from the label window the on-policy study used.
2. `m3` and `m4` do not say which kind of resident error carries the excess:
   evicting a reusable state when a non-reusable candidate was sampled,
   misordering candidates that are all reusable, or only breaking a tie of
   the minimum label differently from the reference.

Questions, fixed before the computation: (a) on a fixed logged candidate
population, does `pi3` choose victims with lower label excess than `pi0` in
the cells where its utility is higher, and the reverse where it is lower;
(b) does the correspondence between `m3` and utility hold on the on-policy
label window; (c) how are the frozen ranker's decisions, and its label
excess, divided among the three error kinds, and how do its conditional error
rates compare with recency and with a uniformly drawn victim on the same
candidate sets?

## Fixed surface

- Logs: the held-out decision populations of the published on-policy run,
  `results/onpolicy_full_feb30eb_001/populations/test/<trace × cell × target ×
  seed>/pi{0,3}.npz`, loaded by the SHA-256 the published manifest records. Each
  is a reservoir of at most 40,000 complete decisions drawn uniformly from the
  eligible held-out decisions of that policy's own replay (timestamp at or
  after the split and at least 600 seconds before the trace end), with the
  candidates' raw features, exact label primitives, the tie-break the store
  used, the arrival index and the logged victim.
- Scorers: the published `pi0` and `pi3` models of the same trace × cell ×
  target × seed, loaded by SHA-256.
- A scorer's victim on a logged decision is the first minimum, in stored
  candidate order, of `(score on the stored features, stored tie-break)` —
  the published rule. No hybrid, protection or override is applied.
- Labels: `label = −log1p(min(next_use_delta_s, 600))`; a candidate is
  *reusable* when its next use is at most 600 seconds away (the `binary`
  convention). `m3` of a set of decisions is the mean of
  `label(victim) − min label`; `m4` is the share whose victim is reusable
  while some candidate is not.
- 2 traces × 6 cells × 2 targets × 5 seeds. Because the reservoir samples
  whole decisions uniformly, a mean over a population estimates the same mean
  over all eligible decisions of that replay. Runs of consecutive decisions
  cannot be reconstructed from a reservoir and are not estimated.

## Required checks

- Every population and model hash equals the published manifest's.
- A scorer applied to its own population reproduces the logged victim; the
  mismatch count is reported per population, and the diagnosis stops as
  unresolved if it exceeds 0.1% of the decisions of any population.
- The scorer-on-own-population `m3` is computed twice, from the logged victim
  and from the recomputed one, and both are published.
- The published utilities used in reading 2 are matched to the populations on
  trace, cell, target, seed and iteration, with none missing.
- Tests on constructed populations: the four-way matrix, the three error
  kinds as a partition of the decisions, the conditional rates, and the
  expectation of a uniformly drawn victim.

## Readings, fixed before the computation

"Consistent" means the same sign in all five seed-paired values.

1. **Selection on a fixed population.** For each target and trace × cell, and
   for each population `P` in {`pi0`, `pi3`}: `Δsel(P) = m3(P, pi3) − m3(P,
   pi0)`, seed-paired, with its mean and sign reading; and the own-policy
   change `Δown = m3(pi3, pi3) − m3(pi0, pi0)`. With `ΔU` the published
   seed-paired `U(pi3) − U(pi0)` on the full evaluation window, a trace × cell
   reads *carried by selection* when the mean of `−Δsel(P)` has the sign of
   the mean `ΔU` on both populations, *population-dependent* when on exactly
   one, and *not carried by selection* when on neither. Counts out of 12 per
   target. The same table is given for `m4`. These are descriptions of four
   means; no share of a change is attributed to selection or to population.
2. **Window alignment.** With `ΔU_label` the published seed-paired change of
   label-window extra avoided tokens (`label_window_avoided_prefill_tokens`
   minus `label_window_l1_avoided_tokens`) in points of label-window input
   tokens: the number of trace × cell, per target, in which the mean of
   `−Δown` has the sign of the mean `ΔU_label`, out of 12, next to the same
   count against `ΔU`; and every cell in which `ΔU_label` is consistent while
   `−Δown` has the other sign.
3. **Kinds of error of the frozen ranker.** For the `pi0` population of each
   target and trace × cell, each logged decision falls in exactly one class by
   its logged victim: *no excess, reference victim* (the victim is the first
   minimum of `(label, tie-break)`); *no excess, other tie-break* (the victim
   has the minimum label and is not that first minimum); *avoidable reusable
   eviction* (the victim is reusable and some candidate is not); *order error
   among reusable candidates* (every candidate is reusable and the victim's
   label is above the minimum). Reported per class: the share of decisions,
   the share of the population's total label excess, and both again separately
   for decisions whose victim is the arrival and whose victim is a resident.
4. **Conditional rates against two baselines.** On the same `pi0`
   populations: the avoidable-eviction rate over the decisions that contain
   both a reusable and a non-reusable candidate, and the order-error rate
   over the decisions whose candidates are all reusable and not all of one
   label, for three choosers of the victim — the frozen ranker, recency (the
   first minimum of the stored tie-break), and a uniformly drawn candidate
   (computed as an expectation, with no random draw). A trace × cell reads
   *better than recency* on a rate when the ranker's is lower in all five
   seeds, *worse* when higher in all five, else *mixed*; counts out of 12 per
   rate and target.
5. **Concentration, descriptive.** Over the avoidable reusable evictions of
   each `pi0` population: the number of distinct victim states, the share of
   those evictions on states that are such a victim more than once in the
   sample, and the share on the 10% of those states with the most. Sampling
   makes repeat counts lower bounds.

Interpretation boundaries. The logs are held-out decisions of `pi0` and `pi3`
replays under the published mechanism; a scorer applied to another policy's
population is evaluated on states that policy kept, not on a trajectory of its
own. A class share is a share of a statistic, not of lost utility: decisions
differ in what a different victim would have saved. Nothing here triggers a
feature, a target, a fit or a policy.

## Execution order

1. Commit this plan alone.
2. Implement a read-only script with tests on constructed populations; review
   and commit before any saved log is read.
3. Run once from a clean tree into `results/paper/ranker_error_diagnosis_001/`,
   recording plan and code commits and the hashes of every log, model and
   published table read.
4. Report the five readings in a findings document, including every cell
   that goes against the error-location reading.

## Addendum, recorded before the computation (2026-10-02)

Written after the implementation was reviewed and before any saved log was
read; nothing has been computed. The text above is unchanged. This addendum
corrects one statement that cannot hold as written and fixes the readings the
implementation had to choose.

- **The four classes of reading 3 are not disjoint in one case.** A victim
  whose next use is at exactly 600 seconds is reusable (delta ≤ 600 s) and
  carries the clipped label of a candidate with no reuse. With such a
  candidate present the decision has no label excess and is also an avoidable
  reusable eviction, so "exactly one class" fails there. The classes are
  taken in the order listed: such a decision counts in a no-excess class, and
  the number of decisions this applies to is published per class
  (`overlap_decisions`). `m4` keeps its own definition and still counts them.
  Excess shares are unaffected, since these decisions have no excess.
- **Which victim each reading uses.** Readings 1 and 2 use the recomputed
  victim of a scorer for all four entries of the matrix, the own scorer
  included; the same quantities from the logged victim are published beside
  them. Readings 3 to 5 use the logged victim of the `pi0` population.
- **Reading 2** compares the five-seed mean of `−Δown` with the reading of
  `ΔU_label`; a cell is listed when `ΔU_label` is consistent and that mean has
  the other, non-zero sign.
- **Reading 4.** Only the comparison with recency is the registered reading;
  the same rule against the uniform expectation is published as a
  description.
- **Reading 5** counts every avoidable reusable eviction of class 3, including
  decisions whose victim is the arrival, and repeats the three numbers
  separately for resident victims and for arrival victims. "The 10% of those
  states with the most" is the ceiling of 10% of the distinct states.
- **Sign conventions.** "Has the sign of" matches zero with zero only. A
  sign reading of `Δsel` or `Δown` is of the raw difference, so a consistent
  gain means `m3` rose in every seed.
- **Additional checks that stop the run.** The SHA-256 of every published
  table equals the one the published run config records; model paths and
  hashes agree across the published manifests; each population's row,
  decision and eligible counts, window, horizon and reservoir seed equal the
  published table's; `pi0` and `pi3` share requested and L1-avoided tokens on
  both windows.
