# Matched-horizon diagnosis on saved decision logs: the frozen ranker's errors at the reuse boundary that works for the cell

Status: pre-registration. Nothing below has been computed. The diagnosis
re-reads the saved held-out decision logs that the
[ranker-error diagnosis](ranker-error-diagnosis-plan.md) read; it runs no
replay and fits nothing. This plan is committed before any implementation, in
one commit with the [matched class-order control](matched-class-order-plan.md);
the implementation is reviewed and committed before the computation.

## Why, and what was already seen

The ranker-error diagnosis ([findings](ranker-error-diagnosis-findings.md))
read the frozen `pi0` ranker's errors with the reuse boundary at 600 seconds:
at least 99.3% of its label excess comes from evicting a state that is
requested within 600 seconds while a sampled candidate is not, and on its own
candidate sets the `next_use` ranker makes that error less often than recency
in 9/12 trace × cell and more often in the three smallest conversation cells.

The [horizon control and its fill-in](horizon-control-findings.md) then found
that the horizon at which an exact reuse bit with recency reaches the exact
`next_use` label differs by L2 capacity, and that 600 seconds is the wrong
boundary in six trace × cell. Published values, conversation / tool-agent:

| cell | horizon that works, `h*` | `S_h*` | `S_600` | `R(evict_binary_learned)` at 600 s |
|---|---:|---:|---:|---:|
| 0.25%×1 | 60 s | 0.028 / 0.001 | 0.822 / 0.824 | 0.350 / 0.330 |
| 0.25%×4 | 150 s | −0.035 / −0.028 | 0.461 / 0.452 | 0.605 / 0.558 |
| 1%×1 | 150 s | −0.036 / −0.036 | 0.385 / 0.367 | 0.470 / 0.450 |
| 1%×4 | 600 s | −0.001 / 0.000 | −0.001 / 0.000 | 0.998 / 0.987 |
| 2%×1 | 300 s | −0.038 / −0.031 | 0.097 / 0.062 | 0.938 / 0.956 |
| 2%×4 | 600 s | 0.001 / −0.003 | 0.001 / −0.003 | 1.043 / 1.042 |

`h*` is the horizon of the grids run ({6, 15, 60, 300, 600} s everywhere, and
{90, 120, 150, 180, 240} s on the 1% cells) that minimises `S_h`, read after
those runs; it is the same on both traces.

So two findings stand side by side without a bridge: the ranker's errors
were counted at the 600-second boundary, and the boundary that works for the
cell is 60, 150 or 300 seconds in eight of the twelve trace × cell. Whether
the frozen ranker fails to tell the classes apart at the boundary that works
for its cell has not been read. Every class, share and rate of the diagnosis
is defined at 600 seconds only.

## Question

On the frozen ranker's own logged decisions, with the reuse boundary at the
horizon that works for the cell: how are its decisions and its label excess
divided among the error kinds; how often does it evict a state reusable
within that horizon while a candidate is not, compared with recency and a
uniformly drawn victim on the same candidate sets; and how well does its
score separate the two classes within a decision? The same quantities are
read at every horizon of {60, 150, 300, 600} s for the whole matrix, so that
the matched reading is seen beside its neighbours.

## Fixed surface

- Logs: the `pi0` held-out decision populations of the published on-policy
  run, `results/onpolicy_full_feb30eb_001/populations/test/<trace × cell ×
  target × seed>/pi0.npz`, both targets (`next_use`, `binary`), 2 traces × 6
  cells × 2 targets × 5 seeds = 120 populations, loaded by the SHA-256 the
  published manifest records, exactly as the ranker-error diagnosis loads
  them. The `pi3` populations are not read.
- Scorers: the published `pi0` model of each lineage. Its victim on a logged
  decision is the first minimum, in stored candidate order, of `(score on the
  stored features, stored tie-break)`, the published rule; no hybrid,
  protection or override.
- Horizons `h` ∈ {60, 150, 300, 600} s. Every logged decision lies at least
  600 seconds before the trace end, so every candidate's label is observable
  at every horizon. The matched horizon `h*` of a cell is the table above.
- At horizon `h`: `label_h = −log1p(min(next_use_delta_s, h))`; a candidate
  is *reusable at h* when its next use is at most `h` seconds away. `m3_h`,
  `m4_h`, the four error kinds (reference victim, other tie-break, avoidable
  reusable eviction, order error among reusable candidates, with the
  precedence the diagnosis addendum fixed) and the conditional rates are the
  diagnosis's definitions with `h` in place of 600.
- Separation, new: over the decisions with both a reusable-at-`h` and a
  non-reusable-at-`h` candidate, the *within-decision concordance* of a
  chooser's key is the share of ordered pairs (one reusable, one not) in
  which the key places the non-reusable candidate first, that is, would evict
  it before the reusable one; an exact tie of the key counts one half. It is
  averaged over decisions with each decision weighing one. The ranker's key
  is `(score, stored tie-break)`, recency's the stored tie-break alone, and a
  uniformly drawn victim has concordance 0.5 exactly.
- Subsets, as in the diagnosis: all decisions, those whose logged victim is
  the arrival, those whose logged victim is a resident.

## Required checks

- Every population and model hash equals the published manifest's; the
  scorer applied to its own population reproduces the logged victim, with the
  diagnosis's stop rule (more than 0.1% mismatches in any population stops
  the computation as unresolved).
- At `h` = 600 the class shares, excess shares and conditional rates equal
  the published `results/paper/ranker_error_diagnosis_001/` tables
  (`classes.csv`, `rates.csv`, `rates_seeds.csv`) on their common columns.
  Nothing is published otherwise.
- Tests on constructed populations: the error kinds partition the decisions
  at every horizon; a decision reusable at 600 s and not at 60 s changes
  class accordingly; concordance is 1 for a key that orders every
  non-reusable candidate first, 0 for the reverse, 0.5 for a constant key;
  the matched map holds the six cells above.
- No existing replay path, runner or table is changed; the existing tests
  pass.

## Readings, fixed before the computation

"Consistent" means the same sign in all five seed-paired values. Every value
at every horizon is reported, not only the matched one.

1. **Composition at the matched horizon.** Per trace × cell × target: the
   share of decisions and of `label_h*` excess in each error kind, by subset.
   Prediction: avoidable reusable eviction carries at least 0.99 of the
   excess in every trace × cell × target, as it does at 600 seconds.
2. **Conditional avoidable rate at the matched horizon.** Over the decisions
   with both classes present, the ranker's avoidable-eviction rate against
   recency's and a uniform victim's on the same candidate sets, seed-paired:
   better, worse or mixed, counted out of 12 per target at `h*` and at each
   `h`. Context: at 600 seconds the `next_use` ranker is better than recency
   in 9/12 and worse in 3/12.
3. **Separation at the matched horizon.** The within-decision concordance of
   the ranker's key for the reuse bit at each `h`, beside recency and 0.5.
   Registered prediction, the fit-horizon hypothesis: in the eight trace ×
   cell with `h*` < 600 s, the `next_use` ranker's concordance for the
   `h*`-bit is below its concordance for the 600-second bit on the same
   population, in the five-seed mean. The prediction fails wherever it is
   not. Both published rankers were fitted to 600-second targets.
4. **Bridge, descriptive.** Per trace × cell and target: the ranker's `m4_h*`
   over all its decisions and over the resident-victim subset, beside the
   published `U(label) − U(learned)` and `U(label_binary_h*) − U(learned)`.
   The Spearman rank correlation over the 12 trace × cell of each target is
   reported as a description of twelve points, not as a test.

## Interpretation boundaries

- The populations are the frozen ranker's own store, sampled as reservoirs
  of 40,000 decisions. A store kept by the exact bit would present other
  candidate sets; these logs do not contain them.
- `h*` was read from replays of the same two traces after the fact. Nothing
  here chooses a horizon before the fact or predicts it from capacity.
- Concordance and rates on logged candidates are properties of a score on
  those sets, not utility; a decision-level statistic was shown not to be a
  utility proxy by the error-location control.
- Nothing here fits. Whether the bit at `h*` can be learned from the 23
  history features is not measured; the published rankers were fitted to
  600-second targets, and a ranker fitted to another horizon is a separate
  experiment with its own pre-registration. Phase 0.9 fitted single-tier
  rankers to 60- and 300-second binary targets with small gains over 600
  seconds ([target change](target-change-findings.md)).
- Two traces of one deployment family, one linear ranker family.

## Execution order

1. Commit this plan, alone with the matched class-order plan.
2. Implement a separate module and runner on `errordiag` with the horizon
   as a parameter and the concordance added; tests on constructed
   populations; no change to an existing path. Review and commit before any
   log is read.
3. Run once with 10 workers from a clean tree whose execution source is the
   code commit, into `results/paper/matched_horizon_diagnosis_001/`,
   recording plan and code commits and the hashes of every population, model
   and table read.
4. Report the four readings next to the diagnosis's, including a failed
   prediction.
