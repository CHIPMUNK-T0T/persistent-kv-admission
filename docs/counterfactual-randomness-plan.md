# Counterfactual randomness and hindsight falsification

Status: pre-registration for a new experiment; no new continuation outcomes
have been generated. Commit this plan before implementation execution on real
traces. Review and commit implementation and tests before prepare and smoke.

## Question and scope

Are the previously observed counterfactual action-value differences stable
under continuation randomness, or largely specific to choosing a winner after
observing a random continuation? This is the first stage of the approved
falsification sequence: randomness/hindsight, simple residence mechanisms,
then label representation/sampling controls. The latter stages are not part
of this run. No model fit, feature, retention policy, arrival intervention,
or semantic experiment is introduced.

The source experiment is `results/counterfactual_action_001`, reported at
commit `aed0959`, with selection manifest SHA256
`5db02a59c058a5934d308b8f893cec855021a62fe1284071ade9d08f50ba80be`.
The published `results/paper/counterfactual_action_001/run_config.json` has
SHA256 `175970ef039c76e930398745c633598b3463109b8205191a6c90ce0f66ebbb7a`.
Its outcomes motivated this experiment and have been inspected. This is not
an independent workload confirmation. All source results remain immutable.

For a fixed sampled simulator state S, action a, frozen continuation policy
pi, and post-draw random stream omega, Q600(S,a,omega) is avoided prefill
tokens in `(t,t+600s]`. The future request trace remains fixed. The expectation
of interest averages continuation sampling randomness, not future workloads.
The old mean of stream-specific maxima is not the maximum of expected Q.

## Fixed population and computation

Use exactly the first hash-ranked decision (`hash_rank=0`) from each of the
40 original lineages: two real traces, cells 0.0025x1 and 0.02x4, next-use
pi0/pi3, and lineage seeds 0..4. This selects 40 snapshots and 678 actions,
using identities only, without inspecting outcomes to choose snapshots.
Keep every original candidate and its order (16 or 17 actions per state).
Do not resample a state after zero effects, ties, adverse results, or failed
integrity checks. The eight trace/cell/source-policy strata each contain
five states; pi0 and pi3 states are different populations.

Generate exactly 16 fresh continuation streams per snapshot, numbered 1..16.
The new namespace is `persistent-kv-counterfactual-rng-crossfit-v1`.
Derive the integer seed from SHA256 of canonical compact JSON encoding
`[namespace, lineage, group_index, ordinal, stream_id]` with UTF-8, using the
full digest as an unsigned big-endian integer. Freeze all resulting seed
values in the new selection manifest. Reject duplicate new seeds and any
collision with the old sensitivity seeds for the same snapshot. Neither the
old captured RNG continuation nor old sensitivity streams enter estimation.

The 10,848 fresh action/stream branches are accompanied by one captured-RNG
actual-action validation branch per state (40 additional branches). The
validation branch is not an estimation stream. All 40 uninterrupted parent
replays must reproduce their published reference outcomes.

Use the exact original fork boundary, already-drawn candidates, suspended
admission state, unchanged replay engine and frozen models. Reseed only future
L2 sampling after the current candidate draw. Within a stream, every action
starts with the same seed (common random numbers). This does not guarantee
the same later candidate identities, call counts, or reduced variance: cache
contents and RNG consumption can diverge. Override exactly one victim, then
resume the original policy, including any remaining overflow rounds.

Run each branch through trace end once. Primary rewards use 600 seconds;
trace-end rewards are a predeclared secondary window. Requests already served
at the decision timestamp are excluded. Preserve the original actual-action
parent reward/digest/terminal-state checks, exact candidate/features/scores
checks, and total-versus-L2 action-difference invariants.
Also compare each captured actual-action reward/digest and terminal state
against the original hash-verified captured actual-action branch.

## Independent selection and evaluation

Fix fold A to streams 1..8 and B to streams 9..16 before outcomes. For each
state, select a_A as the action maximizing mean Q600 over A. Evaluate
`Q600(a_A)-Q600(actual)` on each B stream, pairing both actions within that
stream. Reverse A and B to obtain a_B and its held-out A differences.
The primary per-state statistic is the equal average of the two directional
held-out mean gains. Each direction independently evaluates its selected
action conditional on its training fold. The symmetric statistic is a
descriptive cross-fit estimate of an eight-stream selection procedure using
two potentially different actions, not the true best expected action.

Break equal training means by original live `last_group` ascending, then
original candidate draw index ascending. Never break ties using held-out
outcomes, trace-end outcomes, or previously inspected experiment outcomes.
Never truncate negative held-out gains to zero. A constant-Q training set
still follows this fixed tie-break and remains in the denominator.

Apply the same procedure restricted to the exact next-use minimum-label tie
set, and separately to the count minimum-label tie set. Evaluate both the
unrestricted and restricted choices on the same held-out streams. Their
paired difference measures the performance of these finite-sample selectors;
it is not automatically the gap between optimal expected unrestricted and
restricted actions. Report identical tie sets/selections explicitly rather
than counting them as independent evidence.

Secondary contrasts compare the cross-fitted selections with fixed exact
next-use/count, LRU, and LFU selectors from the original snapshot. Each fixed
selector uses the original full key/tie-break. Retain direct paired contrasts
between exact and actual actions over all 16 streams. For trace-end rewards,
evaluate the SAME actions selected with training Q600; do not select new
actions using trace-end utility. Do not choose folds, stream counts, states,
horizons, or reporting strata from the new outcomes.

Report as diagnostics, clearly separate from primary held-out gain:
stream-specific hindsight regret; training-set gain; evaluation-set hindsight
maximum; actionwise mean/SD; paired-difference mean/SD/SE; fold-selected action
agreement and tie counts. A maximum over all-16-stream means remains a fitted
diagnostic, not held-out evidence. No hindsight regret is a recoverable
trace-wide token estimate.

## Uncertainty and interpretation

Publish every directional per-stream paired difference, each directional
mean/SD/SE, and its approximate conditional 95% t interval (8 held-out
streams, df=7). Small samples and nonnormal effects can limit this interval.
These intervals condition on the selection fold and state, describe only
continuation-stream sampling, and are not simultaneous guarantees. The two
directions reuse data for selection and evaluation and are dependent. Do not
pool their 16 differences into a naive interval for the symmetric statistic.
Stratum and overall means of the symmetric statistic are descriptive, with
state-level values and sign counts retained. Do not treat candidate actions,
snapshots sharing requests, or five lineage seeds as independent workloads.

Stable positive held-out gains support a persistent conditional action-value
difference under the fixed trace/state/continuation experiment; the selector
still uses simulated
future requests and is not deployable. Large training/hindsight gains without
held-out support weaken the current interpretation, but do not establish that
expected action values are identical. Eight selection streams may choose
poorly, and eight evaluation streams may have low precision. Report negative,
zero and inconclusive results. There is no outcome-triggered stream expansion
or automatic claim that randomness explains a particular fraction of the
original regret. Any additional experiment requires a new recorded design.

Neither result establishes semantic necessity, history impossibility, a new
policy, or workload-general conclusions. A positive result motivates the
separate minimal residence-mechanism test; it does not prove complex
trajectory effects dominate. That next stage remains unexecuted here.

## Validation, freeze, and execution order

1. Commit this plan alone. Implement in separate new files, preserving the
   original runner, simulator, plan and results; reuse existing validated
   components where possible. Review and commit code/tests before real runs.
2. Test constructed reward matrices with an in-sample winner that loses on
   held-out streams, negative held-out gains, constant outcomes, training
   ties, restricted-label ties, fold-swap symmetry, and selection invariance
   to changes in that direction's evaluation fold. Test fresh deterministic
   stream derivation, same seeds across actions, captured-control exclusion,
   exact parent continuation, partial overflow/timestamp boundaries through
   existing regression tests, and strict resume/hash/failure checks.
3. Prepare a new run directory. Validate and freeze the original manifest,
   chosen identities, model/population/trace/reference hashes, new stream
   seeds/folds, plan/source hashes, committed HEAD, and protected doc diffs.
   Never overwrite or extend the old run. Reject uncommitted execution code.
4. Smoke: conversation 0.0025x1 pi0 seed0 hash-rank0 only, fresh stream 1
   across all actions plus captured actual-action validation. Smoke outcomes
   belong to the registered run and may be reused only with identical frozen
   configuration. Use it for integrity/resource checks, not design changes.
5. Run the fixed full grid after smoke review, with bounded concurrency and
   memory guard. Only operational worker-count changes are allowed. Resume
   only identity/hash-matched units. Any integrity failure aborts; missing
   branches cannot be silently omitted. Freeze source through aggregation.
6. Publish selected identities, config/hashes, action-stream Q, directional
   selection/evaluation rows, per-state and stratum summaries, readable
   figures, validation results, and a findings document with limitations.

Protected pre-existing changes in README.md,
docs/decision-population-findings.md and docs/experiment-plan.md remain
untouched and unstaged. Implementation is delegated to GPT-6 Sol xHigh;
the primary agent reviews the design, code, verification, and interpretation.
