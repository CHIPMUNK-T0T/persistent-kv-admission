# Fresh-stream counterfactual cross-fit runner

The fixed design is in [the randomness pre-registration](counterfactual-randomness-plan.md).
The implementation consists of [the replay/aggregation runner](../scripts/run_counterfactual_randomness.py)
and [the cross-fit library](../src/persistent_kv_admission/counterfactual_randomness.py).
It replays exactly the first hash-ranked state in each of the original 40
lineages. The source policy and sampled-L2 eviction mechanism are unchanged.

The new controller inherits the original fork boundary and snapshot checks.
One captured-RNG actual-action branch per state verifies the uninterrupted
parent and the hash-verified original captured branch. It is excluded from
estimation. Sixteen fresh post-draw seeds are derived from the full SHA-256
digest of compact JSON `[namespace, lineage, group_index, ordinal, stream_id]`.
Seeds are frozen in the new manifest; duplicate seeds and collisions with the
two original sensitivity seeds for the same state abort preparation. Each
fresh stream runs every originally drawn candidate, including the original
action. A stream's paired differences use that stream's own actual-action
control. All branches run through trace end, with 600-second and trace-end
reward windows recorded separately.

Before any real branch execution, the runner requires committed execution
source and checks the committed pre-registration. `prepare` verifies the
published old run config and selection manifest hashes, reselects each first
state from its published decision population, checks all models and traces,
and freezes the 40 states, 678 legal actions, 640 fresh seeds, source hashes,
reference hashes, folds, and protected documentation diffs in a new run
directory. The original audit and paper directories are read only.

After implementation and tests are reviewed and committed, the sequence is:

```bash
.venv/bin/python scripts/run_counterfactual_randomness.py prepare \
  --output-dir results/counterfactual_randomness_001
.venv/bin/python scripts/run_counterfactual_randomness.py smoke \
  --output-dir results/counterfactual_randomness_001
.venv/bin/python scripts/run_counterfactual_randomness.py full \
  --output-dir results/counterfactual_randomness_001 --workers 6 --detach
.venv/bin/python scripts/run_counterfactual_randomness.py status \
  --output-dir results/counterfactual_randomness_001
.venv/bin/python scripts/run_counterfactual_randomness.py aggregate \
  --output-dir results/counterfactual_randomness_001 \
  --paper-dir results/paper/counterfactual_randomness_001
```

At the implementation checkpoint, these real-run commands have not been
executed. Resume only after the implementation and test commit is reviewed.
The 2026-09-26 checkpoint passed all 235 tests under `.venv/bin/python` and
system `python3` (the latter skips one existing sklearn-dependent test).
The suite includes 15 new constructed counterfactual/cross-fit tests. Run it
with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src`
and `<interpreter> -m unittest discover -s tests`. Plots were reviewed using
constructed values only; no fresh real-trace continuation was generated.
Run `prepare` on that exact committed HEAD, inspect the selection manifest,
then run and review `smoke` before `full`. The manifest pins the full source
hash set and code commit. Editing execution source after preparation invalidates
the run; use a fresh run directory and a new reviewed commit in that case.
`full` may resume incomplete, hash-matched branches under the same manifest;
`aggregate` requires a completed full status and a new publication directory.

The smoke runs only the designated conversation/0.0025×1/pi0/seed0 state:
all its actions in fresh stream 1 plus its captured actual-action check.
Those completed branch files are reused by the full worker after hash and
identity checks. The full run requires 10,848 fresh branches and 40 captured
validation branches. It bounds workers to six and further limits concurrency
using smoke peak RSS and available memory. A run lock excludes concurrent
smoke, full, and aggregation coordinators. Incomplete lineages may be resumed;
completed markers must match the exact action/stream set, identities, and
file hashes. A worker failure stops active process groups and leaves the full
status incomplete. The frozen source and protected documentation are checked
again before completion and publication.

For each state, streams 1–8 form A and 9–16 form B. The candidate maximizing
mean 600-second Q on A is evaluated against the original action on each B
stream, and vice versa. Equal training means use live `last_group` ascending,
then original draw index ascending. The primary state value is the mean of
the two directional held-out mean paired gains, including negative values.
The same procedure is repeated within the minimum exact next-use and count
ties. Trace-end gains evaluate the candidates selected from 600-second Q;
trace-end outcomes never select actions.

The published `action_stream.csv` retains all candidate identities and
stream values. `directional.csv` and `heldout_stream.csv` retain each
selection and individual held-out paired difference. Each direction reports
the sample SD, SE, and an approximate conditional 95% t interval from eight
evaluation streams. The two directions share data and do not yield a pooled
16-stream interval. `state_crossfit.csv`, `state_contrasts.csv`, and
`stratum_summary.csv` contain primary and restricted results by the eight
trace/cell/source-policy strata. `directional_contrasts.csv` compares the
unrestricted selected action with restricted and fixed exact/LRU/LFU choices
on the same held-out streams. `actionwise.csv` reports both Q SD and paired
action-minus-actual SD, which need not agree despite common random seeds.
`stream_hindsight.csv` keeps per-stream hindsight maxima separate;
`state_diagnostics.csv` distinguishes their average from the fitted maximum
of all-16-stream actionwise means. Figures show training versus held-out gain
and the resulting selection optimism in separate panels. All reported gains
are avoided prefill tokens conditional on these fixed states and requests;
they do not estimate deployment-wide improvement.
