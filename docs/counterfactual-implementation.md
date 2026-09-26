# One-step counterfactual runner

The fixed design is in [the pre-registration](counterfactual-action-value-plan.md).
The runner is [run_counterfactual.py](../scripts/run_counterfactual.py). It
uses the published on-policy test populations and frozen next-use models.
There is no fit or new deployed policy.

The optional sampled-union override hook in
[twotier.py](../src/persistent_kv_admission/twotier.py) runs after the ordinary
decision observer and before removal. A None result leaves the original
victim unchanged. A legal integer index changes that one victim. At the
selected hook, the replay worker forks once per candidate action. The child
inherits the current L1 eviction/admission stack, cache insertion orders,
scorer history, counters, and post-draw RNG state. It disables further
interventions, resumes the same timestamp group's pending insertion work,
and runs the frozen source policy to trace end. Requests at the decision's
timestamp were already served and are excluded from both future reward
windows.

The parent never changes its selected action. Each captured-stream
actual-action child must match the parent's future request reward digest,
600-second and trace-end token counts, complete returned replay counters,
and terminal mutable-state digest. The parent also matches its published
on-policy utility row. An integrity mismatch aborts the lineage. The
terminal digest covers L1/L2 resident order, heap/version/history/counters,
L2 RNG state, scorer causal history, and request accounting. Frozen trace and
model inputs are identified separately by their SHA-256 hashes.

## Execution

After the plan and implementation/tests have separate reviewed commits, use
a new ignored run directory:

    .venv/bin/python scripts/run_counterfactual.py prepare \
      --output-dir results/counterfactual_action_001
    .venv/bin/python scripts/run_counterfactual.py smoke \
      --output-dir results/counterfactual_action_001
    .venv/bin/python scripts/run_counterfactual.py full \
      --output-dir results/counterfactual_action_001 --workers 1 --detach
    .venv/bin/python scripts/run_counterfactual.py status \
      --output-dir results/counterfactual_action_001
    .venv/bin/python scripts/run_counterfactual.py aggregate \
      --output-dir results/counterfactual_action_001 \
      --paper-dir results/paper/counterfactual_action_001

Prepare verifies all published CSV/model/population/trace hashes, freezes
all 320 decision identities and source hashes, and must finish before the
smoke. The designated smoke is the first hash-ranked decision in conversation
0.25% × 1, pi0, seed 0. Its per-action outputs can be reused by the full run
only under the identical frozen manifest. The full runner rejects a worker
count above the memory limit inferred from the smoke's parent and child peak
RSS. That projection is a capacity guard, not a precise runtime forecast.
It writes a status file and per-worker logs; the detach option additionally
writes a launcher PID/log record. An exclusive run lock prevents simultaneous
smoke, full, and aggregate coordinators in one directory. Incomplete lineages
can be resumed; a completed lineage and each retained branch are accepted
only after identity and hash checks. A failed worker stops active process
groups and leaves the full status incomplete. No partial table directory is
published.

The full run keeps one request-reward digest per sampled parent snapshot and
one branch file per action and continuation stream. All actions from one
snapshot begin at exactly the same post-draw state. The two sensitivity
streams reseed only future L2 draws with a SHA-256-derived integer and use
their own actual-action controls. Their three-stream means average paired
regrets within each stream. Branches execute in dedicated single-threaded
subprocesses, fork one child at a time, and reject a worker that has multiple
native threads. This implementation requires Linux/POSIX fork.

Aggregate writes candidate/action Q, per-decision regret and exact-label
tie envelopes, five-seed summaries, arrival-rejection strata, sensitivity,
figures, and a self-contained run config to a new paper directory. Q is
future avoided prefill tokens for one forced action followed by the source
policy. The primary window is (t, t+600 s]; the secondary window is
(t, trace end]. Regret is the best Q among that decision's legal candidates
minus the selected action's Q. The Q maximum is recomputed within each
snapshot and random stream. These finite-trace, sampled-state outcomes do
not estimate optimal full-policy performance. pi0 and pi3 create different
snapshot populations, so their aggregate bars are descriptive. The source
traces and test windows have already informed earlier research decisions.
