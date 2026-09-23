# On-policy learning runner

Implementation for the fixed experiment in [the pre-registration](onpolicy-learning-plan.md). The runner and tests were committed before the smoke and full results were used. Both commands require a new `--output-dir`; an existing directory is rejected.

The executed commands below are archival. The runner rejects an existing
output directory. To run again, choose new `--output-dir` paths and a new
`--paper-dir` so these published files stay intact.

```bash
.venv/bin/python scripts/run_onpolicy_learning.py data/raw/conversation_trace.jsonl \
  --smoke --workers 1 --output-dir results/onpolicy_smoke_feb30eb_001

.venv/bin/python scripts/run_onpolicy_learning.py \
  data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl \
  --workers 8 --output-dir results/onpolicy_full_feb30eb_001 \
  --paper-dir results/paper
```

The smoke run used one trace, one cell, one target, and one lineage seed through all four iterations. Its paper-style tables and figures stayed in its separate run directory; they cannot yield a five-seed A–D verdict. The smoke measurements supported eight full-run workers. The full run completed with exit status 0, 480 training-prefix and 480 cold held-out replays, and published the fixed tables, three figures, and config into [`results/paper/onpolicy_learning/`](../results/paper/onpolicy_learning/README.md). Training populations and models remain in the run directory for audit. The full-run config records a peak worker RSS of about 1,895 MiB and elapsed time of about 2 h 56 min.

The runner verifies the pre-registration and committed code tree, published Phase 0.97 fit and replay references, available Phase 0.98b `A_none` loss references, Phase 1 trace hashes, complete model/population manifests, tuple argmin, attribution identities, and unchanged source/reference/document hashes before publication. Phase 0.98b has published `A_none` loss rows for only three of the six cells; all six cells still get new attribution tables and partition checks. The config and loss table mark cells without a published comparison.

Phase 0.97 did not publish hashes for individual candidate-log NPZ files. The runner verifies their measured kept-decision and row counts against published metadata, checks replay counters, records current SHA-256 hashes, and labels those hashes as contemporaneous evidence. A historical hash match is not claimed. Population cap binding and constant-label shares are diagnostics requiring review; the runner does not infer a new threshold for them. Numerical diagonal disagreement, non-finite ranking, and an observed fit-to-terminal training shift remain unresolved in the verdict. The test windows come from traces already used for research decisions and are not an independent validation set.

The full run verified all four `pi0` fits, exact replay outcomes for all 120 published Phase 0.97 `A_none` rows, 60 available Phase 0.98b loss references, 3,840 cross-score entries, 960 recorded-tuple evaluations, and all attribution partition identities. The numerical diagonal check passed. All 480 training and 480 test reservoirs reached the 40,000-decision cap; its possible effect requires manual interpretation rather than an automatic dominance label. The outcomes and interpretation are in [the findings](onpolicy-learning-findings.md).
