# On-policy learning runner

Implementation for the fixed experiment in [the pre-registration](onpolicy-learning-plan.md). Commit the runner and tests before using it. Both commands require a new `--output-dir`; an existing directory is rejected.

```bash
python scripts/run_onpolicy_learning.py data/raw/conversation_trace.jsonl \
  --smoke --workers 1 --output-dir results/onpolicy_smoke_001

python scripts/run_onpolicy_learning.py \
  data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl \
  --workers WORKERS_FROM_SMOKE --output-dir results/onpolicy_full_001 \
  --paper-dir results/paper
```

The smoke run uses one trace, one cell, one target, and one lineage seed through all four iterations. Its paper-style tables and figures stay in `results/onpolicy_smoke_001/paper/`; they are diagnostic and cannot yield a five-seed A–D verdict. Measure smoke wall time, worker peak RSS, and serialized population/model bytes in its config before choosing the full worker count. The full run writes fixed CSV tables, three figures, and a config into `results/paper/onpolicy_learning/` after all checks pass. Training populations and models remain in the run directory for audit.

The runner verifies the pre-registration and committed code tree, published Phase 0.97 fit and replay references, available Phase 0.98b `A_none` loss references, Phase 1 trace hashes, complete model/population manifests, tuple argmin, attribution identities, and unchanged source/reference/document hashes before publication. Phase 0.98b has published `A_none` loss rows for only three of the six cells; all six cells still get new attribution tables and partition checks. The config and loss table mark cells without a published comparison.

Phase 0.97 did not publish hashes for individual candidate-log NPZ files. The runner verifies their measured kept-decision and row counts against published metadata, checks replay counters, records current SHA-256 hashes, and labels those hashes as contemporaneous evidence. A historical hash match is not claimed. Population cap binding and constant-label shares are diagnostics requiring review; the runner does not infer a new threshold for them. Numerical diagonal disagreement, non-finite ranking, and an observed fit-to-terminal training shift remain unresolved in the verdict. The test windows come from traces already used for research decisions and are not an independent validation set.

The code and unit/integration checks are prepared; neither the real-trace smoke run nor the full run has been executed yet. Therefore exact 120-row replay reproduction and real-trace outcomes are not established by this implementation work.
