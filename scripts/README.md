# Scripts

Download the three official Mooncake FAST'25 traces:

```bash
./scripts/download_mooncake_traces.sh [optional-output-directory]
```

Run the full characterization and fixed-budget replay:

```bash
python3 scripts/run_characterization.py data/raw/*_trace.jsonl
```

Useful controls are `--snapshots`, `--precision-k`, `--ridge`, `--budgets`, `--block-size`, `--bytes-per-token`, `--output-dir`, and `--paper-dir`. Run `python3 scripts/run_characterization.py --help` for details. Dependencies are NumPy and Matplotlib; the ridge linear ranking implementation does not require scikit-learn.

The script validates the input schema before producing results. It does not implement or integrate a new vLLM/LMCache policy.

Run the Phase 0.5 cross-workload prediction and replay:

```bash
python3 scripts/run_cross_workload.py data/raw/*_trace.jsonl
```

Run the Phase 0.75 predictability–retention gap decomposition (oracle replay
over several seeds, eviction-candidate logging, standardisation robustness):

```bash
python3 scripts/run_predictability_gap.py data/raw/*_trace.jsonl --seeds 5 --workers 24
```

Controls are `--budgets`, `--seeds`, `--workers`, `--snapshots`, `--precision-k`,
`--max-logged-decisions`, `--dump-decisions`, `--skip-standardization`, and
`--skip-replay`. The full dump goes to `results/predictability_gap/` (untracked);
the summary CSVs, figures, and `gap_run_config.json` go to `results/paper/`.

Check whether a stronger history model ranks the eviction candidates better
than the linear ranker (gradient boosting on the same 23 features over the
logged candidate sets, time-split with a horizon embargo). This one needs
scikit-learn, which the system Python does not ship; use a virtualenv:

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install scikit-learn
.venv/bin/python scripts/run_candidate_models.py data/raw/*_trace.jsonl --workers 12
```

Outputs `results/paper/candidate_model_check.csv`, `fig9_candidate_model_check.png`,
and `candidate_model_run_config.json`; the full dump goes to `results/candidate_models/`.
