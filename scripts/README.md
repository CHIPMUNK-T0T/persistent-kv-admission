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

Run the Phase 0.9 target change (same ranker, features, hyperparameters, and
eviction as Phase 0.5; only the prediction target changes):

```bash
python3 scripts/run_target_change.py data/raw/*_trace.jsonl --seeds 5 --workers 24
python3 scripts/run_target_change.py --figure-only   # redraw fig10 from the saved summary
```

Controls are `--budgets`, `--seeds`, `--workers`, `--snapshots`, and
`--oracle-csv` (the Phase 0.75 summary drawn as reference lines). Outputs
`results/paper/target_change.csv`, `target_change_config.json` (fit metadata
and standardised coefficients), and `fig10_target_change.png`; per-seed rows
and raw JSONL go to `results/target_change/` (untracked).

Run the Phase 0.95 two-tier victim characterization (fixed L1, victim event
log, generic and offline L2 arms on the same stream):

```bash
python3 scripts/run_two_tier.py data/raw/*_trace.jsonl --workers 24
python3 scripts/run_two_tier.py --figure-only   # redraw fig11–13 from the saved CSVs
```

Controls are `--l1-budgets`, `--l2-multipliers`, `--max-total`,
`--l1-policies`, and `--workers`. Outputs `results/paper/victim_events_summary.csv`,
`victim_events_by_depth.csv`, `two_tier_replay.csv`, `two_tier_gate.csv`,
`two_tier_single_tier_reference.csv`, `two_tier_offline_tiebreak.csv` (the
offline tie-break diagnostic, run as the last step), `two_tier_config.json`, and
`fig11_victim_stream.png` / `fig12_two_tier_gain.png` / `fig13_depth_allocation.png`;
the full event logs (one JSONL per trace × L1 policy × L1 budget) and raw
replay rows go to `results/two_tier/` (untracked). Everything is
deterministic; the run takes about five minutes on 24 workers.
