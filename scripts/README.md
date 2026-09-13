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

Run the Phase 0.97 decision-population-matched learning experiment (fixed
two-tier setting; the training population of the L2 ranker is the only
varied factor):

```bash
python3 scripts/run_decision_population.py data/raw/*_trace.jsonl --workers 24
python3 scripts/run_decision_population.py data/raw/conversation_trace.jsonl --smoke   # one cell, one seed, for timing
python3 scripts/run_decision_population.py --figure-only                                # redraw fig14–16
```

Controls are `--l1-budgets`, `--l2-multipliers`, `--seeds`, `--targets`,
`--workers`, `--output-dir`, `--paper-dir`. Outputs
`results/paper/decision_population_fits.csv`, `decision_population_predictive.csv`,
`decision_population_replay.csv` (aggregated over seeds),
`decision_population_replay_seeds.csv`, `decision_population_onpolicy.csv`,
`decision_population_config.json`, and `fig14_decision_population.png` /
`fig15_population_ladder.png` / `fig16_onpolicy_decisions.png`; decision logs
(`.npz`, about 640 MB) and raw rows go to `results/decision_population/`
(untracked). The full grid takes about 1.5 hours on 24 workers (1,962 replays).

Run the Phase 0.98 eviction-decision attribution (descriptive; the Phase 0.97
arms are rebuilt by the Phase 0.97 code and replayed unchanged, with three
read-only hooks recording where the lost reuse came from):

```bash
python3 scripts/run_decision_attribution.py data/raw/conversation_trace.jsonl data/raw/toolagent_trace.jsonl --workers 24
python3 scripts/run_decision_attribution.py data/raw/conversation_trace.jsonl --smoke   # one cell, one seed, three arms
python3 scripts/run_decision_attribution.py --figure-only                                # redraw fig17
```

Controls are `--seeds`, `--workers`, `--output-dir`, `--paper-dir`; the grid
(two real traces, the cells 0.25% × 1, 1% × 4, 2% × 4, the three generic
sampled arms and A_none / B / C_lru / C_union on the next-use and the binary
target) is fixed in the script. Outputs
`results/paper/decision_attribution_losses.csv` (aggregated, with the
pre-registered readings), `decision_attribution_losses_seeds.csv`,
`decision_attribution_decisions.csv`, `decision_attribution_orphaning.csv`,
`decision_attribution_config.json`, and `fig17_decision_attribution.png`; raw
rows and the rebuilt decision logs go to `results/decision_attribution/`
(untracked). Every replay is checked against
`decision_population_replay_seeds.csv` and the run aborts before writing if any
arm does not reproduce Phase 0.97 exactly. The full grid is 330 replays, about
25 minutes on 24 workers.
