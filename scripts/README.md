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
