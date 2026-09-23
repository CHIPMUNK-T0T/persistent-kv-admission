# Exploratory retention decision diagnostics

This is a post hoc analysis of the frozen on-policy run. See the
[diagnostic plan and interpretation](../../../docs/retention-decision-diagnostics.md).
The registered A–D outcomes remain those of the original experiment.

Reproduce from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_retention_decisions
PYTHONPATH=src .venv/bin/python scripts/analyze_retention_decisions.py
```

The script verifies the published CSV hashes, both trace hashes, all 240
selected NPZ hashes, one actual victim per complete group, recorded tuple
argmin, and `pi3` model rescoring on its own test population. It reads one
NPZ at a time with one worker. No model fitting or replay occurs. The
[config](analysis_config.json) records source/output hashes, code versions,
and 9.6 million verified sampled decisions.

| Artifact | Meaning |
|---|---|
| [Window utility, seed](window_utility_seeds.csv) | 960 rows: all four iterations, two request windows, both targets, all traces/cells/seeds; paired raw-token and input-token-point differences. The terminal common-population ranking sign is attached only to `pi3`. |
| [Window utility, aggregate](window_utility_aggregate.csv) | 192 five-seed summaries, with all five direction counts. `full` is the registered utility window; `label` is descriptive. |
| [Actual victim quality](victim_quality_seeds.csv) | `pi0`/`pi3` actual victims on each policy's own saved test population, split by all decisions, arrival rejection, and resident eviction. |
| [Fixed population choices](terminal_fixed_population_choices.csv) | Canonical `pi0` and `pi3` proposed victims on the same `D_test(pi3)` candidate sets. The `all` rows are directly comparable. Decision-type rows classify each scorer's proposed choice, so their subsets may differ. |
| [Population integrity](population_integrity.csv) | Verified NPZ path/hash, row/group counts, argmin checks, and actual rejection/eviction counts for each of 240 files. |
| [Window sensitivity](window_sensitivity.png) | Five-seed `pi3-pi0` mean points for every cell and both targets, full versus label-observable request windows. Horizontal labels are L1 working-set percent × L2/L1 multiplier. |
| [Victim proxy rate](victim_proxy_positive_rate.png) | `next_use` only: mean positive candidate-relative proxy-gap rate across five seeds. Each `pi0`/`pi3` bar uses that policy's own sampled test population; this figure is not the fixed-`D_test(pi3)` comparison. Horizontal labels are L1 working-set percent × L2/L1 multiplier. |

The target-native value uses the original `target_column` transform, with
larger values preferred for retention. `native_min_correct_rate_all` counts a
victim as correct if it is tied for minimum. The uniform baseline is the
per-decision fraction of candidates tied for minimum. The discriminative
rates exclude constant-label decisions and show their denominator. A
candidate-relative gap is `actual_victim_value - min(candidate_values)`;
the reuse-count and token-count columns use the same subtraction. The token
proxy is `count_within_h × incremental block_tokens`, not prefix tokens.
Its reported sum is over reservoir-sampled decisions, not an estimate of
trace-wide avoided tokens or causal regret. Top 1%/10% shares select
`ceil(q × decisions)` largest gaps among **all** sampled decisions in the
specified row. An all-zero gap total yields an undefined share (`nan`).

The `all` row exists for every log. A decision-type row is omitted when that
file has zero decisions of that type; its zero count remains explicit in the
integrity CSV. The label-observable sample is capped at 40,000 decisions per
file. The full-window and label-window utility denominators are each that
window's requested tokens. The 600-second labels may observe future reuse
beyond the label-window request boundary, and ranking is a decision-level
metric rather than request-level utility.
