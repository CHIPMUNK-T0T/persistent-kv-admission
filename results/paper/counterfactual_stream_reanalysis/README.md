# Published three-stream CSV reanalysis

This directory contains a small, post hoc arithmetic check of the published
[one-step counterfactual CSVs](../counterfactual_action_001/). It does not contain
new branch replays or results from the registered 16-fresh-stream experiment.
The [audit note](../../../docs/counterfactual-randomness-reanalysis.md) defines
the subset, tie rules, denominators, and interpretation limits.

[`summary.json`](summary.json) is the output of
[`scripts/analyze_counterfactual_streams.py`](../../../scripts/analyze_counterfactual_streams.py)
for the 80 decisions with `hash_rank` 0 or 1 in each of 40 lineages. The JSON
records the checkout HEAD, script and input SHA-256 values, unrounded results,
and the five rounded 600-second checks: 34,419 fixed exact; 4,114 hindsight
best within the exact-label tie in the same stream; 35,613 best within that
tie selected on another stream; 37,360 best among all actions selected on
another stream; and 35,107 actual learned regret, all in avoided-prefill
tokens. The mean cross-stream Q-rank Spearman is 0.0711161 at 600 seconds and
0.0319674 through trace end.

To regenerate, use an **absent** output directory, then compare its
`summary.json` with this file. For example:

```sh
python3 scripts/analyze_counterfactual_streams.py --output-dir /tmp/counterfactual-stream-audit-new
```

The script checks the published CSV hashes against `run_config.json` and the
source hashes against its source manifest. The held-out comparison has 480
ordered stream directions over the same 80 states; those directions are not
independent samples. These three streams cannot estimate an ICC, a noise
percentage, or stable expected action values.
