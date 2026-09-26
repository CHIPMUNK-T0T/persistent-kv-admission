# One-step counterfactual action values

This directory reports avoided-prefill-token Q for every legal candidate at 320 sampled L2 decisions. Each branch changes one victim and then resumes the frozen source policy. The 600-second window is primary; trace end is secondary.

selected_decisions.csv identifies the fixed snapshots. branch_actions.csv gives every candidate and stream outcome. decision_regrets.csv compares selectors with the best Q in that same candidate set; label_tie_best/worst_regret show the range among candidates tied on the primary exact label. seed_regrets.csv and stratified_regrets.csv summarize primary and sensitivity streams without treating candidates as independent observations. sensitivity.csv averages paired regrets over each snapshot's three continuation streams.

The two figures show mean candidate regret by selector and source policy for each trace/cell. pi0 and pi3 bars come from different policy-created states, so their difference is descriptive, not a causal comparison of the policies. The figures do not establish a trace-wide recoverable loss or a deployment-general effect. Use the run config, frozen selection manifest, and retained branch JSON files for reproduction and audit.
