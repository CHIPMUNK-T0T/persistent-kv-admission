# Ranker-error diagnosis on saved decision logs

Pre-registration: `docs/ranker-error-diagnosis-plan.md`. Logs: the held-out decision populations of `pi0` and `pi3` of the published on-policy run, 2 traces x 6 cells x 2 targets x 5 seeds, each a reservoir of at most 40,000 whole decisions drawn uniformly from the decisions at or after the split and at least 600 s before the trace end, loaded by the SHA-256 the published table records. Scorers: the published `pi0` and `pi3` models of the same lineage. A scorer's victim on a logged decision is the first minimum, in stored candidate order, of (score on the stored features, stored tie-break); no hybrid, protection or override. Labels: `label = -log1p(min(next_use_delta_s, 600))`; a candidate is reusable when its next use is at most 600 s away. `m3` is the mean of `label(victim) - min label` over a set of decisions, `m4` the share whose victim is reusable while some candidate is not. `U` is extra avoided prefill tokens (avoided minus L1-avoided) in points (100 x share) of input tokens: `ΔU` on the full evaluation window, `ΔU_label` on the label window. "Consistent" means the same sign in all five seed-paired values; seed signs list seeds 0-4 as +, -, 0, or n for nan; "has the sign of" matches zero with zero only.

Nothing was replayed or fitted. A scorer applied to another policy's population is evaluated on states that policy kept, not on a trajectory of its own. A class share is a share of a statistic, not of lost utility. Repeat counts in a reservoir are lower bounds, and runs of consecutive decisions are not estimated.

## `checks.csv`

Required checks, one row per population (trace x cell x target x seed x pi0/pi3).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `iteration`: policy iteration that logged the population (0 or 3)
- `policy`: pi0 or pi3
- `population_path`: the population file named by `onpolicy_test_populations.csv`
- `population_sha256`: SHA-256 of the bytes the population was loaded from
- `published_sha256`: SHA-256 recorded in `onpolicy_test_populations.csv`
- `sha256_matches`: whether the two are equal (every row must be True)
- `rows`: candidate rows of the population
- `decisions`: logged decisions in the reservoir
- `decisions_eligible`: eligible held-out decisions of the replay the reservoir drew from
- `cap_bound`: whether more decisions were eligible than kept (the reservoir is a sample)
- `recorded_argmin_mismatches`: decisions whose logged victim is not the first minimum of the recorded (arm_score, arm_tiebreak); not a stop criterion
- `own_model_path`: the published model of the population's own policy
- `own_model_sha256`: its published SHA-256
- `own_victim_mismatches`: decisions in which that model's victim (first minimum of score and stored tie-break) differs from the logged victim
- `own_victim_mismatch_share`: own_victim_mismatches / decisions
- `exceeds_stop_rule`: own_victim_mismatches > 0.1% of decisions; any True stops the diagnosis as unresolved
- `own_score_exact`: the recomputed score equals the stored arm_score in every row
- `own_score_max_abs_diff`: largest |recomputed score - stored arm_score|

## `matrix_seeds.csv`

The four-way matrix: m3 and m4 of every population under each scorer's victims, and under the logged victims, per seed.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `population`: the policy that logged the decisions (pi0 or pi3)
- `victim_source`: `logged` (the logged victim), `scorer_pi0` or `scorer_pi3` (the first minimum of that model's score and the stored tie-break, in stored candidate order)
- `own_scorer`: the scorer is the population's own policy
- `decisions`: decisions of the population
- `excess_sum`: sum over decisions of label(victim) - min label
- `m3`: excess_sum / decisions
- `avoidable_count`: decisions whose victim is reusable while some candidate is not
- `m4`: avoidable_count / decisions

## `selection_seeds.csv`

Readings 1 and 2 per seed: published utilities and the seed-paired changes of m3 and m4.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `requested_tokens`: input tokens of the full evaluation window (shared by pi0 and pi3)
- `extra_tokens_pi0`: pi0 avoided_prefill_tokens - l1_avoided_tokens, full window
- `extra_tokens_pi3`: the same for pi3
- `u_pi0_points`: U(pi0) = 100 x extra_tokens_pi0 / requested_tokens
- `u_pi3_points`: U(pi3), the same
- `delta_u_points`: ΔU = U(pi3) - U(pi0): the published `input_token_points_vs_pi0` of the pi3 row
- `label_window_requested_tokens`: input tokens of the label window (shared by pi0 and pi3)
- `label_window_extra_tokens_pi0`: pi0 label_window_avoided_prefill_tokens - label_window_l1_avoided_tokens
- `label_window_extra_tokens_pi3`: the same for pi3
- `u_label_pi0_points`: 100 x label_window_extra_tokens_pi0 / label_window_requested_tokens
- `u_label_pi3_points`: the same for pi3
- `delta_u_label_points`: ΔU_label = 100 x (label_window_extra_tokens_pi3 - label_window_extra_tokens_pi0) / label_window_requested_tokens
- `m3_pi0pop_logged`: m3 of the pi0 population with the victim of `logged` (as in `matrix_seeds.csv`)
- `m3_pi0pop_scorer_pi0`: m3 of the pi0 population with the victim of `scorer_pi0` (as in `matrix_seeds.csv`)
- `m3_pi0pop_scorer_pi3`: m3 of the pi0 population with the victim of `scorer_pi3` (as in `matrix_seeds.csv`)
- `m3_pi3pop_logged`: m3 of the pi3 population with the victim of `logged` (as in `matrix_seeds.csv`)
- `m3_pi3pop_scorer_pi0`: m3 of the pi3 population with the victim of `scorer_pi0` (as in `matrix_seeds.csv`)
- `m3_pi3pop_scorer_pi3`: m3 of the pi3 population with the victim of `scorer_pi3` (as in `matrix_seeds.csv`)
- `delta_sel_m3_pi0pop`: Δsel(pi0) = m3(pi0 population, pi3 scorer) - m3(pi0 population, pi0 scorer)
- `delta_sel_m3_pi3pop`: Δsel(pi3) = m3(pi3 population, pi3 scorer) - m3(pi3 population, pi0 scorer)
- `delta_own_m3`: Δown = m3(pi3 population, pi3 scorer) - m3(pi0 population, pi0 scorer), recomputed victims
- `delta_own_m3_logged`: the same change with the logged victims of both populations
- `m4_pi0pop_logged`: m4 of the pi0 population with the victim of `logged` (as in `matrix_seeds.csv`)
- `m4_pi0pop_scorer_pi0`: m4 of the pi0 population with the victim of `scorer_pi0` (as in `matrix_seeds.csv`)
- `m4_pi0pop_scorer_pi3`: m4 of the pi0 population with the victim of `scorer_pi3` (as in `matrix_seeds.csv`)
- `m4_pi3pop_logged`: m4 of the pi3 population with the victim of `logged` (as in `matrix_seeds.csv`)
- `m4_pi3pop_scorer_pi0`: m4 of the pi3 population with the victim of `scorer_pi0` (as in `matrix_seeds.csv`)
- `m4_pi3pop_scorer_pi3`: m4 of the pi3 population with the victim of `scorer_pi3` (as in `matrix_seeds.csv`)
- `delta_sel_m4_pi0pop`: Δsel(pi0) = m4(pi0 population, pi3 scorer) - m4(pi0 population, pi0 scorer)
- `delta_sel_m4_pi3pop`: Δsel(pi3) = m4(pi3 population, pi3 scorer) - m4(pi3 population, pi0 scorer)
- `delta_own_m4`: Δown = m4(pi3 population, pi3 scorer) - m4(pi0 population, pi0 scorer), recomputed victims
- `delta_own_m4_logged`: the same change with the logged victims of both populations

## `selection.csv`

Reading 1, per trace x cell x target x statistic (m3 and m4).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `statistic`: m3 or m4
- `seeds`: seed-paired values
- `delta_u_mean`: five-seed mean of ΔU (points of full-window input tokens)
- `delta_u_seed_signs`: signs of ΔU, seeds 0-4
- `delta_u_reading`: consistent_gain / consistent_loss / mixed of ΔU
- `delta_sel_pi0pop_mean`: five-seed mean of Δsel(pi0) = statistic(pi0 population, pi3 scorer) - statistic(pi0 population, pi0 scorer)
- `delta_sel_pi0pop_seed_signs`: signs of Δsel(pi0), seeds 0-4
- `delta_sel_pi0pop_reading`: sign reading of Δsel(pi0) as published (consistent_gain: positive in every seed, i.e. the pi3 scorer's victims have the larger statistic; consistent_loss; mixed)
- `delta_sel_pi0pop_agrees`: the mean of -Δsel(pi0) has the sign of the mean ΔU (zero matching zero only)
- `delta_sel_pi3pop_mean`: five-seed mean of Δsel(pi3) = statistic(pi3 population, pi3 scorer) - statistic(pi3 population, pi0 scorer)
- `delta_sel_pi3pop_seed_signs`: signs of Δsel(pi3), seeds 0-4
- `delta_sel_pi3pop_reading`: sign reading of Δsel(pi3) as published (consistent_gain: positive in every seed, i.e. the pi3 scorer's victims have the larger statistic; consistent_loss; mixed)
- `delta_sel_pi3pop_agrees`: the mean of -Δsel(pi3) has the sign of the mean ΔU (zero matching zero only)
- `delta_own_mean`: five-seed mean of Δown (recomputed victims)
- `delta_own_seed_signs`: signs of Δown, seeds 0-4
- `delta_own_reading`: sign reading of Δown as published (as for Δsel)
- `delta_own_agrees`: the mean of -Δown has the sign of the mean ΔU
- `delta_own_logged_mean`: five-seed mean of Δown with the logged victims
- `selection_reading`: carried_by_selection (both delta_sel_*_agrees), population_dependent (exactly one) or not_carried_by_selection (neither)

## `selection_counts.csv`

Reading 1 counts, per target x statistic, out of the trace x cells.

- `target`: training target
- `statistic`: m3 or m4
- `cells`: trace x cells counted (12)
- `carried_by_selection`: cells reading carried_by_selection
- `population_dependent`: cells reading population_dependent
- `not_carried_by_selection`: cells reading not_carried_by_selection
- `agrees_pi0pop`: cells where the mean of -Δsel(pi0) has the sign of the mean ΔU
- `agrees_pi3pop`: the same on the pi3 population
- `delta_own_agrees`: cells where the mean of -Δown has the sign of the mean ΔU

## `window.csv`

Reading 2, per trace x cell x target, on m3 (Δown from recomputed victims).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seeds`: seed-paired values
- `delta_own_m3_mean`: five-seed mean of Δown on m3; the rule uses its negative
- `delta_own_m3_seed_signs`: signs of Δown, seeds 0-4
- `delta_u_label_mean`: five-seed mean of ΔU_label
- `delta_u_label_seed_signs`: signs of ΔU_label, seeds 0-4
- `delta_u_label_reading`: consistent_gain / consistent_loss / mixed of ΔU_label
- `delta_u_mean`: five-seed mean of ΔU
- `delta_u_seed_signs`: signs of ΔU, seeds 0-4
- `delta_u_reading`: consistent_gain / consistent_loss / mixed of ΔU
- `agrees_label`: the mean of -Δown has the sign of the mean ΔU_label
- `agrees_full`: the mean of -Δown has the sign of the mean ΔU
- `contradicts_label`: ΔU_label is consistent while the mean of -Δown has the other, non-zero sign

## `window_counts.csv`

Reading 2 counts, per target, out of the trace x cells.

- `target`: training target
- `cells`: trace x cells counted (12)
- `agrees_label`: cells where the mean of -Δown has the sign of the mean ΔU_label
- `agrees_full`: cells where it has the sign of the mean ΔU
- `contradictions`: cells with contradicts_label
- `contradiction_list`: those cells, trace/cell

## `classes_seeds.csv`

Reading 3 per seed: the four classes of the pi0 population's decisions by their logged victim, for every decision and separately for decisions whose victim is the arrival or a resident.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `subset`: `all`, `arrival` (the victim is the arrival) or `resident`
- `class`: `reference_victim` (the victim is the first minimum of (label, tie-break)), `other_tiebreak` (minimum label, not that first minimum), `avoidable_reusable` (the victim is reusable and some candidate is not), `order_error` (every candidate is reusable and the victim's label is above the minimum)
- `decisions`: decisions of the subset in the class
- `excess_sum`: their summed label excess
- `overlap_decisions`: decisions of this class that also meet a later class's definition: a victim whose next use is exactly at the horizon has no excess and is also an avoidable reusable eviction; the plan's order assigns it to the no-excess class
- `subset_decisions`: decisions in the subset
- `subset_excess_sum`: summed label excess of the subset
- `decision_share`: decisions / subset_decisions
- `excess_share`: excess_sum / subset_excess_sum
- `decision_share_of_population`: decisions / all decisions of the population
- `excess_share_of_population`: excess_sum / the population's total label excess

## `classes.csv`

Reading 3 over seeds, per trace x cell x target x subset x class.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `subset`: as in classes_seeds.csv
- `class`: as in classes_seeds.csv
- `seeds`: seeds
- `decisions_mean`: mean decisions in the class
- `decision_share_mean`: mean over the seeds of the class's share of the subset's decisions (nan when any seed is nan)
- `decision_share_min`: minimum over the seeds of the class's share of the subset's decisions (nan when any seed is nan)
- `decision_share_max`: maximum over the seeds of the class's share of the subset's decisions (nan when any seed is nan)
- `excess_share_mean`: mean over the seeds of the class's share of the subset's label excess (nan when any seed is nan)
- `excess_share_min`: minimum over the seeds of the class's share of the subset's label excess (nan when any seed is nan)
- `excess_share_max`: maximum over the seeds of the class's share of the subset's label excess (nan when any seed is nan)
- `decision_share_of_population_mean`: mean over seeds of decisions / all decisions of the population
- `excess_share_of_population_mean`: mean over seeds of excess_sum / the population's total label excess
- `overlap_decisions_total`: overlap_decisions summed over seeds

## `rates_seeds.csv`

Reading 4 per seed, on the pi0 population.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `rate`: `avoidable`: share of the decisions with both a reusable and a non-reusable candidate whose victim is reusable; `order`: share of the decisions whose candidates are all reusable and not all of one label whose victim's label is above the minimum
- `eligible_decisions`: decisions the rate is taken over
- `ranker`: the rate of the frozen ranker (pi0), by its logged victim
- `recency`: the rate of recency: the first minimum of the stored tie-break
- `uniform`: the expected rate of a victim drawn uniformly from the candidates

## `rates.csv`

Reading 4 over seeds, per trace x cell x target x rate.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `rate`: as in rates_seeds.csv
- `seeds`: seeds
- `eligible_decisions_mean`: mean eligible decisions
- `ranker_mean`: five-seed mean of the ranker's rate
- `recency_mean`: five-seed mean of recency's rate
- `uniform_mean`: five-seed mean of the uniform expectation
- `ranker_minus_recency_mean`: mean of the seed-paired ranker - recency
- `ranker_minus_recency_seed_signs`: signs of ranker - recency, seeds 0-4
- `vs_recency`: better (ranker lower in all five seeds), worse (higher in all five) or mixed
- `ranker_minus_uniform_mean`: mean of the seed-paired ranker - uniform
- `ranker_minus_uniform_seed_signs`: signs of ranker - uniform, seeds 0-4
- `vs_uniform`: the same rule against the uniform expectation; descriptive, not registered (the plan fixes the rule against recency)

## `rates_counts.csv`

Reading 4 counts, per target x rate, out of the trace x cells.

- `target`: training target
- `rate`: avoidable or order
- `cells`: trace x cells counted (12)
- `better_than_recency`: cells reading better against recency
- `worse_than_recency`: cells reading worse against recency
- `mixed_vs_recency`: cells reading mixed against recency
- `better_than_uniform`: cells reading better against the uniform expectation (descriptive, not registered)
- `worse_than_uniform`: cells reading worse against it (descriptive, not registered)
- `mixed_vs_uniform`: cells reading mixed against it (descriptive, not registered)

## `concentration_seeds.csv`

Reading 5 per seed: the avoidable reusable evictions of the pi0 population (the `avoidable_reusable` class of classes_seeds.csv) by victim state, for all of them and separately for those whose victim is the arrival or a resident; each subset counts its own distinct states, repeats and top states.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `subset`: `all` (every avoidable reusable eviction), `arrival` (the victim is the arrival) or `resident` (the victim is a resident), as in classes_seeds.csv
- `evictions`: avoidable reusable evictions of the subset in the sample
- `distinct_states`: distinct victim states among them
- `repeated_states`: states that are such a victim more than once in the subset
- `repeat_share`: share of the subset's evictions on those states
- `top_states`: 10% of distinct_states, rounded up
- `top_share`: share of the subset's evictions on the top_states states with the most

## `concentration.csv`

Reading 5 over seeds, per trace x cell x target x subset.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `subset`: as in concentration_seeds.csv
- `seeds`: seeds
- `evictions_mean`: mean avoidable reusable evictions in the sample
- `distinct_states_mean`: mean distinct victim states
- `repeat_share_mean`: mean over the seeds of repeat_share (nan when any seed is nan)
- `repeat_share_min`: minimum over the seeds of repeat_share (nan when any seed is nan)
- `repeat_share_max`: maximum over the seeds of repeat_share (nan when any seed is nan)
- `top_share_mean`: mean over the seeds of top_share (nan when any seed is nan)
- `top_share_min`: minimum over the seeds of top_share (nan when any seed is nan)
- `top_share_max`: maximum over the seeds of top_share (nan when any seed is nan)

## `run_config.json`

Plan and code commits, source manifest, the SHA-256 of every population, model and published table read, check counts, timing and memory.
