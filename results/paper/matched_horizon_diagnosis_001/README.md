# Matched-horizon diagnosis on saved decision logs

Pre-registration: `docs/matched-horizon-diagnosis-plan.md`. Logs: the held-out `pi0` decision populations of the published on-policy run, 2 traces x 6 cells x 2 targets x 5 seeds, each a reservoir of at most 40,000 whole decisions at least 600 s before the trace end, loaded by the SHA-256 the published table records; no `pi3` population is read. Scorer: the published `pi0` model of the lineage; its victim on a logged decision is the first minimum, in stored candidate order, of (score on the stored features, stored tie-break); no hybrid, protection or override. Horizons h in {60, 150, 300, 600} s: `label_h = -log1p(min(next_use_delta_s, h))`, and a candidate is reusable at h when its next use is at most h away. The matched horizon h* of a cell: 60 s at 0.25%x1, 150 s at 0.25%x4 and 1%x1, 300 s at 2%x1, 600 s at 1%x4 and 2%x4, the same on both traces. The error kinds, their precedence, `m3`, `m4` and the conditional rates are the ranker-error diagnosis's with h in place of 600, computed by its own code; at h = 600 the class, rate and per-seed rate tables equal its published ones value for value (`reproduction.csv`). The within-decision concordance of a key for the reuse bit at h is, over the decisions with both a reusable and a non-reusable candidate, the share of (reusable, non-reusable) candidate pairs in which the key is lower for the non-reusable one (it would be evicted first), an exact tie counting one half, averaged with each decision weighing one; the ranker's key is (score, stored tie-break), recency's the stored tie-break alone, and a uniformly drawn victim has 0.5. `U` is extra avoided prefill tokens (avoided minus L1-avoided) in points of input tokens, from the published `all16` replay rows. "Consistent" means the same sign in all five seed-paired values; seed signs list seeds 0-4 as +, -, 0, or n for nan.

Nothing was replayed or fitted. The populations are the frozen ranker's own store; a store kept by the exact bit would present other candidate sets. h* was read from replays of the same two traces after the fact. Concordance and rates on logged candidates are properties of a score on those sets, not utility. Both published rankers were fitted to 600-second targets.

## `checks.csv`

Required checks, one row per pi0 population (trace x cell x target x seed); the diagnosis's check table.

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

## `classes_seeds.csv`

Reading 1 per seed and horizon: the four error kinds of the pi0 population's decisions by their logged victim, with labels and reuse at h, for every decision and separately for decisions whose victim is the arrival or a resident, built by the diagnosis's own functions (its summary at h = 600 is compared in reproduction.csv).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
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

Reading 1 over seeds, per trace x cell x target x horizon x subset x class. The h = 600 rows equal the published diagnosis's `classes.csv` (reproduction.csv).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
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

Reading 2 per seed and horizon, on the pi0 population, with reuse at h. The h = 600 rows equal the published diagnosis's `rates_seeds.csv` (reproduction.csv).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
- `rate`: `avoidable`: share of the decisions with both a reusable and a non-reusable candidate whose victim is reusable; `order`: share of the decisions whose candidates are all reusable and not all of one label whose victim's label is above the minimum
- `eligible_decisions`: decisions the rate is taken over
- `ranker`: the rate of the frozen ranker (pi0), by its logged victim
- `recency`: the rate of recency: the first minimum of the stored tie-break
- `uniform`: the expected rate of a victim drawn uniformly from the candidates

## `rates.csv`

Reading 2 over seeds, per trace x cell x target x horizon x rate. The h = 600 rows equal the published diagnosis's `rates.csv` (reproduction.csv).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
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

Reading 2 counts, per horizon selection x target x rate, out of the trace x cells.

- `horizon`: `60`, `150`, `300` or `600` (every cell at that h), or `matched` (every cell at its own h*)
- `matched`: the row counts every cell at its own h*
- `target`: training target
- `rate`: avoidable or order
- `cells`: trace x cells counted (12)
- `better_than_recency`: cells reading better against recency
- `worse_than_recency`: cells reading worse against recency
- `mixed_vs_recency`: cells reading mixed against recency
- `better_than_uniform`: cells reading better against the uniform expectation (descriptive, not registered)
- `worse_than_uniform`: cells reading worse against it (descriptive, not registered)
- `mixed_vs_uniform`: cells reading mixed against it (descriptive, not registered)

## `statistics_seeds.csv`

m3 and m4 at each horizon, per seed, for the logged victim and for the published pi0 model's recomputed victim (the first minimum of its score and the stored tie-break), over every decision and over the decisions whose victim (of that source) is the arrival or a resident.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
- `victim_source`: `logged` or `scorer_pi0`
- `subset`: `all`, `arrival` (the victim is the arrival) or `resident`
- `decisions`: decisions in the subset
- `excess_sum`: their summed label_h(victim) - min label_h
- `m3`: excess_sum / decisions
- `avoidable_count`: decisions whose victim is reusable at h while some candidate is not
- `m4`: avoidable_count / decisions

## `concordance_seeds.csv`

Reading 3 per seed and horizon: the within-decision concordance of each chooser's key for the reuse bit at h, over the decisions with both a reusable and a non-reusable candidate.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `seed`: lineage seed: the replays' sampling seed and the reservoir's seed
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
- `decisions`: decisions with both a reusable-at-h and a non-reusable-at-h candidate
- `ranker`: mean over those decisions (each weighing one) of the share of (reusable, non-reusable) candidate pairs in which the key (score of the published pi0 model, stored tie-break), compared lexicographically, is lower for the non-reusable candidate, i.e. would evict it first; an exact tie of the key counts one half
- `recency`: the same with the stored tie-break alone as the key
- `uniform`: a uniformly drawn victim: 0.5 exactly

## `concordance.csv`

Reading 3 over seeds, per trace x cell x target x horizon.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
- `seeds`: seeds
- `decisions_mean`: mean decisions the concordance is taken over
- `ranker_mean`: mean over the seeds of the ranker's concordance (nan when any seed is nan)
- `ranker_min`: minimum over the seeds of the ranker's concordance (nan when any seed is nan)
- `ranker_max`: maximum over the seeds of the ranker's concordance (nan when any seed is nan)
- `ranker_ci95_half`: 95% t half-width over the seeds of the ranker's concordance (nan when any seed is nan)
- `recency_mean`: mean over the seeds of recency's concordance (nan when any seed is nan)
- `recency_min`: minimum over the seeds of recency's concordance (nan when any seed is nan)
- `recency_max`: maximum over the seeds of recency's concordance (nan when any seed is nan)
- `recency_ci95_half`: 95% t half-width over the seeds of recency's concordance (nan when any seed is nan)
- `uniform`: a uniformly drawn victim's concordance, 0.5
- `ranker_minus_recency_mean`: mean of the seed-paired ranker - recency
- `ranker_minus_recency_seed_signs`: signs of ranker - recency, seeds 0-4
- `ranker_minus_recency_reading`: consistent_gain (the ranker's key separates better in every seed), consistent_loss or mixed; descriptive
- `ranker_minus_uniform_mean`: mean of ranker - 0.5
- `ranker_minus_uniform_seed_signs`: signs of ranker - 0.5, seeds 0-4
- `ranker_minus_uniform_reading`: consistent_gain / consistent_loss / mixed; descriptive

## `concordance_reading.csv`

Reading 3's registered prediction (the fit-horizon hypothesis), per trace x cell x target: the ranker's concordance for the h*-bit against the 600-second bit on the same populations, five-seed means. Registered for `next_use` in the trace x cells with h* < 600 s; the other rows are descriptive.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h_star`: the cell's matched horizon h* in seconds
- `seeds`: seed-paired values
- `registered`: the prediction is registered here (target next_use and h* < 600 s)
- `ranker_at_h_star_mean`: five-seed mean of the ranker's concordance at h* (concordance.csv)
- `ranker_at_600_mean`: five-seed mean of the ranker's concordance at 600 s
- `h_star_minus_600_mean`: mean of the seed-paired concordance at h* - at 600 s
- `h_star_minus_600_seed_signs`: signs of that difference, seeds 0-4
- `recency_at_h_star_mean`: five-seed mean of recency's concordance at h*
- `recency_at_600_mean`: five-seed mean of recency's concordance at 600 s
- `below_600`: ranker_at_h_star_mean < ranker_at_600_mean (strictly; equal is not below)
- `prediction`: `holds` (registered and below_600), `fails` (registered and not below_600) or `not_registered`

## `concordance_counts.csv`

Reading 3 counts, per target.

- `target`: training target
- `cells`: trace x cells (12)
- `cells_h_star_below_600`: trace x cells with h* < 600 s (8)
- `registered`: trace x cells where the prediction is registered (8 for next_use, 0 for binary)
- `holds`: registered trace x cells where it holds
- `fails`: registered trace x cells where it fails
- `below_600`: trace x cells with h* < 600 s where the h* concordance is below the 600-second one (for binary: descriptive)

## `composition_reading.csv`

Reading 1's prediction, per trace x cell x target: the avoidable reusable eviction's share of the label_h* excess of every decision (subset `all` of classes.csv at h*).

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h_star`: the cell's matched horizon h* in seconds
- `seeds`: seeds
- `excess_share_mean`: five-seed mean of the share at h* (classes.csv)
- `excess_share_min`: its minimum over seeds
- `excess_share_max`: its maximum over seeds
- `decision_share_mean`: five-seed mean of the class's share of decisions at h*
- `excess_share_at_600_mean`: the same share at 600 s (the diagnosis's)
- `holds`: excess_share_mean >= 0.99 (the registered prediction; nan does not hold)
- `holds_every_seed`: excess_share_min >= 0.99 (every seed; descriptive)

## `composition_counts.csv`

Reading 1 counts, per target and over both (`all`, out of 24).

- `target`: next_use, binary, or all
- `cells`: trace x cell x targets counted
- `holds`: rows of composition_reading.csv where the prediction holds
- `holds_every_seed`: rows where it holds in every seed

## `bridge.csv`

Reading 4, descriptive, per trace x cell x target x horizon: the ranker's m4 at h by its logged victim, beside the published utility gaps of the trace x cell against that target's ranker's own replay (all16; U is extra avoided prefill tokens over L1 alone in points of input tokens). The registered reading is the `matched` rows.

- `trace`: trace name
- `l1_fraction`: L1 capacity as a fraction of the trace's packed working set
- `l2_multiplier`: L2 capacity as a multiple of the L1 capacity
- `cell`: cell label `l1=<fraction>,l2x<multiplier>`
- `target`: training target of the published rankers (`next_use` or `binary`)
- `h`: reuse horizon h in seconds: label_h = -log1p(min(next_use_delta_s, h)); a candidate is reusable at h when its next use is at most h away
- `h_star`: the cell's matched horizon h* in seconds (the plan's table, the same on both traces)
- `matched`: h equals h*
- `seeds`: seeds
- `ranker_arm`: the ranker's own published all16 replay in error_location_001 the gaps are taken against: `learned` (frozen pi0 next_use) for next_use, `pi0_binary` for binary
- `m4_all_mean`: mean over the seeds of m4 over every decision (nan when any seed is nan)
- `m4_all_min`: minimum over the seeds of m4 over every decision (nan when any seed is nan)
- `m4_all_max`: maximum over the seeds of m4 over every decision (nan when any seed is nan)
- `m4_all_ci95_half`: 95% t half-width over the seeds of m4 over every decision (nan when any seed is nan)
- `m4_resident_mean`: mean over the seeds of m4 over the decisions whose logged victim is a resident (nan when any seed is nan)
- `m4_resident_min`: minimum over the seeds of m4 over the decisions whose logged victim is a resident (nan when any seed is nan)
- `m4_resident_max`: maximum over the seeds of m4 over the decisions whose logged victim is a resident (nan when any seed is nan)
- `m4_resident_ci95_half`: 95% t half-width over the seeds of m4 over the decisions whose logged victim is a resident (nan when any seed is nan)
- `resident_decisions_mean`: mean number of resident-victim decisions
- `u_label_minus_ranker_mean`: mean over the seeds of U(label) - U(ranker_arm), seed by seed from the published error_location_001 replay rows (each seed's token difference in points of its input tokens) (nan when any seed is nan)
- `u_label_minus_ranker_min`: minimum over the seeds of U(label) - U(ranker_arm), seed by seed from the published error_location_001 replay rows (each seed's token difference in points of its input tokens) (nan when any seed is nan)
- `u_label_minus_ranker_max`: maximum over the seeds of U(label) - U(ranker_arm), seed by seed from the published error_location_001 replay rows (each seed's token difference in points of its input tokens) (nan when any seed is nan)
- `u_label_binary_minus_ranker_mean`: mean over the seeds of U(label_binary_h) - U(ranker_arm), the same way; nan where label_binary_h was not replayed for the cell (150 s outside the fill-in's cells) (nan when any seed is nan)
- `u_label_binary_minus_ranker_min`: minimum over the seeds of U(label_binary_h) - U(ranker_arm), the same way; nan where label_binary_h was not replayed for the cell (150 s outside the fill-in's cells) (nan when any seed is nan)
- `u_label_binary_minus_ranker_max`: maximum over the seeds of U(label_binary_h) - U(ranker_arm), the same way; nan where label_binary_h was not replayed for the cell (150 s outside the fill-in's cells) (nan when any seed is nan)
- `label_binary_arm`: the published arm read for U(label_binary_h), empty when none
- `label_binary_source`: its published directory (horizon_control_001 or horizon_fill_001), empty when none

## `bridge_spearman.csv`

Reading 4's description: Spearman's rank correlation (Pearson's on average ranks, ties averaged) over the trace x cells of a target between a five-seed mean m4 and a five-seed mean U gap of bridge.csv. Twelve points, not a test.

- `target`: training target
- `ranker_arm`: the replay the target's U gaps are taken against (as in bridge.csv)
- `horizon`: `matched` (each cell at its own h*: the registered description) or a fixed h for every cell
- `matched`: horizon is `matched`
- `m4_subset`: `all` (m4_all_mean) or `resident` (m4_resident_mean)
- `u_gap`: `label` (U(label) - U(ranker_arm)) or `label_binary` (U(label_binary_h) - U(ranker_arm) at the row's h)
- `cells`: trace x cells of the target
- `points`: of those, cells with both values finite
- `spearman`: the rank correlation; nan unless points equals cells

## `reproduction.csv`

Required check: the h = 600 rows of classes.csv, rates.csv and rates_seeds.csv against the published ranker-error diagnosis tables, value for value as written (exact; no tolerance). The run publishes nothing unless every row is equal.

- `table`: the table compared
- `published_path`: the published table read
- `published_sha256`: its SHA-256
- `published_rows`: its rows
- `rows_at_600`: rows of this run's table at h = 600
- `key_columns`: the columns identifying a row
- `common_columns`: published columns that this table also has, all compared
- `published_columns_absent`: published columns this table lacks (must be empty)
- `missing_rows`: published rows without a row here
- `extra_rows`: rows here without a published row
- `duplicate_keys`: repeated row keys on either side
- `compared_values`: values compared
- `differing_values`: values not equal as written
- `first_difference`: the first of them, empty when none
- `equal`: no missing, extra, duplicate or differing value and no absent column

## `run_config.json`

Plan and code commits, source manifest, the SHA-256 of every population, model and published table read, the horizons and the matched map, check counts, reading counts, timing and memory.
