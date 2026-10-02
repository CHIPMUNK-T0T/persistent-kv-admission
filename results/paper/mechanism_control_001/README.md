# Mechanism control: score ladder under four L2 mechanisms

Pre-registration: `docs/mechanism-control-plan.md`. Rungs: `lru` (sampled LRU key), `learned` (frozen pi0 `next_use` ranker), `label` (exact training target at the decision), `offline` (heap offline comparator's key on the sampled path). Mechanisms: eligibility `all` | `leaf` x sample width K = 16 | 64; `all16` is the published one. U is extra avoided prefill tokens over L1 alone; points are 100 x the share of window input tokens. `H_off` and `H_lru` are the published heap rows of `decision_population_replay_seeds.csv`, not rerun. Seed intervals describe sampling-seed variability only.

- `replay_seeds.csv`: one row per replay (trace x cell x mechanism x rung x seed) with the replay counters, the Phase 0.98b attribution columns, the reproduction mark of the base mechanism, timing and worker memory.
- `replay.csv`: five-seed mean, std, 95% t half-width, min and max of the main replay metrics per trace x cell x mechanism x rung.
- `decomposition_seeds.csv`: per seed, the heap references, the four U's, the four terms of `T = H_off - U(lru)` in exact integer tokens, in points and as shares of T, the per-seed dominant label and the per-seed ladder pairs.
- `decomposition.csv`: the same aggregated over seeds, with the shares and the dominant label taken on the five-seed means, the seed sign counts of every term, and whether the dominant term has one sign in every seed.
- `mechanism_effects.csv`: seed-paired differences of U between mechanisms (`leaf16 - all16`, `all64 - all16`, `leaf64 - leaf16`, `leaf64 - all16`) for every rung, with the seed sign counts and the consistent-gain / consistent-loss / mixed reading.
- `ladder.csv` and `ladder_inversions.csv`: for every trace x cell x mechanism, how many seeds satisfy each adjacent pair of `lru <= learned <= label <= offline`, the ordered flag, and every pair that fails in at least one seed.
- `capacity_pattern.csv`: mean and seed signs of `U(learned) - U(lru)` and `U(label) - U(lru)` per trace x cell x mechanism.
- `ladder_by_mechanism.png`: U per rung, one line per mechanism with its seed min-max band, heap references as horizontal lines.
- `gap_decomposition.png`: the four terms of T per mechanism as stacked bars (negative terms below zero), T marked.
- `run_config.json`: plan and code commits, source manifest, trace, model and reference hashes, grid, check counts, timing and memory.
