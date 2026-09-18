# Phase 1 — Arrival-protection intervention findings

The intervention follows the [pre-registered plan](phase1-intervention-plan.md).
When an L1-victim arrival has a direct child in the live L2, `direct_child`
protects that arriving block only during overflow rounds of its own `admit`;
`all` applies the same protection to every feasible arrival. Other eligible
residents remain subject to the original B score and eviction rule.

## Result

The pre-registered `direct_child` intervention produced a directional benefit
at the primary `2% × 4` cell. With the `next_use` target, its paired gain over
the original B arm (`none`) was **+1.5602 input percentage points** on the
conversation trace and **+0.6318 points** on the tool-agent trace; every one of
the five seed differences was positive on both traces. The fixed `binary`
robustness target gave the same direction.

This does not establish an additional benefit from selecting arrivals by
ancestry. Protecting all feasible arrivals (`all`) achieved similar gains, and
`direct_child − all` had mixed seed signs on both traces. The intervention also
hurt both traces at `0.25% × 1`, while the middle cell was mixed. The supported
reading is therefore a capacity-local repair of the weak B trajectory. It does
not identify ancestry-based selection as the source of the repair, and it does
not make B competitive with the best matching heap policy.

## Confirmatory primary result

The table reports paired five-seed means. Intervals are 95% t-interval
half-widths over five sampling seeds; they describe seed variability only and
are not workload-population confidence intervals.

| Target | Trace | Mean net tokens | Net input points (mean ± CI half-width) | Seed range, tokens | Direction |
|---|---|---:|---:|---:|---|
| `next_use` | conversation | +916,489.6 | +1.5602 ± 0.2313 | +792,064 to +1,081,280 | 5/5 positive |
| `next_use` | tool-agent | +524,544.2 | +0.6318 ± 0.1391 | +402,317 to +628,570 | 5/5 positive |
| `binary` | conversation | +776,379.8 | +1.3217 ± 0.1774 | +672,880 to +897,526 | 5/5 positive |
| `binary` | tool-agent | +518,297.8 | +0.6243 ± 0.1065 | +462,008 to +638,163 | 5/5 positive |

This satisfies the pre-registered directional rule for `direct_child − none`
on the primary target and its fixed robustness target: the mean is positive
and all five seed-paired differences are positive on both traces.

General arrival protection achieved similar gains in this experiment. At `next_use`, `all − none` was +1.5166 points on conversation and
+0.6021 on tool-agent, again positive in all five seeds. The ancestry contrast
was only +0.0436 points and +0.0297 points, respectively, with mixed signs:

| Target | Contrast | Conversation, points | Tool-agent, points | Seed-direction reading |
|---|---|---:|---:|---|
| `next_use` | `all − none` | +1.5166 | +0.6021 | 5/5 positive on both |
| `next_use` | `direct_child − all` | +0.0436 | +0.0297 | mixed on both |
| `binary` | `all − none` | +1.3025 | +0.7622 | 5/5 positive on both |
| `binary` | `direct_child − all` | +0.0192 | −0.1380 | mixed on both |

Thus the benefit is achievable by general arrival protection; an additional
benefit from selecting arrivals by ancestry is not established.

## Capacity controls and policy context

The fixed `next_use` capacity controls rule out a uniform improvement:

| Cell | Conversation `direct_child − none` | Tool-agent `direct_child − none` | Pre-registered reading |
|---|---:|---:|---|
| `0.25% × 1` | −0.1437 points (5/5 negative) | −0.0965 points (5/5 negative) | consistent harm |
| `1% × 4` | −0.0287 points (mixed) | +0.0101 points (mixed) | heterogeneous / unresolved |
| `2% × 4` | +1.5602 points (5/5 positive) | +0.6318 points (5/5 positive) | directional benefit |

The binary controls have the same coarse pattern: both traces are negative in
all five seeds at `0.25% × 1`, negative in all five seeds at `1% × 4`, and
positive in all five seeds at the primary cell. The next-use result remains the
primary verdict.

Even after the intervention, B remains far below the best matching Phase 0.97
heap generic arm at the primary cell. The `direct_child` gap is −8,059,933.4
tokens (**−13.7209 input points**) on conversation and −8,416,576.2 tokens
(**−10.1372 points**) on tool-agent. Arrival protection repairs only a small
part of the observed B collapse; it is not evidence for a new best policy.

## Request-level trade-off

The net gains aggregate substantial movement in both directions. These are
five-seed means over paired request vectors:

| Trace | Saved requests | Lost requests | Saved tokens | Lost tokens | Net tokens |
|---|---:|---:|---:|---:|---:|
| conversation | 597.0 | 168.4 | 1,165,068.0 (1.9834 points) | 248,578.4 (0.4232 points) | +916,489.6 |
| tool-agent | 509.2 | 217.6 | 971,182.2 (1.1697 points) | 446,638.0 (0.5379 points) | +524,544.2 |

Every seed and aggregate row satisfies `net = saved − lost`. These differences
are outcomes of complete replay trajectories. They do not assign a changed
request to one particular earlier admission or eviction.

## Intervention exposure and state diagnostics

The predicate fired often enough to make the primary test substantive. Each
entry below is `whole trace / evaluation window`, averaged over five seeds.
`First override` means the unprotected first-round argmin would have been the
arrival.

| Trace | Variant | Protected offers | Protected offers with overflow | First overrides | Oversized ineligible |
|---|---|---:|---:|---:|---:|
| conversation | `direct_child` | 220,906.0 / 89,390.2 | 167,155.2 / 74,396.2 | 88,282.2 / 40,100.0 | 0 / 0 |
| conversation | `all` | 261,582.0 / 107,839.0 | 199,427.4 / 89,322.2 | 106,824.8 / 48,463.6 | 0 / 0 |
| tool-agent | `direct_child` | 218,492.2 / 87,870.2 | 167,226.8 / 73,650.6 | 43,246.2 / 22,082.2 | 0 / 0 |
| tool-agent | `all` | 255,969.0 / 104,814.0 | 188,895.4 / 84,082.4 | 50,547.8 / 25,505.8 | 0 / 0 |

Present-but-unusable tokens worsened despite the positive net hit result. On
conversation they rose from 2,220,134.2 (3.7795 input points) under `none` to
3,537,653.4 (6.0223 points) under `direct_child`; on tool-agent they rose from
2,368,568.6 (2.8528 points) to 3,510,135.4 (4.2277 points). This diagnostic is
a resident-state snapshot, so the net gain must not be described as an orphan
or present-unusable reduction.

The unchanged Phase 0.98b accounting was retained for every replay. All 180
per-seed rows satisfied the request partition and per-block identities, the 60
matched trace × cell × target × seed groups preserved requested tokens, the L1
prefix outcome, and compulsory charges across variants, and all unexplained
counts were zero. These checks show that the accounting closes. They do not
turn a whole-trajectory intervention into a one-decision causal recovery of a
Phase 0.98b loss class.

## Figures and limits

- `fig19_phase1_intervention.png` shows paired net differences across the three
  cells and two targets. It makes the primary directional gain, small-cell
  harm, and mixed ancestry contrast visible. It cannot establish statistical
  generalization beyond the two traces or ancestry specificity by itself.
- `fig20_phase1_saved_lost.png` shows saved and lost input shares together with
  their net and seed interval. It demonstrates that the positive primary net
  contains substantial request-level churn. It cannot attribute any one saved
  or lost request to a single earlier decision.

No arm was selected after seeing the results. The primary verdict remains
`direct_child − none`, `next_use`, at `2% × 4`; binary is the fixed robustness
check, and the other cells are capacity controls.

## Execution integrity and reproduction

The confirmatory run used pre-registration commit `2490b11` and code commit
`bcea656`. It completed 180 replays in 843.22 replay seconds (846.48 seconds
wall time) with 20 workers. The runner reconstructed and checked 12 fixed B
fits with maximum coefficient deviation 0, reproduced all 60 Phase 0.97
original-B rows and all 60 Phase 0.98b original-B rows, and matched 210 generic
and offline reference rows. The git HEAD and the SHA-256 manifest of the runner
and imported source files were identical at run start and end. Trace paths and
SHA-256 values are recorded in `phase1_intervention_config.json`.

The raw public trace JSONL files and Phase 0.97 victim-row `.npz` files are
ignored. Starting from a checkout that already contains the published Phase
0.97 and Phase 0.98b CSV references in `results/paper/`, first download the
traces, then create the fixed victim artifacts without overwriting those
published references:

```bash
./scripts/download_mooncake_traces.sh data/raw
mkdir -p /tmp/phase1-prerequisite-paper
.venv/bin/python scripts/run_decision_population.py \
  data/raw/conversation_trace.jsonl \
  data/raw/toolagent_trace.jsonl \
  --workers 24 \
  --output-dir results/decision_population \
  --paper-dir /tmp/phase1-prerequisite-paper
```

The temporary paper directory receives the prerequisite run's summaries. The
Phase 1 runner continues to verify the checked-in Phase 0.97 coefficients,
replays, config, and Phase 0.98b reference rows in `results/paper/`; it uses
only `results/decision_population/logs/victims_*.npz` from the prerequisite
run. Run the fixed confirmatory grid and redraw the figures with:

```bash
.venv/bin/python scripts/run_phase1_intervention.py \
  data/raw/conversation_trace.jsonl \
  data/raw/toolagent_trace.jsonl \
  --workers 20 \
  --output-dir results/phase1_intervention \
  --paper-dir results/paper

.venv/bin/python scripts/plot_phase1_intervention.py \
  --input-dir results/paper \
  --output-dir results/paper
```

The smoke run, if used, must point both output directories elsewhere and is not
part of the confirmatory artifacts.
