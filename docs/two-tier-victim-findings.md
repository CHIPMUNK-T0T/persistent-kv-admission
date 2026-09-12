# L1 Victim Stream and Tree Allocation (Phase 0.95)

Question: a persistent KV tier does not decide about every state ever seen;
it decides about the states an upper tier evicts. On that population, is there
a capacity-allocation problem that is specific to the prefix tree (ancestors
must be held for descendants to be usable, branches share ancestors, capacity
is freed only by cutting branches), and how many tokens does it account for
against generic policies?

This is a characterization. No policy is introduced, nothing is fitted, and
nothing is tuned per trace. Pipeline: `scripts/run_two_tier.py`. Artifacts:
`results/paper/victim_events_summary.csv`, `victim_events_by_depth.csv`,
`two_tier_replay.csv`, `two_tier_gate.csv`,
`two_tier_single_tier_reference.csv`, `two_tier_config.json`,
`fig11_victim_stream.png`, `fig12_two_tier_gain.png`,
`fig13_depth_allocation.png`; the full event logs (one JSONL per trace × L1
policy × L1 budget) and raw replay rows are in `results/two_tier/` (untracked).

## 1. Design

### Setting

Two tiers. **L1** is the existing prefix-closed cache (`replay.py`): every
block of every arriving request is inserted, hit or miss, and eviction removes
leaves only, by exact heap LRU or LFU. **L2** receives each L1 victim at the
moment of eviction and decides what to keep. Because every arriving block
enters L1 whatever tier served it, L1's evolution does not depend on L2: the
victim stream is identical for every L2 arm, exactly, and there is no closed
loop to approximate. A closed loop would exist only in a system that serves a
block from L2 without materialising it in L1; that design is not modelled.

The evaluation window, the byte accounting (packed incremental block bytes,
2048 bytes per token), and the trace split (first 60% warms the caches) are
those of every earlier phase. Capacities are fractions of the packed
unique-state working set: L1 ∈ {0.25, 1, 2}%, L2 ∈ {1, 2, 4, 8, 16} × L1,
cells with L1 + L2 > 10% dropped. The reusable set of both real traces fits in
about 5% of the working set (the offline comparator saturates there), so L1
sizes above 2% leave a lower tier nothing to do.

### Three models, fixed before the run

| model | hit rule | L2 constraint | role |
|---|---|---|---|
| **union closure, tree hit rule** (primary) | a block is reused when its whole ancestor chain is present in L1 ∪ L2 at request time | none; L2 is exclusive: a block requested while in L2 is materialised in L1 by that request (served from L2 if usable, recomputed otherwise) and leaves L2 | the persistent tier as a backing store |
| **independent blocks** (control) | a block in either tier counts as reused whether or not its ancestors are present | same contents rule | reachable only by a system that recomputes the holes; the difference to the tree rule is the cost of prefix dependency |
| **standalone closure** (sensitivity) | tree rule | L2 must be prefix-closed by itself: admitting a victim copies its missing ancestors from L1 and charges them; L2 is inclusive and evicts leaves only | a self-contained persistent store |

Under the union model a block in L2 whose parent is in neither tier is not
worthless: it becomes usable again when the parent is recomputed and lands in
L1. The dependency therefore lives in the hit accounting, not in an eviction
constraint.

### Arms

On every cell: L2 = LRU, LFU, 2-hit LRU (admit only states requested at least
twice), and the offline comparator (evict the farthest next use; greedy, not
optimal; same tie-break as the single-tier comparator: among equal next use,
the shorter prefix first), each under the tree rule and the independent rule,
plus LRU / LFU / offline under the standalone closure. L2 priorities are
computed once at admission, from the state's history at eviction (recency,
lifetime frequency) or its next occurrence; they cannot go stale because any
later request moves the block back to L1. Single-tier references (LRU, LFU,
offline at L1 + L2 capacity) are added for context. Everything is
deterministic; there is one seed and no confidence interval.

### Victim events

Unit: `state × eviction event` (the same state can be evicted several times).
Causal fields, known at eviction: depth, block and prefix tokens, lifetime
request count, seconds since last use, prior evictions of the state, retained
siblings, whether the parent becomes a leaf, children observed so far.
Evaluation-only fields, filled when the state is next requested: seconds
until that request; requests, arriving bytes, and distinct block bytes
strictly between the eviction's timestamp group and the reuse's timestamp
group (each state counted once, whatever the number of requests touching it;
simultaneous requests are excluded whatever their order in the file, since
they all observe the same pre-batch cache); the number
of leading blocks of the request's chain still in L1 at that moment; the
missing ancestor blocks and bytes; the tokens L2 would rescue if it held only
the block (`self`, usable only when every ancestor is still in L1) and if it
held the block plus its missing ancestors (`with ancestors`). Simultaneous
requests in one timestamp group that share the state each count, as they each
count as a hit in replay. A state not requested again before the trace ends is
right-censored, and its distances to the trace end are recorded instead.

### Go / stop rule, fixed before the run

Tree allocation becomes the centre of Research 1 only if, on both real
traces, some cell passes all three:

1. **absolute** — the offline L2 adds at least 5% of the evaluation-window
   input tokens in avoided prefill over L1 alone;
2. **dependency** — for at least one L2 policy, the independent-block control
   exceeds the tree rule by at least 10% of the control's gain;
3. **room** — the offline L2 exceeds the best generic L2 (LRU, LFU, 2-hit) by
   at least 20% of its own gain.

## 2. The victim stream

Measured window (the last 40% of each trace), L1 = LRU. LFU rows are in the
CSV and differ by at most 0.04 in the shares and 1–5 points in the ceilings.

| trace | L1 | events | distinct states | repeat evictions | requested again before trace end | of those: every ancestor still in L1 | median missing ancestor blocks | median s to reuse | median distinct bytes to reuse ÷ L1 | ceiling: L2 holds block + missing ancestors | L2 holds the block only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conversation | 0.25% | 112k | 78.4k | 0.37 | 0.30 | 0.054 | 15 | 105 | 17.1 | 35.8% | 1.9% |
| conversation | 1% | 111k | 78.6k | 0.36 | 0.30 | 0.054 | 15 | 96 | 3.9 | 34.9% | 1.8% |
| conversation | 2% | 108k | 78.1k | 0.35 | 0.29 | 0.052 | 16 | 87 | 1.8 | 32.2% | 1.7% |
| tool-agent | 0.25% | 109k | 77.0k | 0.35 | 0.30 | 0.060 | 14 | 99 | 15.7 | 23.8% | 1.5% |
| tool-agent | 1% | 108k | 77.0k | 0.34 | 0.29 | 0.059 | 14 | 90 | 3.5 | 22.4% | 1.3% |
| tool-agent | 2% | 105k | 76.5k | 0.32 | 0.28 | 0.057 | 15 | 81 | 1.6 | 20.4% | 1.2% |
| synthetic | 0.25% | 63k | 19.6k | 0.89 | 0.69 | 0.024 | 28 | 29 | 33.4 | 88.7% | 2.0% |
| synthetic | 1% | 60k | 19.7k | 0.88 | 0.68 | 0.024 | 28 | 30 | 8.6 | 83.7% | 1.9% |
| synthetic | 2% | 55k | 19.9k | 0.87 | 0.65 | 0.025 | 27 | 35 | 4.5 | 76.5% | 1.8% |

The ceilings are avoided prefill by an infinite L2 as a share of the
evaluation-window input tokens (58.7M conversation, 83.0M tool-agent, 32.1M
synthetic); L1 alone avoids 5.5% / 36.8% (conversation / tool-agent at 1%).

- **Victim reuse is substantial on the real traces.** About 30% of measured
  victim events are requested again before the trace ends (70% are
  right-censored by the 24-minute window), and a third of the events are
  repeat evictions of a state already evicted once. A lower tier of unbounded
  size would add 20–36 points of input tokens on the real traces, 77–89 on
  synthetic. The reuse is nearly the same for LRU and LFU victims.
- **A victim block is almost never usable on its own.** When a victim is
  requested again, all of its ancestors are still in L1 in only 5–7% of the
  cases (2% on synthetic); the median reuse finds 14–16 ancestor blocks
  missing. A lower tier that held only the evicted block would add 1.2–1.9%
  of input tokens; the rest of the ceiling requires the tier to hold the
  chain. This is the tree dependency in its raw form: the value of a victim
  is coupled to its ancestors, which arrive in L2 later (L1 evicts leaves, so
  a parent follows its children).
- **Distinct competing bytes until reuse, in units of L1 capacity**, is the
  working-set distance of the earlier phases on the victim population. Its
  median is 16–17 × L1 at 0.25%, 3.5–4 × at 1%, 1.6–1.8 × at 2% on the real
  traces. The share of reused victims whose distance is within 4 × L1 is
  0.05 / 0.51 / 0.73 (conversation, 0.25 / 1 / 2%) and 0.06 / 0.54 / 0.78
  (tool-agent), and it tracks what an L2 of 4 × L1 under LRU recovers in §3.
- Median seconds to reuse are 81–111 s and nearly independent of the L1 size;
  seconds since last use at eviction are 6 / 24 / 45 s (LRU) and 0 (LFU),
  which is the residence time of the upper tier, not a property of the states.

## 3. Lower-tier gain on the same stream

Extra avoided prefill over L1 alone, as a share of the evaluation-window
input tokens, on the same victim stream. L1 = LRU; the LFU rows in the CSV
give the same picture with ceilings 1–5 points lower. "Room" is
(offline − best generic) ÷ offline. The last two columns are one prefix-closed
cache of L1 + L2 bytes, for context.

**conversation**

| L1 | L2 ÷ L1 | L1 + L2 | ceiling | L2 offline | offline ÷ ceiling | best generic L2 | room | one cache: offline | one cache: LRU |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.25% | 1 | 0.5% | 35.8 | 8.4 | 0.24 | LFU 1.7 | 0.80 | 14.4 | 0.1 |
| 0.25% | 2 | 0.75% | 35.8 | 13.7 | 0.38 | LFU 3.5 | 0.74 | 18.2 | 0.5 |
| 0.25% | 4 | 1.25% | 35.8 | 21.2 | 0.59 | LFU 5.7 | 0.73 | 23.5 | 1.3 |
| 0.25% | 8 | 2.25% | 35.8 | 28.6 | 0.80 | 2-hit 11.0 | 0.61 | 29.7 | 4.6 |
| 0.25% | 16 | 4.25% | 35.8 | 35.0 | 0.98 | 2-hit 16.4 | 0.53 | 35.3 | 13.9 |
| 1% | 1 | 2% | 34.9 | 22.1 | 0.63 | 2-hit 6.2 | 0.72 | 27.7 | 2.6 |
| 1% | 2 | 3% | 34.9 | 28.7 | 0.82 | 2-hit 11.4 | 0.60 | 31.7 | 7.5 |
| 1% | 4 | 5% | 34.9 | 34.4 | 0.99 | 2-hit 16.1 | 0.53 | 34.9 | 15.5 |
| 1% | 8 | 9% | 34.9 | 34.9 | 1.00 | LRU 22.7 | 0.35 | 34.9 | 22.7 |
| 2% | 1 | 4% | 32.2 | 27.4 | 0.85 | 2-hit 11.3 | 0.59 | 31.3 | 9.4 |
| 2% | 2 | 6% | 32.2 | 32.2 | 1.00 | LRU 15.5 | 0.52 | 32.2 | 15.5 |
| 2% | 4 | 10% | 32.2 | 32.2 | 1.00 | LRU 21.6 | 0.33 | 32.2 | 21.6 |

**tool-agent**

| L1 | L2 ÷ L1 | L1 + L2 | ceiling | L2 offline | offline ÷ ceiling | best generic L2 | room | one cache: offline | one cache: LRU |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0.25% | 1 | 0.5% | 23.8 | 6.5 | 0.27 | LFU 1.9 | 0.72 | 10.5 | 0.8 |
| 0.25% | 2 | 0.75% | 23.8 | 10.1 | 0.42 | LFU 3.1 | 0.69 | 13.2 | 1.1 |
| 0.25% | 4 | 1.25% | 23.8 | 15.6 | 0.66 | LFU 4.7 | 0.70 | 17.0 | 1.7 |
| 0.25% | 8 | 2.25% | 23.8 | 20.5 | 0.86 | 2-hit 8.8 | 0.57 | 21.3 | 4.0 |
| 0.25% | 16 | 4.25% | 23.8 | 23.8 | 1.00 | 2-hit 12.6 | 0.47 | 23.8 | 11.0 |
| 1% | 1 | 2% | 22.4 | 15.4 | 0.69 | 2-hit 4.6 | 0.70 | 19.1 | 1.9 |
| 1% | 2 | 3% | 22.4 | 19.9 | 0.89 | 2-hit 8.4 | 0.58 | 21.8 | 5.7 |
| 1% | 4 | 5% | 22.4 | 22.4 | 1.00 | 2-hit 11.7 | 0.48 | 22.4 | 11.2 |
| 1% | 8 | 9% | 22.4 | 22.4 | 1.00 | LRU 16.5 | 0.26 | 22.4 | 16.5 |
| 2% | 1 | 4% | 20.4 | 18.9 | 0.92 | 2-hit 8.3 | 0.56 | 20.4 | 6.9 |
| 2% | 2 | 6% | 20.4 | 20.4 | 1.00 | LRU 11.0 | 0.46 | 20.4 | 11.0 |
| 2% | 4 | 10% | 20.4 | 20.4 | 1.00 | LRU 15.3 | 0.25 | 20.4 | 15.3 |

Synthetic (LRU L1): the offline L2 adds 7–58 points and reaches 0.08–0.70 of
its ceiling; the best generic adds 2–34; room 0.42–0.69.

- **The room over generic policies is large everywhere**, 0.25–0.80 of the
  offline gain on the real traces, and in absolute terms the offline L2 adds
  2–5 × what the best generic L2 adds. Which generic policy is best changes
  with L2 size: LFU at tiny L2, 2-hit LRU in the middle, LRU when L1 + L2
  approaches the size of the reusable set.
- **Two tiers with a fixed L1 lose little against one cache of the same
  bytes.** An exclusive L2 under LRU reproduces the single LRU of L1 + L2
  bytes to within three blocks (≤ 0.02%) in the 24 real-trace cells and nine
  blocks (≤ 0.4%) on synthetic; LFU likewise within four and six blocks.
  This is not an identity in general: it is exact only with
  uniform block sizes, one request per timestamp group, and capacities that
  are multiples of the block size. Otherwise the split wastes the packing
  slack of both tiers and recency ties are broken differently, which is why
  synthetic, with its varied block sizes, is the trace that falls short.
  The offline L2 is 4–6 points of input below the single-cache offline at
  L2 = L1, 1.4–2.3 points at 0.25% × 4, and within 0.5 points once L1 + L2
  reaches about 4% of the working set. The headroom seen here is the
  headroom of Phases 0–0.9, on the population a persistent tier actually
  receives.
- **Per-block greedy ordering reaches the ceiling once L2 is a few times
  L1.** The offline L2 recovers 0.98–1.00 of the infinite-L2 ceiling at
  L2 ≥ 4 × L1 (1% and 2% L1) and at 16 × L1 (0.25%), so there is no
  set-level allocation left to find in those cells. Below that it recovers
  0.24–0.89, limited by capacity: the reused victims have distinct-byte
  distances larger than L1 + L2 (§2).

## 4. Prefix dependency: what it costs

The control removes the ancestor requirement from the hit accounting and
keeps the L2 contents rule. Dependency cost = (independent − tree) ÷
independent, in extra avoided tokens.

| L2 policy | real traces, 48 cells | synthetic, 12 cells |
|---|---|---|
| LRU | **0 tokens in every cell** | 0 in every cell |
| LFU | **0 tokens in every cell** | 0 in every cell |
| 2-hit LRU | **0 tokens in every cell** | 0 in every cell |
| offline next use | 0.10–0.13 at L2 = L1 (0.25%), 0.08–0.10 at 2 × L1 (0.25%), ≤ 0.035 elsewhere, 0 where the ceiling is reached | 0.35 / 0.20 / 0.09 at 0.25% × 1 / 2 / 4, 0.01–0.10 elsewhere |

**Why the generic policies pay nothing.** Recency and lifetime frequency are
monotone along an ancestor chain: every request that touches a block touches
its parent, so the parent's last use is at least as recent and its count at
least as large. L1 evicts leaves only, so a parent enters L2 after all of its
children, with a priority at least as high, and it leaves L2 no earlier
(ties go to the earlier admission, the child). Ancestors therefore outlive
descendants across L1 ∪ L2 under LRU, LFU, and 2-hit LRU, and a block in L2 is
never orphaned: the tree rule and the independent rule count the same hits.
The event log says the same thing from the other side: when a victim is
reused its ancestors are almost never in L1 (§2), yet under these policies
they are in L2, because they were evicted from L1 later and ranked no lower.

**Why the offline arm pays a little, and only at tiny L2.** Next-use time is
monotone too (a parent's next use is at or before its child's), but when
parent and child share their next use the comparator breaks the tie by
shorter prefix first, which evicts the parent before the child and orphans
it. A diagnostic step of the same run (`two_tier_offline_tiebreak.csv`;
`run_two_tier(..., offline_tiebreak="deeper_first")`, twelve cells at
0.25% × 1 / 2 and 1% × 1 on both real traces, both L1 policies, plus the
same cells on synthetic) re-runs the offline L2 with the deeper block
evicted first among equal next uses:

| cell (L1, L1 size, L2 ÷ L1) | dependency cost, prefix-first ties | dependency cost, deeper-first ties |
|---|---:|---:|
| conversation LRU 0.25% × 1 / × 2 / 1% × 1 | 0.119 / 0.088 / 0.017 | 0.000 / 0.000 / 0.000 |
| conversation LFU 0.25% × 1 / × 2 / 1% × 1 | 0.128 / 0.095 / 0.021 | 0.000 / 0.000 / 0.000 |
| tool-agent LRU 0.25% × 1 / × 2 / 1% × 1 | 0.099 / 0.091 / 0.014 | 0.000 / 0.000 / 0.000 |
| tool-agent LFU 0.25% × 1 / × 2 / 1% × 1 | 0.116 / 0.081 / 0.012 | 0.000 / 0.000 / 0.000 |

With a depth-consistent tie-break the offline L2 pays nothing either, and its
tree-rule gain rises to the independent-rule gain in every diagnostic cell
(synthetic: 0.35 / 0.20 / 0.10 → 0.000 as well).
The grid keeps the original tie-break so that the offline arm is the same
comparator as in every earlier phase; the diagnostic is reported, not
substituted.

**Standalone closure.** Forcing L2 to be prefix-closed on its own, with
ancestor copies charged, changes the offline L2 gain by −11% to +8% on the
real traces (negative means the inclusive standalone tier did better, which
happens at tiny L2 because it keeps blocks that were reused from L2 instead
of handing them back to L1). For L2-LRU the copies cost 1–72% of the union
gain, so the closure choice matters for a generic policy and not for the
comparator. Neither closure creates a dependency problem the union model
lacks.

**What this leaves for tree-aware allocation.** With ancestors never orphaned
and the greedy offline L2 at 0.98–1.00 of the ceiling once L1 + L2 reaches
about 4% of the working set (0.25% × 16, 1% × 4, 2% × 2 on both real
traces), an allocation that reasons about the set (ancestor cost, shared
ancestors, branch cutting) has at most 2% of the offline gain to add there.
Below that total the offline L2 sits at 0.22–0.92 of the ceiling (0.59–0.66
at 0.25% × 4); the gap is capacity, and the part of it attributable to
missing ancestors is the tie-break artefact above. That bounds what
dependency explains. It does not test whether a different allocation across
branches or depths would beat per-block ordering there, because the
independent-block control keeps the same contents and changes only the hit
accounting.

## 5. Depth allocation

L1 = LRU 1%, L2 = 4 × L1, union closure, tree rule (`fig13_depth_allocation.png`,
`victim_events_by_depth.csv`). Shares by block depth; "bytes held" is L2
byte-seconds inside the evaluation window, the same window as the gains:

| depth | conversation: victims / rescue ceiling / L2-LRU bytes held / L2-LRU gain / offline bytes held / offline gain | tool-agent: same |
|---|---|---|
| 2–3 | 0.09 / 0.10 / 0.08 / 0.11 / 0.06 / 0.10 | 0.08 / 0.10 / 0.08 / 0.11 / 0.05 / 0.10 |
| 4–7 | 0.14 / 0.15 / 0.14 / 0.16 / 0.10 / 0.15 | 0.15 / 0.16 / 0.15 / 0.16 / 0.09 / 0.16 |
| 8–15 | 0.21 / 0.21 / 0.21 / 0.22 / 0.14 / 0.21 | 0.25 / 0.22 / 0.24 / 0.22 / 0.12 / 0.22 |
| 16–31 | 0.25 / 0.23 / 0.25 / 0.23 / 0.17 / 0.23 | 0.23 / 0.24 / 0.24 / 0.23 / 0.14 / 0.24 |
| 32–63 | 0.19 / 0.17 / 0.19 / 0.16 / 0.16 / 0.17 | 0.17 / 0.17 / 0.17 / 0.16 / 0.13 / 0.17 |
| 64+ | 0.13 / 0.13 / 0.14 / 0.11 / 0.38 / 0.13 | 0.11 / 0.12 / 0.12 / 0.11 / 0.46 / 0.12 |

(Depth 1 holds no victims on the real traces: root blocks are shared by so
many chains that L1 never reaches them.)

- Where the victims come from and where the rescue value lies coincide by
  depth within 0.03. Capacity is not consumed at one depth and earned at
  another.
- L2-LRU's held bytes follow its gain by depth within 0.03; shallow blocks
  (2–7) yield slightly more per byte and the deepest (64+) slightly less
  (0.14 held for 0.11 gained, conversation).
- The offline L2 keeps far more deep bytes than it earns from them (0.38 /
  0.46 of its byte-seconds at depth 64+ for 0.13 / 0.12 of its gain): those
  are long chains with a certain but distant next use, held because the
  comparator knows they will be needed. That is a symptom of foresight, not
  of a depth preference to imitate.
- Ancestors are still in L1 at reuse for 53% of depth-2–3 victims and for
  none deeper than 7, which is why the "block only" value in §2 is concentrated
  at shallow depth and small.

## 6. Required statements

### Confirmed

- The persistent tier's population has plenty of reuse: ~30% of L1 victim
  events on the real traces are requested again within the window, an
  infinite lower tier would add 20–36 points of input tokens, and the offline
  L2 adds 5–35 points at every L1 / L2 size tried (gate 1 passes in every
  cell).
- Generic lower-tier policies leave 25–80% of the offline L2 gain on the
  table in every cell (gate 3 passes everywhere); the offline L2 adds 2–5 ×
  what the best of LRU / LFU / 2-hit adds. This is the Phase 0–0.9 headroom
  seen from the victim stream.
- The value of a victim block is coupled to its ancestors (only 5–7% of
  reuses find them in L1; median 14–16 missing), so a lower tier has to hold
  chains.
- Under LRU, LFU, and 2-hit LRU it does so for free: recency and frequency
  are monotone along the chain and L1 evicts leaves, so ancestors outlive
  descendants across both tiers. The dependency cost is exactly zero tokens in
  all 60 cells, for all three policies. The same holds for next-use ordering
  with a depth-consistent tie-break (diagnostic, 12 cells).
- Two tiers with a fixed L1 behave like one cache of L1 + L2 bytes on the
  real traces: LRU reproduces single-tier LRU within nine blocks (exact only
  under uniform blocks, single-request timestamp groups, and block-aligned
  capacities); offline reaches 0.98–1.00 of the infinite-L2 ceiling once
  L1 + L2 is about 4% of the working set or more (0.25% × 16, 1% × 4,
  2% × 2) and 0.22–0.92 below that.
- Distinct-byte reuse distance in units of L1 capacity (the working-set
  axis) describes the victim population: median 16–17 × L1 at 0.25%, 3.5–4 ×
  at 1%, 1.6–1.8 × at 2%, and the share within 4 × L1 tracks the LRU L2 gain.

### Refuted

- That the policies tested lose reuse to missing ancestors. Gate 2 fails in
  45 of 48 real-trace cells; the three passes (0.25% L1 with L2 = L1) come
  only from the offline comparator's tie-break, which the diagnostic
  removes. Independent-block accounting adds nothing to any generic policy.
  What this refutes is narrower than "tree-aware allocation has no
  headroom": the control keeps each policy's contents and changes only the
  hit accounting, so it shows that no ancestor-loss exists for these
  policies, not that allocating capacity differently across branches or
  depths could not gain. Where the greedy offline L2 reaches 0.98–1.00 of
  the ceiling (L1 + L2 at about 4% of the working set or more) any such
  allocation has ≤ 2% of the offline gain to add; below that the question is
  open, and no positive grounds for making it the centre were found.
- That capacity is spent at one depth and earned at another. Victim bytes,
  rescue value, and L2-LRU's held bytes coincide by depth within 0.03.
- That the closure model decides the answer. Standalone prefix-closed L2
  changes the offline gain by −11% to +8%.

### Unresolved

- Whether victim reuse can be ranked from causal event fields (lifetime
  count, seconds since last use, prior evictions, depth, retained siblings)
  well enough to close part of the 25–80% room. This is the Phase 0.5–0.9
  signal question on the population the persistent tier sees; it was not
  measured here, and the earlier result that history features fail on the
  eviction-candidate population is a warning, not an answer, because the
  populations differ.
- Whether any of this survives an L1 that is tiny relative to a persistent
  tier of hours of traffic, or arrival rates that vary. The reusable set of
  these traces fits in 5% of the working set; L2 ≥ 16 × L1 already reaches
  the ceiling at 0.25% L1.
- The offline L2 is greedy per block; with L1 + L2 below about 4% of the
  working set it reaches 0.22–0.92 of the ceiling on the real traces, and an
  optimal set-level allocation could in principle sit above it. The
  independent-block control bounds what missing ancestors explain of that
  gap (≤ 13%, and 0 with the tie-break fixed), not what a better allocation
  or ordering could add.
- Synthetic differs in kind: 69% of victims return, chains are deeper, and
  the offline dependency cost reaches 0.35 at the smallest L2. It is not
  used for the decision.

### Research decision

Tree allocation is not made the centre of Research 1: under the go / stop
rule fixed before the run, the dependency criterion fails on both real
traces except through a tie-break artefact, so no ancestor-loss was observed
for the policies tested and no positive grounds for a tree-specific
allocation problem were found. This is a provisional judgement about
grounds, not a proof that no tree-aware allocation could gain (§7). The
two-tier setting stays as the
evaluation setting, because it is the population a persistent tier decides
about and it reproduces the single-tier headroom. What remains open on that
population is the same question as before, now more sharply posed: ranking
victims by future reuse, where generic policies leave 25–80% of the offline
gain, and where the prefix tree costs nothing as long as the ranking is
monotone along chains. The next measurement, listed in
`docs/experiment-plan.md`, should be taken on the lower tier's own decision
population, an arriving victim against the L2 residents it would displace,
rather than on the victim stream as a whole, so that the population
mismatch of Phase 0.5 is not repeated; the choice is the user's.

## 7. Method limits

- The upper tier is exact heap LRU or LFU; the sampled-leaf eviction of
  Phases 0.5–0.9 is not used, so the victim streams are deterministic and
  reproducible, and no seed dispersion is reported. Ties between equal
  scores are broken by insertion order, not by set iteration, so the
  results do not depend on the interpreter's hash seed (tested).
- Byte-seconds and admitted bytes by depth are restricted to the
  evaluation window, like every token counter; event counts (evictions,
  admissions, rejections) are whole-trace, and each event carries its own
  window flag.
- Reuse distances exclude the reuse's own timestamp group, so a reuse in the
  very next group has distance zero; distances are therefore comparable
  across traces with different batching, but not identical to a per-request
  reuse distance.
- The offline L2 is greedy farthest-next-use per block. It reaches the
  infinite-L2 ceiling at L2 ≥ 4 × L1 on the real traces, which bounds what any
  set-level allocation could add there, but it is not an optimum at smaller
  L2.
- The infinite-L2 ceiling is the sum of rescue tokens over victim events whose
  reuse falls in the evaluation window; it assumes L2 also holds every missing
  ancestor at reuse time.
- The independent-block control is an accounting change for the same L2
  contents, not a policy that exploits independence. It bounds the reuse a
  given policy loses to missing ancestors; it says nothing about whether
  allocating capacity differently across branches or depths would gain, so
  "no dependency cost" is not "no room for tree-aware allocation".
- The two-tier / single-cache correspondence is empirical on these traces.
  It is exact only with uniform block sizes, one request per timestamp
  group, and block-aligned capacities; elsewhere the two-tier union is
  bounded above by the single cache up to packing slack and recency
  tie-breaks (≤ 1.0% short on synthetic).
- The standalone closure keeps blocks reused from L2 (inclusive); the union
  closure moves them back to L1 (exclusive). The two therefore differ in more
  than ancestor copies, and small negative "costs" of the standalone arm at
  tiny L2 come from that.
- Real traces are 59 minutes long with a 24-minute evaluation window; about
  70% of measured victim events are censored. Nothing here speaks to
  persistence across hours, restarts, or user sessions.
- Prefix-closed retention and the contiguous-prefix hit rule are this
  repository's model. The Mooncake paper describes block-level storage with
  LRU eviction (§3.2.1); the tree rule is not claimed to be what every system
  does.
