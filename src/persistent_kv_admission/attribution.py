"""Phase 0.98 — which eviction decisions cost the reuse, and what they stranded.

Purely descriptive. Nothing here fits a model, adds a feature, or changes a
policy: the Phase 0.97 arms are replayed exactly as they were and three
read-only hooks on `run_two_tier` record what the replay was already doing.

Three measurements, all on the union closure under the tree hit rule:

* **Loss attribution.** For one measured request the blocks beyond the L1
  prefix are either served from L2 (a contiguous run under the tree rule), or
  the run stops at the first block `m` that L2 does not hold. The tokens of `m`
  are the *root loss* and are charged to the last decision that removed `m`
  from L2 — a rejection of the arriving victim, an eviction of a resident, or
  "compulsory" when L2 never held it. The blocks after `m` that L2 *does* hold
  are *present-unusable* (the hole at `m` makes them unreachable) and are
  charged to the same decision; the blocks after `m` that L2 does not hold are
  *downstream-absent* and are charged to nothing, because the root loss already
  explains why the request stopped. The four classes partition the blocks
  beyond the prefix exactly, which `LossPartitionError` enforces per request.
* **Orphaning.** When a resident is evicted, its L2-resident descendants become
  present-unusable at that instant. The descendants are walked from the evicted
  state downwards and the walk stops at a child L2 does not hold, because a
  subtree below an existing hole was already unreachable.
* **Decision-type regret and the ranking split**, computed from the decisions
  `decisionpop.OnPolicyDecisionLogger` already records: the score tuples the
  store really compared and the candidate it really removed.

The categories are properties of the replay, not of any model, and every count
is a count of events the replay performed.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from .crossworkload import HYPERPARAMETERS
from .decisionpop import MIN_DISTINCT_LABELS, spearman, target_column
from .phase05 import _average_ranks, fast_ranking_metrics
from .trace import Trace
from .twotier import L2_REMOVAL_KINDS

# Why a block a request needed was not in L2 at that instant. "unexplained"
# must stay empty: a state that was requested before and is in neither tier now
# was necessarily offered to L2 at its L1 eviction, so it carries a record.
LOSS_CATEGORIES = ("rejected", "evicted", "compulsory", "unexplained")
# The categories a decision can be blamed for; the pre-registered "total loss"
# is the root plus present-unusable tokens of exactly these two.
DECISION_CATEGORIES = ("rejected", "evicted")
# Phase 0.97's precision@k argument; only the AUC is read here, which does not
# depend on it, but the call is kept identical to the one it is compared with.
PRECISION_K = 100


class LossPartitionError(AssertionError):
    """The blocks beyond the L1 prefix did not partition into the four classes."""


def children_map(trace: Trace) -> dict[str, list[str]]:
    """`parent -> children`, each list in trace order.

    Never a set: a set of strings iterates in an order that depends on
    per-process hash randomisation, and a diagnostic that has to be replayable
    cannot afford that even where only a sum is taken.
    """
    out: dict[str, list[str]] = {}
    for state_id, meta in trace.states.items():
        if meta.parent_id is not None:
            out.setdefault(meta.parent_id, []).append(state_id)
    return out


class AttributionCollector:
    """The request hook, the removal hook, and the orphaning walk in one object.

    Handed to `run_two_tier` as `l2_request_hook=collector.on_request` and
    `l2_removal_hook=collector`; the replay calls `attach` with the union store
    so that the orphaning walk can read the resident set. Every counter is a
    plain integer total, so a whole-trace replay costs no memory beyond the
    last-removal record (one short string per state that ever left L2).

    Window handling follows the replay's own: the loss attribution counts only
    requests the replay measured, and the eviction counters are kept both for
    the whole trace (the warm-up is part of the mechanism) and for the window.
    """

    def __init__(
        self,
        trace: Trace,
        measure_from_ms: float | None = None,
        bytes_per_token: int = HYPERPARAMETERS["bytes_per_token"],
        size_model: str = HYPERPARAMETERS["size_model"],
    ) -> None:
        self.trace = trace
        self.states = trace.states
        self.occurrences = trace.occurrences_ms
        self.children = children_map(trace)
        self.measure_from_ms = measure_from_ms
        self.bytes_per_token = bytes_per_token
        self.size_model = size_model
        self.store = None
        # --- last removal per state: the decision the next loss is charged to
        self.last_removal: dict[str, str] = {}
        # --- loss attribution, window requests only
        self.root_tokens = {name: 0 for name in LOSS_CATEGORIES}
        self.root_blocks = {name: 0 for name in LOSS_CATEGORIES}
        self.unusable_tokens = {name: 0 for name in LOSS_CATEGORIES}
        self.unusable_blocks = {name: 0 for name in LOSS_CATEGORIES}
        self.downstream_absent_tokens = 0
        self.downstream_absent_blocks = 0
        self.l2_hit_tokens = 0
        self.l2_hit_blocks = 0
        self.beyond_prefix_tokens = 0
        self.beyond_prefix_blocks = 0
        self.measured_requests = 0
        self.fully_served_requests = 0
        self.unexplained_states: set[str] = set()
        self.unexplained_after_promotion = 0
        # --- removals
        self.rejections = 0
        self.rejections_window = 0
        self.promotions = 0
        self.evictions = 0
        self.evictions_window = 0
        self.evictions_with_orphans = 0
        self.evictions_with_orphans_window = 0
        self.orphaned_blocks = 0
        self.orphaned_blocks_window = 0
        self.orphaned_bytes = 0
        self.orphaned_bytes_window = 0

    # --- plumbing ------------------------------------------------------------

    def attach(self, store) -> None:
        self.store = store

    def state_bytes(self, state_id: str) -> int:
        tokens = (
            self.states[state_id].block_tokens if self.size_model == "packed"
            else self.trace.block_size
        )
        return tokens * self.bytes_per_token

    def _in_window(self, timestamp_ms: float) -> bool:
        return self.measure_from_ms is None or timestamp_ms >= self.measure_from_ms

    # --- the request hook ----------------------------------------------------

    def on_request(self, ids, prefix, l2_hits, present, timestamp_ms, group_index,
                   measured) -> None:
        """Classify one request's blocks beyond the L1 prefix."""
        if not measured:
            return
        states = self.states
        self.measured_requests += 1
        beyond = sum(states[ids[index]].block_tokens for index in range(prefix, len(ids)))
        self.beyond_prefix_tokens += beyond
        self.beyond_prefix_blocks += len(ids) - prefix
        hit_tokens = sum(states[ids[index]].block_tokens for index in l2_hits)
        self.l2_hit_tokens += hit_tokens
        self.l2_hit_blocks += len(l2_hits)
        # Under the tree rule the hits are the contiguous run that starts at the
        # prefix, so the first block L2 does not hold sits right after them. The
        # attribution is defined for that rule only; under the independent hit
        # model there is no single root and the check below refuses the input.
        root = prefix + len(l2_hits)
        if l2_hits and l2_hits != list(range(prefix, root)):
            raise LossPartitionError(
                f"the L2 hits {l2_hits} are not the contiguous run from the prefix {prefix}: "
                "the loss attribution is defined for the tree hit rule only"
            )
        if root >= len(ids):
            self.fully_served_requests += 1
            if hit_tokens != beyond:
                raise LossPartitionError(
                    f"every block beyond the prefix was an L2 hit but {hit_tokens} != {beyond}"
                )
            return
        category = self._category(ids[root], timestamp_ms)
        self.root_tokens[category] += states[ids[root]].block_tokens
        self.root_blocks[category] += 1
        unusable_tokens = unusable_blocks = 0
        present_after = set()
        for index in present:
            if index > root:
                present_after.add(index)
                unusable_tokens += states[ids[index]].block_tokens
                unusable_blocks += 1
        self.unusable_tokens[category] += unusable_tokens
        self.unusable_blocks[category] += unusable_blocks
        downstream_tokens = downstream_blocks = 0
        for index in range(root + 1, len(ids)):
            if index in present_after:
                continue
            downstream_tokens += states[ids[index]].block_tokens
            downstream_blocks += 1
        self.downstream_absent_tokens += downstream_tokens
        self.downstream_absent_blocks += downstream_blocks
        total = hit_tokens + states[ids[root]].block_tokens + unusable_tokens + downstream_tokens
        if total != beyond:
            raise LossPartitionError(
                f"blocks beyond the prefix do not partition: {total} != {beyond} "
                f"(hits {hit_tokens}, root {states[ids[root]].block_tokens}, "
                f"unusable {unusable_tokens}, downstream {downstream_tokens})"
            )

    def _category(self, state_id: str, timestamp_ms: float) -> str:
        """Which decision the loss of `state_id` is charged to."""
        kind = self.last_removal.get(state_id)
        if kind in DECISION_CATEGORIES:
            return kind
        if kind is None and bisect.bisect_left(self.occurrences[state_id], timestamp_ms) == 0:
            # First occurrence: no decision could have kept it.
            return "compulsory"
        self.unexplained_states.add(state_id)
        if kind == "promoted":
            # The state left L2 for L1 and is in neither tier now, so an L1
            # eviction must have offered it to L2 since. Counted, never zero'd.
            self.unexplained_after_promotion += 1
        return "unexplained"

    # --- the removal hook ----------------------------------------------------

    def __call__(self, state_id: str, kind: str, timestamp_ms: float, group_index: int,
                 decision_index: int) -> None:
        if kind not in L2_REMOVAL_KINDS:
            raise ValueError(f"unknown removal kind {kind!r}")
        self.last_removal[state_id] = kind
        window = self._in_window(timestamp_ms)
        if kind == "promoted":
            self.promotions += 1
            return
        if kind == "rejected":
            self.rejections += 1
            self.rejections_window += window
            return
        self.evictions += 1
        self.evictions_window += window
        blocks, size = self.orphans(state_id)
        self.orphaned_blocks += blocks
        self.orphaned_bytes += size
        self.evictions_with_orphans += blocks > 0
        if window:
            self.orphaned_blocks_window += blocks
            self.orphaned_bytes_window += size
            self.evictions_with_orphans_window += blocks > 0

    def orphans(self, state_id: str) -> tuple[int, int]:
        """Blocks and bytes of the L2-resident subtree under `state_id`.

        The walk stops at a child L2 does not hold: whatever hangs below that
        child was unreachable before this eviction too, so charging it here
        would count one hole twice.
        """
        if self.store is None:
            raise RuntimeError("the orphaning walk needs the L2 store; call attach() first")
        cached = self.store.cached
        children = self.children
        blocks = 0
        size = 0
        stack = list(children.get(state_id, ()))
        while stack:
            child = stack.pop()
            if child not in cached:
                continue
            blocks += 1
            size += self.state_bytes(child)
            stack.extend(children.get(child, ()))
        return blocks, size

    # --- output --------------------------------------------------------------

    def loss_row(self) -> dict[str, float]:
        """The attributed tokens and blocks of the evaluation window."""
        row: dict[str, float] = {
            "attribution_measured_requests": self.measured_requests,
            "fully_served_requests": self.fully_served_requests,
            "beyond_prefix_tokens": self.beyond_prefix_tokens,
            "beyond_prefix_blocks": self.beyond_prefix_blocks,
            "l2_hit_tokens": self.l2_hit_tokens,
            "l2_hit_blocks_checked": self.l2_hit_blocks,
            "downstream_absent_tokens": self.downstream_absent_tokens,
            "downstream_absent_blocks": self.downstream_absent_blocks,
            "unexplained_states": len(self.unexplained_states),
            "unexplained_after_promotion": self.unexplained_after_promotion,
        }
        for name in LOSS_CATEGORIES:
            row[f"root_{name}_tokens"] = self.root_tokens[name]
            row[f"root_{name}_blocks"] = self.root_blocks[name]
            row[f"unusable_after_{name}_tokens"] = self.unusable_tokens[name]
            row[f"unusable_after_{name}_blocks"] = self.unusable_blocks[name]
        row["root_loss_tokens"] = sum(self.root_tokens.values())
        row["unusable_tokens"] = sum(self.unusable_tokens.values())
        row["decision_loss_tokens"] = sum(
            self.root_tokens[name] + self.unusable_tokens[name] for name in DECISION_CATEGORIES
        )
        return row

    def orphaning_row(self) -> dict[str, float]:
        """Rejections, resident evictions, and what the evictions stranded."""
        return {
            "removal_rejections": self.rejections,
            "removal_rejections_window": self.rejections_window,
            "removal_promotions": self.promotions,
            "resident_evictions": self.evictions,
            "resident_evictions_window": self.evictions_window,
            "evictions_with_orphans": self.evictions_with_orphans,
            "evictions_with_orphans_window": self.evictions_with_orphans_window,
            "share_evictions_with_orphans": (
                self.evictions_with_orphans / self.evictions if self.evictions else math.nan
            ),
            "share_evictions_with_orphans_window": (
                self.evictions_with_orphans_window / self.evictions_window
                if self.evictions_window else math.nan
            ),
            "orphaned_blocks": self.orphaned_blocks,
            "orphaned_blocks_window": self.orphaned_blocks_window,
            "orphaned_bytes": self.orphaned_bytes,
            "orphaned_bytes_window": self.orphaned_bytes_window,
            "orphaned_blocks_per_eviction": (
                self.orphaned_blocks / self.evictions if self.evictions else math.nan
            ),
        }

    def check_against(self, result) -> None:
        """Every counter the replay keeps independently must agree with mine."""
        expected_beyond = result.requested_tokens - result.l1_avoided_tokens
        checks = (
            ("beyond_prefix_tokens", self.beyond_prefix_tokens, expected_beyond),
            ("l2_avoided_tokens", self.l2_hit_tokens, result.l2_avoided_tokens),
            ("l2_hit_blocks", self.l2_hit_blocks, result.l2_hit_blocks),
            ("l2_present_unusable_tokens", sum(self.unusable_tokens.values()),
             result.l2_present_unusable_tokens),
            ("l2_present_unusable_blocks", sum(self.unusable_blocks.values()),
             result.l2_present_unusable_blocks),
            ("l2_rejections", self.rejections, result.l2_rejections),
            ("l2_evictions", self.evictions, result.l2_evictions),
        )
        for name, mine, theirs in checks:
            if mine != theirs:
                raise LossPartitionError(f"{name}: attribution {mine} != replay {theirs}")


# --- decision-type regret and the ranking split -------------------------------


def _labels(record, target: str, horizon_seconds: float) -> np.ndarray:
    return target_column(record.next_use_delta_ms, record.count_within_h, target,
                         horizon_seconds)


def _is_rejection(record) -> bool:
    """The arriving victim lost its own first round.

    `arriving_index` is 0 in the first round of an admission and -1 afterwards,
    and the store places the arrival first, so this is exactly the store's own
    `first_round and victim == arriving`.
    """
    return record.arriving_index == 0 and record.victim_index == 0


def decision_regret(logger) -> dict[str, float]:
    """Rejections and resident evictions of the window, and what they cost.

    The label is the one the on-policy logger already stores: the removed state
    is "reused within H" when its next occurrence is at most H seconds after the
    decision. A decision is "avoidable" when some candidate in the same set was
    *not* reused within H, so keeping the removed state was possible without
    losing another reuse. Computed over the logger's reservoir, which is a
    uniform sample of the window decisions when the cap binds.
    """
    horizon_ms = logger.horizon_seconds * 1000.0
    out: dict[str, float] = {
        "decisions_offered": float(logger.decisions_offered),
        "decisions_in_window": float(logger.decisions_seen),
        "decisions_kept": float(len(logger.kept)),
    }
    for name in ("rejected", "evicted"):
        kept = reused = avoidable = 0
        for record in logger.kept:
            if _is_rejection(record) != (name == "rejected"):
                continue
            kept += 1
            if record.next_use_delta_ms[record.victim_index] > horizon_ms:
                continue
            reused += 1
            avoidable += int((record.next_use_delta_ms > horizon_ms).any())
        out[f"{name}_decisions"] = float(kept)
        out[f"{name}_reused_within_h_share"] = reused / kept if kept else math.nan
        out[f"{name}_avoidable_share"] = avoidable / reused if reused else math.nan
    return out


def victim_rank_fraction(scores: np.ndarray, index: int = 0) -> float:
    """Where one candidate sits in its decision's score order, 0 = first out.

    Average ranks, so a tie puts the candidate in the middle of its tie group,
    which is the same convention the ranking metrics use.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2:
        return math.nan
    return float(_average_ranks(scores)[index] / (len(scores) - 1))


def ranking_split(logger, target: str) -> dict[str, float]:
    """Where the ordering failed: placing the arrival, or ordering the residents.

    Three views of the same decisions, all read off the lexicographic order of
    the score tuples the store actually compared:

    * `victim_vs_resident_metric` — over the first rounds only, every
      (arrival, resident) pair with different labels, scored 1 when the
      higher-labelled one has the higher score and 0.5 on a score tie. Below
      0.5 means the arrival is systematically mis-placed among the residents.
    * `residents_only_metric` — the within-decision metric of Phase 0.97 on the
      candidate set with the arrival taken out, so it measures only the order
      among residents.
    * `whole_set_metric` — the Phase 0.97 within-decision metric itself, for
      reference.
    """
    horizon = logger.horizon_seconds
    binary = target == "binary"
    concordant = pairs = 0.0
    fractions: list[float] = []
    first_rounds = 0
    residents: list[float] = []
    residents_constant = 0
    for record in logger.kept:
        labels = _labels(record, target, horizon)
        order = record.order
        arrival = record.arriving_index
        if arrival == 0:
            first_rounds += 1
            for index in range(1, len(order)):
                if labels[index] == labels[0]:
                    continue
                pairs += 1.0
                if order[0] == order[index]:
                    concordant += 0.5
                elif (labels[0] > labels[index]) == (order[0] > order[index]):
                    concordant += 1.0
            fraction = victim_rank_fraction(order, 0)
            if math.isfinite(fraction):
                fractions.append(fraction)
        if arrival >= 0:
            keep = np.ones(len(order), dtype=bool)
            keep[arrival] = False
            block_labels, block_scores = labels[keep], order[keep]
        else:
            block_labels, block_scores = labels, order
        if len(block_labels) < 2 or len(np.unique(block_labels)) < MIN_DISTINCT_LABELS:
            residents_constant += 1
            continue
        if binary:
            residents.append(
                fast_ranking_metrics(block_labels.astype(np.int8), block_scores, PRECISION_K)[0]
            )
        else:
            value = spearman(block_scores, block_labels)
            if math.isfinite(value):
                residents.append(value)
            else:
                residents_constant += 1
    whole = logger.metrics(target)
    return {
        "victim_vs_resident_metric": concordant / pairs if pairs else math.nan,
        "victim_vs_resident_pairs": pairs,
        "first_round_decisions": float(first_rounds),
        "victim_rank_fraction_mean": float(np.mean(fractions)) if fractions else math.nan,
        "residents_only_metric": float(np.mean(residents)) if residents else math.nan,
        "residents_only_decisions": float(len(residents)),
        "residents_constant_label": float(residents_constant),
        "whole_set_metric": whole["within_decision_macro"],
        "whole_set_decisions_scored": whole["decisions_scored"],
        "evicted_lowest_label_rate": whole["evicted_lowest_label_rate"],
        "victim_matches_argmin_rate": whole["victim_matches_argmin_rate"],
        "prevalence": whole["prevalence"],
    }
