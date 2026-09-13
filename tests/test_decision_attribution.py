"""Tests for the Phase 0.98 eviction-decision attribution.

Every case here is hand-computable: a trace of a handful of blocks, a request
whose classification can be written out by hand, and a decision whose pairwise
comparisons can be counted on one hand. The point of the phase is that the
numbers are attributions rather than estimates, so the tests check the exact
integers and not a tolerance.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from persistent_kv_admission.attribution import (
    LOSS_CATEGORIES,
    AttributionCollector,
    LossPartitionError,
    children_map,
    decision_regret,
    ranking_split,
    victim_rank_fraction,
)
from persistent_kv_admission.decisionpop import OnPolicyDecisionLogger, _OnPolicyDecision
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.twotier import run_two_tier

HORIZON = 10.0
HORIZON_MS = HORIZON * 1000.0


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path)


def chain_records():
    """One six-block chain, then a two-block branch off its second block.

    Blocks are `int:1 .. int:6` down the chain and `int:7` under `int:2`, so a
    request can be given a hole at a chosen depth and the subtree under a block
    is known by inspection.
    """
    return [
        {"timestamp": 0, "input_length": 512 * 6, "output_length": 1,
         "hash_ids": [1, 2, 3, 4, 5, 6]},
        {"timestamp": 1000, "input_length": 512 * 3, "output_length": 1,
         "hash_ids": [1, 2, 7]},
        {"timestamp": 2000, "input_length": 512 * 6, "output_length": 1,
         "hash_ids": [1, 2, 3, 4, 5, 6]},
    ]


def mixed_records(steps=240):
    """A hot chain, a small branching subtree, and cold leaves, one per second."""
    out = []
    for step in range(steps):
        if step % 4 == 0:
            chain = [1, 2]
        elif step % 7 == 0:
            chain = [1, 2, 3 + (step % 5)]
        else:
            chain = [1, 100 + step]
        out.append({"timestamp": step * 1000, "input_length": 512 * len(chain),
                    "output_length": 1, "hash_ids": chain})
    return out


class _FakeStore:
    """Only the resident set is read by the orphaning walk."""

    def __init__(self, cached):
        self.cached = {state_id: None for state_id in cached}


class ChildrenMapTests(unittest.TestCase):
    def test_children_are_listed_in_trace_order(self):
        temporary, trace = build_trace(chain_records())
        self.addCleanup(temporary.cleanup)
        children = children_map(trace)
        self.assertEqual(children["int:2"], ["int:3", "int:7"])
        self.assertEqual(children["int:5"], ["int:6"])
        self.assertNotIn("int:6", children)
        self.assertNotIn("int:7", children)


class LossAttributionTests(unittest.TestCase):
    """One request, one hole, hand-counted tokens."""

    def setUp(self):
        self.temporary, self.trace = build_trace(chain_records())
        self.addCleanup(self.temporary.cleanup)
        self.ids = ("int:1", "int:2", "int:3", "int:4", "int:5", "int:6")
        self.block = 512  # every block of this trace is a full one

    def collector(self):
        return AttributionCollector(self.trace, measure_from_ms=None, bytes_per_token=1,
                                    size_model="packed")

    def one_request(self, collector, prefix=1, l2_hits=(1, 2), present=(1, 2, 4),
                    timestamp_ms=2000.0):
        collector.on_request(self.ids, prefix, list(l2_hits), list(present), timestamp_ms, 2, True)

    def test_the_four_classes_partition_the_blocks_beyond_the_prefix(self):
        collector = self.collector()
        collector.last_removal["int:4"] = "evicted"
        # prefix = 1 (int:1 in L1); L2 holds int:2, int:3 and int:5; the hole is
        # at int:4, so int:5 is present-unusable and int:6 is downstream-absent.
        self.one_request(collector, prefix=1, l2_hits=(1, 2), present=(1, 2, 4))
        self.assertEqual(collector.beyond_prefix_tokens, 5 * self.block)
        self.assertEqual(collector.beyond_prefix_blocks, 5)
        self.assertEqual(collector.l2_hit_tokens, 2 * self.block)
        self.assertEqual(collector.root_tokens["evicted"], self.block)
        self.assertEqual(collector.unusable_tokens["evicted"], self.block)
        self.assertEqual(collector.downstream_absent_tokens, self.block)
        self.assertEqual(collector.downstream_absent_blocks, 1)
        total = (collector.l2_hit_tokens + sum(collector.root_tokens.values())
                 + sum(collector.unusable_tokens.values()) + collector.downstream_absent_tokens)
        self.assertEqual(total, collector.beyond_prefix_tokens)

    def test_a_rejected_root_carries_its_present_unusable_descendant(self):
        collector = self.collector()
        collector.last_removal["int:4"] = "rejected"
        self.one_request(collector)
        self.assertEqual(collector.root_tokens["rejected"], self.block)
        self.assertEqual(collector.root_blocks["rejected"], 1)
        self.assertEqual(collector.unusable_tokens["rejected"], self.block)
        self.assertEqual(collector.unusable_blocks["rejected"], 1)
        for name in ("evicted", "compulsory", "unexplained"):
            self.assertEqual(collector.root_tokens[name], 0)
            self.assertEqual(collector.unusable_tokens[name], 0)

    def test_an_evicted_root_is_charged_to_the_eviction(self):
        collector = self.collector()
        collector.last_removal["int:4"] = "evicted"
        self.one_request(collector)
        self.assertEqual(collector.root_tokens["evicted"], self.block)
        self.assertEqual(collector.unusable_tokens["evicted"], self.block)
        self.assertEqual(collector.root_tokens["rejected"], 0)

    def test_a_first_occurrence_is_compulsory_and_a_later_one_is_not(self):
        # int:7 first occurs at 1000 ms; at that instant no decision could have
        # kept it, at 2000 ms one could have.
        ids = ("int:1", "int:2", "int:7")
        first = self.collector()
        first.on_request(ids, 2, [], [], 1000.0, 1, True)
        self.assertEqual(first.root_tokens["compulsory"], self.block)
        self.assertEqual(first.root_blocks["compulsory"], 1)
        self.assertEqual(len(first.unexplained_states), 0)
        later = self.collector()
        later.on_request(ids, 2, [], [], 2000.0, 2, True)
        self.assertEqual(later.root_tokens["compulsory"], 0)
        self.assertEqual(later.root_tokens["unexplained"], self.block)
        self.assertEqual(later.unexplained_states, {"int:7"})

    def test_every_absent_block_is_charged_to_its_own_last_removal(self):
        # The root int:4 was rejected and the downstream-absent int:6 was
        # evicted, so the two charges differ: the root-only attribution names
        # the rejection alone, the per-block one names both.
        collector = self.collector()
        collector.last_removal["int:4"] = "rejected"
        collector.last_removal["int:6"] = "evicted"
        self.one_request(collector, prefix=1, l2_hits=(1, 2), present=(1, 2, 4))
        self.assertEqual(collector.root_tokens["rejected"], self.block)
        self.assertEqual(collector.downstream_tokens["evicted"], self.block)
        self.assertEqual(collector.downstream_blocks["evicted"], 1)
        self.assertEqual(collector.downstream_tokens["rejected"], 0)
        self.assertEqual(collector.absent_tokens["rejected"], self.block)
        self.assertEqual(collector.absent_tokens["evicted"], self.block)
        for name in LOSS_CATEGORIES:
            self.assertEqual(collector.absent_tokens[name],
                             collector.root_tokens[name] + collector.downstream_tokens[name], name)
            self.assertEqual(collector.absent_blocks[name],
                             collector.root_blocks[name] + collector.downstream_blocks[name], name)
        self.assertEqual(sum(collector.absent_tokens.values()),
                         sum(collector.root_tokens.values())
                         + collector.downstream_absent_tokens)
        # The present-unusable block keeps the root's decision under both charges.
        self.assertEqual(collector.unusable_tokens["rejected"], self.block)
        self.assertEqual(collector.unusable_tokens["evicted"], 0)

    def test_a_downstream_block_at_its_first_occurrence_is_compulsory(self):
        # int:2 was evicted and is the root; int:7 first occurs at this very
        # request, so no decision could have kept it and the per-block charge
        # says so instead of calling it unexplained.
        collector = self.collector()
        collector.last_removal["int:2"] = "evicted"
        collector.on_request(("int:1", "int:2", "int:7"), 1, [], [], 1000.0, 1, True)
        self.assertEqual(collector.absent_tokens["evicted"], self.block)
        self.assertEqual(collector.absent_tokens["compulsory"], self.block)
        self.assertEqual(collector.downstream_tokens["compulsory"], self.block)
        self.assertEqual(collector.downstream_blocks["compulsory"], 1)
        self.assertEqual(collector.root_tokens["compulsory"], 0)
        self.assertEqual(collector.unexplained_states, set())

    def test_the_loss_row_carries_the_per_block_columns(self):
        collector = self.collector()
        collector.last_removal["int:4"] = "rejected"
        collector.last_removal["int:6"] = "evicted"
        self.one_request(collector, prefix=1, l2_hits=(1, 2), present=(1, 2, 4))
        row = collector.loss_row()
        self.assertEqual(row["absent_rejected_tokens"], self.block)
        self.assertEqual(row["absent_rejected_blocks"], 1)
        self.assertEqual(row["downstream_evicted_tokens"], self.block)
        self.assertEqual(row["downstream_evicted_blocks"], 1)
        self.assertEqual(row["absent_loss_tokens"],
                         row["root_loss_tokens"] + row["downstream_absent_tokens"])
        self.assertEqual(row["perblock_decision_loss_tokens"],
                         row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
                         + row["unusable_after_rejected_tokens"]
                         + row["unusable_after_evicted_tokens"])
        # Two absent blocks charged instead of one: the per-block loss is above
        # the root-only loss by exactly the downstream block.
        self.assertEqual(row["perblock_decision_loss_tokens"],
                         row["decision_loss_tokens"] + self.block)

    def test_a_request_served_entirely_from_l2_attributes_nothing(self):
        collector = self.collector()
        collector.on_request(self.ids, 2, [2, 3, 4, 5], [2, 3, 4, 5], 2000.0, 2, True)
        self.assertEqual(collector.fully_served_requests, 1)
        self.assertEqual(sum(collector.root_tokens.values()), 0)
        self.assertEqual(collector.downstream_absent_tokens, 0)
        self.assertEqual(collector.l2_hit_tokens, collector.beyond_prefix_tokens)

    def test_requests_outside_the_window_are_not_attributed(self):
        collector = AttributionCollector(self.trace, measure_from_ms=1500.0, bytes_per_token=1)
        collector.on_request(self.ids, 1, [], [], 1000.0, 1, False)
        self.assertEqual(collector.measured_requests, 0)
        self.assertEqual(collector.beyond_prefix_tokens, 0)

    def test_an_inconsistent_hit_set_is_refused(self):
        collector = self.collector()
        with self.assertRaises(LossPartitionError):
            # A hit set that is not the contiguous run from the prefix puts the
            # root on a block the caller also called a hit, so the classes
            # overlap and no longer add up; the partition check catches it.
            collector.on_request(self.ids, 1, [1, 3], [1, 3], 2000.0, 2, True)


class OrphaningTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(chain_records())
        self.addCleanup(self.temporary.cleanup)

    def collector(self, cached):
        collector = AttributionCollector(self.trace, bytes_per_token=1, size_model="packed")
        collector.attach(_FakeStore(cached))
        return collector

    def test_the_resident_subtree_of_an_evicted_ancestor_is_counted(self):
        # int:2 leaves L2 while int:3, int:4 and the branch int:7 are resident.
        collector = self.collector(("int:3", "int:4", "int:7"))
        collector("int:2", "evicted", 2000.0, 2, 7)
        self.assertEqual(collector.orphaned_blocks, 3)
        self.assertEqual(collector.orphaned_bytes, 3 * 512)
        self.assertEqual(collector.evictions, 1)
        self.assertEqual(collector.evictions_with_orphans, 1)
        self.assertEqual(collector.last_removal["int:2"], "evicted")

    def test_the_walk_stops_at_a_hole_that_was_already_there(self):
        # int:4 is not resident, so int:5 and int:6 were unusable before this
        # eviction and are not charged to it.
        collector = self.collector(("int:3", "int:5", "int:6"))
        collector("int:2", "evicted", 2000.0, 2, 7)
        self.assertEqual(collector.orphaned_blocks, 1)
        self.assertEqual(collector.orphaned_bytes, 512)

    def test_a_leaf_eviction_orphans_nothing_and_a_rejection_is_not_measured(self):
        collector = self.collector(("int:3", "int:4"))
        collector("int:6", "evicted", 2000.0, 2, 7)
        self.assertEqual(collector.orphaned_blocks, 0)
        self.assertEqual(collector.evictions_with_orphans, 0)
        collector("int:5", "rejected", 2000.0, 2, 8)
        self.assertEqual(collector.rejections, 1)
        self.assertEqual(collector.evictions, 1)
        self.assertEqual(collector.orphaned_blocks, 0)

    def test_promotion_is_recorded_but_is_not_a_removal_cause(self):
        collector = self.collector(("int:3", "int:4"))
        collector("int:2", "promoted", 2000.0, 2, -1)
        self.assertEqual(collector.promotions, 1)
        self.assertEqual(collector.evictions, 0)
        self.assertEqual(collector.rejections, 0)
        self.assertEqual(collector.last_removal["int:2"], "promoted")

    def test_the_window_counters_follow_measure_from_ms(self):
        collector = AttributionCollector(self.trace, measure_from_ms=1500.0, bytes_per_token=1)
        collector.attach(_FakeStore(("int:3",)))
        collector("int:2", "evicted", 1000.0, 1, 0)
        collector("int:2", "evicted", 2000.0, 2, 1)
        self.assertEqual(collector.evictions, 2)
        self.assertEqual(collector.evictions_window, 1)
        self.assertEqual(collector.orphaned_blocks, 2)
        self.assertEqual(collector.orphaned_blocks_window, 1)

    def test_an_unknown_removal_kind_is_refused(self):
        collector = self.collector(())
        with self.assertRaises(ValueError):
            collector("int:2", "dropped", 0.0, 0, -1)


def _decision(deltas, order, victim_index, arriving_index, timestamp_ms=0.0):
    """One decision with hand-chosen labels and a hand-chosen score order."""
    deltas = np.asarray(deltas, dtype=float)
    counts = (deltas <= HORIZON_MS).astype(float)
    order = np.asarray(order, dtype=float)
    return _OnPolicyDecision(
        timestamp_ms=timestamp_ms, group_index=0,
        states=np.arange(len(deltas), dtype=np.int32),
        scores=order, tiebreaks=np.zeros(len(deltas)), order=order,
        next_use_delta_ms=deltas, count_within_h=counts,
        victim_index=victim_index, arriving_index=arriving_index,
    )


class RankingSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(mixed_records(40))
        self.addCleanup(self.temporary.cleanup)

    def logger(self, decisions):
        logger = OnPolicyDecisionLogger(self.trace, HORIZON, 0.0, self.trace.end_ms)
        logger.kept = list(decisions)
        logger.decisions_seen = logger.decisions_offered = len(logger.kept)
        return logger

    def test_the_split_separates_a_misplaced_arrival_from_a_good_resident_order(self):
        # Labels 1, 0, 1, 0; scores 0, 1, 3, 2. The arrival (index 0) is reused
        # within H and is scored lowest of all, so every arrival pair with a
        # different label is discordant: victim-vs-resident AUC = 0/2 = 0.
        # Among the residents the reused one (score 3) outranks both others, so
        # the residents-only AUC is 1. The whole-set AUC averages the two:
        # concordant pairs (3>1, 3>2) out of four = 0.5.
        logger = self.logger([_decision([5000, math.inf, 5000, math.inf], [0, 1, 3, 2], 0, 0)])
        split = ranking_split(logger, "binary")
        self.assertEqual(split["victim_vs_resident_pairs"], 2.0)
        self.assertAlmostEqual(split["victim_vs_resident_metric"], 0.0)
        self.assertAlmostEqual(split["residents_only_metric"], 1.0)
        self.assertAlmostEqual(split["whole_set_metric"], 0.5)
        self.assertAlmostEqual(split["victim_rank_fraction_mean"], 0.0)
        self.assertEqual(split["first_round_decisions"], 1.0)

    def test_a_score_tie_between_the_arrival_and_a_resident_counts_a_half(self):
        # Labels 1, 0, 0; the arrival ties the first resident and beats the
        # second: (0.5 + 1) / 2 = 0.75.
        logger = self.logger([_decision([5000, math.inf, math.inf], [1, 1, 0], 2, 0)])
        split = ranking_split(logger, "binary")
        self.assertEqual(split["victim_vs_resident_pairs"], 2.0)
        self.assertAlmostEqual(split["victim_vs_resident_metric"], 0.75)
        # Average ranks put the tie group at 1.5 of a 0..2 range.
        self.assertAlmostEqual(split["victim_rank_fraction_mean"], 0.75)

    def test_later_rounds_contribute_to_the_residents_only_view_only(self):
        later = _decision([5000, math.inf], [1, 0], 1, -1)
        logger = self.logger([later])
        split = ranking_split(logger, "binary")
        self.assertTrue(math.isnan(split["victim_vs_resident_metric"]))
        self.assertEqual(split["first_round_decisions"], 0.0)
        self.assertAlmostEqual(split["residents_only_metric"], 1.0)

    def test_a_constant_label_decision_is_counted_not_scored(self):
        logger = self.logger([_decision([5000, 5000, 5000], [0, 1, 2], 0, 0)])
        split = ranking_split(logger, "binary")
        self.assertTrue(math.isnan(split["victim_vs_resident_metric"]))
        self.assertTrue(math.isnan(split["residents_only_metric"]))
        self.assertEqual(split["residents_constant_label"], 1.0)

    def test_the_graded_target_uses_spearman_on_the_residents(self):
        # Residents' next-use labels are strictly decreasing in delta and their
        # scores are increasing in it, so the rank correlation is exactly -1.
        logger = self.logger([_decision([0.0, 1000, 4000, 9000], [0, 1, 2, 3], 0, 0)])
        split = ranking_split(logger, "next_use")
        self.assertAlmostEqual(split["residents_only_metric"], -1.0)

    def test_victim_rank_fraction_matches_the_average_rank_convention(self):
        self.assertAlmostEqual(victim_rank_fraction([0, 1, 2, 3], 0), 0.0)
        self.assertAlmostEqual(victim_rank_fraction([3, 1, 2, 0], 0), 1.0)
        self.assertAlmostEqual(victim_rank_fraction([1, 1, 0], 0), 0.75)
        self.assertTrue(math.isnan(victim_rank_fraction([5.0], 0)))


class DecisionRegretTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace = build_trace(mixed_records(40))
        self.addCleanup(self.temporary.cleanup)

    def logger(self, decisions):
        logger = OnPolicyDecisionLogger(self.trace, HORIZON, 0.0, self.trace.end_ms)
        logger.kept = list(decisions)
        logger.decisions_seen = logger.decisions_offered = len(logger.kept)
        return logger

    def test_rejections_and_evictions_are_split_and_scored_separately(self):
        decisions = [
            # A rejection that threw away a reuse while a never-reused
            # candidate was in the same set: avoidable.
            _decision([5000, math.inf, math.inf], [0, 1, 2], 0, 0),
            # A rejection of a state that is not reused within H at all.
            _decision([math.inf, 5000, 5000], [0, 1, 2], 0, 0),
            # A resident eviction that threw away a reuse, with no unreused
            # candidate available: not avoidable.
            _decision([5000, 5000, 5000], [0, 1, 2], 0, -1),
        ]
        regret = decision_regret(self.logger(decisions))
        self.assertEqual(regret["decisions_kept"], 3.0)
        self.assertEqual(regret["rejected_decisions"], 2.0)
        self.assertAlmostEqual(regret["rejected_reused_within_h_share"], 0.5)
        self.assertAlmostEqual(regret["rejected_avoidable_share"], 1.0)
        self.assertEqual(regret["evicted_decisions"], 1.0)
        self.assertAlmostEqual(regret["evicted_reused_within_h_share"], 1.0)
        self.assertAlmostEqual(regret["evicted_avoidable_share"], 0.0)

    def test_a_first_round_that_keeps_the_arrival_is_an_eviction(self):
        # arriving_index 0 with a victim elsewhere: the arrival stayed and a
        # resident left, which is an eviction however the round was numbered.
        regret = decision_regret(self.logger([_decision([math.inf, 5000], [1, 0], 1, 0)]))
        self.assertEqual(regret["rejected_decisions"], 0.0)
        self.assertEqual(regret["evicted_decisions"], 1.0)

    def test_an_empty_log_gives_nan_shares_and_zero_counts(self):
        regret = decision_regret(self.logger([]))
        self.assertEqual(regret["decisions_kept"], 0.0)
        self.assertEqual(regret["rejected_decisions"], 0.0)
        self.assertTrue(math.isnan(regret["rejected_reused_within_h_share"]))


class ReplayIntegrationTests(unittest.TestCase):
    """The hooks against a real replay: they observe and they never intervene."""

    def setUp(self):
        self.temporary, self.trace = build_trace(mixed_records())
        self.addCleanup(self.temporary.cleanup)

    def _run(self, policy="lru", eviction="sampled", **extra):
        return run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, policy, bytes_per_token=1,
                            l2_eviction=eviction, l2_sample_width=4, l2_seed=0, **extra)

    def test_hooks_left_at_none_reproduce_the_replay_exactly(self):
        for policy, eviction in (("lru", "sampled"), ("lfu", "sampled"),
                                 ("lru_2hit", "sampled"), ("lru", "heap"),
                                 ("offline_next_use", "heap")):
            with self.subTest(policy=policy, eviction=eviction):
                plain = self._run(policy, eviction)
                collector = AttributionCollector(self.trace, bytes_per_token=1)
                hooked = self._run(policy, eviction, l2_request_hook=collector.on_request,
                                   l2_removal_hook=collector)
                self.assertEqual(plain.as_row(), hooked.as_row())

    def test_every_counter_the_replay_keeps_agrees_with_the_attribution(self):
        for policy in ("lru", "lfu", "lru_2hit"):
            with self.subTest(policy=policy):
                collector = AttributionCollector(self.trace, bytes_per_token=1)
                result = self._run(policy, l2_request_hook=collector.on_request,
                                   l2_removal_hook=collector)
                collector.check_against(result)          # raises on any disagreement
                self.assertGreater(collector.measured_requests, 0)
                self.assertEqual(collector.rejections, result.l2_rejections)
                self.assertEqual(collector.evictions, result.l2_evictions)

    def test_no_loss_is_left_unexplained_on_a_real_replay(self):
        for policy in ("lru", "lfu", "lru_2hit"):
            with self.subTest(policy=policy):
                collector = AttributionCollector(self.trace, bytes_per_token=1)
                self._run(policy, l2_request_hook=collector.on_request, l2_removal_hook=collector)
                self.assertEqual(collector.root_tokens["unexplained"], 0)
                self.assertEqual(collector.unusable_tokens["unexplained"], 0)
                self.assertEqual(collector.unexplained_states, set())
                self.assertGreater(collector.root_tokens["compulsory"], 0)

    def test_the_window_restricts_the_attribution_the_way_the_replay_does(self):
        split = 120_000.0
        collector = AttributionCollector(self.trace, measure_from_ms=split, bytes_per_token=1)
        result = run_two_tier(self.trace, "lru", 3 * 512, 6 * 512, "lru", bytes_per_token=1,
                              l2_eviction="sampled", l2_sample_width=4, l2_seed=0,
                              measure_from_ms=split, l2_request_hook=collector.on_request,
                              l2_removal_hook=collector)
        collector.check_against(result)
        self.assertEqual(collector.measured_requests, result.measured_requests)
        self.assertLessEqual(collector.evictions_window, collector.evictions)
        self.assertLessEqual(collector.rejections_window, collector.rejections)

    def test_the_two_hit_rule_rejects_before_ranking_and_is_still_recorded(self):
        collector = AttributionCollector(self.trace, bytes_per_token=1)
        result = self._run("lru_2hit", l2_request_hook=collector.on_request,
                           l2_removal_hook=collector)
        self.assertGreater(result.l2_rejections, result.l2_decisions)
        self.assertEqual(collector.rejections, result.l2_rejections)

    def test_a_promotion_is_the_last_removal_record_of_a_reused_state(self):
        collector = AttributionCollector(self.trace, bytes_per_token=1)
        self._run("lru", l2_request_hook=collector.on_request, l2_removal_hook=collector)
        self.assertGreater(collector.promotions, 0)
        self.assertEqual(set(collector.last_removal.values()) - {"promoted"},
                         {"rejected", "evicted"} & set(collector.last_removal.values()))

    def test_the_loss_row_sums_to_the_blocks_beyond_the_prefix(self):
        collector = AttributionCollector(self.trace, bytes_per_token=1)
        self._run("lru", l2_request_hook=collector.on_request, l2_removal_hook=collector)
        row = collector.loss_row()
        total = (row["l2_hit_tokens"] + row["downstream_absent_tokens"]
                 + sum(row[f"root_{name}_tokens"] for name in LOSS_CATEGORIES)
                 + sum(row[f"unusable_after_{name}_tokens"] for name in LOSS_CATEGORIES))
        self.assertEqual(total, row["beyond_prefix_tokens"])
        self.assertEqual(row["decision_loss_tokens"],
                         row["root_rejected_tokens"] + row["root_evicted_tokens"]
                         + row["unusable_after_rejected_tokens"]
                         + row["unusable_after_evicted_tokens"])
        self.assertEqual(row["absent_loss_tokens"],
                         row["root_loss_tokens"] + row["downstream_absent_tokens"])
        self.assertEqual(row["perblock_decision_loss_tokens"],
                         row["absent_rejected_tokens"] + row["absent_evicted_tokens"]
                         + row["unusable_after_rejected_tokens"]
                         + row["unusable_after_evicted_tokens"])

    def test_the_compulsory_charge_is_the_same_for_every_arm(self):
        # A block beyond the L1 prefix at its first occurrence is in neither
        # tier whatever L2 decided, and L1 never consults L2, so the compulsory
        # part of the per-block charge and the blocks beyond the prefix are
        # properties of the trace and the capacities alone. This is the
        # invariance the Phase 0.98b run checks over the whole grid.
        rows = []
        for policy in ("lru", "lfu", "lru_2hit"):
            collector = AttributionCollector(self.trace, bytes_per_token=1)
            result = self._run(policy, l2_request_hook=collector.on_request,
                               l2_removal_hook=collector)
            collector.check_against(result)
            rows.append(collector.loss_row())
        for column in ("absent_compulsory_tokens", "absent_compulsory_blocks",
                       "beyond_prefix_tokens", "beyond_prefix_blocks"):
            values = [row[column] for row in rows]
            self.assertEqual(values, [values[0]] * len(values), column)
        self.assertGreater(rows[0]["absent_compulsory_tokens"], 0)
        for row in rows:
            self.assertEqual(row["absent_unexplained_tokens"], 0)
            self.assertEqual(row["downstream_unexplained_tokens"], 0)
            self.assertEqual(row["unexplained_states"], 0)

    def test_the_orphaning_walk_needs_the_store(self):
        collector = AttributionCollector(self.trace, bytes_per_token=1)
        with self.assertRaises(RuntimeError):
            collector("int:1", "evicted", 0.0, 0, 0)


if __name__ == "__main__":
    unittest.main()
