"""Tests for the working-set ratio check (docs/working-set-ratio-plan.md).

The real traces are not processed before this is reviewed, so everything is
checked on constructed streams: the stack distance on hand-built streams
(peak outstanding size, offers that never return, returns outside the window,
the tie rule of an offer and a return in one timestamp group), the fast
segment-tree path against the literal definition, the equality
`D + size <= C` against a brute-force admit-all tier on randomised streams,
the weighted quantiles, the verdict boundaries, the threshold intervals and
the declinable share. The offer stream is checked against the simulator
itself: identical under every L2 arm, returns equal to the replay's own reuse
log, sizes equal to what the store charges. The runner's derivation is run end
to end on two constructed traces and constructed published rows.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from persistent_kv_admission import twotier
from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import load_mooncake_trace
from persistent_kv_admission.workingset import (
    Offer,
    RangeAddMax,
    agreement,
    declinable,
    group_timestamps,
    in_transition,
    observation,
    offers_from_victims,
    predicted_sign,
    ratio,
    record_offer_stream,
    reference_tier_held,
    return_positions,
    stack_distances,
    stack_distances_bruteforce,
    threshold_intervals,
    verdict,
    weighted_quantile,
    weighted_quantiles,
    working_set,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BLOCK = 512
BLOCK_BYTES = BLOCK * 2048


def _runner():
    path = REPOSITORY / "scripts" / "run_working_set_ratio.py"
    spec = importlib.util.spec_from_file_location("_working_set_ratio_runner_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _runner()


# --- constructed streams -------------------------------------------------------------------


def make_offer(group, size, returns=None, state=None, tokens=None, step_ms=1000.0):
    """An offer at timestamp group `group` (timestamp group x step_ms)."""
    return Offer(
        state_id=state if state is not None else f"s{group}_{size}_{returns}_{random.random()}",
        group_index=group, timestamp_ms=group * step_ms, size_bytes=size,
        tokens=tokens if tokens is not None else size,
        return_group=returns,
        return_timestamp_ms=None if returns is None else returns * step_ms,
    )


def random_stream(rng, count, max_size=5, never=0.2, span=8, group_step=(0, 1, 1, 2)):
    """A valid offer stream: groups nondecreasing, every return strictly after
    its offer group, some offers never returning. Several offers share a
    group, and returns land in groups that also carry offers, so the tie rule
    is exercised."""
    group = 0
    offers = []
    for index in range(count):
        group += rng.choice(group_step)
        returns = None if rng.random() < never else group + rng.randint(1, span)
        offers.append(make_offer(group, rng.randint(1, max_size), returns, state=f"o{index}"))
    return offers


# --- constructed traces --------------------------------------------------------------------


def session_records(steps=260, seed=11, start=1000):
    """Growing sessions under two shared prefixes, some timestamps shared, so
    the tree has deep chains, shared ancestors and multi-request groups."""
    rng = random.Random(seed)
    sessions, out, next_id = [], [], start
    for step in range(steps):
        if sessions and rng.random() < 0.65:
            chain = sessions[rng.randrange(len(sessions))]
            for _ in range(rng.randint(1, 2)):
                chain.append(next_id)
                next_id += 1
            if len(chain) > 9:
                sessions.remove(chain)
        else:
            chain = ([1, 2] if rng.random() < 0.6 else [1, 3]) + [next_id]
            next_id += 1
            sessions.append(chain)
        timestamp = (step - (1 if step % 6 == 5 else 0)) * 1000
        out.append({"timestamp": timestamp, "input_length": BLOCK * len(chain),
                    "output_length": 1, "hash_ids": list(chain)})
    return out


def build_trace(records, name="trace"):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / f"{name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path, BLOCK)


# --- the segment tree ----------------------------------------------------------------------


class RangeAddMaxTests(unittest.TestCase):
    def test_matches_a_plain_array_under_random_operations(self):
        rng = random.Random(3)
        for size in (1, 2, 3, 7, 8, 9, 33):
            tree = RangeAddMax(size)
            plain = [0] * size
            for _ in range(300):
                start = rng.randrange(size)
                end = rng.randint(start + 1, size)
                if rng.random() < 0.5:
                    value = rng.randint(0, 9)
                    tree.add(start, end, value)
                    for index in range(start, end):
                        plain[index] += value
                else:
                    self.assertEqual(tree.max(start, end), max(plain[start:end]))

    def test_an_empty_query_is_refused(self):
        with self.assertRaises(ValueError):
            RangeAddMax(4).max(2, 2)


# --- the stack distance on hand-built streams -------------------------------------------------


class HandBuiltStackDistanceTests(unittest.TestCase):
    def test_peak_outstanding_size_is_the_maximum_over_the_interval(self):
        # A waits from group 0 to 5. B (3) is outstanding at groups 1-2 only,
        # C (4) from 3, D (2, never returns) from 4: the peak is C + D = 6.
        offers = [make_offer(0, 5, 5, "A"), make_offer(1, 3, 2, "B"),
                  make_offer(3, 4, 6, "C"), make_offer(4, 2, None, "D")]
        self.assertEqual(return_positions(offers), [4, 2, 4, 4])
        expected = [6, 0, 2, None]
        self.assertEqual(stack_distances(offers), expected)
        self.assertEqual(stack_distances_bruteforce(offers), expected)
        self.assertEqual(reference_tier_held(offers, 11)[0], True)    # 6 + 5 <= 11
        self.assertEqual(reference_tier_held(offers, 10)[0], False)

    def test_a_peak_that_has_drained_before_the_return_still_counts(self):
        # B (5) comes and goes long before A returns; at A's return only C (1)
        # is outstanding, but the tier pushed A out when B arrived.
        offers = [make_offer(0, 1, 10, "A"), make_offer(1, 5, 2, "B"), make_offer(3, 1, 11, "C")]
        distances = stack_distances(offers)
        self.assertEqual(distances[0], 5)
        self.assertEqual(stack_distances_bruteforce(offers)[0], 5)
        self.assertFalse(reference_tier_held(offers, 5)[0])
        self.assertTrue(reference_tier_held(offers, 6)[0])

    def test_older_offers_do_not_count(self):
        offers = [make_offer(0, 9, 4, "old"), make_offer(1, 2, 3, "x"), make_offer(2, 1, 5, "y")]
        # x waits from group 1 to 3: only y (newer) counts, never "old".
        self.assertEqual(stack_distances(offers)[1], 1)

    def test_offers_that_never_return(self):
        offers = [make_offer(0, 2, None, "a"), make_offer(1, 3, 4, "b"), make_offer(2, 4, None, "c")]
        distances = stack_distances(offers)
        self.assertIsNone(distances[0])
        self.assertIsNone(distances[2])
        self.assertEqual(distances[1], 4)        # c is newer and still outstanding
        self.assertEqual(return_positions(offers), [3, 3, 3])
        held = reference_tier_held(offers, 100)
        self.assertEqual(held, [None, True, None])

    def test_offers_of_one_group_keep_the_eviction_order(self):
        # Both evicted while inserting group 0, A first: B is newer than A.
        offers = [make_offer(0, 2, 3, "A"), make_offer(0, 3, 3, "B")]
        self.assertEqual(stack_distances(offers), [3, 0])
        self.assertEqual(stack_distances_bruteforce(offers), [3, 0])

    def test_tie_rule_returns_precede_the_offers_of_their_group(self):
        # A and B return at group 2; C is offered at group 2. Returns come
        # first, so C never coexists with A: D(A) = 1 (B), and A is held at its
        # return in a tier of 2. Were the offers first, C would push A out.
        offers = [make_offer(0, 1, 2, "A"), make_offer(1, 1, 2, "B"), make_offer(2, 1, None, "C")]
        self.assertEqual(return_positions(offers), [2, 2, 3])
        self.assertEqual(stack_distances(offers)[0], 1)
        self.assertEqual(stack_distances_bruteforce(offers)[0], 1)
        self.assertEqual(reference_tier_held(offers, 2), [True, True, None])

    def test_a_state_offered_again_at_its_return_group_is_newer_but_not_outstanding_for_it(self):
        offers = [make_offer(0, 4, 2, "s"), make_offer(2, 4, 4, "s")]
        self.assertEqual(stack_distances(offers), [0, 0])
        self.assertEqual(reference_tier_held(offers, 4), [True, True])

    def test_returns_must_follow_their_offer_group(self):
        with self.assertRaises(ValueError):
            return_positions([make_offer(3, 1, 3, "a")])
        with self.assertRaises(ValueError):
            return_positions([make_offer(3, 1, 5, "a"), make_offer(2, 1, 5, "b")])


class RandomisedEquivalenceTests(unittest.TestCase):
    """The fast path against the literal definition, and `D + size <= C`
    against the brute-force admit-all tier, on randomised small streams."""

    CAPACITIES = (0, 1, 2, 3, 5, 8, 13, 21, 1000)

    def test_fast_distances_equal_the_literal_definition(self):
        rng = random.Random(17)
        for trial in range(250):
            offers = random_stream(rng, rng.randint(0, 40))
            self.assertEqual(stack_distances(offers), stack_distances_bruteforce(offers),
                             msg=f"trial {trial}")

    def test_held_at_return_iff_d_plus_size_fits(self):
        rng = random.Random(29)
        checked = 0
        for trial in range(250):
            offers = random_stream(rng, rng.randint(1, 40))
            distances = stack_distances(offers)
            for capacity in self.CAPACITIES:
                held = reference_tier_held(offers, capacity)
                for offer, distance, kept in zip(offers, distances, held):
                    if offer.return_group is None:
                        self.assertIsNone(kept)
                        continue
                    checked += 1
                    self.assertEqual(distance + offer.size_bytes <= capacity, kept,
                                     msg=f"trial {trial} capacity {capacity}")
        self.assertGreater(checked, 10000)

    def test_both_sides_of_the_equality_occur(self):
        rng = random.Random(5)
        outcomes = set()
        for _ in range(50):
            offers = random_stream(rng, 30)
            outcomes.update(kept for kept in reference_tier_held(offers, 8) if kept is not None)
        self.assertEqual(outcomes, {True, False})


# --- window, quantiles, ratio --------------------------------------------------------------


class WindowTests(unittest.TestCase):
    def test_only_returns_inside_the_window_enter_the_working_set(self):
        split = 5000.0
        offers = [
            make_offer(0, 1, 3, "returns_before_window", tokens=1),
            make_offer(1, 2, 7, "offered_before_returns_inside", tokens=2),
            make_offer(6, 4, 9, "offered_and_returns_inside", tokens=4),
            make_offer(6, 8, None, "never_returns", tokens=8),
            make_offer(4, 16, 5, "returns_at_the_split", tokens=16),
        ]
        offers.sort(key=lambda offer: offer.group_index)
        distances = stack_distances(offers)
        summary = working_set(offers, distances, split)
        self.assertEqual(summary["offers"], 5)
        self.assertEqual(summary["returning_offers"], 4)
        self.assertEqual(summary["window_returning_offers"], 3)
        self.assertEqual(summary["window_returning_tokens"], 2 + 4 + 16)
        # Values D + size of the three in-window returns, weighted by tokens.
        by_state = {offer.state_id: distance + offer.size_bytes
                    for offer, distance in zip(offers, distances) if distance is not None}
        inside = [by_state[s] for s in ("offered_before_returns_inside",
                                        "offered_and_returns_inside", "returns_at_the_split")]
        self.assertEqual(summary["quantiles"],
                         weighted_quantiles(inside, [2, 4, 16]))

    def test_the_window_test_is_the_simulators(self):
        # A return exactly at the split is measured, as in run_two_tier.
        offers = [make_offer(0, 1, 5, "a")]
        self.assertEqual(working_set(offers, stack_distances(offers), 5000.0)
                         ["window_returning_offers"], 1)
        self.assertEqual(working_set(offers, stack_distances(offers), 5000.5)
                         ["window_returning_offers"], 0)
        empty = working_set(offers, stack_distances(offers), 9e9)
        self.assertTrue(math.isnan(empty["quantiles"][0.5]))


class WeightedQuantileTests(unittest.TestCase):
    def test_lower_quantile_with_equal_weights(self):
        values = [40, 10, 30, 20]
        self.assertEqual(weighted_quantile(values, [1] * 4, 0.5), 20)
        self.assertEqual(weighted_quantile(values, [1] * 4, 0.25), 10)
        self.assertEqual(weighted_quantile(values, [1] * 4, 0.75), 30)
        self.assertEqual(weighted_quantile(values, [1] * 4, 1.0), 40)
        self.assertEqual(weighted_quantile(values, [1] * 4, 0.0), 10)

    def test_weights_move_the_quantile(self):
        values = [10, 20, 30, 40]
        self.assertEqual(weighted_quantile(values, [1, 1, 1, 5], 0.5), 40)
        self.assertEqual(weighted_quantile(values, [1, 1, 1, 5], 0.25), 20)
        self.assertEqual(weighted_quantile(values, [5, 1, 1, 1], 0.5), 10)
        # Cumulative weight reaching the target exactly stops at that value.
        self.assertEqual(weighted_quantile([1, 2, 3], [2, 2, 4], 0.5), 2)

    def test_zero_weights_are_ignored_and_empty_is_nan(self):
        self.assertEqual(weighted_quantile([1, 100], [0, 3], 0.25), 100)
        self.assertTrue(math.isnan(weighted_quantile([], [], 0.5)))
        self.assertTrue(math.isnan(weighted_quantile([5], [0], 0.5)))
        with self.assertRaises(ValueError):
            weighted_quantile([1, 2], [1, -1], 0.5)
        with self.assertRaises(ValueError):
            weighted_quantile([1], [1, 2], 0.5)

    def test_token_weighting_differs_from_counting(self):
        offers = [make_offer(0, 1, 9, "small", tokens=1), make_offer(0, 1, 9, "small2", tokens=1),
                  make_offer(0, 1, 9, "big", tokens=10)]
        summary = working_set(offers, stack_distances(offers), None)
        # D + size: 3, 2, 1 in offer order; the heavy one has the smallest value.
        self.assertEqual(summary["quantiles"][0.5], 1)
        self.assertEqual(weighted_quantile([3, 2, 1], [1, 1, 1], 0.5), 2)

    def test_ratio(self):
        self.assertEqual(ratio(5, 5), 1.0)
        self.assertEqual(ratio(5, 10), 0.5)
        self.assertEqual(ratio(20, 10), 2.0)
        with self.assertRaises(ValueError):
            ratio(1, 0)


# --- signs, agreement, verdict, threshold --------------------------------------------------


def _row(gamma, observed, declining=1):
    return {"gamma": gamma, "observed_sign": observed, "declining_sign": declining}


class PredictionTests(unittest.TestCase):
    def test_gamma_exactly_one_falls_on_the_le_side(self):
        self.assertEqual(predicted_sign(1.0, 1), -1)
        self.assertEqual(predicted_sign(1.0, -1), 1)
        self.assertEqual(predicted_sign(math.nextafter(1.0, 2.0), 1), 1)
        self.assertEqual(predicted_sign(math.nextafter(1.0, 2.0), -1), -1)
        self.assertEqual(predicted_sign(ratio(7, 7), 1), -1)
        with self.assertRaises(ValueError):
            predicted_sign(2.0, 0)

    def test_observation_readings(self):
        self.assertEqual(observation([3.0]), (1, "consistent_gain"))
        self.assertEqual(observation([-3.0]), (-1, "consistent_loss"))
        self.assertEqual(observation([0.0]), (0, "zero"))
        self.assertEqual(observation([1, 2, 3, 4, 5]), (1, "consistent_gain"))
        self.assertEqual(observation([-1, -2, -3, -4, -5]), (-1, "consistent_loss"))
        self.assertEqual(observation([1, 2, -3, 4, 5]), (0, "mixed"))
        self.assertEqual(observation([1, 2, 0, 4, 5]), (0, "mixed"))
        self.assertEqual(observation([0, 0, 0, 0, 0]), (0, "zero"))

    def test_mixed_and_zero_are_misses(self):
        counted = agreement([_row(3.0, 1), _row(3.0, 0), _row(0.2, 0), _row(0.2, -1)])
        self.assertEqual(counted, {"agreement": 2, "compared": 4, "misses": [1, 2]})

    def test_outcome_three_has_the_opposite_sign(self):
        counted = agreement([_row(3.0, -1, -1), _row(0.2, 1, -1), _row(3.0, 1, -1)])
        self.assertEqual(counted["misses"], [2])


class VerdictTests(unittest.TestCase):
    def test_boundaries_of_the_transition_are_included(self):
        self.assertTrue(in_transition(0.5))
        self.assertTrue(in_transition(1.0))
        self.assertTrue(in_transition(2.0))
        self.assertFalse(in_transition(math.nextafter(0.5, 0.0)))
        self.assertFalse(in_transition(math.nextafter(2.0, 3.0)))

    def test_verdicts(self):
        self.assertEqual(verdict([]), "located")
        self.assertEqual(verdict([0.5, 1.0, 2.0]), "located outside the transition")
        self.assertEqual(verdict([0.5, 2.0000001]), "not located")
        self.assertEqual(verdict([math.nextafter(0.5, 0.0)]), "not located")
        self.assertEqual(verdict([10.0]), "not located")

    def test_a_miss_at_gamma_exactly_one(self):
        # gamma == 1 predicts the "<=" sign; a gain there is a miss in the band.
        rows = [_row(1.0, 1), _row(4.0, 1), _row(0.1, -1)]
        counted = agreement(rows)
        self.assertEqual(counted["misses"], [0])
        self.assertEqual(verdict([rows[i]["gamma"] for i in counted["misses"]]),
                         "located outside the transition")


class ThresholdTests(unittest.TestCase):
    def test_a_monotone_pattern_switching_around_one(self):
        rows = [_row(0.1, -1), _row(0.3, -1), _row(3.0, 1), _row(10.0, 1)]
        found = threshold_intervals(rows)
        self.assertEqual(found["max_agreement"], 4)
        self.assertEqual(found["intervals"], [(0.3, 3.0)])
        self.assertEqual(agreement(rows)["agreement"], 4)

    def test_a_monotone_pattern_away_from_one(self):
        rows = [_row(0.1, -1), _row(0.3, -1), _row(3.0, -1), _row(10.0, 1)]
        found = threshold_intervals(rows)
        self.assertEqual(found["max_agreement"], 4)
        self.assertEqual(found["intervals"], [(3.0, 10.0)])
        self.assertEqual(agreement(rows, 1.0)["agreement"], 3)

    def test_a_non_monotone_pattern_has_a_lower_maximum_on_several_intervals(self):
        rows = [_row(0.1, 1), _row(0.3, -1), _row(3.0, 1), _row(10.0, -1)]
        found = threshold_intervals(rows)
        self.assertEqual(found["max_agreement"], 2)
        self.assertEqual(found["intervals"], [(0.0, 0.1), (0.3, 3.0), (10.0, math.inf)])

    def test_adjacent_maximising_intervals_merge(self):
        rows = [_row(0.2, -1), _row(0.5, 0), _row(2.0, 1)]
        found = threshold_intervals(rows)
        self.assertEqual(found["max_agreement"], 2)
        self.assertEqual(found["intervals"], [(0.2, 2.0)])

    def test_each_interval_reproduces_its_count(self):
        rng = random.Random(41)
        for _ in range(100):
            rows = [_row(rng.choice([0.1, 0.5, 1.0, 2.0, 7.0, rng.uniform(0.01, 50)]),
                         rng.choice([-1, 0, 1]), rng.choice([-1, 1])) for _ in range(8)]
            found = threshold_intervals(rows)
            for low, high in found["intervals"]:
                probe = low if math.isinf(high) else (low + high) / 2
                self.assertEqual(agreement(rows, probe)["agreement"], found["max_agreement"])
                self.assertEqual(agreement(rows, low)["agreement"], found["max_agreement"])
            best = max(agreement(rows, t)["agreement"]
                       for t in [0.0] + sorted(row["gamma"] for row in rows))
            self.assertEqual(found["max_agreement"], best)

    def test_outcome_three_sign(self):
        found = threshold_intervals([_row(0.1, 1, -1), _row(10.0, -1, -1)])
        self.assertEqual((found["max_agreement"], found["intervals"]), (2, [(0.1, 10.0)]))


# --- the declinable share ------------------------------------------------------------------


class DeclinableTests(unittest.TestCase):
    def test_never_returning_and_beyond_capacity_offers_are_declinable(self):
        split = 2000.0
        offers = [
            make_offer(0, 4, 9, "before_window"),          # outside the window share
            make_offer(2, 2, 3, "fits"),                   # D = 0: 2 <= 5
            make_offer(4, 3, 9, "displaced"),              # D = 3 (later "late"): 6 > 5
            make_offer(5, 3, None, "late"),                # never returns
        ]
        distances = stack_distances(offers)
        # before_window: "displaced" + "late" outstanding together at group 5.
        self.assertEqual(distances, [6, 0, 3, None])
        counts = declinable(offers, distances, 5, split)
        self.assertEqual(counts["window_offers"], 3)
        self.assertEqual(counts["window_offered_bytes"], 8)
        self.assertEqual(counts["window_declinable_bytes"], 6)
        self.assertEqual(counts["window_never_return_bytes"], 3)
        self.assertEqual(counts["window_beyond_capacity_bytes"], 3)
        self.assertEqual(counts["window_declinable_offers"], 2)
        # Whole trace: "before_window" has D + size = 9 > 5 too.
        self.assertEqual((counts["offers"], counts["declinable_offers"]), (4, 3))
        self.assertEqual(counts["declinable_bytes"], 10)
        # A larger tier declines only what never returns.
        self.assertEqual(declinable(offers, distances, 100, split)["window_declinable_bytes"], 3)

    def test_declinable_agrees_with_the_reference_tier(self):
        rng = random.Random(8)
        for _ in range(60):
            offers = random_stream(rng, 25)
            distances = stack_distances(offers)
            for capacity in (3, 7, 12):
                held = reference_tier_held(offers, capacity)
                expected = sum(offer.size_bytes for offer, kept in zip(offers, held) if not kept)
                self.assertEqual(declinable(offers, distances, capacity, None)["declinable_bytes"],
                                 expected)


# --- the offer stream from the simulator ----------------------------------------------------


class OfferStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary, cls.trace = build_trace(session_records())
        cls.groups = _occurrence_groups(cls.trace)
        cls.l1 = 5 * BLOCK_BYTES
        cls.offers, cls.result = record_offer_stream(cls.trace, "lru", cls.l1, cls.groups,
                                                     measure_from_ms=150_000.0)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_the_stream_is_substantive(self):
        self.assertGreater(len(self.offers), 200)
        self.assertEqual(len(self.offers), self.result.l1_evictions)
        self.assertEqual(self.result.l2_capacity_bytes, 0)
        self.assertTrue(any(offer.return_group is None for offer in self.offers))
        self.assertTrue(any(offer.return_group is not None for offer in self.offers))
        groups = [offer.group_index for offer in self.offers]
        self.assertEqual(groups, sorted(groups))
        self.assertGreater(len(groups), len(set(groups)))          # several per group

    def test_the_stream_does_not_depend_on_the_l2_arm(self):
        arms = {
            "heap lru, 1x": dict(l2_capacity_bytes=self.l1, l2_policy="lru"),
            "heap lru, 4x": dict(l2_capacity_bytes=4 * self.l1, l2_policy="lru"),
            "heap lru_2hit, 4x": dict(l2_capacity_bytes=4 * self.l1, l2_policy="lru_2hit"),
            "heap offline, 1x": dict(l2_capacity_bytes=self.l1, l2_policy="offline_next_use"),
            "sampled lru, 1x, seed 3": dict(l2_capacity_bytes=self.l1, l2_policy="lru",
                                            l2_eviction="sampled", l2_sample_width=4, l2_seed=3),
            "sampled lru_2hit, 4x": dict(l2_capacity_bytes=4 * self.l1, l2_policy="lru_2hit",
                                         l2_eviction="sampled", l2_sample_width=4),
            "sampled lru, protection all": dict(l2_capacity_bytes=self.l1, l2_policy="lru",
                                                l2_eviction="sampled", l2_sample_width=4,
                                                l2_arrival_protection="all"),
        }
        for name, options in arms.items():
            offers, result = record_offer_stream(self.trace, "lru", self.l1, self.groups,
                                                 measure_from_ms=150_000.0, **options)
            self.assertGreater(result.l2_capacity_bytes, 0, msg=name)
            self.assertEqual(offers, self.offers, msg=name)
            self.assertEqual(result.l1_avoided_tokens, self.result.l1_avoided_tokens, msg=name)
            self.assertEqual(result.l2_admissions + result.l2_rejections, len(self.offers), msg=name)

    def test_returns_equal_the_replays_own_reuse_log(self):
        events = []
        recorded, _ = record_offer_stream(self.trace, "lru", self.l1, self.groups,
                                          event_sink=events)
        self.assertEqual(recorded, self.offers)
        self.assertEqual(len(events), len(self.offers))
        timestamps = group_timestamps(self.trace)
        for event, offer in zip(events, self.offers):
            self.assertEqual((event.state_id, event.group_index, event.timestamp_ms),
                             (offer.state_id, offer.group_index, offer.timestamp_ms))
            if event.censored:
                self.assertIsNone(offer.return_group)
            else:
                self.assertEqual(offer.return_group, event.reuse_group)
                self.assertEqual(offer.return_timestamp_ms, timestamps[event.reuse_group])
                self.assertEqual(offer.return_timestamp_ms, event.reuse_timestamp_ms)

    def test_sizes_are_what_the_store_charges(self):
        # An admit-all heap L2 large enough for everything admits every offer,
        # and its admitted bytes are the store's own `state_bytes`.
        offers, result = record_offer_stream(
            self.trace, "lru", self.l1, self.groups, l2_capacity_bytes=10 ** 15, l2_policy="lru")
        self.assertEqual(result.l2_admissions, len(offers))
        self.assertEqual(sum(result.l2_admitted_bytes_by_depth.values()),
                         sum(offer.size_bytes for offer in offers))
        for offer in offers:
            meta = self.trace.states[offer.state_id]
            self.assertEqual(offer.tokens, meta.block_tokens)
            self.assertEqual(offer.size_bytes, meta.block_tokens * 2048)

    def test_partial_blocks_keep_their_packed_size(self):
        records = [{"timestamp": t * 1000, "input_length": 3 * BLOCK + 100 + t,
                    "output_length": 1, "hash_ids": [1, 2, 3, 100 + t]} for t in range(40)]
        temporary, trace = build_trace(records)
        self.addCleanup(temporary.cleanup)
        offers, _ = record_offer_stream(trace, "lru", 4 * BLOCK_BYTES)
        self.assertTrue(any(offer.tokens < BLOCK for offer in offers))
        for offer in offers:
            self.assertEqual(offer.size_bytes, trace.states[offer.state_id].block_tokens * 2048)

    def test_distances_and_the_tier_equivalence_on_the_simulator_stream(self):
        distances = stack_distances(self.offers)
        self.assertEqual(distances, stack_distances_bruteforce(self.offers))
        for capacity in (BLOCK_BYTES, 4 * BLOCK_BYTES, 20 * BLOCK_BYTES, 60 * BLOCK_BYTES):
            held = reference_tier_held(self.offers, capacity)
            for offer, distance, kept in zip(self.offers, distances, held):
                if offer.return_group is not None:
                    self.assertEqual(distance + offer.size_bytes <= capacity, kept)

    def test_an_offer_again_before_its_return_is_refused(self):
        timestamps = group_timestamps(self.trace)
        returning = next(offer for offer in self.offers if offer.return_group is not None)
        victim = (returning.state_id, returning.timestamp_ms, returning.group_index)
        with self.assertRaises(RuntimeError):
            offers_from_victims(self.trace, [victim, victim], self.groups, timestamps)
        never = next(offer for offer in self.offers if offer.return_group is None)
        victim = (never.state_id, never.timestamp_ms, never.group_index)
        with self.assertRaises(RuntimeError):
            offers_from_victims(self.trace, [victim, victim], self.groups, timestamps)
        with self.assertRaises(ValueError):
            offers_from_victims(self.trace, [(never.state_id, never.timestamp_ms + 1.0,
                                              never.group_index)], self.groups, timestamps)


# --- the runner, end to end on constructed traces -----------------------------------------------


def _published(streams, heap, sampled, protection):
    """Published rows shaped like the three reference CSVs (strings), built on
    the constructed streams so every identifier the runner checks matches.
    `heap`, `sampled`, `protection` map (trace, f, m) to the token difference
    (one value, five seed values, and {target: five seed values})."""
    decision, mechanism, phase1 = [], [], []
    for stream in streams:
        name, fraction = stream["trace"], stream["l1_fraction"]
        requested = stream["requested_tokens"]
        base = 1000
        for multiplier in runner.MULTIPLIERS:
            key = (name, fraction, multiplier)
            capacity = stream["capacities"][multiplier]["l2_capacity_bytes"]
            common = {"trace": name, "l1_fraction": str(fraction), "l2_multiplier": str(multiplier),
                      "requested_tokens": str(requested),
                      "l1_evictions": str(stream["l1_evictions"]),
                      "l1_capacity_bytes": str(stream["l1_capacity_bytes"]),
                      "l2_capacity_bytes": str(capacity),
                      "l1_avoided_tokens": str(stream["l1_avoided_tokens"]),
                      "l1_policy": "lru", "hit_model": "tree", "closure": "union"}
            for seed in runner.SEEDS:
                decision.append({**common, "kind": "heap", "arm": "lru", "target": "",
                                 "seed": str(seed), "avoided_prefill_tokens": str(base)})
                decision.append({**common, "kind": "heap", "arm": "lru_2hit", "target": "",
                                 "seed": str(seed), "avoided_prefill_tokens": str(base + heap[key])})
                decision.append({**common, "kind": "sampled", "arm": "lru_s", "target": "",
                                 "seed": str(seed), "avoided_prefill_tokens": str(base + seed)})
                decision.append({**common, "kind": "sampled", "arm": "lru_2hit_s", "target": "",
                                 "seed": str(seed),
                                 "avoided_prefill_tokens": str(base + seed + sampled[key][seed])})
            cell = f"l1={fraction:g},l2x{multiplier:g}"
            offers = stream["l1_evictions"]
            for rung, rejections in (("label", offers // 4), ("learned", offers // 2),
                                     ("lru", 0)):
                mechanism.append({"trace": name, "l1_fraction": str(fraction),
                                  "l2_multiplier": str(multiplier), "cell": cell,
                                  "mechanism": "all16", "rung": rung,
                                  "l2_admissions_mean": str(float(offers - rejections)),
                                  "l2_rejections_mean": str(float(rejections)),
                                  "l1_capacity_bytes": common["l1_capacity_bytes"],
                                  "l2_capacity_bytes": common["l2_capacity_bytes"],
                                  "requested_tokens": common["requested_tokens"],
                                  "l1_avoided_tokens": common["l1_avoided_tokens"]})
            if (fraction, multiplier) not in ((0.0025, 1.0), (0.01, 4.0), (0.02, 4.0)):
                continue
            for target, values in protection[key].items():
                for seed in runner.SEEDS:
                    for left, right, net in (("all", "none", values[seed]),
                                             ("direct_child", "none", 7)):
                        phase1.append({
                            "trace": name, "l1_fraction": str(fraction),
                            "l2_multiplier": str(multiplier), "cell": cell, "target": target,
                            "seed": str(seed), "left_variant": left, "right_variant": right,
                            "comparison": f"{left}-{right}", "requested_tokens": str(requested),
                            "left_avoided_tokens": str(base + net),
                            "right_avoided_tokens": str(base),
                            "net_avoided_tokens": str(net),
                            "net_input_percentage_points": repr(100.0 * net / requested)})
    return decision, mechanism, phase1


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporaries, traces, cls.trace_paths = [], {}, []
        for index, name in enumerate(runner.TRACES):
            temporary, trace = build_trace(session_records(steps=220, seed=31 + index,
                                                           start=5000 * (index + 1)), name)
            cls.temporaries.append(temporary)
            cls.trace_paths.append(Path(temporary.name) / f"{name}.jsonl")
            traces[name] = trace
        cls.saved = (dict(runner.rdp._SHARED), dict(runner.SHARED))
        runner.rdp._SHARED["working_set"] = {name: working_set_bytes(t) for name, t in traces.items()}
        runner.SHARED.update(traces=traces,
                             groups={name: _occurrence_groups(t) for name, t in traces.items()},
                             splits={name: 0.6 * t.end_ms for name, t in traces.items()})
        tasks = [(name, fraction) for name in runner.TRACES for fraction in runner.FRACTIONS]
        cls.streams = runner.compute_streams(tasks, 1)
        cls.ratios = runner.ratio_rows(cls.streams)
        gamma = {(row["trace"], row["l1_fraction"], row["l2_multiplier"]): row["gamma"]
                 for row in cls.ratios}
        # Outcomes that follow the rule everywhere, except chosen misses.
        cls.heap = {key: (5 if g > 1 else -5) for key, g in gamma.items()}
        cls.sampled = {key: [(3 if g > 1 else -3)] * 5 for key, g in gamma.items()}
        first = min(gamma)
        cls.sampled[first] = [1, -1, 1, 1, 1]                         # one mixed miss
        cls.protection = {key: {"next_use": [(-4 if g > 1 else 4)] * 5,
                                "binary": [(4 if g > 1 else -4)] * 5}  # all wrong
                          for key, g in gamma.items()}
        cls.gamma = gamma
        cls.decision, cls.mechanism, cls.phase1 = _published(cls.streams, cls.heap, cls.sampled,
                                                             cls.protection)

    @classmethod
    def tearDownClass(cls):
        runner.rdp._SHARED.clear()
        runner.rdp._SHARED.update(cls.saved[0])
        runner.SHARED.clear()
        runner.SHARED.update(cls.saved[1])
        for temporary in cls.temporaries:
            temporary.cleanup()

    def test_streams_pass_the_published_identity_checks(self):
        for stream in self.streams:
            for entry in stream["capacities"].values():
                self.assertEqual(entry["tier_equivalence_violations"], 0)
            self.assertGreater(stream["working_set"]["window_returning_offers"], 0)
        self.assertEqual(runner.check_streams(self.streams, self.decision, self.mechanism,
                                              self.phase1), [])

    def test_a_published_mismatch_is_reported(self):
        decision = [dict(row) for row in self.decision]
        decision[0]["l1_evictions"] = str(int(decision[0]["l1_evictions"]) + 1)
        mechanism = [dict(row) for row in self.mechanism]
        mechanism[0]["l2_rejections_mean"] = str(float(mechanism[0]["l2_rejections_mean"]) + 1)
        problems = runner.check_streams(self.streams, decision, mechanism, self.phase1)
        self.assertTrue(any("l1_evictions" in line for line in problems))
        self.assertTrue(any("admissions + rejections" in line for line in problems))

    def test_capacities_are_the_phase097_rule(self):
        for row in self.ratios:
            expected = runner.rdp._capacity(row["trace"], row["l1_fraction"] * row["l2_multiplier"])
            self.assertEqual(row["l2_capacity_bytes"], expected)
            self.assertEqual(row["gamma"], row["W_bytes"] / expected)

    def test_derived_tables(self):
        tables = runner.derive_tables(self.streams, self.decision, self.phase1, self.mechanism)
        self.assertEqual(len(tables["working_set"]), 6)
        self.assertEqual(len(tables["ratio"]), 12)
        verdicts = {(row["outcome"], row["target"]): row for row in tables["verdicts"]}
        heap = verdicts[("heap_lru_2hit_minus_lru", "")]
        self.assertEqual((heap["agreement"], heap["cells_compared"], heap["verdict"]),
                         (12, 12, "located"))
        sampled = verdicts[("sampled_lru_2hit_s_minus_lru_s", "")]
        self.assertEqual((sampled["agreement"], sampled["misses"]), (11, 1))
        self.assertIn("(mixed)", sampled["miss_list"])
        expected = "located outside the transition" if in_transition(self.gamma[min(self.gamma)]) \
            else "not located"
        self.assertEqual(sampled["verdict"], expected)
        primary = verdicts[("protection_all_minus_none", "next_use")]
        self.assertTrue(primary["primary"])
        self.assertEqual((primary["cells_compared"], primary["agreement"]), (6, 6))
        self.assertEqual(primary["verdict"], "not computable: 6 of 12 cells published")
        self.assertEqual(primary["verdict_on_compared_cells"], "located")
        secondary = verdicts[("protection_all_minus_none", "binary")]
        self.assertFalse(secondary["primary"])
        self.assertEqual(secondary["agreement"], 0)
        # Agreement rows carry the plan's sign convention and the points.
        rows = [row for row in tables["agreement"] if row["outcome"] == "heap_lru_2hit_minus_lru"]
        for row in rows:
            self.assertEqual(row["predicted_sign"], 1 if row["gamma"] > 1 else -1)
            self.assertEqual(row["pairing"], "deterministic")
            requested = next(s["requested_tokens"] for s in self.streams if s["trace"] == row["trace"])
            self.assertAlmostEqual(row["observed_points"],
                                   100.0 * self.heap[(row["trace"], row["l1_fraction"],
                                                      row["l2_multiplier"])] / requested)
        # Threshold rows: agreement at 1 equals the agreement count.
        for row in tables["threshold"]:
            self.assertEqual(row["agreement_at_1"],
                             verdicts[(row["outcome"], row["target"])]["agreement"])
        located = [row for row in tables["threshold"] if row["outcome"] == "heap_lru_2hit_minus_lru"]
        self.assertTrue(any(row["contains_1"] for row in located))
        # Reading 4 sits next to the published rejection shares.
        for row in tables["declinable"]:
            self.assertAlmostEqual(row["label_rejection_share_all16"],
                                   (int(row["offers"]) // 4) / int(row["offers"]))
            self.assertAlmostEqual(row["learned_rejection_share_all16"],
                                   (int(row["offers"]) // 2) / int(row["offers"]))
            self.assertTrue(0.0 <= row["declinable_share"] <= 1.0)

    def test_publish_writes_every_artifact(self):
        tables = runner.derive_tables(self.streams, self.decision, self.phase1, self.mechanism)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            runner.publish(target, tables, {"phase": "test"})
            names = {path.name for path in target.iterdir()}
            self.assertEqual(names, {"working_set.csv", "ratio.csv", "agreement.csv",
                                     "verdicts.csv", "threshold.csv", "declinable.csv",
                                     "effect_vs_gamma.png", "README.md", "run_config.json"})
            with self.assertRaises(FileExistsError):
                runner.publish(target, tables, {})

    def test_main_end_to_end_on_constructed_inputs(self):
        """`main` itself, with git and the split stubbed and the three published
        tables replaced by constructed ones: it loads, checks, derives and
        publishes what `derive_tables` gives on the same streams."""
        def git(*arguments):
            return "" if arguments[0] == "status" else "0" * 40

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, rows in (("decision", self.decision), ("phase1", self.phase1),
                               ("mechanism", self.mechanism)):
                paths[name] = root / f"{name}.csv"
                runner._write(paths[name], rows)
            target = root / "run"
            arguments = [str(path) for path in self.trace_paths] + [
                "--output-dir", str(target), "--workers", "1"]
            with mock.patch.object(runner, "_git", side_effect=git), \
                    mock.patch.object(runner, "sources_differing_from_head", return_value=[]), \
                    mock.patch.object(runner, "horizon_for",
                                      side_effect=lambda trace: (600.0, 0.6 * trace.end_ms, [])), \
                    mock.patch.object(runner, "DECISION_SEEDS", paths["decision"]), \
                    mock.patch.object(runner, "PHASE1_PAIRS", paths["phase1"]), \
                    mock.patch.object(runner, "MECHANISM_REPLAY", paths["mechanism"]), \
                    mock.patch.object(sys, "argv", ["run_working_set_ratio.py", *arguments]), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            tables = runner.derive_tables(self.streams, self.decision, self.phase1, self.mechanism)
            written = runner._read(target / "verdicts.csv")
            self.assertEqual([row["verdict"] for row in written],
                             [row["verdict"] for row in tables["verdicts"]])
            self.assertEqual(len(runner._read(target / "agreement.csv")), len(tables["agreement"]))
            config = json.loads((target / "run_config.json").read_text())
            self.assertEqual(config["checks"]["problems"], 0)
            self.assertEqual(config["checks"]["tier_equivalence_violations"], 0)
            self.assertEqual(set(config["trace_files"]), set(runner.TRACES))
            self.assertEqual(len(config["references"]), 3)
            self.assertEqual(config["phase1_primary_target"], "next_use")
            self.assertTrue((target / "effect_vs_gamma.png").stat().st_size > 0)

    def test_phase1_inventory_and_target_selection(self):
        inventory = runner.phase1_inventory(self.phase1)
        targets = {(entry["comparison"], entry["target"]) for entry in inventory}
        self.assertEqual(targets, {("all-none", "next_use"), ("all-none", "binary"),
                                   ("direct_child-none", "next_use"),
                                   ("direct_child-none", "binary")})
        variants = runner.observations(self.decision, self.phase1)
        self.assertEqual([(v["outcome"], v["target"], v["primary"]) for v in variants], [
            ("heap_lru_2hit_minus_lru", "", True),
            ("sampled_lru_2hit_s_minus_lru_s", "", True),
            ("protection_all_minus_none", "next_use", True),
            ("protection_all_minus_none", "binary", False),
        ])

    def test_published_rows_are_validated(self):
        decision = [dict(row) for row in self.decision]
        heap = next(row for row in decision if row["kind"] == "heap" and row["arm"] == "lru")
        heap["avoided_prefill_tokens"] = "1"
        with self.assertRaises(SystemExit):
            runner.heap_outcome(decision)
        decision = [row for row in self.decision
                    if not (row["arm"] == "lru_s" and row["seed"] == "4")]
        with self.assertRaises(SystemExit):
            runner.sampled_outcome(decision)
        phase1 = [dict(row) for row in self.phase1]
        phase1[0]["net_avoided_tokens"] = str(int(phase1[0]["net_avoided_tokens"]) + 1)
        with self.assertRaises(SystemExit):
            runner.protection_outcome(phase1, phase1[0]["target"])


class RunnerRefusalTests(unittest.TestCase):
    def _main(self, arguments):
        with mock.patch.object(sys, "argv", ["run_working_set_ratio.py", *arguments]):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main()

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                self._main(["trace.jsonl", "--output-dir", directory])
            self.assertIn("exists", str(raised.exception))

    def test_an_uncommitted_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "new")
            with mock.patch.object(runner, "_git", return_value="?? scripts/new.py"):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["trace.jsonl", "--output-dir", target])
            self.assertIn("not clean", str(raised.exception))
            with mock.patch.object(runner, "_git", return_value=""), \
                    mock.patch.object(runner, "sources_differing_from_head",
                                      return_value=["src/persistent_kv_admission/workingset.py"]):
                with self.assertRaises(SystemExit) as raised:
                    self._main(["trace.jsonl", "--output-dir", target])
            self.assertIn("differs from HEAD", str(raised.exception))
            self.assertFalse(Path(target).exists())


if __name__ == "__main__":
    unittest.main()
