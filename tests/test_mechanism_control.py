"""Tests for the mechanism control (docs/mechanism-control-plan.md).

Three things have to hold before a single real replay of the new arms means
anything: the published sampled path must be exactly what it was (same
candidates, same draws, same result) with the new options at their defaults;
leaf eligibility must be what the plan says, with an exact resident-child
count behind it; and the two new rungs must be the scores they claim to be —
the training label itself, and the heap comparator's own key. The pure
decomposition helpers are checked on hand-made numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

import persistent_kv_admission
from persistent_kv_admission import twotier
from persistent_kv_admission.attribution import AttributionCollector, children_map
from persistent_kv_admission.decisionpop import RawFeatureScorer, _Labeller, target_column
from persistent_kv_admission.mechanism import (
    GAPS,
    RUNGS,
    ExactLabelScorer,
    decompose,
    dominant_label,
    ladder_holds,
    ladder_ordered,
    sign_counts,
    sign_reading,
)
from persistent_kv_admission.replay import _occurrence_groups
from persistent_kv_admission.trace import StateMeta, Trace, load_mooncake_trace

SOURCE_ROOT = str(Path(list(persistent_kv_admission.__path__)[0]).resolve().parent)
L1_BYTES = 4 * 512
L2_BYTES = 8 * 512
MEASURE_FROM_MS = 120_000.0
LABEL_HORIZON = 30.0


# --- fixtures -------------------------------------------------------------------


def conversation_records(steps=360, seed=7):
    """A real-shaped trace: two shared system prompts and growing sessions.

    Each request either opens a session under one of two system-prompt
    prefixes or extends an open one by a block or two, so the tree has shared
    ancestors, deep chains and many leaves, and the published mechanism does
    orphan blocks on it (the leaf tests rely on that to have teeth). Every
    seventh request shares the timestamp of the one before it, so timestamp
    groups with several requests occur. Blocks are full, so the packed sizes of
    a state never disagree between requests.
    """
    rng = random.Random(seed)
    sessions = []
    next_id = 1000
    out = []
    for step in range(steps):
        if sessions and rng.random() < 0.65:
            chain = sessions[rng.randrange(len(sessions))]
            for _ in range(rng.randint(1, 2)):
                chain.append(next_id)
                next_id += 1
            if len(chain) > 10:
                sessions.remove(chain)
        else:
            chain = ([1, 2] if rng.random() < 0.6 else [1, 3]) + [next_id]
            next_id += 1
            sessions.append(chain)
        ids = list(chain)
        timestamp = (step - (1 if step % 7 == 6 else 0)) * 1000
        out.append({"timestamp": timestamp, "input_length": 512 * len(ids),
                    "output_length": 1, "hash_ids": ids})
    return out


def build_trace(records):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return temporary, load_mooncake_trace(path), path


class _LinearRanker:
    """A fixed linear score of the raw history features: a learned-like arm
    whose order moves with the clock, without fitting anything."""

    COEFFICIENTS = (0.7, 1.3, -0.4, 0.2, 0.9, -0.6, 0.1, -0.3, 0.05, 0.4, 0.2, -0.1,
                    0.3, 0.1, 0.6, -0.2, 0.5, 0.25, -0.35, 0.15, -0.05, 0.45, 0.3)

    def score_row(self, row):
        total = 0.0
        for coefficient, value in zip(self.COEFFICIENTS, row):
            total += coefficient * value
        return total


def rung_arguments(trace, rung):
    """`run_two_tier` arguments of one rung of the ladder on a test trace."""
    if rung == "lru":
        return dict(l2_policy="lru")
    if rung == "learned":
        return dict(l2_policy="learned", l2_scorer=RawFeatureScorer(trace, _LinearRanker()))
    if rung == "label":
        return dict(l2_policy="learned", l2_scorer=ExactLabelScorer(trace, LABEL_HORIZON))
    if rung == "offline":
        return dict(l2_policy="offline_next_use", l2_sampled_offline=True)
    raise ValueError(rung)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def published_path_fingerprints(trace):
    """Row and decision-stream digests of the published arms at default options.

    Only arguments that existed before the mechanism control are passed, so the
    same function runs against the code as it stood at the plan commit; the
    frozen values below were produced that way.
    """
    arms = {
        "lru_s/0": dict(l2_policy="lru", l2_eviction="sampled", l2_sample_width=4, l2_seed=0),
        "lru_s/3": dict(l2_policy="lru", l2_eviction="sampled", l2_sample_width=4, l2_seed=3),
        "lru_s/w16": dict(l2_policy="lru", l2_eviction="sampled", l2_sample_width=16, l2_seed=0),
        "lfu_s/1": dict(l2_policy="lfu", l2_eviction="sampled", l2_sample_width=4, l2_seed=1),
        "lru_2hit_s/2": dict(l2_policy="lru_2hit", l2_eviction="sampled", l2_sample_width=4,
                             l2_seed=2),
        "learned/0": dict(l2_policy="learned", l2_eviction="sampled", l2_sample_width=4,
                          l2_seed=0, scorer=True),
        "learned_direct_child/0": dict(l2_policy="learned", l2_eviction="sampled",
                                       l2_sample_width=4, l2_seed=0, scorer=True,
                                       l2_arrival_protection="direct_child"),
        "learned_all/1": dict(l2_policy="learned", l2_eviction="sampled", l2_sample_width=4,
                              l2_seed=1, scorer=True, l2_arrival_protection="all"),
        "heap_lru": dict(l2_policy="lru"),
        "heap_lfu": dict(l2_policy="lfu"),
        "heap_lru_2hit": dict(l2_policy="lru_2hit"),
        "heap_offline": dict(l2_policy="offline_next_use"),
    }
    out = {}
    for name, arguments in arms.items():
        arguments = dict(arguments)
        if arguments.pop("scorer", False):
            arguments["l2_scorer"] = RawFeatureScorer(trace, _LinearRanker())
        decisions = []

        def hook(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
            decisions.append([list(candidates), [list(score) for score in scores], victim_index,
                              timestamp_ms, group_index, arriving_index])

        if arguments.get("l2_eviction") == "sampled":
            arguments["l2_decision_hook"] = hook
        result = twotier.run_two_tier(trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                                      measure_from_ms=MEASURE_FROM_MS, **arguments)
        out[name] = [_digest(result.as_row()), _digest(decisions),
                     result.avoided_prefill_tokens]
    return out


# Produced by `published_path_fingerprints` against the source tree of commit
# 3052319 (the plan commit, before any mechanism-control change), with the
# same trace. A change on the published path moves the row digest, the
# decision digest (candidates, score tuples, victims: RNG consumption), or both.
PUBLISHED_PATH = {
    "heap_lfu": [
        "f0a333870d9ecfd32360b5f8d67c1ba426bc5c18a6c977308b6bbbabda2973f0",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        248832,
    ],
    "heap_lru": [
        "c4fcbcf805c60b936c1dee97c54f629082f6c925d0a159b4569f457b8c6e4513",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        238592,
    ],
    "heap_lru_2hit": [
        "64b5242659a9c42045b796675f4743b76d4621988d8fed0d338cc8eb10df9c77",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        245760,
    ],
    "heap_offline": [
        "dfe9c9cd529751d0a917f27044e22cb7bd039fa6daf6c61431818c30566b3aa8",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        287744,
    ],
    "learned/0": [
        "873e51effca1c33f4a7c3dfa7ac174f3ab72c606c5eb594f3dbb920c5fd9335b",
        "7b06d351903f3855aef473c5c014df53ad0a45e0c8f03c42b0fc222009aaebc1",
        252416,
    ],
    "learned_all/1": [
        "f01a8c80d14759b790a55af17d5f52ef0e2e28b4c2299a34a16b3837623240ef",
        "ba37d4f5c2c590786dbb6d61d37764ab470939e15b9ca51e1a45387e9ff6b374",
        252416,
    ],
    "learned_direct_child/0": [
        "417ea2344a3b613c8e46175f5f49593367e342ef5904361ac710018d128a1853",
        "12630d577ac9ef01fa7f741e990042558a670d6c4736e4e43a90534a285203e4",
        252416,
    ],
    "lfu_s/1": [
        "3efca8336ea6805a4988c5e5c0a5e446d5f011ac442ecba81b21bb029b14dcef",
        "bb1f2f94912c4f7ddfb36aa2a94fe8e824ea1ecf93c28256bae6041084f94727",
        250880,
    ],
    "lru_2hit_s/2": [
        "d691d24f02270baa8ffd68e0ffb8b08baf1f97684d385fedd4812a5132eef737",
        "c7afae9145a2ee0455d43d87e097ec93d92f5c1193eded7fc3f5234380f6abde",
        239616,
    ],
    "lru_s/0": [
        "19334f67edd3d974b57a98aec41cae5775b893fe09a5f3e32d8fa0e74f8dd696",
        "c39262424580753b9dfcbd7c77567e3fe169594b2cbf7c94c76c181706f6e278",
        224768,
    ],
    "lru_s/3": [
        "c6ae4733b548d86d759817fa633e1deb385d2ee08cca82613c0502f08f24206a",
        "8284257f4855f0222b4bebbd17cccae0ec8fe362b680883a47f140933b6b4189",
        222720,
    ],
    "lru_s/w16": [
        "8fb9b44cfff60e5ccba14eb2aaef45c21f10a22dae302736268d19119afb5d03",
        "40acde13962d4f7e5ef4ec449ec725309b6fbda95cfb08db2c955f46ef02ce49",
        230912,
    ],
}


# --- an instrumented store ----------------------------------------------------------


class _CheckedStore(twotier._VictimStore):
    """The union store, with `resident_children` recomputed by brute force
    after every insertion and every removal, whatever caused it."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checks = Counter()
        type(self).instances.append(self)

    def _brute_force(self):
        expected = {state_id: 0 for state_id in self.cached}
        for state_id in self.cached:
            parent = self.trace.states[state_id].parent_id
            if parent is not None and parent in expected:
                expected[parent] += 1
        return expected

    def _check(self, event):
        self.checks[event] += 1
        expected = self._brute_force() if self.eligibility == "leaf" else {}
        if self.resident_children != expected:
            raise AssertionError(f"resident_children wrong after {event}")

    def _count_insert(self, state_id):
        super()._count_insert(state_id)
        self._check("insert")

    def _remove(self, state_id):
        super()._remove(state_id)
        self._check("remove")


class _ClosureWatch:
    """Removal hook and L1-victim hook in one: records every removal kind, every
    leaf-rule violation, and how many arrivals had a resident direct child."""

    def __init__(self, trace):
        self.children = children_map(trace)
        self.store = None
        self.kinds = Counter()
        self.violations = []
        self.arrivals_with_resident_child = 0

    def attach(self, store):
        self.store = store

    def _has_resident_child(self, state_id):
        return any(child in self.store.cached for child in self.children.get(state_id, ()))

    def on_victim(self, state_id, timestamp_ms, group_index):
        if self.store is not None and self._has_resident_child(state_id):
            self.arrivals_with_resident_child += 1

    def __call__(self, state_id, kind, timestamp_ms, group_index, decision_index):
        self.kinds[kind] += 1
        if kind in ("rejected", "evicted") and self._has_resident_child(state_id):
            self.violations.append((state_id, kind, timestamp_ms))


def _tree_trace(parents):
    """The smallest Trace a store needs: a forest of unit-size states."""
    states, children = {}, defaultdict(set)

    def depth(state_id):
        value, parent = 1, parents[state_id]
        while parent is not None:
            value, parent = value + 1, parents[parent]
        return value

    for state_id, parent in parents.items():
        states[state_id] = StateMeta(state_id=state_id, parent_id=parent, depth=depth(state_id),
                                     prefix_tokens=depth(state_id), block_tokens=1)
        if parent is not None:
            children[parent].add(state_id)
    return Trace(name="leaf-store", requests=[], states=states, occurrences_ms={},
                 children=dict(children), terminal_branches={}, block_size=1)


class _MappingScorer:
    time_varying = True

    def __init__(self, scores):
        self.scores = scores

    def observe(self, requests, timestamp_ms):
        pass

    def score(self, state_id, timestamp_ms):
        return float(self.scores[state_id])


def _leaf_store(trace, capacity, scores, decisions=None, removals=None, sample_width=16, seed=0):
    def record_decision(candidates, score_tuples, victim_index, timestamp_ms, group_index,
                        arriving_index):
        decisions.append((tuple(candidates), victim_index, arriving_index))

    def record_removal(state_id, kind, timestamp_ms, group_index, decision_index):
        removals.append((state_id, kind))

    return twotier._VictimStore(
        trace, capacity, "learned", {}, lambda state_id: 1, eviction="sampled",
        sample_width=sample_width, seed=seed, scorer=_MappingScorer(scores),
        decision_hook=record_decision if decisions is not None else None,
        removal_hook=record_removal if removals is not None else None,
        frequency=defaultdict(int), last_group=defaultdict(int), eligibility="leaf",
    )


# --- the published path ---------------------------------------------------------------


class PublishedPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, self.path = build_trace(conversation_records())
        self.addCleanup(self.temporary.cleanup)

    def test_default_options_reproduce_the_plan_commit_exactly(self):
        fingerprints = published_path_fingerprints(self.trace)
        self.assertEqual(set(fingerprints), set(PUBLISHED_PATH))
        for name, expected in PUBLISHED_PATH.items():
            with self.subTest(arm=name):
                self.assertEqual(fingerprints[name], expected)

    def test_explicit_all_is_the_default(self):
        for policy, extra in (("lru", {}), ("lfu", {}), ("lru_2hit", {}),
                              ("learned", {"scorer": True})):
            with self.subTest(policy=policy):
                rows = []
                for explicit in (False, True):
                    arguments = dict(l2_eviction="sampled", l2_sample_width=4, l2_seed=1,
                                     measure_from_ms=MEASURE_FROM_MS)
                    if extra:
                        arguments["l2_scorer"] = RawFeatureScorer(self.trace, _LinearRanker())
                    if explicit:
                        arguments["l2_eligibility"] = "all"
                    rows.append(twotier.run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES,
                                                     policy, bytes_per_token=1,
                                                     **arguments).as_row())
                self.assertEqual(rows[0], rows[1])

    def test_the_default_path_keeps_no_resident_child_count(self):
        stores = []
        with mock.patch.object(twotier, "_VictimStore", _CheckedStore):
            _CheckedStore.instances = stores
            twotier.run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES, "lru", bytes_per_token=1,
                                 l2_eviction="sampled", l2_sample_width=4)
        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0].resident_children, {})
        self.assertEqual(stores[0].checks["insert"], 0)

    def test_eligibilities_are_fixed(self):
        self.assertEqual(twotier.L2_ELIGIBILITIES, ("all", "leaf"))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, _ = build_trace(conversation_records(60))
        self.addCleanup(self.temporary.cleanup)

    def _run(self, **arguments):
        return twotier.run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
                                    **arguments)

    def test_leaf_refuses_what_it_is_not_defined_for(self):
        refused = (
            dict(l2_policy="lru_2hit", l2_eviction="sampled", l2_eligibility="leaf"),
            dict(l2_policy="lru", l2_eviction="heap", l2_eligibility="leaf"),
            dict(l2_policy="offline_next_use", l2_eviction="heap", l2_eligibility="leaf"),
            dict(l2_policy="lru", l2_eviction="sampled", l2_eligibility="leaf",
                 l2_arrival_protection="direct_child"),
            dict(l2_policy="lru", l2_eviction="sampled", l2_eligibility="leaf",
                 l2_arrival_protection="all"),
            dict(l2_policy="lru", l2_eviction="sampled", l2_eligibility="leaf",
                 l2_sample_width=0),
            dict(l2_policy="lru", l2_eviction="sampled", l2_eligibility="leaf",
                 closure="standalone"),
            dict(l2_policy="lru", l2_eviction="sampled", l2_eligibility="leaves"),
        )
        for arguments in refused:
            with self.subTest(**{k: str(v) for k, v in arguments.items()}):
                with self.assertRaises(ValueError):
                    self._run(**arguments)

    def test_the_store_refuses_the_same_settings(self):
        common = dict(frequency=defaultdict(int), last_group=defaultdict(int))
        with self.assertRaises(ValueError):
            twotier._VictimStore(self.trace, 512, "lru_2hit", {}, lambda s: 512,
                                 eviction="sampled", eligibility="leaf", **common)
        with self.assertRaises(ValueError):
            twotier._VictimStore(self.trace, 512, "lru", {}, lambda s: 512, eviction="heap",
                                 eligibility="leaf")
        with self.assertRaises(ValueError):
            twotier._VictimStore(self.trace, 512, "lru", {}, lambda s: 512, eviction="sampled",
                                 eligibility="leaf", arrival_protection="direct_child", **common)

    def test_the_sampled_offline_comparator_needs_the_explicit_flag(self):
        with self.assertRaises(ValueError):
            self._run(l2_policy="offline_next_use", l2_eviction="sampled")
        with self.assertRaises(ValueError):
            self._run(l2_policy="lru", l2_eviction="sampled", l2_sampled_offline=True)
        with self.assertRaises(ValueError):
            self._run(l2_policy="offline_next_use", l2_eviction="heap", l2_sampled_offline=True)
        result = self._run(l2_policy="offline_next_use", l2_eviction="sampled",
                           l2_sampled_offline=True)
        self.assertGreater(result.l2_decisions, 0)


# --- leaf eligibility on a replay ---------------------------------------------------------


class LeafReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, self.path = build_trace(conversation_records())
        self.addCleanup(self.temporary.cleanup)

    def _run(self, rung, eligibility, width=4, seed=0, **hooks):
        return twotier.run_two_tier(
            self.trace, "lru", L1_BYTES, L2_BYTES, bytes_per_token=1,
            measure_from_ms=MEASURE_FROM_MS, l2_eviction="sampled", l2_sample_width=width,
            l2_seed=seed, l2_eligibility=eligibility, **rung_arguments(self.trace, rung), **hooks)

    def test_resident_child_counts_are_exact_after_every_event(self):
        for rung in RUNGS:
            with self.subTest(rung=rung):
                stores = []
                watch = _ClosureWatch(self.trace)
                with mock.patch.object(twotier, "_VictimStore", _CheckedStore):
                    _CheckedStore.instances = stores
                    result = self._run(rung, "leaf", l2_removal_hook=watch)
                store = stores[0]
                # Every way in and out of the store was exercised and checked.
                self.assertEqual(store.checks["insert"], result.l2_admissions + result.l2_rejections)
                self.assertEqual(store.checks["remove"], sum(watch.kinds.values()))
                self.assertGreater(watch.kinds["evicted"], 0)
                self.assertGreater(watch.kinds["promoted"], 0)
                if rung != "lru":
                    self.assertGreater(watch.kinds["rejected"], 0)

    def test_no_parent_with_a_resident_child_is_ever_rejected_or_evicted(self):
        for rung in RUNGS:
            for width in (1, 4, 64):
                with self.subTest(rung=rung, width=width):
                    watch = _ClosureWatch(self.trace)
                    result = self._run(rung, "leaf", width=width, l2_removal_hook=watch,
                                       victim_hook=watch.on_victim)
                    self.assertEqual(watch.violations, [])
                    self.assertGreater(watch.arrivals_with_resident_child, 0)
                    self.assertGreater(result.l2_evictions, 0)
                    # The rule is not vacuous: under "all" the same rungs do it.
                    if width == 4:
                        loose = _ClosureWatch(self.trace)
                        self._run(rung, "all", width=width, l2_removal_hook=loose)
                        self.assertGreater(len(loose.violations), 0)

    def test_leaf_leaves_nothing_present_but_unusable_and_all_does(self):
        for rung in RUNGS:
            with self.subTest(rung=rung):
                for width in (4, 16):
                    collector = AttributionCollector(self.trace, measure_from_ms=MEASURE_FROM_MS,
                                                     bytes_per_token=1)
                    leaf = self._run(rung, "leaf", width=width,
                                     l2_request_hook=collector.on_request,
                                     l2_removal_hook=collector)
                    collector.check_against(leaf)
                    self.assertEqual(leaf.l2_present_unusable_tokens, 0)
                    self.assertEqual(leaf.l2_present_unusable_blocks, 0)
                    self.assertEqual(collector.orphaned_blocks, 0)
                    self.assertEqual(collector.root_tokens["unexplained"], 0)
                    self.assertGreater(leaf.l2_avoided_tokens, 0)
                loose = self._run(rung, "all")
                self.assertGreater(loose.l2_present_unusable_tokens, 0)

    def test_leaf_replays_do_not_depend_on_the_string_hash_seed(self):
        child = (
            "import json, sys\n"
            "from persistent_kv_admission.trace import load_mooncake_trace\n"
            "from persistent_kv_admission.twotier import run_two_tier\n"
            "from persistent_kv_admission.mechanism import ExactLabelScorer\n"
            "trace = load_mooncake_trace(sys.argv[1])\n"
            "out = {}\n"
            "for name, extra in (('lru', dict(l2_policy='lru')),\n"
            "                    ('label', dict(l2_policy='learned',\n"
            "                                   l2_scorer=ExactLabelScorer(trace, 30.0))),\n"
            "                    ('offline', dict(l2_policy='offline_next_use',\n"
            "                                     l2_sampled_offline=True))):\n"
            "    r = run_two_tier(trace, 'lru', 4 * 512, 8 * 512, bytes_per_token=1,\n"
            "                     l2_eviction='sampled', l2_sample_width=4, l2_seed=5,\n"
            "                     l2_eligibility='leaf', **extra)\n"
            "    out[name] = r.as_row()\n"
            "print(json.dumps(out, sort_keys=True))\n"
        )
        outputs = []
        for hash_seed in ("0", "4242"):
            environment = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=SOURCE_ROOT)
            finished = subprocess.run([sys.executable, "-c", child, str(self.path)],
                                      capture_output=True, text=True, env=environment)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            outputs.append(finished.stdout)
        self.assertEqual(outputs[0], outputs[1])


# --- leaf eligibility on a constructed store ------------------------------------------------


class LeafStoreTests(unittest.TestCase):
    def test_counts_follow_admit_evict_reject_and_promote(self):
        trace = _tree_trace({"p": None, "c1": "p", "c2": "p", "g": "c1", "x": None})
        removals = []
        store = _leaf_store(trace, 3, {"p": 9, "c1": 9, "c2": 9, "g": 9, "x": 0},
                            removals=removals)
        store.admit("g", 1, 0, 0)
        store.admit("c2", 1, 0, 0)
        self.assertEqual(store.resident_children, {"g": 0, "c2": 0})
        store.admit("c1", 1, 0, 0)          # parent after child: counts g
        self.assertEqual(store.resident_children, {"g": 0, "c2": 0, "c1": 1})
        self.assertFalse(store.admit("x", 1, 0, 0))   # a leaf arrival, lowest: rejected
        self.assertEqual(removals[-1], ("x", "rejected"))
        self.assertEqual(store.resident_children, {"g": 0, "c2": 0, "c1": 1})
        store.promote("g")
        self.assertEqual(store.resident_children, {"c2": 0, "c1": 0})
        store.admit("p", 1, 0, 0)           # a parent arriving above two residents
        self.assertEqual(store.resident_children, {"c2": 0, "c1": 0, "p": 2})
        store.promote("c2")
        self.assertEqual(store.resident_children, {"c1": 0, "p": 1})

    def test_a_non_leaf_arrival_is_not_offered_and_cannot_be_rejected(self):
        trace = _tree_trace({"p": None, "c": "p", "x": None})
        decisions, removals = [], []
        store = _leaf_store(trace, 2, {"p": 0, "c": 5, "x": 7}, decisions, removals)
        store.admit("c", 1, 0, 0)
        store.admit("x", 1, 0, 0)
        # p scores lowest of all, but its child c is resident, so it is not a
        # candidate: the first round ranks the other leaves only.
        self.assertTrue(store.admit("p", 1, 1, 1))
        self.assertEqual(decisions, [(("c", "x"), 0, -1)])
        self.assertEqual(removals, [("c", "evicted")])
        self.assertEqual((store.admissions, store.rejections, store.evictions), (3, 0, 1))
        self.assertEqual(list(store.cached), ["x", "p"])

    def test_a_parent_becomes_eligible_after_its_last_resident_child_leaves(self):
        trace = _tree_trace({"p": None, "c1": "p", "c2": "p", "x": None, "y": None,
                             "z": None})
        decisions, removals = [], []
        scores = {"p": 0, "c1": 1, "c2": 2, "x": 5, "y": 6, "z": 7}
        store = _leaf_store(trace, 3, scores, decisions, removals)
        store.admit("c1", 1, 0, 0)
        store.admit("c2", 1, 0, 0)
        store.admit("p", 1, 0, 0)
        self.assertTrue(store.admit("x", 1, 0, 0))
        self.assertEqual(decisions[-1], (("x", "c1", "c2"), 1, 0))   # p is not offered
        self.assertTrue(store.admit("y", 1, 0, 0))
        self.assertEqual(decisions[-1], (("y", "c2", "x"), 1, 0))    # nor with one child left
        self.assertEqual(store.resident_children["p"], 0)
        # Its last resident child has left: p is a leaf now and, scoring lowest, goes.
        self.assertTrue(store.admit("z", 1, 0, 0))
        self.assertEqual(decisions[-1], (("z", "p", "x", "y"), 1, 0))
        self.assertEqual(removals, [("c1", "evicted"), ("c2", "evicted"), ("p", "evicted")])

    def test_promotion_frees_a_parent_too(self):
        trace = _tree_trace({"p": None, "c": "p", "x": None, "y": None, "w": None, "z": None})
        decisions = []
        store = _leaf_store(trace, 3, {"p": 0, "c": 9, "x": 5, "y": 6, "w": 8, "z": 7},
                            decisions)
        store.admit("c", 1, 0, 0)
        store.admit("p", 1, 0, 0)
        store.admit("x", 1, 0, 0)
        self.assertTrue(store.admit("y", 1, 0, 0))
        self.assertEqual(decisions[-1], (("y", "c", "x"), 2, 0))     # p is not offered
        store.promote("c")                   # its only resident child goes to L1
        self.assertEqual(store.resident_children["p"], 0)
        self.assertTrue(store.admit("w", 1, 0, 0))                   # fits: no decision
        self.assertEqual(len(decisions), 1)
        self.assertTrue(store.admit("z", 1, 0, 0))
        self.assertEqual(decisions[-1], (("z", "p", "y", "w"), 1, 0))
        self.assertNotIn("p", store.cached)

    def test_the_first_round_draws_uniformly_from_the_other_leaves_in_insertion_order(self):
        parents = {"r1": None, "r2": None, "p": None, "c": "p", "r3": None, "a": None}
        trace = _tree_trace(parents)
        decisions = []
        store = _leaf_store(trace, 5, {name: 9 for name in parents}, decisions,
                            sample_width=2, seed=11)
        for name in ("r1", "r2", "c", "p", "r3"):
            store.admit(name, 1, 0, 0)
        store.admit("a", 1, 0, 0)
        expected = random.Random(11).sample(["r1", "r2", "c", "r3"], 2)
        self.assertEqual(decisions[0][0], tuple(["a"] + expected))
        self.assertEqual(decisions[0][2], 0)

    def test_an_overflow_without_a_leaf_is_an_invariant_violation(self):
        trace = _tree_trace({"p": None, "c": "p", "x": None})
        store = _leaf_store(trace, 2, {"p": 0, "c": 5, "x": 7})
        store.admit("c", 1, 0, 0)
        store.admit("x", 1, 0, 0)
        for state_id in store.resident_children:
            store.resident_children[state_id] = 1      # corrupt the count on purpose
        with self.assertRaises(AssertionError):
            store.admit("p", 1, 1, 1)


# --- the label rung ----------------------------------------------------------------------------


class ExactLabelScorerTests(unittest.TestCase):
    HORIZON = 600.0

    def setUp(self):
        # State int:1 occurs at 0, twice at 1000, at 5000, at 605000 (exactly
        # 600 s after 5000), and last at 1205001 (600.001 s after 605000).
        stamps = (0, 1000, 1000, 5000, 605000, 1205001)
        records = [{"timestamp": t, "input_length": 512, "output_length": 1, "hash_ids": [1]}
                   for t in stamps]
        self.temporary, self.trace, _ = build_trace(records)
        self.addCleanup(self.temporary.cleanup)
        self.scorer = ExactLabelScorer(self.trace, self.HORIZON)

    def expected(self, delta_ms):
        return float(target_column(np.asarray([delta_ms]), np.asarray([0.0]), "next_use",
                                   self.HORIZON)[0])

    def test_it_equals_the_training_target_at_random_instants(self):
        labeller = _Labeller(self.trace, self.HORIZON)
        rng = np.random.default_rng(3)
        instants = np.concatenate([rng.uniform(-1000.0, 1_300_000.0, 400),
                                   [0.0, 1000.0, 5000.0, 605000.0, 1205001.0]])
        pairs = [labeller("int:1", float(t)) for t in instants]
        labels = target_column(np.asarray([p[0] for p in pairs]),
                               np.asarray([p[1] for p in pairs], dtype=float),
                               "next_use", self.HORIZON)
        mine = np.asarray([self.scorer.score("int:1", float(t)) for t in instants])
        np.testing.assert_array_equal(mine, labels)

    def test_occurrences_at_the_current_instant_are_excluded(self):
        self.assertEqual(self.scorer.score("int:1", 1000.0), -math.log1p(4.0))
        self.assertEqual(self.scorer.score("int:1", 999.0), -math.log1p(0.001))

    def test_the_horizon_clips_at_exactly_h(self):
        at_horizon = self.scorer.score("int:1", 5000.0)         # next use 600 s later
        self.assertEqual(at_horizon, -math.log1p(self.HORIZON))
        self.assertEqual(at_horizon, self.expected(600_000.0))
        just_inside = self.scorer.score("int:1", 5001.0)
        self.assertEqual(just_inside, self.expected(599_999.0))
        self.assertGreater(just_inside, at_horizon)
        beyond = self.scorer.score("int:1", 605000.0)           # 600.001 s: clipped
        self.assertEqual(beyond, at_horizon)

    def test_no_further_use_scores_the_clipped_value(self):
        self.assertEqual(self.scorer.score("int:1", 1205001.0), -math.log1p(self.HORIZON))
        self.assertEqual(self.scorer.score("int:1", 1205001.0), self.expected(math.inf))
        self.assertEqual(self.scorer.score("int:1", 2_000_000.0), self.expected(math.inf))

    def test_it_is_a_scorer_that_ignores_history(self):
        self.assertTrue(ExactLabelScorer.time_varying)
        self.assertIsNone(self.scorer.observe([], 0.0))
        with self.assertRaises(ValueError):
            ExactLabelScorer(self.trace, self.HORIZON, target="next_use_squared")


# --- the offline rung on the sampled path ----------------------------------------------------


class SampledOfflineTests(unittest.TestCase):
    def setUp(self):
        self.temporary, self.trace, _ = build_trace(conversation_records())
        self.addCleanup(self.temporary.cleanup)
        self.groups = _occurrence_groups(self.trace)
        # A heap store only to ask for the heap comparator's key.
        self.heap = twotier._VictimStore(self.trace, L2_BYTES, "offline_next_use", self.groups,
                                         lambda s: 512)

    def _decisions(self, eligibility, width):
        decisions = []

        def hook(candidates, scores, victim_index, timestamp_ms, group_index, arriving_index):
            decisions.append((list(candidates), list(scores), victim_index, group_index))

        twotier.run_two_tier(self.trace, "lru", L1_BYTES, L2_BYTES, "offline_next_use",
                             bytes_per_token=1, occurrence_groups=self.groups,
                             l2_eviction="sampled", l2_sample_width=width, l2_seed=0,
                             l2_eligibility=eligibility, l2_sampled_offline=True,
                             l2_decision_hook=hook)
        return decisions

    def test_every_candidate_is_scored_by_the_heap_comparators_key(self):
        for eligibility in ("all", "leaf"):
            with self.subTest(eligibility=eligibility):
                decisions = self._decisions(eligibility, 4)
                self.assertGreater(len(decisions), 0)
                for candidates, scores, _, group_index in decisions:
                    expected = [self.heap._key(state_id, 0, 0, group_index)
                                for state_id in candidates]
                    self.assertEqual(scores, expected)

    def test_a_full_width_decision_removes_the_comparators_minimum(self):
        decisions = self._decisions("all", 64)
        unique = 0
        for candidates, scores, victim_index, group_index in decisions:
            # Every resident is a candidate: the store holds at most 8 blocks.
            self.assertLessEqual(len(candidates), L2_BYTES // 512 + 1)
            self.assertEqual(scores[victim_index], min(scores))
            self.assertEqual(victim_index, scores.index(min(scores)))
            if scores.count(min(scores)) == 1:
                unique += 1
                ordered = sorted(candidates, key=lambda s: self.heap._key(s, 0, 0, group_index))
                self.assertEqual(candidates[victim_index], ordered[0])
        self.assertGreater(unique, 0)


# --- the decomposition helpers ------------------------------------------------------------


class DecompositionTests(unittest.TestCase):
    def test_the_four_terms_sum_to_t_exactly_on_integers(self):
        rng = random.Random(5)
        for _ in range(500):
            utilities = {rung: rng.randrange(-10**9, 10**9) for rung in RUNGS}
            h_off = rng.randrange(-10**9, 10**9)
            out = decompose(utilities, h_off)
            self.assertEqual(sum(out["gaps"][name] for name in GAPS), out["total"])
            self.assertEqual(out["total"], h_off - utilities["lru"])

    def test_gap_definitions(self):
        out = decompose({"lru": 1, "learned": 3, "label": 6, "offline": 10}, 15)
        self.assertEqual(out["gaps"], {"candidate_search": 5, "objective": 4, "signal": 3,
                                       "achieved": 2})
        self.assertEqual(out["total"], 14)
        self.assertAlmostEqual(out["shares"]["candidate_search"], 5 / 14)
        self.assertEqual(out["dominant"], "mixed")

    def test_dominant_labels(self):
        base = {"lru": 0, "learned": 0, "label": 0, "offline": 0}
        self.assertEqual(decompose(base, 10)["dominant"], "mechanism_bound")
        self.assertEqual(decompose(dict(base, offline=10, label=0), 10)["dominant"],
                         "objective_bound")
        self.assertEqual(decompose({"lru": 0, "learned": 0, "label": 6, "offline": 8}, 10)
                         ["dominant"], "signal_bound")
        self.assertEqual(decompose({"lru": 0, "learned": 5, "label": 7, "offline": 8}, 10)
                         ["dominant"], "achieved")
        # Exactly half is enough.
        self.assertEqual(decompose({"lru": 0, "learned": 1, "label": 2, "offline": 5}, 10)
                         ["dominant"], "mechanism_bound")
        # Nothing reaches half.
        self.assertEqual(decompose({"lru": 0, "learned": 3, "label": 6, "offline": 8}, 10)
                         ["dominant"], "mixed")

    def test_negative_terms_and_no_headroom(self):
        # The candidate-search gap can be negative (sampled offline above the
        # heap); two terms then clear half of T and neither dominates.
        out = decompose({"lru": 0, "learned": 0, "label": 6, "offline": 14}, 10)
        self.assertEqual(out["gaps"]["candidate_search"], -4)
        self.assertEqual(out["dominant"], "mixed")
        out = decompose({"lru": 0, "learned": 0, "label": 2, "offline": 12}, 10)
        self.assertEqual(out["gaps"], {"candidate_search": -2, "objective": 10, "signal": 2,
                                       "achieved": 0})
        self.assertEqual(out["dominant"], "objective_bound")
        for h_off in (0, -3):
            out = decompose({"lru": 0, "learned": 1, "label": 2, "offline": 3}, h_off)
            self.assertEqual(out["dominant"], "mixed")
        self.assertTrue(all(math.isnan(v) for v in decompose(
            {"lru": 4, "learned": 1, "label": 2, "offline": 3}, 4)["shares"].values()))
        self.assertEqual(dominant_label({name: 1.0 for name in GAPS}, math.nan), "mixed")

    def test_missing_rungs_are_refused(self):
        with self.assertRaises(ValueError):
            decompose({"lru": 0, "learned": 0, "label": 0}, 1)

    def test_ladder_order(self):
        ordered = {"lru": 1, "learned": 2, "label": 2, "offline": 5}
        self.assertEqual(ladder_holds(ordered), {"lru_le_learned": True,
                                                 "learned_le_label": True,
                                                 "label_le_offline": True})
        inverted = {"lru": 3, "learned": 2, "label": 4, "offline": 5}
        self.assertEqual(ladder_holds(inverted)["lru_le_learned"], False)
        self.assertTrue(ladder_ordered([ordered, ordered]))
        self.assertFalse(ladder_ordered([ordered, inverted]))
        self.assertFalse(ladder_ordered([]))

    def test_seed_sign_readings(self):
        self.assertEqual(sign_reading([1, 2, 3, 4, 5]), "consistent_gain")
        self.assertEqual(sign_reading([-1, -2, -3, -4, -5]), "consistent_loss")
        self.assertEqual(sign_reading([1, 2, 0, 4, 5]), "mixed")
        self.assertEqual(sign_reading([1, -2, 3, 4, 5]), "mixed")
        self.assertEqual(sign_reading([]), "mixed")
        self.assertEqual(sign_counts([1, 0, -1, 2, 0]), (2, 2, 1))


if __name__ == "__main__":
    unittest.main()
