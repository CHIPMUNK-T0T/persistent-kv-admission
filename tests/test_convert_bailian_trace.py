"""Tests for the Bailian -> Mooncake 512-token converter (scripts/convert_bailian_trace.py).

Small synthetic Bailian records check the grouping of 16-token ids into
512-token blocks, the kept final partial group and its token count, prefix
identity (a shared id group under different prefixes is two states; two
requests sharing their first 64 ids share exactly two blocks), the exact
seconds-to-milliseconds conversion, the stable timestamp ordering, and that
the unmodified `load_mooncake_trace` reads the output with the expected state
metadata. No cache replay is run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from persistent_kv_admission.trace import load_mooncake_trace

REPOSITORY = Path(__file__).resolve().parents[1]


def _script():
    path = REPOSITORY / "scripts" / "convert_bailian_trace.py"
    spec = importlib.util.spec_from_file_location("_convert_bailian_trace_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = _script()


def _record(timestamp, input_length, hash_ids, chat_id=0, parent=-1, output_length=7):
    assert len(hash_ids) == math.ceil(input_length / 16)
    return {"chat_id": chat_id, "parent_chat_id": parent, "timestamp": timestamp,
            "input_length": input_length, "output_length": output_length,
            "type": "text", "turn": 1, "hash_ids": list(hash_ids)}


def _lines(records):
    return [json.dumps(record) + "\n" for record in records]


def _convert(records, block_tokens=512):
    parsed = convert.parse_records(_lines(records))
    return list(convert.convert_records(parsed, block_tokens))


class _LoadedTrace:
    """Write converted records and read them with the unmodified loader."""

    def __init__(self, records, block_tokens=512):
        self.records = _convert(records, block_tokens)
        self._directory = tempfile.TemporaryDirectory()
        path = Path(self._directory.name) / "bailian_test_trace.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in self.records))
        self.trace = load_mooncake_trace(path, block_tokens)

    def close(self):
        self._directory.cleanup()


class GroupingTest(unittest.TestCase):
    def test_consecutive_groups_of_32_become_one_block(self):
        records = _convert([_record(0.0, 1024, range(64))])
        self.assertEqual(len(records[0]["hash_ids"]), 2)
        self.assertEqual(records[0]["hash_ids"], [0, 1])

    def test_block_count_matches_the_loader_rule(self):
        for input_length in (1, 15, 16, 17, 511, 512, 513, 1000, 1008, 1020, 1024, 1025, 5000):
            ids = range(math.ceil(input_length / 16))
            records = _convert([_record(0.0, input_length, ids)])
            self.assertEqual(len(records[0]["hash_ids"]), math.ceil(input_length / 512), input_length)

    def test_block_id_depends_on_every_id_of_the_group(self):
        base = list(range(64))
        for position in (32, 47, 63):
            changed = list(base)
            changed[position] = 999
            records = _convert([_record(0.0, 1024, base), _record(1.0, 1024, changed, chat_id=1)])
            self.assertEqual(records[0]["hash_ids"][0], records[1]["hash_ids"][0])
            self.assertNotEqual(records[0]["hash_ids"][1], records[1]["hash_ids"][1], position)

    def test_other_block_sizes(self):
        records = _convert([_record(0.0, 100, range(7))], block_tokens=32)
        self.assertEqual(len(records[0]["hash_ids"]), 4)  # groups 2+2+2+1
        for bad in (0, 500, 8, -512):
            with self.assertRaises(ValueError):
                convert.group_size(bad)


class PartialGroupTest(unittest.TestCase):
    def test_final_partial_group_is_kept_and_sized_by_the_remainder(self):
        loaded = _LoadedTrace([_record(0.0, 1000, range(63))])  # 32 ids + 31 ids
        try:
            first, second = loaded.records[0]["hash_ids"]
            states = loaded.trace.states
            self.assertEqual((states[f"int:{first}"].block_tokens, states[f"int:{first}"].prefix_tokens),
                             (512, 512))
            self.assertEqual((states[f"int:{second}"].block_tokens, states[f"int:{second}"].prefix_tokens),
                             (488, 1000))
        finally:
            loaded.close()

    def test_partial_final_block_never_aliases_a_full_block(self):
        # Same 64 source ids, but in the first request the last 16-token block
        # is partial (1020 tokens). The final 512-block is then a 508-token
        # state, distinct from the full 512-token block of the second request;
        # one shared id would be rejected by the loader as inconsistent.
        ids = list(range(64))
        loaded = _LoadedTrace([_record(0.0, 1020, ids), _record(1.0, 1100, ids + [64, 65, 66, 67, 68],
                                                                  chat_id=1)])
        try:
            a, b = loaded.records[0]["hash_ids"], loaded.records[1]["hash_ids"]
            self.assertEqual(a[0], b[0])
            self.assertNotEqual(a[1], b[1])
            self.assertEqual(loaded.trace.states[f"int:{a[1]}"].block_tokens, 508)
            self.assertEqual(loaded.trace.states[f"int:{b[1]}"].block_tokens, 512)
        finally:
            loaded.close()

    def test_hash_count_must_match_input_length(self):
        bad = _record(0.0, 1024, range(64))
        bad["hash_ids"] = list(range(63))
        with self.assertRaises(ValueError):
            convert.parse_records(_lines([bad]))
        empty = _record(0.0, 16, [0])
        empty["hash_ids"] = []
        with self.assertRaises(ValueError):
            convert.parse_records(_lines([empty]))


class PrefixIdentityTest(unittest.TestCase):
    def test_sharing_the_first_64_ids_shares_exactly_two_blocks(self):
        shared = list(range(64))
        first = shared + list(range(100, 132))
        second = shared + list(range(200, 232))
        loaded = _LoadedTrace([_record(0.0, 1536, first), _record(1.0, 1536, second, chat_id=1)])
        try:
            a, b = loaded.records[0]["hash_ids"], loaded.records[1]["hash_ids"]
            self.assertEqual(len(set(a) & set(b)), 2)
            self.assertEqual(a[:2], b[:2])
            self.assertNotEqual(a[2], b[2])
            self.assertEqual(len(loaded.trace.states), 4)
            self.assertEqual(loaded.trace.occurrences_ms[f"int:{a[1]}"], [0.0, 1000.0])
        finally:
            loaded.close()

    def test_same_group_under_different_prefixes_is_two_states(self):
        # Bailian To-C/To-B ids are per-block content hashes: the same ids can
        # follow a different prefix. The loader keys a state by id alone, so
        # the output id must encode the whole prefix.
        group = list(range(32))
        loaded = _LoadedTrace([
            _record(0.0, 512, group),
            _record(1.0, 1024, list(range(500, 532)) + group, chat_id=1),
        ])
        try:
            a, b = loaded.records[0]["hash_ids"], loaded.records[1]["hash_ids"]
            self.assertNotEqual(a[0], b[1])
            self.assertEqual(loaded.trace.states[f"int:{b[1]}"].parent_id, f"int:{b[0]}")
            self.assertEqual(len(loaded.trace.states), 3)
        finally:
            loaded.close()

    def test_repeated_group_inside_one_request(self):
        group = list(range(32))
        records = _convert([_record(0.0, 1024, group + group)])
        first, second = records[0]["hash_ids"]
        self.assertNotEqual(first, second)

    def test_ids_are_sequential_in_first_appearance_order(self):
        records = _convert([_record(0.0, 1024, range(64)), _record(1.0, 600, range(1000, 1038), chat_id=1),
                            _record(2.0, 1024, range(64), chat_id=2)])
        self.assertEqual([record["hash_ids"] for record in records], [[0, 1], [2, 3], [0, 1]])


class TimestampTest(unittest.TestCase):
    def test_seconds_to_exact_milliseconds(self):
        self.assertEqual(convert.seconds_to_ms(Decimal("61.114")), 61114)
        self.assertIsInstance(convert.seconds_to_ms(Decimal("61.114")), int)
        self.assertEqual(convert.seconds_to_ms(Decimal("0.0")), 0)
        self.assertEqual(convert.seconds_to_ms(Decimal("7199.97")), 7199970)
        self.assertEqual(convert.seconds_to_ms(3), 3000)
        self.assertEqual(convert.seconds_to_ms(0.29), 290)  # float input, exact via its repr
        self.assertEqual(convert.seconds_to_ms(Decimal("0.0005")), 0.5)
        for bad in (True, "1.0", None, float("nan")):
            with self.assertRaises(ValueError):
                convert.seconds_to_ms(bad)

    def test_records_are_stably_sorted_by_timestamp(self):
        records = _convert([
            _record(2.5, 16, [1], chat_id=10),
            _record(0.007, 16, [2], chat_id=11),
            _record(2.5, 16, [3], chat_id=12),
            _record(0.007, 16, [4], chat_id=13),
            _record(1.0, 16, [5], chat_id=14),
        ])
        self.assertEqual([record["chat_id"] for record in records], [11, 13, 14, 10, 12])
        self.assertEqual([record["timestamp"] for record in records], [7, 7, 1000, 2500, 2500])
        # ids are assigned after sorting, so they increase down the file
        self.assertEqual([record["hash_ids"] for record in records], [[0], [1], [2], [3], [4]])

    def test_ties_survive_and_the_loader_groups_them(self):
        loaded = _LoadedTrace([_record(0.25, 16, [1]), _record(0.25, 16, [2], chat_id=1),
                               _record(0.251, 16, [1], chat_id=2)])
        try:
            groups = [(timestamp, len(requests)) for timestamp, requests in loaded.trace.timestamp_groups()]
            self.assertEqual(groups, [(250.0, 2), (251.0, 1)])
        finally:
            loaded.close()


class FileTest(unittest.TestCase):
    def test_convert_file_is_deterministic_and_loader_compatible(self):
        records = [_record(0.0, 1000, range(63)), _record(0.5, 1536, list(range(64)) + list(range(70, 102)),
                                                          chat_id=1, parent=0),
                   _record(0.5, 20, [7, 8], chat_id=2)]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "qwen_test_blksz_16.jsonl"
            source.write_text("".join(_lines(records)))
            outputs = []
            for name in ("a", "b"):
                output = Path(directory) / name / "bailian_test_trace.jsonl"
                with contextlib.redirect_stdout(io.StringIO()):
                    convert.main([str(source), "--output", str(output)])
                outputs.append(output.read_bytes())
                manifest = json.loads((output.parent / "bailian_test_trace.manifest.json").read_text())
                self.assertEqual(manifest["records"], 3)
                self.assertEqual(manifest["block_tokens"], 512)
                self.assertEqual(manifest["timestamp_unit"], "ms")
            self.assertEqual(outputs[0], outputs[1])
            trace = load_mooncake_trace(Path(directory) / "a" / "bailian_test_trace.jsonl", 512)
            self.assertEqual(trace.name, "bailian_test_trace")
            self.assertEqual(len(trace.requests), 3)
            self.assertEqual([request.timestamp_ms for request in trace.requests], [0.0, 500.0, 500.0])
            first = json.loads(outputs[0].decode().splitlines()[1])
            self.assertEqual(first["parent_chat_id"], 0)
            self.assertEqual(set(first), {"timestamp", "input_length", "output_length", "hash_ids",
                                          "chat_id", "parent_chat_id", "type", "turn"})

    def test_main_rejects_bad_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                convert.main([])
            with self.assertRaises(SystemExit):
                convert.main(["a.jsonl", "b.jsonl", "--output", "c.jsonl"])
        with self.assertRaises(ValueError):
            convert.main(["a.jsonl", "--block-tokens", "500"])


if __name__ == "__main__":
    unittest.main()
