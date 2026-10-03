"""Tests for the Bailian input audit (scripts/audit_bailian_inputs.py).

Constructed inputs only, written to a temporary directory: nine upstream-format
records (integer ids, timestamps with 0-3 decimals, a same-timestamp pair, a
child that extends its parent exactly, one whose parent's last id is replaced
as in the upstream FAQ Q2, one that shares nothing with its parent, one that
shares a short prefix, partial 16-token blocks, inputs longer than 512 tokens,
an id that recurs at two positions after two predecessors, unresolvable and
later parents), converted with `convert_file`, then audited. Every section's
numbers are checked against values worked out by hand in the comments; the
direct section-D computation is checked against the unmodified loader,
`gap.working_set_bytes` and `characterize_external_traces.repeat_shares`, the
capacities against `run_decision_population._capacity`; tampering fails the
right checks; the CLI writes its four outputs with exit status 0, or 1 after
tampering. No real trace is read.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from persistent_kv_admission.gap import working_set_bytes
from persistent_kv_admission.trace import load_mooncake_trace

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "audit_bailian_inputs.py"


def _script():
    spec = importlib.util.spec_from_file_location("_audit_bailian_inputs_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _script()
convert = audit.convert
characterize = audit.characterize
decision_population = audit.decision_population

UPSTREAM = "qwen_traceA_blksz_16.jsonl"
STEM = convert.NAMES[UPSTREAM]  # bailian_toc_trace
A = list(range(100, 165))  # 65 ids = 1040 tokens: two full 512-blocks and 16 tokens

# (chat_id, parent_chat_id, raw timestamp text, input_length, output_length, type, turn, hash_ids)
RECORDS = (
    (1, -1, "0", 1040, 10, "text", 1, A),
    (2, 1, "1.5", 1100, 20, "text", 2, A + [200, 201, 202, 203]),             # extends 1 exactly
    (3, -1, "1.5", 1072, 30, "search", 1, A + [200, 201]),                     # same timestamp as 2
    (4, 2, "2.25", 1152, 40, "text", 3, A + [200, 201, 202, 210, 211, 212, 213]),  # 2's last id replaced
    (5, 3, "3.125", 20, 50, "text", 2, [500, 501]),                            # shares nothing with 3
    (6, -1, "4", 512, 0, "text", 1, [600, 777, 601, 602, 777] + list(range(603, 630))),
    (7, 4, "12.000", 187, 5, "text", 7, A[:10] + [700, 701]),                  # LCP 10 with 4
    (8, 9, "13.5", 16, 7, "file", 12, [800]),                                  # parent comes later
    (9, 99, "20", 520, 100, "image", 2, list(range(900, 933))),                # parent not in file
)


def _line(chat_id, parent, timestamp, input_length, output_length, kind, turn, ids):
    assert len(ids) == math.ceil(input_length / 16)
    record = {"chat_id": chat_id, "parent_chat_id": parent, "timestamp": "@TS@",
              "input_length": input_length, "output_length": output_length, "type": kind,
              "turn": turn, "hash_ids": list(ids)}
    return json.dumps(record).replace('"@TS@"', timestamp) + "\n"


class Inputs:
    """Upstream file, converted file and manifest in a temporary directory."""

    def __init__(self, records=RECORDS):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.source_dir = self.root / "source"
        self.converted_dir = self.root / "converted"
        self.source_dir.mkdir()
        self.source = self.source_dir / UPSTREAM
        self.source.write_text("".join(_line(*record) for record in records), encoding="utf-8")
        self.converted = self.converted_dir / f"{STEM}.jsonl"
        self.manifest_path = self.converted_dir / f"{STEM}.manifest.json"
        self.manifest = convert.convert_file(self.source, self.converted)

    def records(self):
        return audit.read_source(self.source)

    def audit(self, granularities=(16, 512)):
        return audit.audit_source(UPSTREAM, self.source_dir, self.converted_dir, list(granularities))

    def close(self):
        self._directory.cleanup()


def _failed(result):
    return {check["check"] for check in result["checks"] if not check["pass"]}


class _Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = Inputs()
        cls.result = cls.inputs.audit()

    @classmethod
    def tearDownClass(cls):
        cls.inputs.close()


class SchemaTest(_Shared):
    def test_times(self):
        schema = self.result["schema"]
        self.assertEqual(schema["records"], 9)
        self.assertEqual(schema["timestamp_min_s"], 0.0)
        self.assertEqual(schema["timestamp_max_s"], 20.0)
        self.assertEqual(schema["span_s"], 20.0)
        self.assertEqual(schema["max_timestamp_decimals"], 3)  # "3.125" and "12.000"
        self.assertEqual(schema["timestamp_decreases"], 0)
        self.assertEqual(schema["distinct_timestamps"], 8)  # 1.5 twice
        self.assertEqual(schema["multi_record_timestamp_groups"], 1)
        self.assertEqual(schema["records_in_multi_record_groups"], 2)
        self.assertEqual(schema["largest_timestamp_group"], 2)

    def test_lengths(self):
        schema = self.result["schema"]
        # 1040 + 1100 + 1072 + 1152 + 20 + 512 + 187 + 16 + 520
        self.assertEqual(schema["total_input_tokens"], 5619)
        self.assertAlmostEqual(schema["input_mean"], 5619 / 9)
        # sorted: 16 20 187 512 520 1040 1072 1100 1152
        self.assertEqual(schema["input_median"], 520.0)
        self.assertAlmostEqual(schema["input_p95"], 1100 + 0.6 * (1152 - 1100))  # index 7.6
        self.assertEqual(schema["input_max"], 1152)
        self.assertAlmostEqual(schema["output_mean"], 262 / 9)
        self.assertEqual(schema["output_median"], 20.0)
        # input_length % 16: 1100 -> 12, 20 -> 4, 187 -> 11, 520 -> 8
        self.assertEqual(schema["partial16_records"], 4)
        self.assertEqual(schema["partial16_tokens"], 35)
        # input_length % 512: 16 76 48 128 20 (512 -> 0) 187 16 8
        self.assertEqual(schema["partial512_records"], 8)
        self.assertEqual(schema["partial512_tokens"], 499)
        self.assertAlmostEqual(schema["partial512_token_share"], 499 / 5619)

    def test_sessions(self):
        schema = self.result["schema"]
        self.assertEqual(schema["distinct_chat_ids"], 9)
        self.assertTrue(schema["chat_id_unique"])
        self.assertEqual(schema["root_records"], 3)
        self.assertAlmostEqual(schema["root_share"], 3 / 9)
        # 2->1, 4->2, 5->3, 7->4 earlier; 8->9 later; 9->99 missing
        self.assertEqual(schema["resolvable_parent_records"], 5)
        self.assertEqual(schema["resolvable_parent_earlier_records"], 4)
        self.assertEqual(schema["unresolvable_parent_records"], 1)
        self.assertEqual(schema["turn_buckets"],
                         {"1": 3, "2": 3, "3-5": 1, "6-10": 1, ">10": 1, "other": 0})
        self.assertEqual(schema["type_counts"], {"file": 1, "image": 1, "search": 1, "text": 6})

    def test_out_of_order_timestamps_fail_the_check(self):
        records = list(RECORDS)
        records[4], records[5] = records[5], records[4]  # 4 s before 3.125 s
        inputs = Inputs(records)
        try:
            result = inputs.audit()
            self.assertEqual(result["schema"]["timestamp_decreases"], 1)
            self.assertEqual(result["identity"]["fresh"]["records_moved_by_sort"], 2)  # both swapped
            self.assertEqual(_failed(result), {"timestamp_decreases"})
        finally:
            inputs.close()

    def test_duplicate_chat_id_fails_the_check(self):
        records = list(RECORDS)
        records[5] = (5,) + records[5][1:]
        inputs = Inputs(records)
        try:
            result = inputs.audit([512])
            self.assertFalse(result["schema"]["chat_id_unique"])
            self.assertEqual(_failed(result), {"chat_id_unique_per_record"})
        finally:
            inputs.close()


class PrefixIdentityTest(_Shared):
    def test_ids(self):
        prefix = self.result["prefix"]
        # 65 + 4 + 4 + 2 + 31 + 2 + 1 + 33 distinct ids; 65+69+67+72+2+32+12+1+33 occurrences
        self.assertEqual(prefix["ids16_distinct"], 142)
        self.assertEqual(prefix["ids16_occurrences"], 353)
        # 777: positions 1 and 4, predecessors 600 and 602; every other id is prefix-consistent
        self.assertEqual(prefix["ids16_at_multiple_positions"], 1)
        self.assertEqual(prefix["ids16_after_multiple_predecessors"], 1)
        self.assertEqual(prefix["ids16_occurrences_new_position"], 1)
        self.assertEqual(prefix["ids16_occurrences_new_predecessor"], 1)
        self.assertAlmostEqual(prefix["ids16_occurrences_new_predecessor_share"], 1 / 353)
        self.assertAlmostEqual(prefix["ids16_after_multiple_predecessors_share"], 1 / 142)

    def test_parent_prefix(self):
        prefix = self.result["prefix"]
        self.assertEqual(prefix["children_with_parent"], 5)
        self.assertEqual(prefix["lcp_full"], 1)          # 2 over 1: 65 of 65
        self.assertEqual(prefix["lcp_all_but_last"], 1)  # 4 over 2: 68 of 69
        self.assertEqual(prefix["lcp_partial"], 1)       # 7 over 4: 10 of 72
        self.assertEqual(prefix["lcp_none"], 2)          # 5 over 3, 8 over 9
        self.assertAlmostEqual(prefix["lcp_none_share"], 2 / 5)
        self.assertEqual(prefix["children_shorter_than_parent"], 3)  # 5, 7, 8
        self.assertEqual(prefix["parents_single_block"], 0)

    def test_one_block_parent_is_full_or_none(self):
        self.assertEqual(audit.lcp_category(1, 1), "full")
        self.assertEqual(audit.lcp_category(0, 1), "none")
        self.assertEqual(audit.lcp_category(1, 2), "all_but_last")
        self.assertEqual(audit.lcp_category(0, 2), "none")
        self.assertEqual(audit.lcp_category(1, 3), "partial")
        records = convert.parse_records([
            _line(1, -1, "0", 16, 1, "text", 1, [5]),
            _line(2, 1, "1", 32, 1, "text", 2, [5, 6]),
            _line(3, 1, "2", 32, 1, "text", 2, [7, 8]),
        ])
        prefix = audit.prefix_identity(records, audit.parent_indices(records))
        self.assertEqual((prefix["lcp_full"], prefix["lcp_none"], prefix["parents_single_block"]), (1, 1, 2))


class GranularityTest(_Shared):
    def test_window(self):
        records = self.inputs.records()
        self.assertEqual(audit.window_start_ms(records), 0.0 + 0.6 * 20000.0)

    def test_16(self):
        d16 = self.result["granularity"]["16"]
        # new states: 65 + 4 + 0 + 4 + 2 + 32 + 2 + 1 + 33; tokens 1040+60+0+64+20+512+27+16+520
        self.assertEqual(d16["unique_states"], 143)
        self.assertEqual(d16["states_counted"], 143)
        self.assertEqual(d16["working_set_bytes"], 2259 * 2048)
        self.assertEqual(d16["block_occurrences"], 353)
        self.assertEqual(d16["total_input_tokens"], 5619)
        self.assertEqual(d16["token_variant_keys"], 0)
        # strict: 2 -> 1040, 3 -> 1040 (2 is the same timestamp), 4 -> 68 blocks, 7 -> 10 blocks
        self.assertEqual(d16["repeated_tokens"], 1040 + 1040 + 1088 + 160)
        # file order: 3 also sees 2's blocks 65, 66 -> 1072
        self.assertEqual(d16["repeated_tokens_file_order"], 1040 + 1072 + 1088 + 160)
        self.assertEqual(d16["repeat_share"], 3328 / 5619)
        self.assertEqual(d16["repeat_share_file_order"], 3360 / 5619)
        # last 40 %: timestamps >= 12 000 ms -> records 7, 8, 9
        self.assertEqual(d16["last40_requests"], 3)
        self.assertEqual(d16["last40_input_tokens"], 187 + 16 + 520)
        self.assertEqual(d16["last40_repeated_tokens"], 160)
        self.assertEqual(d16["last40_repeated_tokens_file_order"], 160)
        self.assertEqual(d16["repeat_share_last40"], 160 / 723)

    def test_512(self):
        d512 = self.result["granularity"]["512"]
        # 1: 3 blocks (512, 512, 16); 2, 3, 4: a new third block of 76, 48, 128 tokens;
        # 5, 6, 7, 8: one block each; 9: 512 + 8
        self.assertEqual(d512["unique_states"], 12)
        self.assertEqual(d512["unique_states"], self.inputs.manifest["output_blocks"])
        self.assertEqual(d512["working_set_bytes"], 2547 * 2048)
        self.assertEqual(d512["block_occurrences"], 18)
        # only the two full blocks of A repeat, for 2, 3 and 4; 3's partial block differs from 2's
        self.assertEqual(d512["repeated_tokens"], 3 * 1024)
        self.assertEqual(d512["repeated_tokens_file_order"], 3 * 1024)
        self.assertEqual(d512["last40_repeated_tokens"], 0)
        self.assertEqual(d512["repeat_share_last40"], 0.0)

    def test_coarsening(self):
        effect = self.result["coarsening"]
        self.assertEqual(effect["working_set_ratio_512_over_16"], 2547 / 2259)
        self.assertEqual(effect["unique_states_ratio_16_over_512"], 143 / 12)
        self.assertAlmostEqual(effect["repeat_share_diff_16_minus_512"], 256 / 5619)
        self.assertAlmostEqual(effect["repeat_share_file_order_diff_16_minus_512"], 288 / 5619)
        self.assertAlmostEqual(effect["repeat_share_last40_diff_16_minus_512"], 160 / 723)
        self.assertAlmostEqual(effect["repeat_share_last40_file_order_diff_16_minus_512"], 160 / 723)

    def test_capacities_follow_run_decision_population(self):
        cells = self.result["granularity"]["512"]["capacities"]
        self.assertEqual(len(cells), 6)
        by_cell = {(cell["l1_fraction"], cell["l2_multiplier"]): cell for cell in cells}
        # working set 5 216 256 bytes: x 0.0025 = 13 040.64, x 0.04 = 208 650.24
        self.assertEqual(by_cell[(0.0025, 1)]["l1_bytes"], 13041)
        self.assertEqual(by_cell[(0.01, 4)]["l2_bytes"], 208650)
        self.assertEqual(by_cell[(0.02, 4)]["l2_blocks512"], 417300 / (512 * 2048))
        shared = decision_population._SHARED
        saved = dict(shared)
        try:
            for working_set in (2547 * 2048, 185744203776, 7):
                shared["working_set"] = {"t": working_set}
                for cell in audit.cell_capacities(working_set):
                    fraction, multiplier = cell["l1_fraction"], cell["l2_multiplier"]
                    self.assertEqual(cell["l1_bytes"], decision_population._capacity("t", fraction))
                    self.assertEqual(cell["l2_bytes"],
                                     decision_population._capacity("t", fraction * multiplier))
                    self.assertEqual(cell["cell"], decision_population.cell_label(fraction, multiplier))
        finally:
            shared.clear()
            shared.update(saved)


class LoaderCrossCheckTest(_Shared):
    def test_direct_512_equals_loader(self):
        trace = load_mooncake_trace(self.inputs.converted, 512)
        direct = self.result["granularity"]["512"]
        window = trace.start_ms + 0.6 * trace.duration_ms
        self.assertEqual(direct["window_start_ms"], window)
        self.assertEqual(direct["unique_states"], len(trace.states))
        self.assertEqual(direct["working_set_bytes"], working_set_bytes(trace))
        shares = characterize.repeat_shares(trace, window)
        for key, value in shares.items():
            self.assertEqual(direct[key], value, key)
        self.assertEqual(self.result["loader_crosscheck"]["unique_states"], 12)
        self.assertFalse({name for name in _failed(self.result) if name.startswith("loader")})

    def test_direct_16_equals_loader_on_a_16_token_conversion(self):
        path = self.inputs.root / "at16" / "bailian_toc_trace.jsonl"
        convert.convert_file(self.inputs.source, path, block_tokens=16)
        trace = load_mooncake_trace(path, 16)
        direct = self.result["granularity"]["16"]
        window = trace.start_ms + 0.6 * trace.duration_ms
        self.assertEqual(direct["unique_states"], len(trace.states))
        self.assertEqual(direct["working_set_bytes"], working_set_bytes(trace))
        for key, value in characterize.repeat_shares(trace, window).items():
            self.assertEqual(direct[key], value, key)

    def test_other_granularity_matches_the_loader(self):
        records = self.inputs.records()
        direct = audit.granularity_summary(records, 64, audit.window_start_ms(records))
        path = self.inputs.root / "at64" / "bailian_toc_trace.jsonl"
        convert.convert_file(self.inputs.source, path, block_tokens=64)
        trace = load_mooncake_trace(path, 64)
        loader = audit.loader_summary(trace)
        for key in audit.LOADER_KEYS:
            self.assertEqual(direct[key], loader[key], key)

    def test_mooncake_section(self):
        path = self.inputs.root / "mooncake_test_trace.jsonl"
        requests = [(0, 600, [1, 2]), (0, 1024, [1, 3]), (1000, 512, [1]), (5000, 1100, [1, 3, 4])]
        path.write_text("".join(json.dumps({"timestamp": t, "input_length": n, "output_length": 1,
                                            "hash_ids": ids}) + "\n" for t, n, ids in requests))
        result = audit.audit_mooncake(path)
        schema, d512 = result["schema"], result["granularity"]["512"]
        self.assertEqual(result["trace"], "mooncake_test_trace")
        self.assertEqual((schema["records"], schema["span_s"], schema["distinct_timestamps"]), (4, 5.0, 3))
        self.assertEqual((schema["multi_record_timestamp_groups"], schema["records_in_multi_record_groups"],
                          schema["largest_timestamp_group"]), (1, 2, 2))
        self.assertEqual(schema["total_input_tokens"], 3236)
        self.assertEqual(schema["input_median"], 812.0)
        self.assertAlmostEqual(schema["input_p95"], 1024 + 0.85 * 76)
        self.assertEqual((schema["partial512_records"], schema["partial512_tokens"]), (2, 88 + 76))
        self.assertEqual(d512["unique_states"], 4)
        self.assertEqual(d512["working_set_bytes"], (512 + 88 + 512 + 76) * 2048)
        self.assertEqual(d512["repeated_tokens"], 512 + 1024)               # strict
        self.assertEqual(d512["repeated_tokens_file_order"], 512 + 512 + 1024)
        self.assertEqual(d512["last40_input_tokens"], 1100)                 # window 3000 ms
        self.assertEqual(d512["repeat_share_last40"], 1024 / 1100)


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        self.inputs = Inputs()

    def tearDown(self):
        self.inputs.close()

    def test_untouched_inputs_pass_every_check(self):
        result = self.inputs.audit()
        self.assertEqual(_failed(result), set())
        identity = result["identity"]
        self.assertEqual(identity["source_lines"], 9)
        self.assertEqual(identity["converted_lines"], 9)
        self.assertIsNone(identity["first_differing_line"])
        self.assertEqual(identity["fresh"], {"records": 9, "output_blocks": 12, "records_moved_by_sort": 0,
                                             "block_keys_differing_only_in_tokens": 0})
        self.assertEqual(identity["upstream_commit"], audit.UNAVAILABLE)  # not a git checkout
        self.assertEqual(identity["lfs_oid"], audit.UNAVAILABLE)
        names = {check["check"] for check in result["checks"]}
        self.assertNotIn("lfs_oid_matches_source_sha256", names)
        self.assertIn("loader512_repeat_share_last40_file_order", names)

    def _tamper_manifest(self, **changes):
        manifest = json.loads(self.inputs.manifest_path.read_text())
        manifest.update(changes)
        self.inputs.manifest_path.write_text(json.dumps(manifest))

    def test_tampered_manifest(self):
        self._tamper_manifest(output_blocks=13, source_sha256="0" * 64, records=10)
        self.assertEqual(_failed(self.inputs.audit()), {
            "fresh_output_blocks_equal_manifest", "source_sha256_matches_manifest",
            "fresh_records_equal_manifest", "source_lines_equal_manifest_records",
            "converted_lines_equal_manifest_records"})

    def test_manifest_of_another_conversion(self):
        self._tamper_manifest(block_tokens=256, source="qwen_traceB_blksz_16.jsonl")
        self.assertEqual(_failed(self.inputs.audit()),
                         {"manifest_block_tokens", "manifest_source_is_this_file"})

    def test_tampered_converted_line(self):
        lines = self.inputs.converted.read_text().splitlines(keepends=True)
        record = json.loads(lines[2])
        record["output_length"] += 1  # the loader ignores it, so only A fails
        lines[2] = json.dumps(record) + "\n"
        self.inputs.converted.write_text("".join(lines))
        result = self.inputs.audit()
        self.assertEqual(_failed(result), {"converted_sha256_matches_manifest",
                                           "fresh_conversion_byte_identical"})
        self.assertEqual(result["identity"]["first_differing_line"], 3)
        check = next(c for c in result["checks"] if c["check"] == "fresh_conversion_byte_identical")
        self.assertEqual(check["observed"], "first difference at line 3")

    def test_truncated_converted_file(self):
        lines = self.inputs.converted.read_text().splitlines(keepends=True)
        self.inputs.converted.write_text("".join(lines[:-1]))
        result = self.inputs.audit()
        self.assertEqual(result["identity"]["first_differing_line"], 9)
        self.assertTrue({"converted_sha256_matches_manifest", "converted_lines_equal_manifest_records",
                         "fresh_conversion_byte_identical", "loader512_unique_states"} <= _failed(result))

    def test_extra_converted_line(self):
        with self.inputs.converted.open("a") as handle:
            handle.write(self.inputs.converted.read_text().splitlines(keepends=True)[-1])
        result = self.inputs.audit([512])
        self.assertEqual(result["identity"]["first_differing_line"], 10)

    def test_unloadable_converted_file_is_a_failed_check(self):
        self.inputs.converted.write_text("{not json\n")
        result = self.inputs.audit([512])
        self.assertIn("loader512_loads", _failed(result))


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class UpstreamIdentityTest(unittest.TestCase):
    def _git(self, directory, *arguments):
        subprocess.run(["git", "-C", str(directory), *arguments], check=True, capture_output=True)

    def test_commit_of_a_checkout_root_only(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "upstream"
            (root / "inner").mkdir(parents=True)
            (root / UPSTREAM).write_text("x\n")
            self._git(root, "init", "-q")
            self._git(root, "add", UPSTREAM)
            self._git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "t")
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
            identity = audit.upstream_identity(root, UPSTREAM)
            self.assertEqual(identity["upstream_commit"], head)
            self.assertEqual(identity["lfs_oid"], audit.UNAVAILABLE)  # not an LFS file
            inner = audit.upstream_identity(root / "inner", UPSTREAM)
            self.assertEqual(inner["upstream_commit"], audit.UNAVAILABLE)
            outside = audit.upstream_identity(Path(name), UPSTREAM)
            self.assertEqual(outside["upstream_commit"], audit.UNAVAILABLE)


class BlockTokensTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(audit.parse_block_tokens("16,512"), [16, 512])
        self.assertEqual(audit.parse_block_tokens(" 512 , 16,64"), [16, 64, 512])
        for bad in ("", "16,", "0", "-16", "500", "16.0", "abc", "16,16"):
            with self.assertRaises(ValueError, msg=bad):
                audit.parse_block_tokens(bad)

    def test_cli_rejects_bad_values(self):
        for bad in ("500", "16,16", "0"):
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                audit.parse_args(["--block-tokens", bad])
            self.assertEqual(raised.exception.code, 2)
        self.assertEqual(audit.parse_args(["--block-tokens", "512"]).block_tokens, [512])


class CliTest(unittest.TestCase):
    def test_end_to_end(self):
        inputs = Inputs()
        try:
            mooncake = inputs.root / "mooncake_test_trace.jsonl"
            mooncake.write_text("".join(json.dumps({"timestamp": t, "input_length": n, "output_length": 1,
                                                    "hash_ids": ids}) + "\n"
                                        for t, n, ids in ((0, 600, [1, 2]), (2000, 1100, [1, 3, 4]))))
            output, paper, work = inputs.root / "out", inputs.root / "paper", inputs.root / "work"
            command = [sys.executable, str(SCRIPT), "--source-dir", str(inputs.source_dir),
                       "--converted-dir", str(inputs.converted_dir), "--only", UPSTREAM,
                       "--mooncake", str(mooncake), "--output-dir", str(output),
                       "--paper-dir", str(paper), "--work-dir", str(work)]
            finished = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(finished.returncode, 0, finished.stderr + finished.stdout)
            for name in audit.OUTPUTS:
                self.assertTrue((output / name).is_file(), name)
                self.assertEqual((output / name).read_bytes(), (paper / name).read_bytes(), name)
            self.assertTrue((work / f"{STEM}.json").is_file())
            self.assertIn("| unique states @16 |", finished.stdout)

            with (output / "checks.csv").open() as handle:
                checks = list(csv.DictReader(handle))
            self.assertTrue(checks)
            self.assertEqual({row["pass"] for row in checks}, {"True"})
            self.assertEqual({row["trace"] for row in checks}, {STEM, "mooncake_test_trace"})

            with (output / "audit.csv").open() as handle:
                reader = csv.DictReader(handle)
                header, rows = reader.fieldnames, list(reader)
            self.assertEqual([(row["kind"], row["trace"], row["granularity"]) for row in rows],
                             [("bailian", STEM, "16"), ("bailian", STEM, "512"),
                              ("mooncake", "mooncake_test_trace", "512")])
            self.assertEqual(rows[0]["unique_states"], "143")
            self.assertEqual(rows[1]["unique_states"], "12")
            self.assertEqual(rows[0]["records"], rows[1]["records"])
            self.assertEqual(rows[1]["type_text"], "6")
            self.assertEqual(rows[2]["unique_states"], "4")
            self.assertEqual(rows[2]["lcp_full"], "")
            readme = (output / "README.md").read_text()
            for column in header:
                if not column.startswith(("type_", "turn_", "l1_", "l2_")):
                    self.assertIn(f"`{column}`", readme)
            document = json.loads((output / "audit.json").read_text())
            self.assertEqual(sorted(document["traces"]), [STEM])
            self.assertEqual(sorted(document["mooncake"]), ["mooncake_test_trace"])
            self.assertEqual(document["checks_failed"], 0)
            self.assertEqual(document["traces"][STEM]["granularity"]["512"]["unique_states"], 12)

            manifest = json.loads(inputs.manifest_path.read_text())
            manifest["output_blocks"] += 1
            inputs.manifest_path.write_text(json.dumps(manifest))
            finished = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(finished.returncode, 1, finished.stderr + finished.stdout)
            with (output / "checks.csv").open() as handle:
                failed = [row["check"] for row in csv.DictReader(handle) if row["pass"] != "True"]
            self.assertEqual(failed, ["fresh_output_blocks_equal_manifest"])
            self.assertIn("FAIL", finished.stdout)
        finally:
            inputs.close()

    def test_missing_source_is_a_failed_check(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            finished = subprocess.run(
                [sys.executable, str(SCRIPT), "--source-dir", str(root), "--converted-dir", str(root),
                 "--only", UPSTREAM, "--mooncake", "--output-dir", str(root / "out")],
                capture_output=True, text=True)
            self.assertEqual(finished.returncode, 1, finished.stderr)
            with (root / "out" / "checks.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row["check"], row["pass"]) for row in rows], [("audit_completed", "False")])


if __name__ == "__main__":
    unittest.main()
