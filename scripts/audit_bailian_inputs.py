#!/usr/bin/env python3
"""Read-only audit of the Qwen-Bailian inputs and of their 512-token conversion.

Before an external-workload pre-registration this records what the four
upstream files of `scripts/convert_bailian_trace.py` (`NAMES`) are, checks that
the Mooncake-schema files in --converted-dir are exactly what the converter
makes of them, measures what the 16 -> 512-token coarsening changes, and
reports trace-only characteristics. Each upstream file is audited in its own
subprocess, so memory is freed between files and wall time and peak RSS are
per file. Sections:

A. Identity and conversion integrity: sha256 and line counts of the raw and
   converted files against the converter manifest; a fresh in-memory
   conversion at 512 tokens (`parse_records` + `convert_records`) whose counts
   must equal the manifest's and whose JSON lines must be byte-identical to the
   converted file; the upstream git commit and LFS object id when the source
   dir is the root of a git checkout ("unavailable" otherwise).
B. Raw schema and time (`parse_records` validates every record): timestamps
   (decimal places in the raw text, order, ties), input / output lengths,
   chat_id uniqueness and parent_chat_id resolution, turn and type
   distributions, partial 16- and 512-token blocks.
C. Meaning of the 16-token ids: how often an id recurs at another position or
   after another predecessor, and the longest common prefix, in 16-token
   blocks, of every child request with its parent request.
D. Granularity comparison at every --block-tokens value, computed directly from
   the in-memory conversion (`convert_records`, no loader): unique states,
   packed working-set bytes, block occurrences, the infinite-capacity repeat
   share with the definitions of `characterize_external_traces.repeat_shares`,
   and the byte capacities of the six (L1 fraction, L2 multiplier) cells of
   `run_decision_population`. At 512 the direct numbers are checked against
   the unmodified `load_mooncake_trace`, `gap.working_set_bytes` and
   `repeat_shares` on the converted file, which validates the same code at 16.
E. Mooncake comparison: the applicable B fields and D at 512 for the
   --mooncake traces, through the loader.

It runs no cache policy, reads no horizon-dependent reuse statistic and fits
nothing. It only reads its inputs, writes audit.json, audit.csv, checks.csv and
README.md to --output-dir (copied to --paper-dir if given) and prints a
Markdown summary. The exit status is 1 if any check fails; the outputs are
written either way.

  .venv/bin/python scripts/audit_bailian_inputs.py
  .venv/bin/python scripts/audit_bailian_inputs.py --only qwen_traceA_blksz_16.jsonl --mooncake
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np  # noqa: E402

from persistent_kv_admission.crossworkload import HYPERPARAMETERS  # noqa: E402


def _sibling(stem: str):
    """Import scripts/<stem>.py by path, as run_class_order_mix imports its siblings."""
    module = sys.modules.get(stem)
    if module is None:
        spec = importlib.util.spec_from_file_location(stem, REPOSITORY / "scripts" / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


convert = _sibling("convert_bailian_trace")
characterize = _sibling("characterize_external_traces")
decision_population = _sibling("run_decision_population")

SOURCE_BLOCK_TOKENS = convert.SOURCE_BLOCK_TOKENS        # 16: one upstream hash id per 16 tokens
BLOCK_TOKENS = characterize.BLOCK_TOKENS                 # 512: the converted (Mooncake) granularity
TEST_FRACTION_START = characterize.TEST_FRACTION_START   # 0.6: the runners' split, last 40 %
BYTES_PER_TOKEN = HYPERPARAMETERS["bytes_per_token"]     # 2048, packed size model
BLOCK_BYTES = BLOCK_TOKENS * BYTES_PER_TOKEN
L1_FRACTIONS = tuple(decision_population.DEFAULT_L1_BUDGETS)
L2_MULTIPLIERS = tuple(decision_population.DEFAULT_MULTIPLIERS)
DEFAULT_GRANULARITIES = (SOURCE_BLOCK_TOKENS, BLOCK_TOKENS)
DEFAULT_SOURCE_DIR = convert.DEFAULT_SOURCE_DIR
DEFAULT_CONVERTED_DIR = convert.default_output_dir(BLOCK_TOKENS)
DEFAULT_MOONCAKE = (REPOSITORY / "data/raw/conversation_trace.jsonl",
                    REPOSITORY / "data/raw/toolagent_trace.jsonl")
DEFAULT_OUTPUT_DIR = REPOSITORY / "results/bailian_input_audit"
OUTPUTS = ("audit.json", "audit.csv", "checks.csv", "README.md")
TURN_BUCKETS = ("1", "2", "3-5", "6-10", ">10", "other")
SHARE_KEYS = ("repeat_share", "repeat_share_file_order",
              "repeat_share_last40", "repeat_share_last40_file_order")
MANIFEST_STATS = ("records", "output_blocks", "records_moved_by_sort",
                  "block_keys_differing_only_in_tokens")
# Quantities the direct section-D computation at 512 must reproduce exactly from
# the loader, gap.working_set_bytes and characterize_external_traces.repeat_shares.
LOADER_KEYS = ("unique_states", "working_set_bytes", "block_occurrences", "total_input_tokens",
               "window_start_ms", *SHARE_KEYS, "repeated_tokens", "repeated_tokens_file_order",
               "last40_requests", "last40_input_tokens")
LCP_CATEGORIES = ("full", "all_but_last", "partial", "none")
UNAVAILABLE = "unavailable"
COMPLETED = "completed"

if convert.DEFAULT_BLOCK_TOKENS != BLOCK_TOKENS or HYPERPARAMETERS["size_model"] != "packed":
    raise RuntimeError("converter, characterization and size model disagree on the 512-token setup")


def _share(part, whole):
    return part / whole if whole else None


def _check(name: str, expected, observed) -> dict:
    return {"check": name, "expected": expected, "observed": observed,
            "pass": expected == observed}


def _display(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY))
    except ValueError:
        return str(path)


def parse_block_tokens(text: str) -> list[int]:
    """Comma-separated section-D granularities: distinct positive multiples of 16, sorted."""
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"--block-tokens: {part!r} is not an integer") from None
        convert.group_size(value)  # a positive multiple of 16, else ValueError
        if value in values:
            raise ValueError(f"--block-tokens: {value} is given twice")
        values.append(value)
    return sorted(values)


def _block_tokens_arg(text: str) -> list[int]:
    try:
        return parse_block_tokens(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def read_source(path: Path) -> list[dict]:
    """Upstream records via `parse_records`, which validates the schema; the
    audit also needs the Bailian metadata (chat_id, parent_chat_id, type, turn)."""
    with Path(path).open(encoding="utf-8") as handle:
        records = convert.parse_records(handle, str(path))
    for record in records:
        missing = [field for field in convert.METADATA if field not in record]
        if missing:
            raise ValueError(f"{path}:{record['_line']}: missing metadata fields {missing}")
    return records


def parent_indices(records: list[dict]) -> list[int | None]:
    """File index of each record's parent: the first record whose chat_id equals
    its parent_chat_id; None for parent_chat_id == -1 or an id not in the file."""
    first: dict = {}
    for index, record in enumerate(records):
        first.setdefault(record["chat_id"], index)
    return [None if record["parent_chat_id"] == -1 else first.get(record["parent_chat_id"])
            for record in records]


# ---------------------------------------------------------------- A. identity and integrity

def line_count(path: Path) -> int:
    """Physical lines, counting a final line without a newline."""
    lines, last = 0, b"\n"
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    return lines + (last != b"\n")


def _git(directory: Path, *arguments: str) -> str | None:
    completed = subprocess.run(["git", "-C", str(directory), *arguments],
                               capture_output=True, text=True, timeout=300)
    return completed.stdout if completed.returncode == 0 else None


def upstream_identity(source_dir: Path, name: str) -> dict:
    """HEAD of the source dir and the LFS oid of `name`, only when the source dir
    is itself the root of a git checkout (not a directory inside another one)."""
    identity = {"upstream_commit": UNAVAILABLE, "lfs_oid": UNAVAILABLE}
    try:
        top = _git(source_dir, "rev-parse", "--show-toplevel")
        if top is None or Path(top.strip()).resolve() != Path(source_dir).resolve():
            return identity
        head = _git(source_dir, "rev-parse", "HEAD")
        if head and head.strip():
            identity["upstream_commit"] = head.strip()
        for line in (_git(source_dir, "lfs", "ls-files", "--long") or "").splitlines():
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and parts[2] == name:
                identity["lfs_oid"] = parts[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def integrity(name: str, source_path: Path, converted_path: Path, manifest_path: Path,
              records: list[dict]) -> tuple[dict, list[dict]]:
    """Section A without the git lookup: hashes, line counts, fresh conversion."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source_sha = convert._sha256(Path(source_path))
    converted_sha = convert._sha256(Path(converted_path))
    source_lines, converted_lines = line_count(source_path), line_count(converted_path)
    stats: dict = {}
    first_difference, fresh_lines = None, 0
    with Path(converted_path).open("rb") as handle:
        # The generator is always exhausted (its stats are set at the end); the
        # converted file is read only until the first differing line.
        for fresh_lines, record in enumerate(convert.convert_records(records, BLOCK_TOKENS, stats), 1):
            if first_difference is None:
                if handle.readline() != (json.dumps(record) + "\n").encode("utf-8"):
                    first_difference = fresh_lines
        if first_difference is None and handle.readline():
            first_difference = fresh_lines + 1
    identity = {
        "source": name, "source_sha256": source_sha, "source_lines": source_lines,
        "converted": Path(converted_path).name, "converted_sha256": converted_sha,
        "converted_lines": converted_lines, "manifest": manifest,
        "fresh": {key: stats[key] for key in MANIFEST_STATS}, "fresh_lines": fresh_lines,
        "first_differing_line": first_difference,
    }
    checks = [
        _check("source_sha256_matches_manifest", manifest.get("source_sha256"), source_sha),
        _check("converted_sha256_matches_manifest", manifest.get("output_sha256"), converted_sha),
        _check("manifest_source_is_this_file", name, manifest.get("source")),
        _check("manifest_output_is_this_file", Path(converted_path).name, manifest.get("output")),
        _check("manifest_block_tokens", BLOCK_TOKENS, manifest.get("block_tokens")),
        _check("source_lines_equal_manifest_records", manifest.get("records"), source_lines),
        _check("converted_lines_equal_manifest_records", manifest.get("records"), converted_lines),
        *[_check(f"fresh_{key}_equal_manifest", manifest.get(key), stats[key]) for key in MANIFEST_STATS],
        _check("fresh_conversion_byte_identical", "identical",
               "identical" if first_difference is None
               else f"first difference at line {first_difference}"),
    ]
    return identity, checks


# ---------------------------------------------------------------- B. schema and time

def length_time_summary(times_ms: list, input_lengths: list[int]) -> dict:
    """The B fields that also apply to a Mooncake trace (timestamps in ms)."""
    counts = Counter(times_ms)
    multi = [count for count in counts.values() if count > 1]
    lengths = np.asarray(input_lengths, dtype=np.int64)
    total = int(lengths.sum())
    partial = [length % BLOCK_TOKENS for length in input_lengths if length % BLOCK_TOKENS]
    return {
        "records": len(times_ms),
        "span_s": (max(times_ms) - min(times_ms)) / 1000,
        "distinct_timestamps": len(counts),
        "multi_record_timestamp_groups": len(multi),
        "records_in_multi_record_groups": sum(multi),
        "largest_timestamp_group": max(counts.values()),
        "input_mean": float(lengths.mean()),
        "input_median": float(np.median(lengths)),
        "input_p95": float(np.percentile(lengths, 95)),
        "total_input_tokens": total,
        "partial512_records": len(partial),
        "partial512_tokens": sum(partial),
        "partial512_token_share": _share(sum(partial), total),
    }


def _decimal_places(value) -> int:
    """Decimal places of a timestamp as written in the raw JSON text."""
    return max(0, -value.as_tuple().exponent) if isinstance(value, Decimal) else 0


def _turn_bucket(turn) -> str:
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        return "other"
    if turn <= 2:
        return str(turn)
    return "3-5" if turn <= 5 else ("6-10" if turn <= 10 else ">10")


def schema_summary(records: list[dict], parents: list[int | None]) -> dict:
    """Section B over the parsed upstream records, in file order."""
    times = [record["_ms"] for record in records]
    lengths = [record["input_length"] for record in records]
    outputs = np.asarray([record["output_length"] for record in records], dtype=np.int64)
    summary = length_time_summary(times, lengths)
    chat_ids = Counter(record["chat_id"] for record in records)
    roots = sum(1 for record in records if record["parent_chat_id"] == -1)
    resolvable = sum(1 for parent in parents if parent is not None)
    earlier = sum(1 for index, parent in enumerate(parents) if parent is not None and parent < index)
    turns = Counter(_turn_bucket(record["turn"]) for record in records)
    types = Counter(str(record["type"]) for record in records)
    partial16 = [length % SOURCE_BLOCK_TOKENS for length in lengths if length % SOURCE_BLOCK_TOKENS]
    summary.update(
        timestamp_min_s=float(min(record["timestamp"] for record in records)),
        timestamp_max_s=float(max(record["timestamp"] for record in records)),
        max_timestamp_decimals=max(_decimal_places(record["timestamp"]) for record in records),
        timestamp_decreases=sum(1 for before, after in zip(times, times[1:]) if after < before),
        input_max=max(lengths),
        output_mean=float(outputs.mean()),
        output_median=float(np.median(outputs)),
        distinct_chat_ids=len(chat_ids),
        chat_id_unique=len(chat_ids) == len(records),
        root_records=roots,
        root_share=roots / len(records),
        resolvable_parent_records=resolvable,
        resolvable_parent_earlier_records=earlier,
        unresolvable_parent_records=len(records) - roots - resolvable,
        turn_buckets={bucket: turns[bucket] for bucket in TURN_BUCKETS},
        type_counts=dict(sorted(types.items())),
        partial16_records=len(partial16),
        partial16_tokens=sum(partial16),
    )
    return summary


# ---------------------------------------------------------------- C. 16-token id identity

def _common_prefix(left: list, right: list) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def lcp_category(lcp: int, parent_blocks: int) -> str:
    """Exclusive categories, tested in this order: full (lcp == len(parent)),
    all_but_last (lcp == len(parent) - 1 >= 1), partial (0 < lcp < len(parent) - 1),
    none (lcp == 0). A one-block parent is therefore either full or none."""
    if lcp == parent_blocks:
        return "full"
    if lcp > 0 and lcp == parent_blocks - 1:
        return "all_but_last"
    return "partial" if lcp > 0 else "none"


def prefix_identity(records: list[dict], parents: list[int | None]) -> dict:
    """Section C. Positions and predecessors are taken in file order; the
    predecessor of position 0 is a sentinel shared by every request."""
    first_position: dict = {}
    first_predecessor: dict = {}
    multi_position: set = set()
    multi_predecessor: set = set()
    occurrences = new_position = new_predecessor = 0
    for record in records:
        predecessor = None  # the sentinel; ids are integers
        for position, block in enumerate(record["hash_ids"]):
            seen = first_position.get(block)
            if seen is None:
                first_position[block] = position
                first_predecessor[block] = predecessor
            else:
                if seen != position:
                    new_position += 1
                    multi_position.add(block)
                if first_predecessor[block] != predecessor:
                    new_predecessor += 1
                    multi_predecessor.add(block)
            predecessor = block
        occurrences += len(record["hash_ids"])
    distinct = len(first_position)
    del first_position, first_predecessor
    categories = Counter()
    children = shorter = single = 0
    for record, parent in zip(records, parents):
        if parent is None:
            continue
        child_ids, parent_ids = record["hash_ids"], records[parent]["hash_ids"]
        children += 1
        shorter += len(child_ids) < len(parent_ids)
        single += len(parent_ids) == 1
        categories[lcp_category(_common_prefix(child_ids, parent_ids), len(parent_ids))] += 1
    summary = {
        "ids16_distinct": distinct,
        "ids16_occurrences": occurrences,
        "ids16_at_multiple_positions": len(multi_position),
        "ids16_at_multiple_positions_share": _share(len(multi_position), distinct),
        "ids16_after_multiple_predecessors": len(multi_predecessor),
        "ids16_after_multiple_predecessors_share": _share(len(multi_predecessor), distinct),
        "ids16_occurrences_new_position": new_position,
        "ids16_occurrences_new_position_share": _share(new_position, occurrences),
        "ids16_occurrences_new_predecessor": new_predecessor,
        "ids16_occurrences_new_predecessor_share": _share(new_predecessor, occurrences),
        "children_with_parent": children,
        "children_shorter_than_parent": shorter,
        "parents_single_block": single,
    }
    for category in LCP_CATEGORIES:
        summary[f"lcp_{category}"] = categories[category]
        summary[f"lcp_{category}_share"] = _share(categories[category], children)
    return summary


# ---------------------------------------------------------------- D. granularity comparison

def window_start_ms(records: list[dict]) -> float:
    """start + 0.6 * span, as characterize_one computes it from the loaded trace."""
    times = [record["_ms"] for record in records]
    start, end = float(min(times)), float(max(times))
    return start + TEST_FRACTION_START * (end - start)


def capacity(working_set: int, fraction: float) -> int:
    """run_decision_population._capacity for a given working set."""
    return max(1, round(working_set * fraction))


def cell_capacities(working_set: int) -> list[dict]:
    cells = []
    for fraction in L1_FRACTIONS:
        l1 = capacity(working_set, fraction)
        for multiplier in L2_MULTIPLIERS:
            l2 = capacity(working_set, fraction * multiplier)
            cells.append({
                "cell": decision_population.cell_label(fraction, multiplier),
                "l1_fraction": fraction, "l2_multiplier": multiplier,
                "l1_bytes": l1, "l2_bytes": l2,
                "l1_blocks512": l1 / BLOCK_BYTES, "l2_blocks512": l2 / BLOCK_BYTES,
            })
    return cells


def capacity_columns() -> list[str]:
    columns = []
    for fraction in L1_FRACTIONS:
        columns += [f"l1_{fraction:g}_bytes", f"l1_{fraction:g}_blocks512"]
    for fraction in L1_FRACTIONS:
        for multiplier in L2_MULTIPLIERS:
            columns += [f"l2_{fraction:g}x{multiplier:g}_bytes", f"l2_{fraction:g}x{multiplier:g}_blocks512"]
    return columns


def _repeat_fields(totals: Counter) -> dict:
    return {
        "repeat_share": _share(totals["all_strict"], totals["all_tokens"]),
        "repeat_share_file_order": _share(totals["all_order"], totals["all_tokens"]),
        "repeat_share_last40": _share(totals["last40_strict"], totals["last40_tokens"]),
        "repeat_share_last40_file_order": _share(totals["last40_order"], totals["last40_tokens"]),
        "repeated_tokens": totals["all_strict"],
        "repeated_tokens_file_order": totals["all_order"],
        "last40_repeated_tokens": totals["last40_strict"],
        "last40_repeated_tokens_file_order": totals["last40_order"],
        "last40_requests": totals["last40_requests"],
        "last40_input_tokens": totals["last40_tokens"],
    }


_NO_GROUP = object()


def granularity_summary(records: list[dict], block_tokens: int, window_start: float) -> dict:
    """Section D at one granularity, walking `convert_records` without the loader.

    Output ids are BlockIds' sequential integers, so one byte of flags per id
    replaces the loader's state table: EARLIER = seen in an earlier request
    (file order), STRICT = seen at a strictly earlier timestamp (set when the
    next timestamp group starts, as repeat_shares updates seen_strict)."""
    earlier_flag, strict_flag = 1, 2
    flags = bytearray()
    stats: dict = {}
    totals: Counter = Counter()
    group: list[list[int]] = []
    group_timestamp: object = _NO_GROUP
    states = working_tokens = occurrences = 0
    for record in convert.convert_records(records, block_tokens, stats):
        timestamp = record["timestamp"]
        if timestamp != group_timestamp:
            for ids in group:
                for state in ids:
                    flags[state] |= strict_flag
            group, group_timestamp = [], timestamp
        ids, input_length = record["hash_ids"], record["input_length"]
        top = max(ids)
        if top >= len(flags):
            flags.extend(bytes(top + 1 - len(flags)))
        tokens = [min(input_length, (index + 1) * block_tokens) - index * block_tokens
                  for index in range(len(ids))]
        if sum(tokens) != input_length:
            raise AssertionError(f"block tokens do not add up to input_length {input_length}")
        hits = {}
        for variant, flag in (("strict", strict_flag), ("order", earlier_flag)):
            marks = [bool(flags[state] & flag) for state in ids]
            run = marks.index(False) if False in marks else len(marks)
            if any(marks[run:]):
                raise AssertionError(f"block_tokens={block_tokens}: repeated blocks are not a prefix")
            hits[variant] = sum(tokens[:run])
        in_window = float(timestamp) >= window_start
        for scope, use in (("all", True), ("last40", in_window)):
            if use:
                totals[f"{scope}_tokens"] += input_length
                totals[f"{scope}_strict"] += hits["strict"]
                totals[f"{scope}_order"] += hits["order"]
                totals[f"{scope}_requests"] += 1
        for state, size in zip(ids, tokens):
            if not flags[state] & earlier_flag:
                states += 1
                working_tokens += size  # the first time this output id appears
            flags[state] |= earlier_flag
        group.append(ids)
        occurrences += len(ids)
    working_set = working_tokens * BYTES_PER_TOKEN
    return {
        "block_tokens": block_tokens,
        "records": stats["records"],
        "unique_states": stats["output_blocks"],
        "states_counted": states,
        "token_variant_keys": stats["block_keys_differing_only_in_tokens"],
        "working_set_bytes": working_set,
        "block_occurrences": occurrences,
        "total_input_tokens": totals["all_tokens"],
        "window_start_ms": window_start,
        **_repeat_fields(totals),
        "capacities": cell_capacities(working_set),
    }


def coarsening_effect(fine: dict, coarse: dict) -> dict:
    """512 vs 16; fine is the 16-token summary, coarse the 512-token one."""
    effect = {
        "working_set_ratio_512_over_16": _share(coarse["working_set_bytes"], fine["working_set_bytes"]),
        "unique_states_ratio_16_over_512": _share(fine["unique_states"], coarse["unique_states"]),
    }
    for key in SHARE_KEYS:
        effect[f"{key}_diff_16_minus_512"] = (None if fine[key] is None or coarse[key] is None
                                              else fine[key] - coarse[key])
    return effect


def loader_summary(trace) -> dict:
    """The D quantities through the unmodified loader, gap and repeat_shares."""
    from persistent_kv_admission.gap import working_set_bytes

    window = trace.start_ms + TEST_FRACTION_START * trace.duration_ms
    working_set = int(working_set_bytes(trace))
    return {
        "unique_states": len(trace.states),
        "working_set_bytes": working_set,
        "block_occurrences": sum(len(request.hash_ids) for request in trace.requests),
        "total_input_tokens": sum(request.input_length for request in trace.requests),
        "window_start_ms": window,
        **characterize.repeat_shares(trace, window),
        "capacities": cell_capacities(working_set),
    }


def loader_crosscheck(converted_path: Path, direct: dict) -> tuple[dict, list[dict]]:
    from persistent_kv_admission.trace import load_mooncake_trace

    try:
        trace = load_mooncake_trace(converted_path, BLOCK_TOKENS)
    except (OSError, ValueError) as error:
        return {"error": str(error)}, [_check(f"loader{BLOCK_TOKENS}_loads", "loaded", f"failed: {error}")]
    observed = loader_summary(trace)
    del trace
    checks = [_check(f"loader{BLOCK_TOKENS}_loads", "loaded", "loaded")]
    checks += [_check(f"loader{BLOCK_TOKENS}_{key}", direct[key], observed[key]) for key in LOADER_KEYS]
    return {key: observed[key] for key in LOADER_KEYS}, checks


# ---------------------------------------------------------------- per-file audits

def audit_source(name: str, source_dir: Path, converted_dir: Path, granularities: list[int]) -> dict:
    """Sections A-D for one upstream file (run in a subprocess by `main`)."""
    began = time.perf_counter()
    seconds: dict[str, float] = {}
    stem = convert.NAMES[name]
    source_dir, converted_dir = Path(source_dir), Path(converted_dir)
    source_path = source_dir / name
    converted_path = converted_dir / f"{stem}.jsonl"
    manifest_path = converted_dir / f"{stem}.manifest.json"

    clock = time.perf_counter()
    records = read_source(source_path)
    parents = parent_indices(records)
    seconds["parse"] = time.perf_counter() - clock

    clock = time.perf_counter()
    identity, checks = integrity(name, source_path, converted_path, manifest_path, records)
    identity.update(upstream_identity(source_dir, name))
    if identity["lfs_oid"] != UNAVAILABLE:
        checks.append(_check("lfs_oid_matches_source_sha256", identity["lfs_oid"], identity["source_sha256"]))
    seconds["identity"] = time.perf_counter() - clock

    clock = time.perf_counter()
    schema = schema_summary(records, parents)
    checks.append(_check("timestamp_decreases", 0, schema["timestamp_decreases"]))
    checks.append(_check("chat_id_unique_per_record", True, schema["chat_id_unique"]))
    seconds["schema"] = time.perf_counter() - clock

    clock = time.perf_counter()
    prefix = prefix_identity(records, parents)
    seconds["prefix_identity"] = time.perf_counter() - clock
    del parents

    window = window_start_ms(records)
    granularity: dict[str, dict] = {}
    for block_tokens in granularities:
        clock = time.perf_counter()
        summary = granularity_summary(records, block_tokens, window)
        granularity[str(block_tokens)] = summary
        checks.append(_check(f"d{block_tokens}_states_counted_equal_output_blocks",
                             summary["unique_states"], summary["states_counted"]))
        checks.append(_check(f"d{block_tokens}_input_tokens_equal_schema",
                             schema["total_input_tokens"], summary["total_input_tokens"]))
        seconds[f"granularity_{block_tokens}"] = time.perf_counter() - clock
    del records

    pair = (str(SOURCE_BLOCK_TOKENS), str(BLOCK_TOKENS))
    coarsening = (coarsening_effect(granularity[pair[0]], granularity[pair[1]])
                  if all(key in granularity for key in pair) else None)
    loader = None
    if str(BLOCK_TOKENS) in granularity:
        clock = time.perf_counter()
        loader, loader_checks = loader_crosscheck(converted_path, granularity[str(BLOCK_TOKENS)])
        checks += loader_checks
        seconds["loader_crosscheck"] = time.perf_counter() - clock
    return {
        "kind": "bailian", "trace": stem, "source": name,
        "source_path": _display(source_path), "converted_path": _display(converted_path),
        "identity": identity, "schema": schema, "prefix": prefix, "granularity": granularity,
        "coarsening": coarsening, "loader_crosscheck": loader, "checks": checks,
        "seconds": time.perf_counter() - began, "section_seconds": seconds,
        "peak_rss_mib": characterize._rss_mib(),
    }


def audit_mooncake(path: Path) -> dict:
    """Section E for one Mooncake-schema trace, through the loader at 512."""
    from persistent_kv_admission.trace import load_mooncake_trace

    began = time.perf_counter()
    trace = load_mooncake_trace(path, BLOCK_TOKENS)
    schema = length_time_summary([request.timestamp_ms for request in trace.requests],
                                 [request.input_length for request in trace.requests])
    summary = {"block_tokens": BLOCK_TOKENS, "records": len(trace.requests), **loader_summary(trace)}
    return {
        "kind": "mooncake", "trace": trace.name, "source": _display(path), "schema": schema,
        "granularity": {str(BLOCK_TOKENS): summary}, "checks": [],
        "seconds": time.perf_counter() - began, "peak_rss_mib": characterize._rss_mib(),
    }


def _run_child(arguments: list[str], key: str, work_dir: Path) -> dict:
    """One per-file audit in a fresh interpreter; its JSON result and log go to work_dir."""
    result_path, log_path = work_dir / f"{key}.json", work_dir / f"{key}.log"
    result_path.unlink(missing_ok=True)
    clock = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *arguments, "--result", str(result_path)],
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    wall = time.perf_counter() - clock
    if completed.returncode == 0 and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["wall_seconds"] = wall
        result["checks"] = [_check("audit_completed", COMPLETED, COMPLETED), *result["checks"]]
        return result
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    error = lines[-1] if lines else f"exit status {completed.returncode}"
    return {"error": error, "wall_seconds": wall,
            "checks": [_check("audit_completed", COMPLETED, f"failed: {error}")]}


# ---------------------------------------------------------------- outputs

CSV_COLUMNS = (
    ("kind", "`bailian` (an upstream file, sections A-D) or `mooncake` (section E)."),
    ("trace", "Converted trace name (`convert_bailian_trace.NAMES`, the loader's file stem); "
              "for mooncake rows the file stem."),
    ("source", "Upstream file name; for mooncake rows the trace path."),
    ("granularity", "Block size in tokens of the section-D numbers on this row "
                    "(one row per trace and --block-tokens value; 512 for mooncake rows)."),
    # D, per granularity
    ("records", "Requests (records) in the trace."),
    ("unique_states", "D: distinct block states at this granularity (`stats['output_blocks']` of "
                      "`convert_records`; `len(trace.states)` for mooncake rows)."),
    ("states_counted", "D: distinct output ids met while walking the converted records "
                       "(must equal unique_states)."),
    ("token_variant_keys", "D: BlockIds keys that differ from an earlier key only in the block's "
                           "token count (`block_keys_differing_only_in_tokens`)."),
    ("working_set_bytes", "D: packed working set, sum over unique states of the state's block "
                          "tokens x 2048 bytes (`gap.working_set_bytes` for mooncake rows)."),
    ("block_occurrences", "D: sum over requests of the number of blocks at this granularity."),
    ("total_input_tokens", "Sum of input_length over all requests."),
    ("window_start_ms", "start + 0.6 x span of the converted timestamps (ms); requests at or "
                        "after it form the last 40 %."),
    ("repeat_share", "D: infinite-capacity repeat share, strict-earlier-timestamp variant "
                     "(repeated_tokens / total_input_tokens)."),
    ("repeat_share_file_order", "D: the same, file-order variant."),
    ("repeat_share_last40", "D: strict variant over the requests at or after window_start_ms, "
                            "the whole trace before them as history."),
    ("repeat_share_last40_file_order", "D: file-order variant over the last 40 %."),
    ("repeated_tokens", "D: input tokens in repeated blocks, strict variant, whole trace."),
    ("repeated_tokens_file_order", "D: the same, file-order variant."),
    ("last40_repeated_tokens", "D: repeated tokens, strict variant, last 40 % (bailian rows only)."),
    ("last40_repeated_tokens_file_order", "D: the same, file-order variant (bailian rows only)."),
    ("last40_requests", "D: requests at or after window_start_ms."),
    ("last40_input_tokens", "D: their input tokens."),
    ("<capacity>", "D: `l1_<f>_bytes` = max(1, round(working_set_bytes x f)) for f in "
                   "{0.0025, 0.01, 0.02}; `l2_<f>x<m>_bytes` = max(1, round(working_set_bytes x "
                   "(f x m))) for m in {1, 4} (`run_decision_population._capacity`); each "
                   "`_blocks512` column is the bytes / (512 x 2048)."),
    # derived, per trace
    ("working_set_ratio_512_over_16", "Derived: working_set_bytes at 512 / at 16."),
    ("unique_states_ratio_16_over_512", "Derived: unique_states at 16 / at 512."),
    ("repeat_share_diff_16_minus_512", "Derived: repeat_share at 16 minus at 512."),
    ("repeat_share_file_order_diff_16_minus_512", "Derived: the same for the file-order variant."),
    ("repeat_share_last40_diff_16_minus_512", "Derived: the same for the last-40 % strict variant."),
    ("repeat_share_last40_file_order_diff_16_minus_512",
     "Derived: the same for the last-40 % file-order variant."),
    # B, per trace
    ("timestamp_min_s", "B: smallest raw timestamp (s)."),
    ("timestamp_max_s", "B: largest raw timestamp (s)."),
    ("span_s", "B/E: (largest - smallest timestamp) in seconds."),
    ("max_timestamp_decimals", "B: most decimal places of any timestamp in the raw JSON text "
                               "(Decimal exponent; trailing zeros count)."),
    ("timestamp_decreases", "B: records whose timestamp is below the previous record's, file order."),
    ("distinct_timestamps", "B/E: distinct timestamp values."),
    ("multi_record_timestamp_groups", "B/E: timestamp values carried by more than one record."),
    ("records_in_multi_record_groups", "B/E: records in such groups."),
    ("largest_timestamp_group", "B/E: most records sharing one timestamp."),
    ("input_mean", "B/E: mean input_length."),
    ("input_median", "B/E: median input_length."),
    ("input_p95", "B/E: 95th percentile of input_length (numpy linear interpolation, as "
                  "characterize_external_traces)."),
    ("input_max", "B: largest input_length."),
    ("output_mean", "B: mean output_length."),
    ("output_median", "B: median output_length."),
    ("distinct_chat_ids", "B: distinct chat_id values."),
    ("chat_id_unique", "B: every record has its own chat_id."),
    ("root_records", "B: records with parent_chat_id == -1."),
    ("root_share", "B: root_records / records."),
    ("resolvable_parent_records", "B: records whose parent_chat_id (!= -1) is the chat_id of a record "
                                  "in the file."),
    ("resolvable_parent_earlier_records", "B: of those, records whose parent record (the first record "
                                          "with that chat_id) comes earlier in file order."),
    ("unresolvable_parent_records", "B: records with parent_chat_id != -1 that name no chat_id in the file."),
    ("turn_<bucket>", "B: records per turn bucket 1, 2, 3-5, 6-10, >10; `turn_other` counts turn < 1 "
                      "or a non-integer turn."),
    ("type_<name>", "B: records per `type` value."),
    ("partial16_records", "B: records with input_length % 16 != 0."),
    ("partial16_tokens", "B: tokens in their partial final 16-token blocks (sum of input_length % 16)."),
    ("partial512_records", "B/E: records with input_length % 512 != 0."),
    ("partial512_tokens", "B/E: tokens in their partial final 512-token blocks (sum of the nonzero "
                          "input_length % 512)."),
    ("partial512_token_share", "B/E: partial512_tokens / total_input_tokens."),
    # C, per trace
    ("ids16_distinct", "C: distinct upstream 16-token hash ids."),
    ("ids16_occurrences", "C: hash id occurrences (sum of len(hash_ids))."),
    ("ids16_at_multiple_positions", "C: ids seen at more than one index within hash_ids."),
    ("ids16_at_multiple_positions_share", "C: / ids16_distinct."),
    ("ids16_after_multiple_predecessors", "C: ids seen after more than one distinct predecessor (the "
                                          "previous id of the same request; a shared sentinel at index 0)."),
    ("ids16_after_multiple_predecessors_share", "C: / ids16_distinct."),
    ("ids16_occurrences_new_position", "C: occurrences at an index other than the id's first-seen index "
                                       "(file order)."),
    ("ids16_occurrences_new_position_share", "C: / ids16_occurrences."),
    ("ids16_occurrences_new_predecessor", "C: occurrences whose predecessor differs from the id's "
                                          "first-seen predecessor (file order)."),
    ("ids16_occurrences_new_predecessor_share", "C: / ids16_occurrences."),
    ("children_with_parent", "C: records with a resolvable parent (= resolvable_parent_records)."),
    ("lcp_full", "C: children whose longest common hash_ids prefix with the parent equals "
                 "len(parent hash_ids) (the child extends the parent exactly)."),
    ("lcp_full_share", "C: / children_with_parent."),
    ("lcp_all_but_last", "C: children with LCP == len(parent) - 1 >= 1 (only the parent's last block "
                         "differs, the upstream FAQ Q2 case)."),
    ("lcp_all_but_last_share", "C: / children_with_parent."),
    ("lcp_partial", "C: children with 0 < LCP < len(parent) - 1."),
    ("lcp_partial_share", "C: / children_with_parent."),
    ("lcp_none", "C: children with LCP == 0."),
    ("lcp_none_share", "C: / children_with_parent."),
    ("children_shorter_than_parent", "C: children with fewer hash ids than their parent (cannot be full)."),
    ("parents_single_block", "C: children whose parent has one hash id (full or none only)."),
    # A, per trace
    ("source_sha256", "A: sha256 of the raw upstream file."),
    ("converted_sha256", "A: sha256 of the converted file."),
    ("lfs_oid", "A: git-lfs oid of the upstream file, or `unavailable`."),
    ("upstream_commit", "A: HEAD of --source-dir when it is a git checkout root, or `unavailable`."),
    ("source_lines", "A: physical lines of the raw file."),
    ("converted_lines", "A: physical lines of the converted file."),
    ("manifest_records", "A: `records` of the converter manifest."),
    ("manifest_output_blocks", "A: `output_blocks` of the manifest."),
    ("manifest_records_moved_by_sort", "A: `records_moved_by_sort` of the manifest."),
    ("manifest_block_keys_differing_only_in_tokens", "A: `block_keys_differing_only_in_tokens` of the manifest."),
    ("fresh_records", "A: `records` of the fresh in-memory conversion at 512."),
    ("fresh_output_blocks", "A: its `output_blocks`."),
    ("fresh_records_moved_by_sort", "A: its `records_moved_by_sort`."),
    ("fresh_block_keys_differing_only_in_tokens", "A: its `block_keys_differing_only_in_tokens`."),
    ("first_differing_line", "A: first line where the fresh conversion and the converted file differ "
                             "(blank when byte-identical)."),
    # run
    ("checks_failed", "Failed checks of this trace in checks.csv."),
    ("seconds", "Wall time of the per-file audit inside its subprocess (s)."),
    ("wall_seconds", "Wall time of the subprocess including interpreter start-up (s)."),
    ("peak_rss_mib", "Peak resident set size of the per-file subprocess (MiB, ru_maxrss)."),
    ("error", "Last log line of a per-file subprocess that did not complete (other columns blank)."),
)

CHECK_DOCS = (
    ("audit_completed", "the per-file subprocess finished and returned its result."),
    ("source_sha256_matches_manifest", "sha256 of the raw file == manifest `source_sha256`."),
    ("converted_sha256_matches_manifest", "sha256 of the converted file == manifest `output_sha256`."),
    ("manifest_source_is_this_file", "manifest `source` == the upstream file name."),
    ("manifest_output_is_this_file", "manifest `output` == the converted file name."),
    ("manifest_block_tokens", "manifest `block_tokens` == 512."),
    ("source_lines_equal_manifest_records", "physical lines of the raw file == manifest `records`."),
    ("converted_lines_equal_manifest_records", "physical lines of the converted file == manifest `records`."),
    ("fresh_<stat>_equal_manifest", "each of records, output_blocks, records_moved_by_sort, "
                                    "block_keys_differing_only_in_tokens of a fresh in-memory conversion at "
                                    "512 == the manifest's."),
    ("fresh_conversion_byte_identical", "every JSON line of the fresh conversion equals the converted "
                                        "file's line byte for byte, and neither has extra lines "
                                        "(observed names the first differing line)."),
    ("lfs_oid_matches_source_sha256", "the git-lfs oid of the upstream file == its sha256 "
                                      "(only when the oid is available)."),
    ("timestamp_decreases", "no raw record has a timestamp below the previous record's."),
    ("chat_id_unique_per_record", "every record has its own chat_id (parent resolution is unambiguous)."),
    ("d<B>_states_counted_equal_output_blocks", "the direct walk at granularity B meets exactly "
                                                "`output_blocks` distinct ids."),
    ("d<B>_input_tokens_equal_schema", "the direct walk at B sums the same input tokens as B."),
    ("loader512_loads", "the unmodified loader reads the converted file at 512."),
    ("loader512_<quantity>", "the direct computation at 512 == the loader + gap.working_set_bytes + "
                             "repeat_shares on the converted file, exactly, for unique_states, "
                             "working_set_bytes, block_occurrences, total_input_tokens, window_start_ms, "
                             "the four repeat shares, repeated_tokens, repeated_tokens_file_order, "
                             "last40_requests and last40_input_tokens."),
)


def _flat_capacities(summary: dict) -> dict:
    row = {}
    for cell in summary["capacities"]:
        fraction, multiplier = cell["l1_fraction"], cell["l2_multiplier"]
        row[f"l1_{fraction:g}_bytes"] = cell["l1_bytes"]
        row[f"l1_{fraction:g}_blocks512"] = cell["l1_blocks512"]
        row[f"l2_{fraction:g}x{multiplier:g}_bytes"] = cell["l2_bytes"]
        row[f"l2_{fraction:g}x{multiplier:g}_blocks512"] = cell["l2_blocks512"]
    return row


def _flat_schema(schema: dict) -> dict:
    row = {key: value for key, value in schema.items() if key not in ("turn_buckets", "type_counts")}
    for bucket, count in schema.get("turn_buckets", {}).items():
        row[f"turn_{bucket}"] = count
    for kind, count in schema.get("type_counts", {}).items():
        row[f"type_{kind}"] = count
    return row


def _flat_identity(identity: dict) -> dict:
    row = {key: identity[key] for key in ("source_sha256", "converted_sha256", "lfs_oid",
                                          "upstream_commit", "source_lines", "converted_lines",
                                          "first_differing_line")}
    for key in MANIFEST_STATS:
        row[f"manifest_{key}"] = identity["manifest"].get(key)
        row[f"fresh_{key}"] = identity["fresh"][key]
    return row


def _merge(row: dict, extra: dict) -> None:
    for key, value in extra.items():
        if key in row and row[key] != value:
            raise RuntimeError(f"audit.csv column {key}: {row[key]!r} != {value!r}")
        row[key] = value


def flat_rows(results: list[dict]) -> list[dict]:
    """One audit.csv row per trace x granularity, with the per-trace fields repeated."""
    rows = []
    for result in results:
        base = {"kind": result["kind"], "trace": result["trace"], "source": result["source"]}
        if "error" in result:
            rows.append({**base, "error": result["error"], "wall_seconds": result["wall_seconds"],
                         "checks_failed": sum(not check["pass"] for check in result["checks"])})
            continue
        per_trace = _flat_schema(result["schema"])
        if result["kind"] == "bailian":
            _merge(per_trace, _flat_identity(result["identity"]))
            _merge(per_trace, result["prefix"])
            _merge(per_trace, result["coarsening"] or {})
        _merge(per_trace, {"checks_failed": sum(not check["pass"] for check in result["checks"]),
                           "seconds": result["seconds"], "wall_seconds": result.get("wall_seconds"),
                           "peak_rss_mib": result["peak_rss_mib"]})
        for key in sorted(result["granularity"], key=int):
            summary = result["granularity"][key]
            row = {**base, "granularity": int(key)}
            _merge(row, {name: value for name, value in summary.items()
                         if name not in ("capacities", "block_tokens")})
            _merge(row, _flat_capacities(summary))
            _merge(row, per_trace)
            rows.append(row)
    return rows


def csv_columns(rows: list[dict]) -> list[str]:
    """Documented column order; raises if a row carries an undocumented column."""
    present = set().union(*rows) if rows else set()
    columns: list[str] = []
    for column, _ in CSV_COLUMNS:
        if column == "<capacity>":
            columns += capacity_columns()
        elif column == "turn_<bucket>":
            columns += [f"turn_{bucket}" for bucket in TURN_BUCKETS]
        elif column == "type_<name>":
            columns += sorted(key for key in present if key.startswith("type_"))
        else:
            columns.append(column)
    undocumented = present - set(columns)
    if undocumented:
        raise RuntimeError(f"undocumented audit.csv columns: {sorted(undocumented)}")
    return columns


def _cell(value):
    return "" if value is None else value


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})


def readme_text(config: dict, columns: list[str]) -> str:
    column_lines = "\n".join(f"| `{name}` | {text} |" for name, text in CSV_COLUMNS)
    check_lines = "\n".join(f"| `{name}` | {text} |" for name, text in CHECK_DOCS)
    granularities = ",".join(str(value) for value in config["block_tokens"])
    return f"""# Bailian input audit

Written by `scripts/audit_bailian_inputs.py`: a read-only audit of the
Qwen-Bailian upstream traces (`{config['source_dir']}`) and of their conversion
to the Mooncake 512-token schema (`{config['converted_dir']}`, made by
`scripts/convert_bailian_trace.py`). It records identities, checks schema and
conversion integrity, measures what the 16 -> 512-token coarsening changes and
reports trace-only characteristics. It runs no cache policy, reads no
horizon-dependent reuse statistic and fits nothing.

## Commands

This run:

```
{config['command']}
```

Produce the converted inputs, then rerun the audit and its tests:

```
.venv/bin/python scripts/convert_bailian_trace.py --all
.venv/bin/python scripts/audit_bailian_inputs.py --block-tokens {granularities}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m unittest tests.test_audit_bailian_inputs
```

Each upstream file (and each --mooncake trace) is audited in its own
subprocess; `seconds` and `peak_rss_mib` are per file. The exit status is 1 if
any check fails; all four outputs are written either way.

## Files

* `audit.json`: everything, nested per trace (`traces` for the upstream files,
  `mooncake` for section E), with the run configuration and every check.
* `audit.csv`: one row per trace x section-D granularity; per-trace fields are
  repeated on each of its rows. Mooncake rows have the B/E and D-at-512 columns
  only.
* `checks.csv`: one row per check; `pass` is `True` when `observed` equals
  `expected` exactly.
* `README.md`: this file.

## Definitions

* Records are the upstream JSONL lines parsed by
  `convert_bailian_trace.parse_records` (fields validated, integer types,
  `len(hash_ids) == ceil(input_length / 16)`, timestamps parsed as exact
  decimals). "File order" is the order of the raw file; the converter stably
  sorts by timestamp, so file order and converted order coincide when
  `records_moved_by_sort` is 0.
* Parent resolution: the parent record of a record is the first record in file
  order whose `chat_id` equals its `parent_chat_id`; `parent_chat_id == -1`
  is a root.
* C, 16-token ids: an occurrence is one entry of a record's `hash_ids`; its
  position is its index there; its predecessor is the previous entry, or a
  sentinel shared by every request at index 0. "First-seen" is the first
  occurrence in file order. LCP is the length of the longest common prefix of
  the child's and the parent's `hash_ids`, in 16-token blocks; the categories
  are tested in the order full, all_but_last (requires LCP >= 1), partial,
  none, so they partition the children and a one-block parent is full or none.
* D, at block size B: the records are converted in memory by
  `convert_records(records, B, stats)`. The token count of block i of a request
  is `min(input_length, (i + 1) * B) - i * B`. Unique states are the distinct
  output ids (`stats['output_blocks']`); the packed working set sums the token
  count of each output id the first time it appears, times
  {config['bytes_per_token']} bytes per token. Repeat share, as
  `characterize_external_traces.repeat_shares`: the input tokens of a request
  in blocks whose state occurred before it, over all input tokens. The strict
  variant counts a state as occurred only at a strictly earlier timestamp
  (every request of one timestamp sees the cache before that timestamp); the
  file-order variant also counts earlier requests of the same timestamp. The
  repeated blocks of a request must be a leading run (asserted). "Last 40 %"
  is the requests with timestamp >= start + {config['test_fraction_start']} x
  span (`window_start_ms`), with the whole trace before them as history.
  Capacities follow `run_decision_population._capacity` with the L1 fractions
  {list(config['l1_fractions'])} and L2 multipliers
  {list(config['l2_multipliers'])}. At 512 the direct numbers are compared with
  `load_mooncake_trace(converted, 512)`, `gap.working_set_bytes` and
  `repeat_shares` (checks `loader512_*`), which validates the same direct code
  that produces the 16-token numbers.
* E: the Mooncake traces are read with `load_mooncake_trace(path, 512)`; B/E
  columns use the loader's millisecond timestamps.

## checks.csv

Columns: `trace`, `check`, `expected`, `observed`, `pass`.

| check | passes when |
|---|---|
{check_lines}

## audit.csv columns

| column | meaning |
|---|---|
{column_lines}

Written columns, in order: {", ".join(f"`{name}`" for name in columns)}.

## Run record

* script sha256: `{config['script_sha256']}`
* repository commit: `{config['repository_commit']}` (script git status: {config['script_git_status']})
"""


SUMMARY_ROWS = (
    ("records", None, "records", "{:,}"),
    ("span (s)", None, "span_s", "{:,.1f}"),
    ("max timestamp decimals", None, "max_timestamp_decimals", "{}"),
    ("timestamp decreases", None, "timestamp_decreases", "{}"),
    ("distinct timestamps", None, "distinct_timestamps", "{:,}"),
    ("timestamps with > 1 record", None, "multi_record_timestamp_groups", "{:,}"),
    ("input mean / p95", None, ("input_mean", "input_p95"), "{:,.0f} / {:,.0f}"),
    ("partial 512-block token share", None, "partial512_token_share", "{:.4f}"),
    ("chat_id unique", None, "chat_id_unique", "{}"),
    ("root share", None, "root_share", "{:.3f}"),
    ("resolvable parents (earlier)", None, ("resolvable_parent_records", "resolvable_parent_earlier_records"),
     "{:,} ({:,})"),
    ("LCP full / all-but-last", None, ("lcp_full_share", "lcp_all_but_last_share"), "{:.3f} / {:.3f}"),
    ("LCP partial / none", None, ("lcp_partial_share", "lcp_none_share"), "{:.3f} / {:.3f}"),
    ("16-ids at > 1 position / predecessor", None,
     ("ids16_at_multiple_positions_share", "ids16_after_multiple_predecessors_share"), "{:.4f} / {:.4f}"),
    ("16-id occurrences, new predecessor", None, "ids16_occurrences_new_predecessor_share", "{:.4f}"),
)


def _summary_value(row: dict | None, key, form: str) -> str:
    keys = key if isinstance(key, tuple) else (key,)
    if row is None or any(row.get(name) in (None, "") for name in keys):
        return "n/a"
    return form.format(*(row[name] for name in keys))


def summary_markdown(rows: list[dict], checks: list[dict], granularities: list[int]) -> str:
    traces = list(dict.fromkeys(row["trace"] for row in rows))
    by_key = {(row["trace"], row.get("granularity")): row for row in rows}
    any_row = {trace: next(row for row in rows if row["trace"] == trace) for trace in traces}
    metrics = list(SUMMARY_ROWS)
    for block_tokens in granularities:
        metrics += [
            (f"unique states @{block_tokens}", block_tokens, "unique_states", "{:,}"),
            (f"working set GiB @{block_tokens}", block_tokens, "working_set_gib", "{:,.2f}"),
            (f"repeat share @{block_tokens} (last 40%)", block_tokens,
             ("repeat_share", "repeat_share_last40"), "{:.3f} ({:.3f})"),
            (f"file-order repeat share @{block_tokens} (last 40%)", block_tokens,
             ("repeat_share_file_order", "repeat_share_last40_file_order"), "{:.3f} ({:.3f})"),
        ]
    metrics += [
        ("working set 512 / 16", None, "working_set_ratio_512_over_16", "{:.3f}"),
        ("repeat share 16 - 512", None, "repeat_share_diff_16_minus_512", "{:+.4f}"),
        ("checks failed", None, "checks_failed", "{}"),
        ("seconds", None, "seconds", "{:,.1f}"),
        ("peak RSS (MiB)", None, "peak_rss_mib", "{:,.0f}"),
    ]
    lines = ["| metric | " + " | ".join(traces) + " |", "|---|" + "---:|" * len(traces)]
    for label, granularity, key, form in metrics:
        cells = []
        for trace in traces:
            row = any_row[trace] if granularity is None else by_key.get((trace, granularity))
            if row is not None and "working_set_bytes" in row and row.get("working_set_bytes") not in (None, ""):
                row = {**row, "working_set_gib": row["working_set_bytes"] / 2**30}
            cells.append(_summary_value(row, key, form))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    failed = [check for check in checks if not check["pass"]]
    lines.append("")
    lines.append(f"{len(checks) - len(failed)} of {len(checks)} checks pass.")
    for check in failed:
        lines.append(f"- FAIL {check['trace']} {check['check']}: expected {check['expected']!r}, "
                     f"observed {check['observed']!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="directory holding the four upstream files (default data/raw/qwen_bailian)")
    parser.add_argument("--converted-dir", type=Path, default=DEFAULT_CONVERTED_DIR,
                        help="directory holding <trace>.jsonl and <trace>.manifest.json at 512 tokens "
                             "(default data/raw/qwen_bailian_512)")
    parser.add_argument("--block-tokens", type=_block_tokens_arg, default=list(DEFAULT_GRANULARITIES),
                        help="comma-separated section-D granularities, each a positive multiple of 16 "
                             "(default 16,512; 512 adds the loader cross-check)")
    parser.add_argument("--mooncake", type=Path, nargs="*", default=list(DEFAULT_MOONCAKE),
                        help="Mooncake traces for section E (default the conversation and toolagent "
                             "traces; give the flag alone for none)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="where the four outputs go (default results/bailian_input_audit)")
    parser.add_argument("--paper-dir", type=Path, help="also copy the four outputs here")
    parser.add_argument("--work-dir", type=Path,
                        help="keep each per-file subprocess's JSON result and log here "
                             "(default: a temporary directory removed at exit)")
    parser.add_argument("--only", choices=sorted(convert.NAMES),
                        help="audit only this upstream file name")
    parser.add_argument("--one-source", help=argparse.SUPPRESS)
    parser.add_argument("--one-mooncake", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.one_source is not None or args.one_mooncake is not None) and args.result is None:
        parser.error("--result is required in a per-file subprocess")
    if args.one_source is not None and args.one_source not in convert.NAMES:
        parser.error(f"unknown upstream file {args.one_source}")
    return args


def _git_head(directory: Path) -> str:
    try:
        head = _git(directory, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        head = None
    return head.strip() if head and head.strip() else UNAVAILABLE


def _script_git_status() -> str:
    try:
        status = _git(REPOSITORY, "status", "--porcelain", "--", str(Path(__file__).resolve()))
    except (OSError, subprocess.SubprocessError):
        status = None
    if status is None:
        return UNAVAILABLE
    return "clean" if not status.strip() else f"`{status.strip()}`"


def main(argv: list[str] | None = None) -> int:
    given = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(given)
    if args.one_source is not None:
        result = audit_source(args.one_source, args.source_dir, args.converted_dir, args.block_tokens)
        args.result.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    if args.one_mooncake is not None:
        result = audit_mooncake(args.one_mooncake)
        args.result.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    names = [args.only] if args.only else list(convert.NAMES)
    granularities = ",".join(str(value) for value in args.block_tokens)
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="audit_bailian_") as scratch:
        work_dir = Path(args.work_dir) if args.work_dir is not None else Path(scratch)
        work_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            stem = convert.NAMES[name]
            result = _run_child(["--one-source", name, "--source-dir", str(Path(args.source_dir).resolve()),
                                 "--converted-dir", str(Path(args.converted_dir).resolve()),
                                 "--block-tokens", granularities], stem, work_dir)
            result.setdefault("kind", "bailian")
            result.setdefault("trace", stem)
            result.setdefault("source", name)
            results.append(result)
            print(f"done {stem}: {sum(not c['pass'] for c in result['checks'])} failed checks",
                  file=sys.stderr, flush=True)
        for path in args.mooncake:
            stem = Path(path).stem
            result = _run_child(["--one-mooncake", str(Path(path).resolve())], f"mooncake_{stem}", work_dir)
            result.setdefault("kind", "mooncake")
            result.setdefault("trace", stem)
            result.setdefault("source", _display(path))
            results.append(result)
            print(f"done {stem}", file=sys.stderr, flush=True)

    checks = [{"trace": result["trace"], **check} for result in results for check in result["checks"]]
    config = {
        "command": shlex.join([".venv/bin/python", "scripts/audit_bailian_inputs.py", *given]),
        "source_dir": _display(args.source_dir), "converted_dir": _display(args.converted_dir),
        "block_tokens": args.block_tokens, "mooncake": [_display(path) for path in args.mooncake],
        "only": args.only, "work_dir": None if args.work_dir is None else _display(args.work_dir),
        "source_block_tokens": SOURCE_BLOCK_TOKENS, "converted_block_tokens": BLOCK_TOKENS,
        "bytes_per_token": BYTES_PER_TOKEN, "size_model": HYPERPARAMETERS["size_model"],
        "test_fraction_start": TEST_FRACTION_START,
        "l1_fractions": list(L1_FRACTIONS), "l2_multipliers": list(L2_MULTIPLIERS),
        "script_sha256": convert._sha256(Path(__file__).resolve()),
        "repository_commit": _git_head(REPOSITORY), "script_git_status": _script_git_status(),
    }
    rows = flat_rows(results)
    columns = csv_columns(rows)
    failed = sum(not check["pass"] for check in checks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "config": config,
        "traces": {result["trace"]: result for result in results if result["kind"] == "bailian"},
        "mooncake": {result["trace"]: result for result in results if result["kind"] == "mooncake"},
        "checks": checks, "checks_failed": failed,
    }
    (output_dir / "audit.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    _write_csv(output_dir / "audit.csv", columns, rows)
    _write_csv(output_dir / "checks.csv", ["trace", "check", "expected", "observed", "pass"], checks)
    (output_dir / "README.md").write_text(readme_text(config, columns), encoding="utf-8")
    if args.paper_dir is not None:
        Path(args.paper_dir).mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            shutil.copyfile(output_dir / name, Path(args.paper_dir) / name)
    print(summary_markdown(rows, checks, args.block_tokens))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
