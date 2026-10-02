#!/usr/bin/env python3
"""Convert a Qwen-Bailian 16-token trace to the Mooncake 512-token JSONL schema.

Source: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon (Apache-2.0).
Each source record has chat_id, parent_chat_id, timestamp (seconds from the
start of the file), input_length, output_length, type, turn and hash_ids, one
anonymised SipHash id per 16-token block with len(hash_ids) ==
ceil(input_length / 16). The ids are not uniformly prefix-chained: in the To-C
and To-B files the same id recurs at different positions and after different
predecessors (a per-block content hash), while in the thinking and coder files
every id has exactly one position and one predecessor.

`persistent_kv_admission.trace.load_mooncake_trace` identifies a state by its
hash id alone and requires that id to be a cumulative prefix hash: every
occurrence of an id must have the same parent, depth, prefix_tokens and
block_tokens. It keeps a final partial block (ceil(input_length / block_size)
ids) and sizes it by the remainder, so a partial final block is a different
state from any full block. This converter produces ids that satisfy that rule:

* Grouping. Consecutive groups of G = block_tokens / 16 source ids form one
  output block, so a request has ceil(len(hash_ids) / G) ==
  ceil(input_length / block_tokens) blocks, exactly the count the loader checks.
* Partial group. The final group is kept, as Mooncake keeps its final partial
  block: it holds the remaining 1..G source ids and its token count is
  input_length - (blocks - 1) * block_tokens (it can be partial even with G ids
  when the last 16-token block is itself partial).
* Identity. The output id of block d is a sequential integer (first-appearance
  order in the output file, like the domain remapping of both upstream traces)
  assigned injectively to the key (output id of block d-1, token count of
  block d, the G source ids of block d). Two requests therefore share output
  block d iff they share every source id of blocks 1..d and the token count of
  block d, so two different prefixes cannot meet on one state and a partial
  final block can never alias a full one. No hashing, so no collisions.
* Time. Seconds become milliseconds (the loader's unit) exactly: timestamps are
  parsed as decimals and multiplied by 1000, written as an integer when
  integral (every upstream timestamp has at most 3 decimals) and as a float
  otherwise.
* Order. Records are stably sorted by timestamp (the upstream files are
  already nondecreasing, so this is a no-op on them) before ids are assigned.

The output keeps the source metadata (chat_id, parent_chat_id, type, turn)
after the four Mooncake fields; the loader reads the four fields and ignores
the rest. The script only reads the source and writes the given output plus a
`<output stem>.manifest.json` beside it with hashes and counts. It is
deterministic: the same input and --block-tokens give byte-identical output.

  python scripts/convert_bailian_trace.py --all
  python scripts/convert_bailian_trace.py data/raw/qwen_bailian/qwen_traceA_blksz_16.jsonl
  python scripts/convert_bailian_trace.py IN.jsonl --output OUT.jsonl --block-tokens 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_BLOCK_TOKENS = 16
DEFAULT_BLOCK_TOKENS = 512
DEFAULT_SOURCE_DIR = REPOSITORY / "data/raw/qwen_bailian"
# Upstream file name -> output trace name (the loader names a trace by its file stem).
NAMES = {
    "qwen_traceA_blksz_16.jsonl": "bailian_toc_trace",
    "qwen_traceB_blksz_16.jsonl": "bailian_tob_trace",
    "qwen_thinking_blksz_16.jsonl": "bailian_thinking_trace",
    "qwen_coder_blksz_16.jsonl": "bailian_coder_trace",
}
REQUIRED = ("timestamp", "input_length", "output_length", "hash_ids")
METADATA = ("chat_id", "parent_chat_id", "type", "turn")


def default_output_dir(block_tokens: int) -> Path:
    return REPOSITORY / f"data/raw/qwen_bailian_{block_tokens}"


def group_size(block_tokens: int) -> int:
    if (isinstance(block_tokens, bool) or not isinstance(block_tokens, int)
            or block_tokens <= 0 or block_tokens % SOURCE_BLOCK_TOKENS):
        raise ValueError(f"--block-tokens must be a positive multiple of {SOURCE_BLOCK_TOKENS}, "
                         f"got {block_tokens!r}")
    return block_tokens // SOURCE_BLOCK_TOKENS


def seconds_to_ms(value) -> int | float:
    """Exact seconds -> milliseconds; an int when the result is integral."""
    if isinstance(value, bool) or not isinstance(value, (int, Decimal, float)):
        raise ValueError(f"timestamp must be numeric, got {value!r}")
    if isinstance(value, int):
        return value * 1000
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"timestamp must be finite, got {value!r}")
        value = Decimal(repr(value))
    if not value.is_finite():
        raise ValueError(f"timestamp must be finite, got {value!r}")
    ms = value * 1000
    return int(ms) if ms == ms.to_integral_value() else float(ms)


def parse_records(lines: Iterable[str], source: str = "<input>") -> list[dict]:
    """Parse and validate source records; timestamps keep their exact decimal value."""
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_float=Decimal)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {error}") from error
        missing = [field for field in REQUIRED if field not in record]
        if missing:
            raise ValueError(f"{source}:{line_number}: missing fields {missing}")
        input_length, output_length = record["input_length"], record["output_length"]
        hash_ids = record["hash_ids"]
        if isinstance(input_length, bool) or not isinstance(input_length, int) or input_length <= 0:
            raise ValueError(f"{source}:{line_number}: input_length must be a positive integer")
        if isinstance(output_length, bool) or not isinstance(output_length, int) or output_length < 0:
            raise ValueError(f"{source}:{line_number}: output_length must be a nonnegative integer")
        if not isinstance(hash_ids, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in hash_ids
        ):
            raise ValueError(f"{source}:{line_number}: hash_ids must be a list of integers")
        expected = math.ceil(input_length / SOURCE_BLOCK_TOKENS)
        if len(hash_ids) != expected:
            raise ValueError(
                f"{source}:{line_number}: got {len(hash_ids)} hash ids, expected "
                f"ceil({input_length}/{SOURCE_BLOCK_TOKENS})={expected}"
            )
        record["_ms"] = seconds_to_ms(record["timestamp"])
        record["_line"] = line_number
        records.append(record)
    if not records:
        raise ValueError(f"{source}: no records")
    return records


class BlockIds:
    """Injective map (parent output id, block tokens, source ids) -> sequential int."""

    def __init__(self) -> None:
        self._ids: dict[tuple, int] = {}
        self.token_variants = 0  # keys that differ from an earlier key only in block tokens
        self._by_content: dict[tuple, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def get(self, parent: int | None, tokens: int, group: tuple[int, ...]) -> int:
        key = (parent, tokens, group)
        value = self._ids.get(key)
        if value is None:
            value = len(self._ids)
            self._ids[key] = value
            content = (parent, group)
            if content in self._by_content:
                self.token_variants += 1
            else:
                self._by_content[content] = value
        return value


def coarsen(hash_ids: list[int], input_length: int, block_tokens: int,
            block_ids: BlockIds) -> list[int]:
    """Output block ids of one request (see the module docstring for the rule)."""
    size = group_size(block_tokens)
    blocks = math.ceil(len(hash_ids) / size)
    if blocks != math.ceil(input_length / block_tokens):
        raise ValueError(f"{len(hash_ids)} source ids do not cover input_length {input_length}")
    result: list[int] = []
    parent = None
    for index in range(blocks):
        group = tuple(hash_ids[index * size:(index + 1) * size])
        tokens = min(input_length, (index + 1) * block_tokens) - index * block_tokens
        parent = block_ids.get(parent, tokens, group)
        result.append(parent)
    return result


def convert_records(records: list[dict], block_tokens: int = DEFAULT_BLOCK_TOKENS,
                    stats: dict | None = None) -> Iterator[dict]:
    """Mooncake-schema records in stable timestamp order."""
    group_size(block_tokens)
    ordered = sorted(records, key=lambda record: record["_ms"])  # stable
    block_ids = BlockIds()
    moved = sum(1 for position, record in enumerate(ordered) if record is not records[position])
    for record in ordered:
        output = {
            "timestamp": record["_ms"],
            "input_length": record["input_length"],
            "output_length": record["output_length"],
            "hash_ids": coarsen(record["hash_ids"], record["input_length"], block_tokens, block_ids),
        }
        for field in METADATA:
            if field in record:
                output[field] = record[field]
        yield output
    if stats is not None:
        stats.update(
            records=len(ordered),
            records_moved_by_sort=moved,
            output_blocks=len(block_ids),
            block_keys_differing_only_in_tokens=block_ids.token_variants,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_file(source: Path, output: Path, block_tokens: int = DEFAULT_BLOCK_TOKENS) -> dict:
    with source.open(encoding="utf-8") as handle:
        records = parse_records(handle, str(source))
    stats: dict = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in convert_records(records, block_tokens, stats):
            handle.write(json.dumps(record) + "\n")
    temporary.replace(output)
    manifest = {
        "source": source.name,
        "source_sha256": _sha256(source),
        "output": output.name,
        "output_sha256": _sha256(output),
        "source_block_tokens": SOURCE_BLOCK_TOKENS,
        "block_tokens": block_tokens,
        "timestamp_unit": "ms",
        **stats,
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="*", type=Path, help="upstream Bailian JSONL file(s)")
    parser.add_argument("--all", action="store_true",
                        help=f"convert the four upstream files found in {DEFAULT_SOURCE_DIR}")
    parser.add_argument("--output", type=Path,
                        help="output path (one input only); default "
                             "data/raw/qwen_bailian_<block-tokens>/<name>.jsonl")
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS,
                        help="output block size in tokens, a multiple of 16 (default 512)")
    args = parser.parse_args(argv)
    group_size(args.block_tokens)
    inputs = list(args.inputs)
    if args.all:
        inputs += [DEFAULT_SOURCE_DIR / name for name in NAMES]
    if not inputs:
        parser.error("give input files or --all")
    if args.output is not None and len(inputs) != 1:
        parser.error("--output needs exactly one input")
    for source in inputs:
        if args.output is not None:
            output = args.output
        elif source.name in NAMES:
            output = default_output_dir(args.block_tokens) / f"{NAMES[source.name]}.jsonl"
        else:
            parser.error(f"no default name for {source.name}; give --output")
        manifest = convert_file(source, output, args.block_tokens)
        print(f"{source} -> {output}: {manifest['records']} records, "
              f"{manifest['output_blocks']} unique {args.block_tokens}-token blocks, "
              f"sha256 {manifest['output_sha256']}", flush=True)


if __name__ == "__main__":
    main()
