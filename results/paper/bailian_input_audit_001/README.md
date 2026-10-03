# Bailian input audit

Written by `scripts/audit_bailian_inputs.py`: a read-only audit of the
Qwen-Bailian upstream traces (`data/raw/qwen_bailian`) and of their conversion
to the Mooncake 512-token schema (`data/raw/qwen_bailian_512`, made by
`scripts/convert_bailian_trace.py`). It records identities, checks schema and
conversion integrity, measures what the 16 -> 512-token coarsening changes and
reports trace-only characteristics. It runs no cache policy, reads no
horizon-dependent reuse statistic and fits nothing.

## Commands

This run:

```
.venv/bin/python scripts/audit_bailian_inputs.py --paper-dir results/paper/bailian_input_audit_001 --work-dir /home/ubuntu/.claude/jobs/64a33413/tmp/audit_work
```

Produce the converted inputs, then rerun the audit and its tests:

```
.venv/bin/python scripts/convert_bailian_trace.py --all
.venv/bin/python scripts/audit_bailian_inputs.py --block-tokens 16,512
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
  2048 bytes per token. Repeat share, as
  `characterize_external_traces.repeat_shares`: the input tokens of a request
  in blocks whose state occurred before it, over all input tokens. The strict
  variant counts a state as occurred only at a strictly earlier timestamp
  (every request of one timestamp sees the cache before that timestamp); the
  file-order variant also counts earlier requests of the same timestamp. The
  repeated blocks of a request must be a leading run (asserted). "Last 40 %"
  is the requests with timestamp >= start + 0.6 x
  span (`window_start_ms`), with the whole trace before them as history.
  Capacities follow `run_decision_population._capacity` with the L1 fractions
  [0.0025, 0.01, 0.02] and L2 multipliers
  [1, 4]. At 512 the direct numbers are compared with
  `load_mooncake_trace(converted, 512)`, `gap.working_set_bytes` and
  `repeat_shares` (checks `loader512_*`), which validates the same direct code
  that produces the 16-token numbers.
* E: the Mooncake traces are read with `load_mooncake_trace(path, 512)`; B/E
  columns use the loader's millisecond timestamps.

## checks.csv

Columns: `trace`, `check`, `expected`, `observed`, `pass`.

| check | passes when |
|---|---|
| `audit_completed` | the per-file subprocess finished and returned its result. |
| `source_sha256_matches_manifest` | sha256 of the raw file == manifest `source_sha256`. |
| `converted_sha256_matches_manifest` | sha256 of the converted file == manifest `output_sha256`. |
| `manifest_source_is_this_file` | manifest `source` == the upstream file name. |
| `manifest_output_is_this_file` | manifest `output` == the converted file name. |
| `manifest_block_tokens` | manifest `block_tokens` == 512. |
| `source_lines_equal_manifest_records` | physical lines of the raw file == manifest `records`. |
| `converted_lines_equal_manifest_records` | physical lines of the converted file == manifest `records`. |
| `fresh_<stat>_equal_manifest` | each of records, output_blocks, records_moved_by_sort, block_keys_differing_only_in_tokens of a fresh in-memory conversion at 512 == the manifest's. |
| `fresh_conversion_byte_identical` | every JSON line of the fresh conversion equals the converted file's line byte for byte, and neither has extra lines (observed names the first differing line). |
| `lfs_oid_matches_source_sha256` | the git-lfs oid of the upstream file == its sha256 (only when the oid is available). |
| `timestamp_decreases` | no raw record has a timestamp below the previous record's. |
| `chat_id_unique_per_record` | every record has its own chat_id (parent resolution is unambiguous). |
| `d<B>_states_counted_equal_output_blocks` | the direct walk at granularity B meets exactly `output_blocks` distinct ids. |
| `d<B>_input_tokens_equal_schema` | the direct walk at B sums the same input tokens as B. |
| `loader512_loads` | the unmodified loader reads the converted file at 512. |
| `loader512_<quantity>` | the direct computation at 512 == the loader + gap.working_set_bytes + repeat_shares on the converted file, exactly, for unique_states, working_set_bytes, block_occurrences, total_input_tokens, window_start_ms, the four repeat shares, repeated_tokens, repeated_tokens_file_order, last40_requests and last40_input_tokens. |

## audit.csv columns

| column | meaning |
|---|---|
| `kind` | `bailian` (an upstream file, sections A-D) or `mooncake` (section E). |
| `trace` | Converted trace name (`convert_bailian_trace.NAMES`, the loader's file stem); for mooncake rows the file stem. |
| `source` | Upstream file name; for mooncake rows the trace path. |
| `granularity` | Block size in tokens of the section-D numbers on this row (one row per trace and --block-tokens value; 512 for mooncake rows). |
| `records` | Requests (records) in the trace. |
| `unique_states` | D: distinct block states at this granularity (`stats['output_blocks']` of `convert_records`; `len(trace.states)` for mooncake rows). |
| `states_counted` | D: distinct output ids met while walking the converted records (must equal unique_states). |
| `token_variant_keys` | D: BlockIds keys that differ from an earlier key only in the block's token count (`block_keys_differing_only_in_tokens`). |
| `working_set_bytes` | D: packed working set, sum over unique states of the state's block tokens x 2048 bytes (`gap.working_set_bytes` for mooncake rows). |
| `block_occurrences` | D: sum over requests of the number of blocks at this granularity. |
| `total_input_tokens` | Sum of input_length over all requests. |
| `window_start_ms` | start + 0.6 x span of the converted timestamps (ms); requests at or after it form the last 40 %. |
| `repeat_share` | D: infinite-capacity repeat share, strict-earlier-timestamp variant (repeated_tokens / total_input_tokens). |
| `repeat_share_file_order` | D: the same, file-order variant. |
| `repeat_share_last40` | D: strict variant over the requests at or after window_start_ms, the whole trace before them as history. |
| `repeat_share_last40_file_order` | D: file-order variant over the last 40 %. |
| `repeated_tokens` | D: input tokens in repeated blocks, strict variant, whole trace. |
| `repeated_tokens_file_order` | D: the same, file-order variant. |
| `last40_repeated_tokens` | D: repeated tokens, strict variant, last 40 % (bailian rows only). |
| `last40_repeated_tokens_file_order` | D: the same, file-order variant (bailian rows only). |
| `last40_requests` | D: requests at or after window_start_ms. |
| `last40_input_tokens` | D: their input tokens. |
| `<capacity>` | D: `l1_<f>_bytes` = max(1, round(working_set_bytes x f)) for f in {0.0025, 0.01, 0.02}; `l2_<f>x<m>_bytes` = max(1, round(working_set_bytes x (f x m))) for m in {1, 4} (`run_decision_population._capacity`); each `_blocks512` column is the bytes / (512 x 2048). |
| `working_set_ratio_512_over_16` | Derived: working_set_bytes at 512 / at 16. |
| `unique_states_ratio_16_over_512` | Derived: unique_states at 16 / at 512. |
| `repeat_share_diff_16_minus_512` | Derived: repeat_share at 16 minus at 512. |
| `repeat_share_file_order_diff_16_minus_512` | Derived: the same for the file-order variant. |
| `repeat_share_last40_diff_16_minus_512` | Derived: the same for the last-40 % strict variant. |
| `repeat_share_last40_file_order_diff_16_minus_512` | Derived: the same for the last-40 % file-order variant. |
| `timestamp_min_s` | B: smallest raw timestamp (s). |
| `timestamp_max_s` | B: largest raw timestamp (s). |
| `span_s` | B/E: (largest - smallest timestamp) in seconds. |
| `max_timestamp_decimals` | B: most decimal places of any timestamp in the raw JSON text (Decimal exponent; trailing zeros count). |
| `timestamp_decreases` | B: records whose timestamp is below the previous record's, file order. |
| `distinct_timestamps` | B/E: distinct timestamp values. |
| `multi_record_timestamp_groups` | B/E: timestamp values carried by more than one record. |
| `records_in_multi_record_groups` | B/E: records in such groups. |
| `largest_timestamp_group` | B/E: most records sharing one timestamp. |
| `input_mean` | B/E: mean input_length. |
| `input_median` | B/E: median input_length. |
| `input_p95` | B/E: 95th percentile of input_length (numpy linear interpolation, as characterize_external_traces). |
| `input_max` | B: largest input_length. |
| `output_mean` | B: mean output_length. |
| `output_median` | B: median output_length. |
| `distinct_chat_ids` | B: distinct chat_id values. |
| `chat_id_unique` | B: every record has its own chat_id. |
| `root_records` | B: records with parent_chat_id == -1. |
| `root_share` | B: root_records / records. |
| `resolvable_parent_records` | B: records whose parent_chat_id (!= -1) is the chat_id of a record in the file. |
| `resolvable_parent_earlier_records` | B: of those, records whose parent record (the first record with that chat_id) comes earlier in file order. |
| `unresolvable_parent_records` | B: records with parent_chat_id != -1 that name no chat_id in the file. |
| `turn_<bucket>` | B: records per turn bucket 1, 2, 3-5, 6-10, >10; `turn_other` counts turn < 1 or a non-integer turn. |
| `type_<name>` | B: records per `type` value. |
| `partial16_records` | B: records with input_length % 16 != 0. |
| `partial16_tokens` | B: tokens in their partial final 16-token blocks (sum of input_length % 16). |
| `partial512_records` | B/E: records with input_length % 512 != 0. |
| `partial512_tokens` | B/E: tokens in their partial final 512-token blocks (sum of the nonzero input_length % 512). |
| `partial512_token_share` | B/E: partial512_tokens / total_input_tokens. |
| `ids16_distinct` | C: distinct upstream 16-token hash ids. |
| `ids16_occurrences` | C: hash id occurrences (sum of len(hash_ids)). |
| `ids16_at_multiple_positions` | C: ids seen at more than one index within hash_ids. |
| `ids16_at_multiple_positions_share` | C: / ids16_distinct. |
| `ids16_after_multiple_predecessors` | C: ids seen after more than one distinct predecessor (the previous id of the same request; a shared sentinel at index 0). |
| `ids16_after_multiple_predecessors_share` | C: / ids16_distinct. |
| `ids16_occurrences_new_position` | C: occurrences at an index other than the id's first-seen index (file order). |
| `ids16_occurrences_new_position_share` | C: / ids16_occurrences. |
| `ids16_occurrences_new_predecessor` | C: occurrences whose predecessor differs from the id's first-seen predecessor (file order). |
| `ids16_occurrences_new_predecessor_share` | C: / ids16_occurrences. |
| `children_with_parent` | C: records with a resolvable parent (= resolvable_parent_records). |
| `lcp_full` | C: children whose longest common hash_ids prefix with the parent equals len(parent hash_ids) (the child extends the parent exactly). |
| `lcp_full_share` | C: / children_with_parent. |
| `lcp_all_but_last` | C: children with LCP == len(parent) - 1 >= 1 (only the parent's last block differs, the upstream FAQ Q2 case). |
| `lcp_all_but_last_share` | C: / children_with_parent. |
| `lcp_partial` | C: children with 0 < LCP < len(parent) - 1. |
| `lcp_partial_share` | C: / children_with_parent. |
| `lcp_none` | C: children with LCP == 0. |
| `lcp_none_share` | C: / children_with_parent. |
| `children_shorter_than_parent` | C: children with fewer hash ids than their parent (cannot be full). |
| `parents_single_block` | C: children whose parent has one hash id (full or none only). |
| `source_sha256` | A: sha256 of the raw upstream file. |
| `converted_sha256` | A: sha256 of the converted file. |
| `lfs_oid` | A: git-lfs oid of the upstream file, or `unavailable`. |
| `upstream_commit` | A: HEAD of --source-dir when it is a git checkout root, or `unavailable`. |
| `source_lines` | A: physical lines of the raw file. |
| `converted_lines` | A: physical lines of the converted file. |
| `manifest_records` | A: `records` of the converter manifest. |
| `manifest_output_blocks` | A: `output_blocks` of the manifest. |
| `manifest_records_moved_by_sort` | A: `records_moved_by_sort` of the manifest. |
| `manifest_block_keys_differing_only_in_tokens` | A: `block_keys_differing_only_in_tokens` of the manifest. |
| `fresh_records` | A: `records` of the fresh in-memory conversion at 512. |
| `fresh_output_blocks` | A: its `output_blocks`. |
| `fresh_records_moved_by_sort` | A: its `records_moved_by_sort`. |
| `fresh_block_keys_differing_only_in_tokens` | A: its `block_keys_differing_only_in_tokens`. |
| `first_differing_line` | A: first line where the fresh conversion and the converted file differ (blank when byte-identical). |
| `checks_failed` | Failed checks of this trace in checks.csv. |
| `seconds` | Wall time of the per-file audit inside its subprocess (s). |
| `wall_seconds` | Wall time of the subprocess including interpreter start-up (s). |
| `peak_rss_mib` | Peak resident set size of the per-file subprocess (MiB, ru_maxrss). |
| `error` | Last log line of a per-file subprocess that did not complete (other columns blank). |

Written columns, in order: `kind`, `trace`, `source`, `granularity`, `records`, `unique_states`, `states_counted`, `token_variant_keys`, `working_set_bytes`, `block_occurrences`, `total_input_tokens`, `window_start_ms`, `repeat_share`, `repeat_share_file_order`, `repeat_share_last40`, `repeat_share_last40_file_order`, `repeated_tokens`, `repeated_tokens_file_order`, `last40_repeated_tokens`, `last40_repeated_tokens_file_order`, `last40_requests`, `last40_input_tokens`, `l1_0.0025_bytes`, `l1_0.0025_blocks512`, `l1_0.01_bytes`, `l1_0.01_blocks512`, `l1_0.02_bytes`, `l1_0.02_blocks512`, `l2_0.0025x1_bytes`, `l2_0.0025x1_blocks512`, `l2_0.0025x4_bytes`, `l2_0.0025x4_blocks512`, `l2_0.01x1_bytes`, `l2_0.01x1_blocks512`, `l2_0.01x4_bytes`, `l2_0.01x4_blocks512`, `l2_0.02x1_bytes`, `l2_0.02x1_blocks512`, `l2_0.02x4_bytes`, `l2_0.02x4_blocks512`, `working_set_ratio_512_over_16`, `unique_states_ratio_16_over_512`, `repeat_share_diff_16_minus_512`, `repeat_share_file_order_diff_16_minus_512`, `repeat_share_last40_diff_16_minus_512`, `repeat_share_last40_file_order_diff_16_minus_512`, `timestamp_min_s`, `timestamp_max_s`, `span_s`, `max_timestamp_decimals`, `timestamp_decreases`, `distinct_timestamps`, `multi_record_timestamp_groups`, `records_in_multi_record_groups`, `largest_timestamp_group`, `input_mean`, `input_median`, `input_p95`, `input_max`, `output_mean`, `output_median`, `distinct_chat_ids`, `chat_id_unique`, `root_records`, `root_share`, `resolvable_parent_records`, `resolvable_parent_earlier_records`, `unresolvable_parent_records`, `turn_1`, `turn_2`, `turn_3-5`, `turn_6-10`, `turn_>10`, `turn_other`, `type_api`, `type_coder`, `type_file`, `type_image`, `type_search`, `type_text`, `type_thinking`, `partial16_records`, `partial16_tokens`, `partial512_records`, `partial512_tokens`, `partial512_token_share`, `ids16_distinct`, `ids16_occurrences`, `ids16_at_multiple_positions`, `ids16_at_multiple_positions_share`, `ids16_after_multiple_predecessors`, `ids16_after_multiple_predecessors_share`, `ids16_occurrences_new_position`, `ids16_occurrences_new_position_share`, `ids16_occurrences_new_predecessor`, `ids16_occurrences_new_predecessor_share`, `children_with_parent`, `lcp_full`, `lcp_full_share`, `lcp_all_but_last`, `lcp_all_but_last_share`, `lcp_partial`, `lcp_partial_share`, `lcp_none`, `lcp_none_share`, `children_shorter_than_parent`, `parents_single_block`, `source_sha256`, `converted_sha256`, `lfs_oid`, `upstream_commit`, `source_lines`, `converted_lines`, `manifest_records`, `manifest_output_blocks`, `manifest_records_moved_by_sort`, `manifest_block_keys_differing_only_in_tokens`, `fresh_records`, `fresh_output_blocks`, `fresh_records_moved_by_sort`, `fresh_block_keys_differing_only_in_tokens`, `first_differing_line`, `checks_failed`, `seconds`, `wall_seconds`, `peak_rss_mib`, `error`.

## Run record

* script sha256: `449d53e60d6ce93d20b5240fd1c9b4e9e4cfeaa7c3a264b43cb2cfcc4449e889`
* repository commit: `6f2007076eb3b978287c89fbe5282c7763c4fd58` (script git status: `?? scripts/audit_bailian_inputs.py`)
