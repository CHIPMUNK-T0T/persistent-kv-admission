# Qwen-Bailian inputs: identity, conversion and what the 512-token coarsening changes

Status: input audit, run once on 2026-10-03 before the [external-workload check](bailian-external-check-plan.md) was drafted. It reads the four upstream traces and their converted files, runs no cache policy, reads no horizon-dependent reuse statistic and fits nothing. Script: `scripts/audit_bailian_inputs.py` (tests in `tests/test_audit_bailian_inputs.py`); output: [`results/paper/bailian_input_audit_001/`](../results/paper/bailian_input_audit_001/README.md) (`audit.csv`, `checks.csv`, `audit.json`, README with every definition). 138 of 138 checks pass; 4–20 s and 0.7–2.1 GiB per file.

## What was done before, and what is new

- 2026-10-02 (commit `8805f00`): the converter `scripts/convert_bailian_trace.py` and the trace-only characterization `scripts/characterize_external_traces.py`; the four converted files and manifests in `data/raw/qwen_bailian_512/`; the characterization's output (`results/external_trace_characterization/`, untracked) was read then — requests, span, input lengths, working set, infinite-capacity repeat shares at 512 — and an independent recomputation of the 512-token state counts from the raw files was recorded in the internal notes. No replay, utility or horizon statistic has been computed on these traces.
- This audit (new): identity against the upstream checkout, byte-level reproduction of the conversion, the raw schema and time structure, the meaning of the 16-token ids, the parent–child prefix relation, partial blocks, and the 16-versus-512 comparison, which needs the raw files (the characterization read the converted files only). The 512-token quantities equal the characterization's.

## Identity

Upstream: `alibaba-edu/qwen-bailian-usagetraces-anon` at commit `5f7439c51ec248a0c585f7d90a41a6f57773b912` (git-lfs; the LFS object id of each file equals its sha256). Converter manifests record the same source hashes; the converted files' hashes equal the manifests'; a fresh in-memory conversion of each raw file is byte-identical to the converted file (stable sort moved 0 records; 0 block keys differ only in token count); the unmodified loader reads all four.

| upstream file | trace | records | raw sha256 | converted sha256 |
|---|---|---:|---|---|
| `qwen_traceA_blksz_16.jsonl` (To-C) | `bailian_toc_trace` | 43,058 | `07cedc9e…` | `ea9d739e…` |
| `qwen_traceB_blksz_16.jsonl` (To-B) | `bailian_tob_trace` | 172,800 | `68e3f98e…` | `3f1d3869…` |
| `qwen_thinking_blksz_16.jsonl` | `bailian_thinking_trace` | 10,812 | `41ac36d9…` | `02db3029…` |
| `qwen_coder_blksz_16.jsonl` | `bailian_coder_trace` | 43,011 | `3d74974c…` | `a6355771…` |

## Conversion and loader rules, as they bear on the readings

- The raw files carry one salted-SipHash id per 16-token block, timestamps in seconds with at most 3 decimals from the start of the file, and `chat_id`, `parent_chat_id`, `type`, `turn`. The converter groups 32 consecutive ids into one 512-token block, keeps the final partial group sized by the remainder, and assigns each block a sequential id injective in (parent block id, token count, the 32 source ids): two requests share a block iff they share every source id up to it and its token count, so a partial final block never aliases a full one. Seconds become milliseconds exactly.
- The loader identifies a state by its id, requires consistent chain metadata, keeps the partial final block as a state of its own size, and lets every request of one timestamp see the cache as it was before that timestamp (no hit between simultaneous requests). Capacity is charged in the packed size model, 2048 bytes per token of each state's own block; the working set `W` is the sum over unique states, and the cells are fractions of `W`.

## Time

Spans 7,192–7,200 s (two hours; Mooncake 3,537 s). Timestamps have 3 decimals (milliseconds), none decreases, and ties are rare: 139 / 2,207 / 42 / 127 timestamps carry more than one record (278 / 4,434 / 85 / 254 records, groups of at most 3), where every Mooncake record sits in one of 1,180 groups of up to 28 / 47 records (about 3-second quantisation). The no-hit-between-simultaneous-requests rule is therefore nearly inactive on Bailian: the strict and file-order repeat shares differ by less than 0.0001 on every trace (Mooncake: 0.0000–0.0001 as well, as those batches rarely repeat within themselves).

## Sessions and prefix identity

- `chat_id` is unique per record on every trace. Children (records whose `parent_chat_id` names an earlier record): To-C 19,957 (46%), thinking 1,200 (11%), coder 16,605 (39%); To-B has none (every record is a root at turn 1; `type` api 87%, text 13%). Turn buckets 1 / 2 / 3–5 / 6–10 / >10: To-C 23,101 / 9,012 / 8,413 / 2,052 / 480; coder 26,406 / 6,693 / 6,906 / 2,236 / 770; thinking 9,612 / 509 / 503 / 159 / 29. To-C types: text 31,744, search 8,187, image 1,617, file 1,510.
- Longest common prefix of a child's ids with its parent's, in 16-token blocks: **all but the parent's last block** in 94.2% / 93.3% / 93.1% of children (To-C / thinking / coder), the whole parent in 5.8% / 2.8% / 6.5%, a shorter prefix in 0.0% / 3.7% / 0.4%, nothing in 0.0% / 0.3% / 0.0%. The upstream FAQ explains the first case: the parent's last input block contains padding that the first output token replaces, so its hash changes. At 512 tokens the whole 512-block containing that 16-token block differs, so a child reuses its parent's prefix only up to the previous 512-token boundary; this holds for every arm equally and is part of what the coarsening loses.
- The 16-token ids are per-block content hashes in To-C and To-B (2.1% / 3.0% of ids occur at more than one position, 0.7% / 1.2% after more than one predecessor; 1.3% / 2.9% of occurrences follow a different predecessor than the id's first), and cumulative-like in thinking and coder (0 in every count). The converter's cumulative identity treats both the same.

## Partial blocks

94% of records end in a partial 16-token block (0.3–0.8% of input tokens). At 512 tokens 99.8% of records end in a partial block, holding **10.3% / 25.8% / 4.9% / 4.3%** of input tokens (To-C / To-B / thinking / coder; Mooncake 2.2% / 2.6%). A partial final block is its own state and never serves another request; on To-B, whose inputs are short (mean 915, median 574 tokens), a quarter of the input is in such blocks at 512.

## 16 versus 512 tokens

| | To-C | To-B | thinking | coder |
|---|---:|---:|---:|---:|
| unique states, 16 / 512 | 2,656,400 / 120,177 | 4,638,378 / 290,486 | 1,735,015 / 62,077 | 5,202,517 / 199,350 |
| block occurrences, 16 / 512 | 6.29 M / 218,859 | 9.96 M / 401,630 | 3.22 M / 106,500 | 15.47 M / 505,188 |
| working set GiB, 16 / 512 (ratio) | 80.5 / 95.4 (1.19) | 139.2 / 196.6 (1.41) | 52.8 / 55.5 (1.05) | 158.2 / 173.2 (1.09) |
| repeat share, whole trace, 16 / 512 | 0.580 / 0.502 | 0.538 / 0.348 | 0.462 / 0.435 | 0.665 / 0.633 |
| repeat share, last 40%, 16 / 512 | 0.623 / 0.549 | 0.539 / 0.348 | 0.365 / 0.343 | 0.675 / 0.642 |
| repeatable tokens lost by 512, whole / last 40% (points) | 7.8 / 7.4 | 19.0 / 19.1 | 2.8 / 2.1 | 3.2 / 3.3 |

The repeat share is the infinite-capacity share of input tokens in blocks whose state occurred at a strictly earlier timestamp; "last 40%" is the requests at or after 60% of the span, with the whole trace before them as history (the runners' split). The 512-token working set exceeds the 16-token one because two requests that share part of a 512-token span hold two distinct blocks; the capacity fractions therefore buy more bytes at 512 than they would at 16 (1.05–1.41×). To-B is the trace the coarsening changes most: 19 points of repeatable input become unrepeatable and the working set grows 41%.

Absolute capacities at 512 tokens (`round(f × W)` bytes, shown in MiB; L1 fraction `f`, L2 = L1 × multiplier): 

| trace | W (GiB) | 0.25%: L1, L2×1, L2×4 | 1%: L1, L2×1, L2×4 | 2%: L1, L2×1, L2×4 |
|---|---:|---|---|---|
| To-C | 95.4 | 244, 244, 977 | 977, 977, 3,909 | 1,955, 1,955, 7,819 |
| To-B | 196.6 | 503, 503, 2,013 | 2,013, 2,013, 8,051 | 4,025, 4,025, 16,102 |
| thinking | 55.5 | 142, 142, 568 | 568, 568, 2,274 | 1,137, 1,137, 4,548 |
| coder | 173.2 | 443, 443, 1,773 | 1,773, 1,773, 7,094 | 3,547, 3,547, 14,188 |
| Mooncake conversation | 173.0 | 443, 443, 1,771 | 1,771, 1,771, 7,086 | 3,543, 3,543, 14,171 |
| Mooncake tool-agent | 166.2 | 425, 425, 1,702 | 1,702, 1,702, 6,808 | 3,404, 3,404, 13,616 |

## Beside Mooncake

| | To-C | To-B | thinking | coder | conversation | tool-agent |
|---|---:|---:|---:|---:|---:|---:|
| requests | 43,058 | 172,800 | 10,812 | 43,011 | 12,031 | 23,608 |
| input tokens (M) | 100.4 | 158.0 | 51.5 | 247.2 | 144.8 | 202.9 |
| mean / p95 input | 2,331 / 8,808 | 915 / 2,491 | 4,763 / 16,221 | 5,748 / 13,406 | 12,035 / 39,550 | 8,596 / 26,110 |
| input tokens per second | 13,943 | 21,949 | 7,160 | 34,340 | 40,937 | 57,376 |
| W (GiB) / tokens per second (×10⁶ byte·s/token) | 7.35 | 9.62 | 8.32 | 5.42 | 4.54 | 3.11 |
| repeat share, last 40%, at 512 | 0.549 | 0.348 | 0.343 | 0.642 | 0.404 | 0.592 |

At the same capacity fraction, a Bailian L2 holds 1.2–2.1× the seconds of input traffic of the conversation trace's and 1.7–3.1× the tool-agent trace's (the fifth row), which is the reasoning recorded behind the plan's first prediction. The Bailian traces are four scenarios of one provider sampled over two hours; they are a second workload family, not four deployments.

## Not verified, and the range of what this audit says

- The upstream anonymisation (salted SipHash-2-4, domain remapping) is taken as collision-free; nothing here can check it. The FAQ's explanations (padding in the last block; special tokens stripped between turns) are taken from upstream and are consistent with the prefix counts, not verified against content.
- Whether the two-hour files are the cluster's whole traffic or a sample, and what the serving system's hit rule was, are upstream matters; the loader's rules (prefix identity, partial blocks as separate states, no hit between simultaneous requests, packed charging) are this repository's, applied identically to Mooncake.
- The 16-token figures are computed directly from the converter's records without the loader (validated against the loader at 512 on every trace and at 16 on constructed files); whether a replay at 16 tokens is feasible in memory is measured by the plan's smoke, not here.
- No policy was run and no horizon-dependent statistic was read; nothing here says which horizon, class or rule works on these traces.
