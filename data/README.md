# Mooncake FAST'25 traces

Run `../scripts/download_mooncake_traces.sh` from the repository root. Raw JSONL files are excluded from Git and are downloaded from Mooncake commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`.

Validated inputs used for the initial results:

| File | JSONL records | SHA-256 |
|---|---:|---|
| `conversation_trace.jsonl` | 12,031 | `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df` |
| `toolagent_trace.jsonl` | 23,608 | `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71` |
| `synthetic_trace.jsonl` | 3,993 | `bd070915a98fc0ed264d7cfef2ce746002eb3076a695ec31ba2674c0111ec131` |

The analysis loader independently validates field types, request ordering, block count, and cumulative-prefix parent/depth consistency.
