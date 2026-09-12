#!/usr/bin/env bash
set -euo pipefail
destination="${1:-data/raw}"
mkdir -p "${destination}"
revision="3cca71daccf2a7afb8fe3f0295358f70e3a69fdb"
base="https://raw.githubusercontent.com/kvcache-ai/Mooncake/${revision}/FAST25-release/traces"
for trace in conversation toolagent synthetic; do
  curl -fL "${base}/${trace}_trace.jsonl" -o "${destination}/${trace}_trace.jsonl"
done
sha256sum "${destination}"/*_trace.jsonl
