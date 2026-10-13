#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
until grep -q "V2 DONE" results/run_v2.log 2>/dev/null; do sleep 15; done
for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
uv run --env-file ../.env python structured_alt.py gemma4:26b gemma4:12b qwen3.5:9b gpt-oss:20b 2>&1 | grep -vE "DeprecationWarning|^\s+warnings\." > results/structured_alt.log
echo "AFTER V2 DONE $(date +%T)" >> results/structured_alt.log
