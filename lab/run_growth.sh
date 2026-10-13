#!/usr/bin/env bash
# run_matrix.sh の終了後に多ターン検証を回す
set -u
cd "$(dirname "$0")"
LOG=results/run_growth.log
: > "$LOG"
until grep -q "MATRIX DONE" results/run_matrix.log; do sleep 15; done
for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
for args in "gemma4:26b --turns 16" "gemma4:26b --turns 16 --profile" "gemma4:26b --turns 16 --num-ctx 8192" "gemma4:26b --turns 16 --num-ctx 8192 --profile" "gemma4:12b --turns 16 --profile"; do
  echo "##### $args $(date +%T)" | tee -a "$LOG"
  uv run --env-file ../.env python thread_growth.py $args 2>&1 | grep -vE "DeprecationWarning|^\s+warnings\." >> "$LOG"
done
echo "GROWTH DONE $(date +%T)" | tee -a "$LOG"
