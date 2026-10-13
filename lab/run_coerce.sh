#!/usr/bin/env bash
# 多ターン検証の後に、引数変換ミドルウェアあり/なしの比較実験を回す
set -u
cd "$(dirname "$0")"
LOG=results/run_coerce.log
: > "$LOG"
until grep -q "GROWTH DONE" results/run_growth.log; do sleep 15; done
for c in "gemma4:26b true" "qwen3.8:27b true" "gemma4:12b true"; do
  set -- $c
  echo "##### $1 reasoning=$2 coerce $(date +%T)" | tee -a "$LOG"
  for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
  uv run --env-file ../.env python run_eval.py "$1" --reasoning "$2" --coerce 2>&1 | grep -vE "DeprecationWarning|^\s+warnings\.|list_examples|Deserializing unregistered" >> "$LOG"
done
echo "COERCE DONE $(date +%T)" | tee -a "$LOG"
