#!/usr/bin/env bash
# 全タスク × 全設定を逐次実行する(GPU は 1 つ)。引数で設定を上書きできる: ./run_matrix.sh "gemma4:26b true" "qwen3.5:9b false"
set -u
cd "$(dirname "$0")"
LOG=results/run_matrix.log
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=("gemma4:26b true" "gemma4:12b true" "qwen3.8:27b true" "qwen3.6:27b true" "gpt-oss:20b true" "qwen3.5:9b true" "gemma4:26b false" "qwen3.5:9b false")
fi
for c in "${CONFIGS[@]}"; do
  set -- $c
  echo "##### $1 reasoning=$2 $(date +%T)" | tee -a "$LOG"
  for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
  uv run --env-file ../.env python run_eval.py "$1" --reasoning "$2" 2>&1 | grep -vE "DeprecationWarning|^\s+warnings\.|list_examples|Deserializing unregistered" >> "$LOG"
  echo "##### done $1 reasoning=$2 $(date +%T)" | tee -a "$LOG"
done
echo "MATRIX DONE $(date +%T)" | tee -a "$LOG"
