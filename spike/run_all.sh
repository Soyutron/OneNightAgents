#!/usr/bin/env bash
# 4 モデルを順に計測する(GPU は 1 つなので逐次)
set -u
cd "$(dirname "$0")"
LOG=results/run_all.log
: > "$LOG"
for m in qwen3.5:9b gemma4:12b qwen3:8b gpt-oss:20b; do
  echo "##### $m $(date +%T)" | tee -a "$LOG"
  uv run python 01_tool_calls.py "$m" >> "$LOG" 2>&1
  echo "--- vram after 01 ($m, num_ctx=16384)" >> "$LOG"; ollama ps >> "$LOG"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$LOG"
  uv run python 02_deep_agent.py "$m" --runs 10 >> "$LOG" 2>&1
  uv run python 03_japanese.py "$m" >> "$LOG" 2>&1
  uv run python 01_tool_calls.py "$m" --reasoning >> "$LOG" 2>&1
  echo "##### done $m $(date +%T)" | tee -a "$LOG"
done
echo "ALL DONE $(date +%T)" | tee -a "$LOG"
