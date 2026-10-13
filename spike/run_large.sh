#!/usr/bin/env bash
# 32GB VRAM に収まる上位候補の計測(reasoning=True 固定)
set -u
cd "$(dirname "$0")"
LOG=results/run_large.log
: > "$LOG"
for m in gemma4:26b qwen3.6:27b qwen3.8:27b; do
  echo "##### pull $m $(date +%T)" | tee -a "$LOG"
  ollama pull "$m" 2>&1 | tr '\r' '\n' | grep -E "success|Error|error" >> "$LOG"
done
for m in gemma4:26b qwen3.6:27b qwen3.8:27b nemotron-3-nano:30b; do
  echo "##### $m $(date +%T)" | tee -a "$LOG"
  for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
  uv run python 01_tool_calls.py "$m" --reasoning >> "$LOG" 2>&1
  echo "--- vram ($m, num_ctx=16384)" >> "$LOG"; ollama ps >> "$LOG"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$LOG"
  uv run python 02_deep_agent.py "$m" --runs 10 --reasoning >> "$LOG" 2>&1
  echo "##### done $m $(date +%T)" | tee -a "$LOG"
done
echo "LARGE DONE $(date +%T)" | tee -a "$LOG"
