#!/usr/bin/env bash
# run_all.sh 終了後の追加計測
set -u
cd "$(dirname "$0")"
LOG=results/run_followup.log
: > "$LOG"
until grep -q "ALL DONE" results/run_all.log; do sleep 10; done
echo "##### followup start $(date +%T)" | tee -a "$LOG"
./04_vram.sh >> "$LOG" 2>&1
# seed 修正前に計測した qwen3.5:9b の Deep Agent を再計測
uv run python 02_deep_agent.py qwen3.5:9b --runs 10 >> "$LOG" 2>&1
# ツール呼び出しで reasoning=True が効いた 2 モデルは Deep Agent でも reasoning=True を試す
uv run python 02_deep_agent.py qwen3.5:9b --runs 10 --reasoning >> "$LOG" 2>&1
uv run python 02_deep_agent.py gemma4:12b --runs 10 --reasoning >> "$LOG" 2>&1
# gemma4 の暴走が num_predict 上限で抑えられるか
uv run python 01_tool_calls.py gemma4:12b --num-predict 1024 >> "$LOG" 2>&1
echo "FOLLOWUP DONE $(date +%T)" | tee -a "$LOG"
