#!/usr/bin/env bash
# 検証 5: 各モデルを単独で num_ctx=16384 でロードし、VRAM 使用量と GPU 配置率を記録する
set -u
cd "$(dirname "$0")"
OUT=results/04_vram.txt
: > "$OUT"
for m in qwen3.5:9b gemma4:12b qwen3:8b gpt-oss:20b; do
  for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
  sleep 2
  before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  uv run python - "$m" <<'PY'
import sys
from common import make_llm
make_llm(sys.argv[1]).invoke("こんにちは。一言で返事してください。")
PY
  after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  {
    echo "### $m  num_ctx=16384  nvidia-smi used: before=${before}MiB after=${after}MiB delta=$((after-before))MiB"
    ollama ps
  } | tee -a "$OUT"
done
