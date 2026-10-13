#!/usr/bin/env bash
# run_coerce.sh の終了後に、v2 タスク(tasks_extra を連結)で全モデルを coerce 有効で回す
set -u
cd "$(dirname "$0")"
LOG=results/run_v2.log
: > "$LOG"
until grep -q "COERCE DONE" results/run_coerce.log 2>/dev/null; do sleep 15; done
# v2 タスクを有効化(冪等)
grep -q "tasks_extra" tasks.py || printf '\n# v2 追加タスク(難度高)。末尾で連結して TASK_BY_ID も更新する\nfrom tasks_extra import EXTRA_TASKS  # noqa: E402\n\nTASKS += EXTRA_TASKS\nTASK_BY_ID.update({t.id: t for t in EXTRA_TASKS})\n' >> tasks.py
for c in "gemma4:26b true" "gemma4:12b true" "qwen3.8:27b true" "qwen3.6:27b true" "gpt-oss:20b true" "qwen3.5:9b true"; do
  set -- $c
  echo "##### v2 $1 reasoning=$2 coerce $(date +%T)" | tee -a "$LOG"
  for r in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$r"; done
  uv run --env-file ../.env python run_eval.py "$1" --reasoning "$2" --coerce 2>&1 | grep -vE "DeprecationWarning|^\s+warnings\.|list_examples|Deserializing unregistered" >> "$LOG"
done
echo "V2 DONE $(date +%T)" | tee -a "$LOG"
