"""保存済みの results/*.jsonl を現在の判定関数で採点し直す(判定のバグ修正後に使う)。

LangSmith 側の feedback は変わらないので、必要なら evaluate(experiment=...) で評価器を追加する。
"""

import json
from pathlib import Path

from tasks import TASK_BY_ID

RESULTS = Path(__file__).parent / "results"

changed = 0
for p in sorted(RESULTS.glob("*.jsonl")):
    if p.name.startswith("thread_growth"):
        continue
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for r in rows:
        task = TASK_BY_ID.get(r["task_id"])
        if task is None:
            out.append(r)
            continue
        new = bool(task.check(r)) and not r.get("error")
        if new != r["success"]:
            changed += 1
            print(f"{p.name}: {r['task_id']} {r['success']} -> {new}")
            r.setdefault("success_at_run", r["success"])
            r["success"] = new
        out.append(r)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
print("changed:", changed)
