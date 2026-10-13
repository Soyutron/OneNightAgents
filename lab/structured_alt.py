"""構造化出力の代替方式: ChatOllama.with_structured_output(method="json_schema") を structured タスクで試す。

使い方: uv run --env-file ../.env python structured_alt.py gemma4:26b qwen3.5:9b ...
各モデルで structured タスク 3 件を 3 回ずつ実行し、成功数を出す。Deep Agent の response_format(ToolStrategy)側の成績は
run_eval の structured カテゴリと比較する。
"""

from __future__ import annotations

import json
import sys
import time

from langchain_ollama import ChatOllama

from tasks import TASKS
from tasks_extra import EXTRA_TASKS

STRUCT_TASKS = [t for t in TASKS + EXTRA_TASKS if t.category == "structured"]
STRUCT_TASKS = list({t.id: t for t in STRUCT_TASKS}.values())


def main() -> None:
    models = sys.argv[1:]
    rows = []
    for model in models:
        llm = ChatOllama(model=model, num_ctx=8192, num_predict=1024, temperature=0.3, seed=7, reasoning=True, keep_alive="30m")
        for task in STRUCT_TASKS:
            ok_js = 0
            t_js = []
            structured = llm.with_structured_output(task.response_format, method="json_schema")
            for _ in range(3):
                t0 = time.perf_counter()
                try:
                    out = structured.invoke(task.prompt)
                    d = out.model_dump() if hasattr(out, "model_dump") else out
                except Exception as e:  # noqa: BLE001
                    d = {"error": repr(e)}
                t_js.append(time.perf_counter() - t0)
                ok_js += int(task.check({"structured": d}))
            rows.append({"model": model, "task": task.id, "method": "json_schema", "ok": ok_js, "n": 3, "sec_mean": round(sum(t_js) / 3, 1)})
            print(rows[-1], flush=True)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    (__import__("pathlib").Path(__file__).parent / "results" / "structured_alt.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
