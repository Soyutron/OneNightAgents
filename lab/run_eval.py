"""タスク一覧を LangSmith データセットに同期し、モデル設定ごとに evaluate() で実験を回す。

使い方:
  uv run --env-file ../.env python run_eval.py gemma4:26b [--reasoning true|false|default] [--category tool_use,filesystem] [--repeat 1] [--num-predict 2048]

結果は LangSmith の実験(experiment)と results/<experiment>.jsonl の両方に残す。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from deepagents import __version__ as deepagents_version
from deepagents.backends.utils import create_file_data
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import Client, evaluate

import agent_factory
from agent_factory import ModelConfig, build_agent
from tasks import TASK_BY_ID, TASKS

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PREFIX = "deepagents-local-lab"


class CallRecorder(BaseCallbackHandler):
    """モデル呼び出しごとの時間・トークン・ツール呼び出し名を集める。"""

    def __init__(self, num_predict: int) -> None:
        self.num_predict = num_predict
        self.calls: list[dict] = []
        self.tool_seq: list[str] = []
        self.read_paths: list[str] = []
        self._t: dict = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._t[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kw):
        msg = response.generations[0][0].message
        meta = msg.response_metadata or {}
        self.calls.append(
            {
                "wall_sec": round(time.perf_counter() - self._t.pop(run_id, time.perf_counter()), 3),
                "prompt_eval_count": meta.get("prompt_eval_count"),
                "eval_count": meta.get("eval_count"),
                "tool_calls": [tc["name"] for tc in (msg.tool_calls or [])],
            }
        )
        for tc in msg.tool_calls or []:
            self.tool_seq.append(tc["name"])
            if tc["name"] == "read_file":
                self.read_paths.append(str(tc["args"].get("file_path", "")))

    @property
    def runaway(self) -> bool:
        return any((c["eval_count"] or 0) >= self.num_predict for c in self.calls)


def dataset_name() -> str:
    """タスク内容のハッシュを名前に含め、内容が変わったら別データセットになるようにする。"""
    payload = json.dumps([(t.id, t.prompt, sorted(t.files)) for t in TASKS], ensure_ascii=False, sort_keys=True)
    return f"{DATASET_PREFIX}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


def sync_dataset(client: Client) -> str:
    name = dataset_name()
    if client.has_dataset(dataset_name=name):
        return name
    ds = client.create_dataset(name, description="ローカル LLM × Deep Agents の用途別ベンチマーク(lab/tasks.py から生成)")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"task_id": t.id, "prompt": t.prompt, "files": sorted(t.files)},
                "outputs": {},
                "metadata": {"category": t.category, "lang": t.lang, "features": sorted(t.features)},
            }
            for t in TASKS
        ],
    )
    return name


def make_target(cfg: ModelConfig, log_path: Path):
    def target(inputs: dict) -> dict:
        task = TASK_BY_ID[inputs["task_id"]]
        agent = build_agent(cfg, task)
        rec = CallRecorder(cfg.num_predict)
        agent_factory.custom_calls.clear()
        config = {"configurable": {"thread_id": f"lab:{task.id}:{uuid.uuid4().hex[:8]}"}, "callbacks": [rec]}
        state_in = {"messages": [{"role": "user", "content": task.prompt}]}
        if task.files:
            state_in["files"] = {p: create_file_data(c) for p, c in task.files.items()}
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = agent.invoke(state_in, config=config)
        except Exception as e:  # noqa: BLE001 - 失敗もデータとして残す
            error = f"{type(e).__name__}: {e}"[:300]
        latency = time.perf_counter() - t0
        state = agent.get_state(config).values
        files = {p: (d.get("content", "") if isinstance(d, dict) else str(d)) for p, d in (state.get("files") or {}).items()}
        final = ""
        structured = None
        todos_used = False
        if result is not None:
            last = result["messages"][-1]
            final = last.content if isinstance(last.content, str) else str(last.content)
            sr = result.get("structured_response")
            structured = sr.model_dump() if hasattr(sr, "model_dump") else sr
            todos_used = bool(result.get("todos")) or "write_todos" in rec.tool_seq
        out = {
            "final": final,
            "files": {p: c for p, c in files.items() if p not in task.files or files[p] != task.files[p]},
            "tool_seq": rec.tool_seq,
            "custom_calls": list(agent_factory.custom_calls),
            "read_paths": rec.read_paths,
            "structured": structured,
            "todos_used": todos_used,
            "model_calls": len(rec.calls),
            "latency_s": round(latency, 2),
            "runaway": rec.runaway,
            "limit_hit": final.startswith("Model call limits exceeded"),
            "error": error,
            "prompt_tokens_first": rec.calls[0]["prompt_eval_count"] if rec.calls else None,
            "prompt_tokens_last": rec.calls[-1]["prompt_eval_count"] if rec.calls else None,
            "output_tokens": sum(c["eval_count"] or 0 for c in rec.calls),
        }
        out["success"] = bool(task.check(out)) and error is None
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": task.id, "category": task.category, "lang": task.lang, **out}, ensure_ascii=False) + "\n")
        print(f"[{task.id}] ok={out['success']} calls={out['model_calls']} {latency:.1f}s seq={rec.tool_seq} err={error}", flush=True)
        return out

    return target


# --- 評価関数(決定的)。数値はそのまま LangSmith の feedback になり、実験ごとに平均が出る ---


def ev_success(outputs: dict) -> dict:
    return {"key": "success", "score": 1.0 if outputs.get("success") else 0.0}


def ev_latency(outputs: dict) -> dict:
    return {"key": "latency_s", "score": outputs.get("latency_s", 0.0)}


def ev_calls(outputs: dict) -> dict:
    return {"key": "model_calls", "score": outputs.get("model_calls", 0)}


def ev_runaway(outputs: dict) -> dict:
    return {"key": "runaway", "score": 1.0 if outputs.get("runaway") else 0.0}


def ev_limit(outputs: dict) -> dict:
    return {"key": "limit_hit", "score": 1.0 if outputs.get("limit_hit") else 0.0}


def ev_error(outputs: dict) -> dict:
    return {"key": "error", "score": 1.0 if outputs.get("error") else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--reasoning", default="true", choices=["true", "false", "default"])
    ap.add_argument("--category", default=None, help="カンマ区切りで絞り込む")
    ap.add_argument("--task", default=None, help="task_id をカンマ区切りで絞り込む")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--coerce", action="store_true", help="ツール引数の寛容な変換ミドルウェアを有効にする")
    ap.add_argument("--num-predict", type=int, default=2048, help="1 呼び出しの生成上限(暴走の判定にも使う)")
    args = ap.parse_args()
    cfg = ModelConfig(
        model=args.model,
        reasoning={"true": True, "false": False, "default": None}[args.reasoning],
        coerce=args.coerce,
        num_predict=args.num_predict,
    )

    client = Client()
    ds_name = sync_dataset(client)
    examples = list(client.list_examples(dataset_name=ds_name))
    if args.category:
        cats = set(args.category.split(","))
        examples = [e for e in examples if e.metadata.get("category") in cats]
    if args.task:
        ids = set(args.task.split(","))
        examples = [e for e in examples if e.inputs["task_id"] in ids]

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = RESULTS_DIR / f"{cfg.label.replace(':', '_')}-{stamp}.jsonl"
    print(f"dataset={ds_name} examples={len(examples)} config={cfg.label} log={log_path}")

    res = evaluate(
        make_target(cfg, log_path),
        data=examples,
        evaluators=[ev_success, ev_latency, ev_calls, ev_runaway, ev_limit, ev_error],
        experiment_prefix=cfg.label,
        description=f"{cfg.label} / num_ctx={cfg.num_ctx} num_predict={cfg.num_predict} / deepagents {deepagents_version}",
        metadata={
            "model_name": cfg.model,
            "model_provider": "ollama",
            "reasoning": str(cfg.reasoning),
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.num_predict,
            "deepagents": deepagents_version,
            "coerce": cfg.coerce,
            "dataset": ds_name,
        },
        max_concurrency=1,
        num_repetitions=args.repeat,
    )
    print("experiment:", res.experiment_name)


if __name__ == "__main__":
    main()
