"""Step 0 スパイク共通: モデル生成・計測コールバック・結果出力。"""

import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama

RESULTS_DIR = Path(__file__).parent / "results"


def make_llm(model: str, *, reasoning: bool | None = False, num_ctx: int = 16384, temperature: float = 0.7, seed: int = 42, num_predict: int | None = None) -> ChatOllama:
    """architecture.md 11.1 の接続設定でスパイク用 ChatOllama を作る。"""
    return ChatOllama(
        model=model,
        num_ctx=num_ctx,
        keep_alive="30m",
        temperature=temperature,
        seed=seed,
        reasoning=reasoning,
        num_predict=num_predict,
        validate_model_on_init=True,
        metadata={"ls_provider": "ollama", "ls_model_name": model},
    )


class ModelCallRecorder(BaseCallbackHandler):
    """モデル呼び出しごとの壁時計時間と Ollama の response_metadata を集める。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._started: dict[Any, float] = {}
        # 1 回目の呼び出しで送った system prompt とツール定義を残す(固定コストの計測用)
        self.first_prompt: dict[str, Any] | None = None

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = time.perf_counter()
        if self.first_prompt is None:
            tools = kwargs.get("invocation_params", {}).get("tools") or []
            sys_text = ""
            for m in messages[0]:
                if m.type == "system":
                    sys_text = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            self.first_prompt = {
                "system_chars": len(sys_text),
                "tools_json_chars": len(json.dumps(tools, ensure_ascii=False)),
                "tool_names": [t.get("function", {}).get("name") or t.get("name") for t in tools],
                "system_text": sys_text,
            }

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        meta = (msg.response_metadata if msg is not None else {}) or {}
        self.calls.append(
            {
                "wall_sec": round(elapsed, 3),
                "prompt_eval_count": meta.get("prompt_eval_count"),
                "eval_count": meta.get("eval_count"),
                "eval_duration_sec": _ns(meta.get("eval_duration")),
                "prompt_eval_duration_sec": _ns(meta.get("prompt_eval_duration")),
                "load_duration_sec": _ns(meta.get("load_duration")),
                "tool_calls": [tc["name"] for tc in (getattr(msg, "tool_calls", None) or [])],
                "content_chars": len(msg.content) if msg is not None and isinstance(msg.content, str) else None,
                "has_reasoning": bool(msg is not None and msg.additional_kwargs.get("reasoning_content")),
            }
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        elapsed = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        self.calls.append({"wall_sec": round(elapsed, 3), "error": repr(error)})


def _ns(value: int | None) -> float | None:
    return round(value / 1e9, 3) if value else None


def percentiles(values: list[float]) -> dict[str, float]:
    """p50 / p95 / max を返す。要素が少ないときは近似。"""
    if not values:
        return {}
    s = sorted(values)
    q = statistics.quantiles(s, n=20) if len(s) >= 2 else [s[0]] * 19
    return {"p50": round(statistics.median(s), 2), "p95": round(min(q[18], s[-1]), 2), "max": round(s[-1], 2), "mean": round(statistics.mean(s), 2)}


def write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
