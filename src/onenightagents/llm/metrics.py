"""モデル呼び出しを包むミドルウェア: 計測と、過去の思考の履歴からの除去。"""

from __future__ import annotations

import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class MetricsMiddleware(AgentMiddleware):
    """calls に 1 呼び出し 1 件を追記する。判断ごとに呼び出し側が reset する。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = []

    async def awrap_model_call(self, request, handler):
        t0 = time.perf_counter()
        try:
            response = await handler(request)
        except Exception as e:
            self.calls.append({"wall_sec": round(time.perf_counter() - t0, 3), "error": repr(e)})
            raise
        msg = next((m for m in response.result if isinstance(m, AIMessage)), None)
        meta = (msg.response_metadata if msg is not None else {}) or {}
        self.calls.append(
            {
                "wall_sec": round(time.perf_counter() - t0, 3),
                "prompt_eval_count": meta.get("prompt_eval_count"),
                "eval_count": meta.get("eval_count"),
                "eval_duration_sec": _ns(meta.get("eval_duration")),
                "load_duration_sec": _ns(meta.get("load_duration")),
                "tool_calls": [tc["name"] for tc in (msg.tool_calls if msg is not None else [])],
                "has_reasoning": bool(msg is not None and msg.additional_kwargs.get("reasoning_content")),
            }
        )
        return response


def _ns(value: int | None) -> float | None:
    return round(value / 1e9, 3) if value else None


def _without_reasoning(message: BaseMessage) -> BaseMessage:
    if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
        kwargs = {k: v for k, v in message.additional_kwargs.items() if k != "reasoning_content"}
        return message.model_copy(update={"additional_kwargs": kwargs})
    return message


@wrap_model_call
async def drop_past_reasoning(request, handler):
    """過去の判断の思考(reasoning_content)を送信時だけ履歴から外す。

    langchain-ollama は AIMessage の reasoning_content を thinking として毎回送り返すので、
    暴走した思考(数千トークン)がゲーム終了まで毎回コンテキストに乗る(005)。
    現在の判断(最後の HumanMessage 以降)の思考はツール呼び出しの連続性のために残す。
    """
    messages = list(request.messages)
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    stripped = [_without_reasoning(m) if i < last_human else m for i, m in enumerate(messages)]
    return await handler(request.override(messages=stripped))
