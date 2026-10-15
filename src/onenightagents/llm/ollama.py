"""ChatOllama の生成。ローカルモデル向けの既定は docs/research/local-deepagents.md 5.1。"""

from __future__ import annotations

from langchain_ollama import ChatOllama


def make_llm(
    model: str,
    *,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    reasoning: bool | None = True,
    temperature: float = 0.7,
    top_p: float | None = None,
    seed: int | None = None,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict,  # 暴走の上限
        reasoning=reasoning,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        keep_alive="30m",
        validate_model_on_init=True,
        # これがないと Deep Agents の要約ミドルウェアが num_ctx 内で発火しない
        profile={"max_input_tokens": num_ctx},
        metadata={"ls_provider": "ollama", "ls_model_name": model},
    )
