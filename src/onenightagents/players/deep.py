"""DeepAgentPlayer: create_deep_agent で作る AI プレイヤー(docs/agent-design.md 2.3)。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from onenightagents.game.models import NightRequest, Role, SpeakRequest, VoteRequest
from onenightagents.llm.metrics import MetricsMiddleware, drop_past_reasoning
from onenightagents.players.base import ActionNotSubmitted, NightAction, SpeechAction, VoteAction
from onenightagents.players.mailbox import ActionMailbox, build_tools, coerce_tool_args
from onenightagents.players.prompts import build_system_prompt

NOTES_PATH = "/notes.md"


class DeepAgentPlayer:
    """1 ゲーム = 1 スレッド。GM からのメッセージを同じスレッドに追加して ainvoke し、mailbox から行動を取り出す。"""

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        game_id: str,
        llm: BaseChatModel,
        persona: str | None = None,
        model_call_limit: int = 4,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.id = id
        self.display_name = display_name
        self._on_step = on_step
        self._mailbox = ActionMailbox(id)
        self._metrics = MetricsMiddleware()
        self._last_notes: str | None = None
        self._thread = {"configurable": {"thread_id": f"{game_id}:{id}"}}

        # ChatOllama には既定のハーネスプロファイルがないので、general-purpose サブエージェント(task)を無効化する
        register_harness_profile(
            "ollama", HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
        )
        backend = StateBackend()
        self._agent = create_deep_agent(
            model=llm,
            system_prompt=build_system_prompt(display_name, id, persona),
            tools=build_tools(self._mailbox),
            backend=backend,
            middleware=[
                FilesystemMiddleware(backend=backend, tools=["read_file", "write_file", "edit_file", "ls"]),
                ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"),
                coerce_tool_args,
                ModelRetryMiddleware(max_retries=1),
                drop_past_reasoning,
                self._metrics,
            ],
            checkpointer=InMemorySaver(),
            name=f"player-{id}",
        )

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction:
        phase = "seer_look" if request.role is Role.SEER else "thief_swap"
        self._mailbox.expect(phase, request.candidates)
        return await self._decide(messages, phase, NightAction)

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction:
        self._mailbox.expect("speak", request.others, pass_allowed=request.turn.pass_allowed)
        return await self._decide(messages, "speak", SpeechAction)

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction:
        self._mailbox.expect("vote", request.candidates)
        return await self._decide(messages, "vote", VoteAction)

    async def update_notes(self, messages: list[BaseMessage]) -> str | None:
        await self._run(messages, "notes")
        return self._last_notes

    async def _decide(self, messages: list[BaseMessage], purpose: str, kind: type) -> Any:
        await self._run(messages, purpose)
        action = self._mailbox.take()
        if not isinstance(action, kind):
            raise ActionNotSubmitted(f"{self.id}: {purpose}")
        return action

    async def _run(self, messages: list[BaseMessage], purpose: str) -> None:
        self._metrics.reset()
        config = {
            **self._thread,
            "run_name": f"{self.id}:{purpose}",
            "metadata": {"player_id": self.id, "display_name": self.display_name, "purpose": purpose},
        }
        t0 = time.perf_counter()
        try:
            await self._agent.ainvoke({"messages": messages}, config=config)
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            notes = await self._read_notes()
            notes_changed = notes != self._last_notes
            self._last_notes = notes
            if self._on_step is not None:
                calls = self._metrics.calls
                self._on_step(
                    {
                        "player": self.id,
                        "purpose": purpose,
                        "model_calls": len(calls),
                        "tool_calls": [name for c in calls for name in c.get("tool_calls", [])],
                        "latency_ms": latency_ms,
                        "prompt_tokens": sum(c.get("prompt_eval_count") or 0 for c in calls),
                        "completion_tokens": sum(c.get("eval_count") or 0 for c in calls),
                        "last_prompt_tokens": calls[-1].get("prompt_eval_count") if calls else None,
                        "errors": [c["error"] for c in calls if "error" in c],
                        "calls": calls,
                        "notes": notes if notes_changed else None,
                    }
                )

    async def _read_notes(self) -> str | None:
        state = await self._agent.aget_state(self._thread)
        entry = (state.values.get("files") or {}).get(NOTES_PATH)
        if entry is None:
            return None
        content = entry.get("content", "") if isinstance(entry, dict) else entry
        return "\n".join(content) if isinstance(content, list) else str(content)
