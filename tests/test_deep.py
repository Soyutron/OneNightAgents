"""DeepAgentPlayer のテスト: 偽モデルでエージェント → mailbox → 行動の経路と、未提出時の例外。"""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from onenightagents.game.models import SpeakRequest, Turn, VoteRequest
from onenightagents.players.base import ActionNotSubmitted, SpeechAction, VoteAction
from onenightagents.players.deep import DeepAgentPlayer


class FakeToolModel(GenericFakeChatModel):
    """bind_tools を受け付ける偽モデル。与えた AIMessage を順に返す。"""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self


def tool_call(name: str, args: dict, id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def make_player(replies: list[AIMessage], steps: list[dict]) -> DeepAgentPlayer:
    return DeepAgentPlayer(
        id="p1",
        display_name="太郎",
        game_id="g",
        llm=FakeToolModel(messages=iter(replies)),
        model_call_limit=4,
        on_step=steps.append,
    )


async def test_speak_via_tool_reaches_mailbox() -> None:
    steps: list[dict] = []
    player = make_player(
        [
            tool_call("write_file", {"file_path": "/notes.md", "content": "p2 が怪しい"}),
            tool_call("speak", {"content": "p2 が怪しい", "addressed_to": "p2", "claimed_role": "villager"}, "call-2"),
            AIMessage(content="提出しました"),
        ],
        steps,
    )
    request = SpeakRequest(turn=Turn(speaker="p1", round=1, pass_allowed=True), others=["p2", "p3"])
    action = await player.speak([HumanMessage("発言してください")], request)
    assert action == SpeechAction(content="p2 が怪しい", addressed_to="p2", claimed_role="villager")
    assert len(steps) == 1
    assert steps[0]["purpose"] == "speak"
    assert steps[0]["model_calls"] == 3
    assert steps[0]["tool_calls"] == ["write_file", "speak"]
    assert steps[0]["notes"] == "p2 が怪しい"


async def test_no_tool_call_raises_action_not_submitted() -> None:
    steps: list[dict] = []
    player = make_player([AIMessage(content="p3 に投票します")], steps)
    with pytest.raises(ActionNotSubmitted):
        await player.vote([HumanMessage("投票してください")], VoteRequest(candidates=["p2", "p3"]))
    assert steps[0]["tool_calls"] == []


async def test_invalid_tool_args_then_fix() -> None:
    steps: list[dict] = []
    player = make_player(
        [
            tool_call("vote", {"target": "p1", "reason": "自分"}),  # 自分には投票できない
            tool_call("vote", {"target": "p3", "reason": "怪しい"}, "call-2"),
            AIMessage(content="投票しました"),
        ],
        steps,
    )
    action = await player.vote([HumanMessage("投票してください")], VoteRequest(candidates=["p2", "p3"]))
    assert action == VoteAction(target="p3", reason="怪しい")
    assert steps[0]["tool_calls"] == ["vote", "vote"]


async def test_update_notes_returns_file_content() -> None:
    steps: list[dict] = []
    player = make_player(
        [tool_call("write_file", {"file_path": "/notes.md", "content": "# メモ\n- p2: 村人 CO"}), AIMessage(content="書きました")],
        steps,
    )
    assert await player.update_notes([HumanMessage("メモを書いてください")]) == "# メモ\n- p2: 村人 CO"
    # 変化がなければ notes は None
    player2 = make_player([AIMessage(content="何もしません")], steps)
    assert await player2.update_notes([HumanMessage("メモを書いてください")]) is None
    assert steps[-1]["notes"] is None
