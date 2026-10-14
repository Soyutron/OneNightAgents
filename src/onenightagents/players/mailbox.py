"""行動提出ツールと ActionMailbox(docs/agent-design.md 2.3 / 4.3)。

エージェントはツールを呼ぶことで GM に行動を届ける。ツールはフェーズと候補を検証し、
違反時は理由を返り値で返してモデルに修正させる。最初の有効な提出だけを受理する(冪等)。
"""

from __future__ import annotations

import json
from typing import Literal

from langchain.agents.middleware import wrap_tool_call
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from onenightagents.game.models import CENTER, Role
from onenightagents.players.base import NightAction, SpeechAction, VoteAction

Phase = Literal["seer_look", "thief_swap", "speak", "vote"]

ACCEPTED = "受理しました。これ以上ツールを呼ばず、短く終了してください。"
ALREADY = "提出済みです。これ以上ツールを呼ばず、短く終了してください。"

# 小型モデルは列挙値を日本語で入れてくるので、両方を受け付ける
ROLE_ALIASES = {
    "villager": Role.VILLAGER,
    "werewolf": Role.WEREWOLF,
    "seer": Role.SEER,
    "thief": Role.THIEF,
    "村人": Role.VILLAGER,
    "人狼": Role.WEREWOLF,
    "占い師": Role.SEER,
    "怪盗": Role.THIEF,
}


class SpeakArgs(BaseModel):
    content: str = Field(max_length=200, description="発言本文。日本語で 1〜3 文")
    addressed_to: str | None = Field(
        default=None, description="質問や反論の宛先プレイヤー ID(例: p2)。全員向けなら null"
    )
    claimed_role: str | None = Field(
        default=None,
        description="この発言で主張する自分の役職。嘘でもよい。villager / werewolf / seer / thief のいずれか。主張しないなら null",
    )


class VoteArgs(BaseModel):
    target: str = Field(description="処刑したいプレイヤー ID。自分は不可")
    reason: str = Field(default="", max_length=200, description="理由の要約。他のプレイヤーには公開されない")


class SeerLookArgs(BaseModel):
    target: str = Field(description="占う対象。プレイヤー ID か、中央 2 枚を見るなら 'center'")


class ThiefSwapArgs(BaseModel):
    target: str | None = Field(default=None, description="交換相手のプレイヤー ID。交換しないなら null")


class ActionMailbox:
    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self._expected: Phase | None = None
        self._candidates: list[str] = []
        self._pass_allowed = False
        self._action: NightAction | SpeechAction | VoteAction | None = None

    def expect(self, phase: Phase, candidates: list[str], *, pass_allowed: bool = False) -> None:
        self._expected = phase
        self._candidates = list(candidates)
        self._pass_allowed = pass_allowed
        self._action = None

    def take(self) -> NightAction | SpeechAction | VoteAction | None:
        action = self._action
        self._action = None
        self._expected = None
        return action

    def _accept(self, phase: Phase, action: NightAction | SpeechAction | VoteAction) -> str:
        if self._expected != phase:
            return f"無効: いまは {self._expected} で行動を提出する時間です。"
        if self._action is not None:
            return ALREADY
        self._action = action
        return ACCEPTED

    def _check_candidate(self, target: str) -> str | None:
        if target not in self._candidates:
            return f"無効: {target} は選べません。候補: {', '.join(self._candidates)}"
        return None


def build_tools(mailbox: ActionMailbox) -> list[BaseTool]:
    @tool(args_schema=SpeakArgs)
    def speak(content: str, addressed_to: str | None = None, claimed_role: str | None = None) -> str:
        """発言ターンで発言を提出する。"""
        if addressed_to is not None and (err := mailbox._check_candidate(addressed_to)):
            return err
        role: Role | None = None
        if claimed_role is not None:
            role = ROLE_ALIASES.get(claimed_role.strip().lower())
            if role is None:
                return f"無効: claimed_role は villager / werewolf / seer / thief のいずれかです: {claimed_role}"
        return mailbox._accept("speak", SpeechAction(content=content, addressed_to=addressed_to, claimed_role=role))

    @tool
    def pass_turn() -> str:
        """発言ターンで発言せずにパスする。"""
        if mailbox._expected == "speak" and not mailbox._pass_allowed:
            return "無効: このターンはパスできません。speak で発言してください。"
        return mailbox._accept("speak", SpeechAction(is_pass=True))

    @tool(args_schema=VoteArgs)
    def vote(target: str, reason: str = "") -> str:
        """投票で処刑したい相手を提出する。"""
        if err := mailbox._check_candidate(target):
            return err
        return mailbox._accept("vote", VoteAction(target=target, reason=reason))

    @tool(args_schema=SeerLookArgs)
    def seer_look(target: str) -> str:
        """夜に占い師として占う対象を提出する。"""
        if err := mailbox._check_candidate(target):
            return err
        return mailbox._accept("seer_look", NightAction(target=target))

    @tool(args_schema=ThiefSwapArgs)
    def thief_swap(target: str | None = None) -> str:
        """夜に怪盗として交換相手を提出する。交換しないなら target を null にする。"""
        if target is not None and (err := mailbox._check_candidate(target)):
            return err
        return mailbox._accept("thief_swap", NightAction(target=target))

    return [speak, pass_turn, vote, seer_look, thief_swap]


@wrap_tool_call
async def coerce_tool_args(request, handler):
    """ローカルモデルが崩しがちな引数をスキーマ検証の前に直す(docs/research/local-deepagents.md 4.2 / 4.3)。

    - str 型の引数に dict / list が来たら JSON 文字列にする
    - 文字列の "None" / "null" は None にする
    """
    schema = getattr(request.tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None) if schema is not None else None
    if not fields:
        return await handler(request)
    args = dict(request.tool_call["args"])
    changed = False
    for name, value in args.items():
        ann = str(fields[name].annotation) if name in fields else ""
        if isinstance(value, (dict, list)) and ann == "<class 'str'>":
            args[name] = json.dumps(value, ensure_ascii=False, indent=2)
            changed = True
        elif isinstance(value, str) and value.strip() in ("None", "null") and "None" in ann:
            args[name] = None
            changed = True
    if changed:
        request = request.override(tool_call={**request.tool_call, "args": args})
    return await handler(request)


__all__ = ["CENTER", "ActionMailbox", "build_tools", "coerce_tool_args"]
