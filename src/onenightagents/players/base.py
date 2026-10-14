"""Player プロトコルと、プレイヤーが GM に返す行動(docs/architecture.md 7)。"""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from onenightagents.game.models import NightRequest, Role, SpeakRequest, VoteRequest


class NightAction(BaseModel):
    target: str | None  # 占い師: プレイヤー ID か CENTER。怪盗: プレイヤー ID か None(交換しない)


class SpeechAction(BaseModel):
    content: str = ""
    addressed_to: str | None = None
    claimed_role: Role | None = None
    is_pass: bool = False


class VoteAction(BaseModel):
    target: str
    reason: str = ""


class Player(Protocol):
    id: str

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction: ...

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction: ...

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction: ...

    async def update_notes(self, messages: list[BaseMessage]) -> str | None:
        """推理メモ(/notes.md)の内容。メモを持たないプレイヤーは None。"""
        ...
