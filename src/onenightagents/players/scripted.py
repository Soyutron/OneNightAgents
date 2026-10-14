"""LLM を使わないプレイヤー(テストとベースライン)。"""

from __future__ import annotations

import random

from langchain_core.messages import BaseMessage

from onenightagents.game.models import NightRequest, Role, SpeakRequest, VoteRequest
from onenightagents.players.base import NightAction, SpeechAction, VoteAction


class RandomPlayer:
    """候補からランダムに選ぶ。seed 固定で決定的。"""

    def __init__(self, id: str, rng: random.Random) -> None:
        self.id = id
        self._rng = rng

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction:
        candidates: list[str | None] = list(request.candidates)
        if request.role is Role.THIEF:
            candidates.append(None)
        return NightAction(target=self._rng.choice(candidates))

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction:
        if request.turn.pass_allowed and self._rng.random() < 0.3:
            return SpeechAction(is_pass=True)
        addressed_to = self._rng.choice([None, *request.others])
        claimed_role = self._rng.choice([None, None, Role.VILLAGER, Role.SEER, Role.THIEF])
        return SpeechAction(content=f"{self.id} です。様子を見ます。", addressed_to=addressed_to, claimed_role=claimed_role)

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction:
        return VoteAction(target=self._rng.choice(request.candidates), reason="ランダム")

    async def update_notes(self, messages: list[BaseMessage]) -> str | None:
        return None
