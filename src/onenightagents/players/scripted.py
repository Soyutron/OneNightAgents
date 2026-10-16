"""LLM を使わないプレイヤー(テストとベースライン)。"""

from __future__ import annotations

import random
import re

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


class LiarSeerPlayer:
    """人狼として占い師を騙り、仲間でない 1 人を人狼だと言い続ける(023: 固定ルールの搾取の検証)。

    夜明けの通知から仲間の人狼を読み取り、それ以外からランダムに標的を選ぶ。議論の内容には反応しない。
    """

    PARTNER = re.compile(r"仲間の人狼: (\S+?)。")

    def __init__(self, id: str, rng: random.Random) -> None:
        self.id = id
        self._rng = rng
        self.partner: str | None = None
        self.target: str | None = None

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction:
        # 人狼は夜に呼ばれない。怪盗に奪われても初期役職で呼ばれるので到達しない
        return NightAction(target=None)

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction:
        if self.target is None:
            self.target = self._rng.choice([p for p in request.others if p != self.partner])
        t = self.target
        turn = request.turn
        if turn.reply_to_turn is not None:
            content = f"お答えします。私は占い師で、占ったのは{t}さんだけです。結果は人狼でした。"
        elif turn.round == 0:
            content = f"私は占い師です。{t}さんを占ったところ、人狼でした。"
        else:
            content = f"改めて言います。私は占い師で、{t}さんは人狼です。{t}さんに投票しましょう。"
        return SpeechAction(content=content, claimed_role=Role.SEER)

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction:
        return VoteAction(target=self.target or self._rng.choice(request.candidates), reason="占いの結果、人狼だったため。")

    async def update_notes(self, messages: list[BaseMessage]) -> str | None:
        # 夜明けの通知(messages.py の _dawn)だけが仲間を知る手段
        for m in messages:
            found = self.PARTNER.search(str(m.content))
            if found:
                self.partner = found.group(1)
        return None
