"""ゲームのデータモデル(LLM・LangGraph に依存しない)。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# 占い師が中央カードを見るときの対象 ID
CENTER = "center"

# 日本版の人数範囲(docs/game-design.md 3.1)
MIN_PLAYERS = 3
MAX_PLAYERS = 7


class Team(StrEnum):
    VILLAGE = "village"
    WEREWOLF = "werewolf"


class Role(StrEnum):
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    THIEF = "thief"

    @property
    def team(self) -> Team:
        return Team.WEREWOLF if self is Role.WEREWOLF else Team.VILLAGE


class PlayerConfig(BaseModel):
    id: str
    display_name: str
    kind: Literal["deep", "simple", "scripted", "human"] = "deep"
    model: str | None = None
    persona: str | None = None


class GameConfig(BaseModel):
    game_id: str
    seed: int
    players: list[PlayerConfig]
    max_free_rounds: int = 2
    reply_chain_limit: int = 2
    llm_timeout_sec: float = 30.0
    model_call_limit: int = 4

    @model_validator(mode="after")
    def _check_players(self) -> GameConfig:
        n = len(self.players)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:
            raise ValueError(f"players must be {MIN_PLAYERS}..{MAX_PLAYERS}, got {n}")
        ids = [p.id for p in self.players]
        if len(set(ids)) != n:
            raise ValueError("player ids must be unique")
        if CENTER in ids:
            raise ValueError(f"player id {CENTER!r} is reserved")
        return self

    @property
    def player_ids(self) -> list[str]:
        return [p.id for p in self.players]


class NightRecord(BaseModel):
    seer_target: str | None = None  # プレイヤー ID または CENTER
    seer_result: dict[str, Role] | list[Role] | None = None
    werewolves: list[str] = Field(default_factory=list)
    thief_target: str | None = None
    thief_new_role: Role | None = None


class Utterance(BaseModel):
    turn: int
    round: int
    speaker: str
    content: str
    addressed_to: str | None = None
    claimed_role: Role | None = None
    is_pass: bool = False
    reply_to_turn: int | None = None


class Vote(BaseModel):
    voter: str
    target: str
    reason: str = ""


class GameResult(BaseModel):
    executed: list[str]
    winner: Literal["village", "werewolf", "none"]
    final_roles: dict[str, Role]
    center: list[Role]
