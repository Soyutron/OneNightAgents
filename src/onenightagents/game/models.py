"""ゲームのデータモデル(LLM・LangGraph に依存しない)。"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

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
    llm_timeout_sec: float = 60.0
    model_call_limit: int = 4
    # 023: この席に人狼カードを配り、嘘の占い師 CO を出す scripted プレイヤーが担当する
    liar_seer: str | None = None

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
        if self.liar_seer is not None and self.liar_seer not in ids:
            raise ValueError(f"liar_seer must be a player id, got {self.liar_seer!r}")
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


# --- 議論の進行(docs/game-design.md 5.3) ---------------------------------------------


class Turn(BaseModel):
    """GM がプレイヤーに与える 1 回の発言機会。"""

    speaker: str
    round: int  # 0 はオープニング、1 以降は自由ラウンド
    pass_allowed: bool
    reply_to_turn: int | None = None  # 応答ターンなら、問われた発言の番号
    chain: int = 0  # 応答の連鎖の深さ(元の発言が 0)


class TurnCursor(BaseModel):
    """議論の進行位置。scheduler が読み書きする。"""

    round: int = 0
    seat_index: int = 0  # ラウンド内の席順で次に発言する位置
    pending_reply: Turn | None = None
    replied_this_round: list[str] = Field(default_factory=list)
    passes_this_round: int = 0
    finished: bool = False


# --- プレイヤーへの要求(候補と制約を明示し、できることを推測させない) -----------------------


class NightRequest(BaseModel):
    role: Role
    candidates: list[str]


class SpeakRequest(BaseModel):
    turn: Turn
    others: list[str]  # 宛先に選べるプレイヤー


class VoteRequest(BaseModel):
    candidates: list[str]


class NotesRequest(BaseModel):
    """夜明け後・投票前の推理メモ更新の要求。"""


# --- ゲーム状態(LangGraph の state。正解情報を含む) ------------------------------------


class GameState(TypedDict, total=False):
    config: GameConfig
    phase: str
    initial_roles: dict[str, Role]
    current_roles: dict[str, Role]
    center: list[Role]
    night: NightRecord
    utterances: Annotated[list[Utterance], operator.add]
    delivered_upto: dict[str, int]  # 各プレイヤーに送信済みの発言番号(差分送信用)
    votes: list[Vote]
    result: GameResult | None
    cursor: TurnCursor
