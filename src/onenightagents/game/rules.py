"""配役・夜の解決・投票集計・勝敗判定・行動の検証(すべて純粋関数)。

ルールの根拠は docs/game-design.md 2.2 / 3.1 / 3.3。
"""

from __future__ import annotations

import random
from collections import Counter

from onenightagents.game.models import CENTER, MAX_PLAYERS, MIN_PLAYERS, GameResult, Role, Vote

# 固定の役職カード。残りは村人(docs/game-design.md 3.1)
FIXED_ROLES = [Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.THIEF]
CENTER_CARDS = 2


def role_deck(num_players: int) -> list[Role]:
    """人数 + 2 枚のカード構成を返す。"""
    if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
        raise ValueError(f"num_players must be {MIN_PLAYERS}..{MAX_PLAYERS}, got {num_players}")
    villagers = num_players + CENTER_CARDS - len(FIXED_ROLES)
    return FIXED_ROLES + [Role.VILLAGER] * villagers


def assign_roles(
    player_ids: list[str], rng: random.Random, werewolf: str | None = None
) -> tuple[dict[str, Role], list[Role]]:
    """カードをシャッフルして配り、(初期役職, 中央 2 枚) を返す。

    werewolf を指定すると、その席が人狼でなければ最初の人狼カードと入れ替える(023 の嘘 CO 実験用)。
    """
    deck = role_deck(len(player_ids))
    rng.shuffle(deck)
    if werewolf is not None:
        seat = player_ids.index(werewolf)
        if deck[seat] is not Role.WEREWOLF:
            wolf = deck.index(Role.WEREWOLF)
            deck[seat], deck[wolf] = deck[wolf], deck[seat]
    roles = dict(zip(player_ids, deck))
    return roles, deck[len(player_ids):]


# --- 夜 ---------------------------------------------------------------------


def resolve_seer(target: str, roles: dict[str, Role], center: list[Role]) -> dict[str, Role] | list[Role]:
    """占いの結果を返す。中央なら 2 枚のリスト、プレイヤーなら {id: 役職}。"""
    if target == CENTER:
        return list(center)
    return {target: roles[target]}


def werewolves(roles: dict[str, Role]) -> list[str]:
    """人狼カードを持つプレイヤー ID の一覧(人狼同士の確認用)。"""
    return [pid for pid, role in roles.items() if role is Role.WEREWOLF]


def resolve_thief(thief: str, target: str | None, roles: dict[str, Role]) -> tuple[dict[str, Role], Role | None]:
    """怪盗の交換を適用した新しい役職表と、怪盗が得た役職を返す。交換しなければ役職表はそのまま、結果は None。"""
    if target is None:
        return dict(roles), None
    new_roles = dict(roles)
    new_roles[thief], new_roles[target] = roles[target], roles[thief]
    return new_roles, new_roles[thief]


# --- 投票・判定 -----------------------------------------------------------------


def tally(votes: list[Vote]) -> list[str]:
    """処刑されるプレイヤーを返す。最多得票が複数なら全員、全員 1 票なら誰も処刑されない。"""
    counts = Counter(v.target for v in votes)
    if not counts:
        return []
    top = max(counts.values())
    if top == 1:
        return []
    return sorted(pid for pid, n in counts.items() if n == top)


def judge(executed: list[str], roles: dict[str, Role], center: list[Role]) -> GameResult:
    """終了時点の役職で勝敗を決める。人狼が場にいなければ平和村。"""
    if not werewolves(roles):
        winner = "village" if not executed else "none"
    elif any(roles[pid] is Role.WEREWOLF for pid in executed):
        winner = "village"
    else:
        winner = "werewolf"
    return GameResult(executed=executed, winner=winner, final_roles=dict(roles), center=list(center))


# --- 行動の検証とフォールバック(docs/game-design.md 3.3) ---------------------------------
# 検証関数は無効な理由を返し、有効なら None を返す。理由はそのまま再要求メッセージに使う。


def _other_player(actor: str, target: str, player_ids: list[str]) -> str | None:
    if target not in player_ids:
        return f"存在しないプレイヤーです: {target}"
    if target == actor:
        return "自分自身は選べません"
    return None


def validate_seer_target(actor: str, target: str, player_ids: list[str]) -> str | None:
    if target == CENTER:
        return None
    return _other_player(actor, target, player_ids)


def validate_thief_target(actor: str, target: str | None, player_ids: list[str]) -> str | None:
    if target is None:
        return None
    return _other_player(actor, target, player_ids)


def validate_vote(voter: str, target: str, player_ids: list[str]) -> str | None:
    return _other_player(voter, target, player_ids)


def validate_speech(
    speaker: str,
    content: str,
    addressed_to: str | None,
    is_pass: bool,
    player_ids: list[str],
    pass_allowed: bool,
) -> str | None:
    if is_pass:
        return None if pass_allowed else "このターンはパスできません"
    if not content.strip():
        return "発言が空です"
    if addressed_to is not None:
        return _other_player(speaker, addressed_to, player_ids)
    return None


def fallback_seer_target() -> str:
    return CENTER


def fallback_thief_target() -> None:
    return None


def fallback_vote(voter: str, player_ids: list[str], rng: random.Random) -> str:
    """有効な投票先から seed 固定で 1 人選ぶ。"""
    return rng.choice([pid for pid in player_ids if pid != voter])
