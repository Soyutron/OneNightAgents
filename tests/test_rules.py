"""ルール層の単体テスト(docs/architecture.md 12 節: 同票・全員 1 票・平和村・怪盗交換・無効行動)。"""

import random

import pytest
from pydantic import ValidationError

from onenightagents.game import rules
from onenightagents.game.models import CENTER, GameConfig, PlayerConfig, Role, Vote

P5 = ["p1", "p2", "p3", "p4", "p5"]


def roles_of(*roles: Role) -> dict[str, Role]:
    return dict(zip(P5, roles))


# --- 配役 ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_role_deck_composition(n: int) -> None:
    deck = rules.role_deck(n)
    assert len(deck) == n + 2
    assert deck.count(Role.WEREWOLF) == 2
    assert deck.count(Role.SEER) == 1
    assert deck.count(Role.THIEF) == 1
    assert deck.count(Role.VILLAGER) == n - 2


@pytest.mark.parametrize("n", [2, 8])
def test_role_deck_rejects_out_of_range(n: int) -> None:
    with pytest.raises(ValueError):
        rules.role_deck(n)


def test_assign_roles_is_seeded_and_complete() -> None:
    a = rules.assign_roles(P5, random.Random(42))
    b = rules.assign_roles(P5, random.Random(42))
    assert a == b
    roles, center = a
    assert list(roles) == P5
    assert len(center) == 2
    assert sorted(list(roles.values()) + center) == sorted(rules.role_deck(5))


def test_game_config_validates_players() -> None:
    players = [PlayerConfig(id=f"p{i}", display_name=f"P{i}") for i in range(1, 3)]
    with pytest.raises(ValidationError):
        GameConfig(game_id="g", seed=1, players=players)
    dup = [PlayerConfig(id="p1", display_name="A")] * 3
    with pytest.raises(ValidationError):
        GameConfig(game_id="g", seed=1, players=dup)
    reserved = [PlayerConfig(id=CENTER, display_name="C")] + [
        PlayerConfig(id=f"p{i}", display_name=f"P{i}") for i in range(1, 3)
    ]
    with pytest.raises(ValidationError):
        GameConfig(game_id="g", seed=1, players=reserved)


# --- 夜 ----------------------------------------------------------------------


def test_resolve_seer_player_and_center() -> None:
    roles = roles_of(Role.SEER, Role.WEREWOLF, Role.VILLAGER, Role.THIEF, Role.VILLAGER)
    center = [Role.WEREWOLF, Role.VILLAGER]
    assert rules.resolve_seer("p2", roles, center) == {"p2": Role.WEREWOLF}
    assert rules.resolve_seer(CENTER, roles, center) == center


def test_werewolves_lists_only_wolves() -> None:
    roles = roles_of(Role.WEREWOLF, Role.VILLAGER, Role.WEREWOLF, Role.SEER, Role.THIEF)
    assert rules.werewolves(roles) == ["p1", "p3"]
    assert rules.werewolves(roles_of(*[Role.VILLAGER] * 5)) == []


def test_resolve_thief_swaps_without_mutating_input() -> None:
    roles = roles_of(Role.THIEF, Role.WEREWOLF, Role.VILLAGER, Role.SEER, Role.VILLAGER)
    new_roles, got = rules.resolve_thief("p1", "p2", roles)
    assert got is Role.WEREWOLF
    assert new_roles["p1"] is Role.WEREWOLF
    assert new_roles["p2"] is Role.THIEF
    assert roles["p1"] is Role.THIEF  # 入力は変更しない


def test_resolve_thief_no_swap() -> None:
    roles = roles_of(Role.THIEF, Role.WEREWOLF, Role.VILLAGER, Role.SEER, Role.VILLAGER)
    new_roles, got = rules.resolve_thief("p1", None, roles)
    assert new_roles == roles
    assert got is None


# --- 集計 ---------------------------------------------------------------------


def votes(*targets: str) -> list[Vote]:
    return [Vote(voter=v, target=t) for v, t in zip(P5, targets)]


def test_tally_majority() -> None:
    assert rules.tally(votes("p2", "p2", "p2", "p1", "p1")) == ["p2"]


def test_tally_tie_executes_all() -> None:
    assert rules.tally(votes("p2", "p2", "p3", "p3", "p1")) == ["p2", "p3"]


def test_tally_everyone_one_vote_executes_nobody() -> None:
    assert rules.tally(votes("p2", "p3", "p4", "p5", "p1")) == []


def test_tally_empty() -> None:
    assert rules.tally([]) == []


# --- 勝敗 ---------------------------------------------------------------------

CENTER_VILLAGE = [Role.VILLAGER, Role.VILLAGER]


def test_judge_village_wins_when_wolf_executed() -> None:
    roles = roles_of(Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.THIEF, Role.VILLAGER)
    result = rules.judge(["p1"], roles, CENTER_VILLAGE)
    assert result.winner == "village"
    assert result.executed == ["p1"]
    assert result.final_roles == roles


def test_judge_werewolf_wins_when_villager_executed() -> None:
    roles = roles_of(Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.THIEF, Role.VILLAGER)
    assert rules.judge(["p5"], roles, CENTER_VILLAGE).winner == "werewolf"


def test_judge_werewolf_wins_when_nobody_executed() -> None:
    roles = roles_of(Role.WEREWOLF, Role.VILLAGER, Role.SEER, Role.THIEF, Role.VILLAGER)
    assert rules.judge([], roles, [Role.WEREWOLF, Role.VILLAGER]).winner == "werewolf"


def test_judge_tie_with_a_wolf_is_village_win() -> None:
    roles = roles_of(Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.THIEF, Role.VILLAGER)
    assert rules.judge(["p1", "p5"], roles, CENTER_VILLAGE).winner == "village"


def test_judge_peaceful_village() -> None:
    roles = roles_of(Role.VILLAGER, Role.VILLAGER, Role.SEER, Role.THIEF, Role.VILLAGER)
    center = [Role.WEREWOLF, Role.WEREWOLF]
    assert rules.judge([], roles, center).winner == "village"
    assert rules.judge(["p3"], roles, center).winner == "none"


def test_judge_uses_roles_after_thief_swap() -> None:
    # 怪盗が人狼を取って処刑された: 人狼が処刑されたので村人陣営の勝ち
    initial = roles_of(Role.THIEF, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER)
    roles, _ = rules.resolve_thief("p1", "p2", initial)
    assert rules.judge(["p1"], roles, CENTER_VILLAGE).winner == "village"
    # 元の人狼(いまは怪盗)が処刑されても人狼陣営の勝ち
    assert rules.judge(["p2"], roles, CENTER_VILLAGE).winner == "werewolf"


# --- 検証とフォールバック ------------------------------------------------------------


def test_validate_seer_target() -> None:
    assert rules.validate_seer_target("p1", CENTER, P5) is None
    assert rules.validate_seer_target("p1", "p2", P5) is None
    assert rules.validate_seer_target("p1", "p1", P5) is not None
    assert rules.validate_seer_target("p1", "p9", P5) is not None


def test_validate_thief_target() -> None:
    assert rules.validate_thief_target("p1", None, P5) is None
    assert rules.validate_thief_target("p1", "p2", P5) is None
    assert rules.validate_thief_target("p1", "p1", P5) is not None
    assert rules.validate_thief_target("p1", CENTER, P5) is not None


def test_validate_vote() -> None:
    assert rules.validate_vote("p1", "p2", P5) is None
    assert rules.validate_vote("p1", "p1", P5) is not None
    assert rules.validate_vote("p1", "p9", P5) is not None


def test_validate_speech() -> None:
    assert rules.validate_speech("p1", "こんにちは", None, False, P5, pass_allowed=False) is None
    assert rules.validate_speech("p1", "質問です", "p2", False, P5, pass_allowed=False) is None
    assert rules.validate_speech("p1", "", None, False, P5, pass_allowed=True) is not None
    assert rules.validate_speech("p1", "自問", "p1", False, P5, pass_allowed=True) is not None
    assert rules.validate_speech("p1", "誰?", "p9", False, P5, pass_allowed=True) is not None
    assert rules.validate_speech("p1", "", None, True, P5, pass_allowed=True) is None
    assert rules.validate_speech("p1", "", None, True, P5, pass_allowed=False) is not None


def test_fallbacks_are_deterministic() -> None:
    assert rules.fallback_seer_target() == CENTER
    assert rules.fallback_thief_target() is None
    a = rules.fallback_vote("p1", P5, random.Random(3))
    b = rules.fallback_vote("p1", P5, random.Random(3))
    assert a == b
    assert a in P5 and a != "p1"
