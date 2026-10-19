"""情報隔離のテスト: 全フェーズ・全役職で、他者の秘密がメッセージに含まれない。"""

import pytest

from onenightagents.game import rules
from onenightagents.game.messages import ROLE_NAMES, build_message
from onenightagents.game.models import (
    CENTER,
    GameConfig,
    GameState,
    NightRecord,
    NightRequest,
    NotesRequest,
    PlayerConfig,
    Role,
    SpeakRequest,
    Turn,
    Utterance,
    Vote,
    VoteRequest,
)

IDS = ["p1", "p2", "p3", "p4", "p5"]
INITIAL = dict(zip(IDS, [Role.SEER, Role.THIEF, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER]))


def make_state(seer_target: str = "p3") -> GameState:
    config = GameConfig(game_id="g", seed=0, players=[PlayerConfig(id=i, display_name=f"名前{i}") for i in IDS])
    center = [Role.WEREWOLF, Role.VILLAGER]
    seer_result = rules.resolve_seer(seer_target, INITIAL, center)
    current, new_role = rules.resolve_thief("p2", "p3", INITIAL)  # 怪盗が人狼を取る
    night = NightRecord(
        seer_target=seer_target,
        seer_result=seer_result,
        werewolves=rules.werewolves(INITIAL),
        thief_target="p3",
        thief_new_role=new_role,
    )
    utterances = [
        Utterance(turn=0, round=0, speaker="p1", content="私は占い師です", claimed_role=Role.SEER),
        Utterance(turn=1, round=0, speaker="p2", content="p3 はどう?", addressed_to="p3"),
        Utterance(turn=2, round=0, speaker="p3", content="", is_pass=True, reply_to_turn=1),
    ]
    votes = [Vote(voter=i, target="p3", reason=f"{i} の秘密の理由") for i in IDS]
    return GameState(
        config=config,
        phase="day",
        initial_roles=INITIAL,
        current_roles=current,
        center=center,
        night=night,
        utterances=utterances,
        delivered_upto={},
        votes=votes,
        result=None,
    )


def all_requests(pid: str):
    others = [i for i in IDS if i != pid]
    return [
        NightRequest(role=Role.SEER, candidates=[*others, CENTER], hint=CENTER),
        NightRequest(role=Role.THIEF, candidates=others, hint=others[0]),
        NotesRequest(),
        SpeakRequest(turn=Turn(speaker=pid, round=1, pass_allowed=True), others=others),
        SpeakRequest(turn=Turn(speaker=pid, round=0, pass_allowed=False, reply_to_turn=1, chain=1), others=others),
        VoteRequest(candidates=others),
    ]


@pytest.mark.parametrize("pid", IDS)
def test_no_other_players_secrets(pid: str) -> None:
    state = make_state()
    role = INITIAL[pid]
    for request in all_requests(pid):
        text = build_message(state, pid, request)
        for other in IDS:
            if other == pid:
                continue
            # 占い結果以外で他者の役職を書かない
            if not (role is Role.SEER and other == "p3"):
                assert f"{other} の役職は" not in text
            assert f"{other} の秘密の理由" not in text
        if role is not Role.SEER:
            assert "占いの結果" not in text
            assert "中央の 2 枚は" not in text
        if role is not Role.WEREWOLF:
            assert "仲間の人狼" not in text and "一匹狼" not in text
        if role is not Role.THIEF:
            assert "交換しました" not in text and "現在の役職" not in text
        # 配られた役職は本人のものだけ
        for r, name in ROLE_NAMES.items():
            if r is not role:
                assert f"配られた役職は「{name}」" not in text
        assert "p3 の秘密" not in text


def test_seer_sees_only_own_result() -> None:
    state = make_state()
    text = build_message(state, "p1", NotesRequest())
    assert "p3 の役職は「人狼」" in text
    state_center = make_state(seer_target=CENTER)
    text = build_message(state_center, "p1", NotesRequest())
    assert "中央の 2 枚は「人狼」「村人」" in text
    assert "p3 の役職は" not in text


def test_werewolf_and_thief_dawn_messages() -> None:
    state = make_state()
    assert "一匹狼" in build_message(state, "p3", NotesRequest())
    text = build_message(state, "p2", NotesRequest())
    assert "p3 と交換しました。あなたの現在の役職は「人狼」" in text


def test_intro_only_in_first_message() -> None:
    state = make_state()
    assert "ワンナイト人狼を始めます" in build_message(state, "p1", NightRequest(role=Role.SEER, candidates=[CENTER], hint=CENTER))
    assert "ワンナイト人狼を始めます" not in build_message(state, "p1", NotesRequest())
    assert "ワンナイト人狼を始めます" in build_message(state, "p4", NotesRequest())
    assert "ワンナイト人狼を始めます" not in build_message(state, "p4", VoteRequest(candidates=["p1"]))


def test_public_diff_uses_delivered_upto() -> None:
    state = make_state()
    state["delivered_upto"] = {"p4": 1}
    text = build_message(state, "p4", VoteRequest(candidates=["p1"]))
    assert "#0 p1" not in text
    assert "#1 p2: 「p3 はどう?」(宛先: p3)" in text
    assert "#2 p3: パス" in text


def test_reply_turn_mentions_asker() -> None:
    state = make_state()
    request = SpeakRequest(turn=Turn(speaker="p3", round=0, pass_allowed=False, reply_to_turn=1, chain=1), others=["p1"])
    text = build_message(state, "p3", request)
    assert "p2 からの発言 #1 で問われています" in text
    assert "パスできません" in text
