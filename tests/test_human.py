"""HumanPlayer のテスト: 入力の解析、候補との照合のやり直し、早送り、時間切れのフォールバック、人間向けの要求文。"""

import asyncio
import random
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from onenightagents.cli import RECURSION_LIMIT, make_config
from onenightagents.engine.context import GameContext
from onenightagents.engine.graph import build_graph
from onenightagents.engine.recorder import Recorder
from onenightagents.game.messages import build_message
from onenightagents.game.models import CENTER, NightRequest, Role, SpeakRequest, Turn, VoteRequest
from onenightagents.players.human import HumanPlayer, parse_speech
from onenightagents.players.scripted import RandomPlayer


class Scripted:
    """台本どおりに 1 行ずつ返す Reader。"""

    def __init__(self, lines: list[str]) -> None:
        self._it = iter(lines)
        self.discarded = 0

    async def read(self) -> str:
        return next(self._it)

    def discard(self) -> None:
        self.discarded += 1


def scripted(lines: list[str]) -> Scripted:
    return Scripted(lines)


def test_parse_speech() -> None:
    others = ["p1", "p2"]
    action, err = parse_speech("@p2 CO:占い師 p1 を占ったら人狼でした", others)
    assert err is None and action is not None
    assert (action.addressed_to, action.claimed_role, action.content) == ("p2", Role.SEER, "p1 を占ったら人狼でした")
    action, err = parse_speech("co:villager こんにちは", others)
    assert action is not None and action.claimed_role is Role.VILLAGER and action.addressed_to is None
    assert parse_speech("@p9 こんにちは", others) == (None, "宛先は p1, p2 から選んでください")
    assert parse_speech("CO:神 こんにちは", others)[0] is None
    assert parse_speech("@p1", others) == (None, "発言を入力してください")


async def test_night_and_vote_retry_until_valid() -> None:
    out: list[str] = []
    seer = HumanPlayer("p1", reader=scripted(["p9", CENTER]), write=out.append)
    action = await seer.act_at_night([HumanMessage("夜")], NightRequest(role=Role.SEER, candidates=["p2", CENTER], hint="p2"))
    assert action.target == CENTER and any("候補の中から" in o for o in out)
    thief = HumanPlayer("p3", reader=scripted(["none"]), write=out.append)
    assert (await thief.act_at_night([], NightRequest(role=Role.THIEF, candidates=["p1"], hint="p1"))).target is None
    voter = HumanPlayer("p1", reader=scripted(["p1 自分", "p2  占い師 CO が矛盾"]), write=out.append)
    vote = await voter.vote([], VoteRequest(candidates=["p2", "p3"]))
    assert (vote.target, vote.reason) == ("p2", "占い師 CO が矛盾")


async def test_speak_pass_rules_and_fast_forward() -> None:
    out: list[str] = []
    human = HumanPlayer("p1", reader=scripted(["pass", "こんにちは", "vote"]), write=out.append)
    opening = SpeakRequest(turn=Turn(speaker="p1", round=0, pass_allowed=False), others=["p2"])
    free = SpeakRequest(turn=Turn(speaker="p1", round=1, pass_allowed=True), others=["p2"])
    # オープニングはパスできないので、やり直して発言になる
    assert (await human.speak([], opening)).content == "こんにちは"
    assert any("パスできません" in o for o in out)
    # vote で以後の自由ラウンドは自動でパス(読み取りは起きない)
    assert (await human.speak([], free)).is_pass
    assert (await human.speak([], free)).is_pass and "(自動でパス)" in out


async def test_human_timeout_falls_back_with_human_timeout(tmp_path: Path) -> None:
    """人間の席は human_timeout_sec で打ち切られ、AI の llm_timeout_sec は使われない。"""

    class Never(Scripted):
        async def read(self) -> str:
            await asyncio.Event().wait()
            return ""

    never = Never([])
    config = make_config("g", 3, 4, 1, human="p2", llm_timeout_sec=60.0, human_timeout_sec=0.05)
    assert [p.kind for p in config.players] == ["scripted", "human", "scripted", "scripted"]
    players = {p.id: RandomPlayer(p.id, random.Random(i)) for i, p in enumerate(config.players)}
    out: list[str] = []
    players["p2"] = HumanPlayer("p2", reader=never, write=out.append, timeout_sec=0.05)
    ctx = GameContext(players=players, recorder=Recorder(tmp_path), rng=random.Random(3))
    final = await asyncio.wait_for(
        build_graph().ainvoke({"config": config}, config={"recursion_limit": RECURSION_LIMIT}, context=ctx), 20
    )
    assert final["phase"] == "end"
    skipped = [e for e in (ln for ln in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()) if '"turn_skipped"' in e]
    assert skipped and all('"p2"' in e and "制限時間" in e for e in skipped)
    # 人間には GM のメッセージが表示されている(配役の案内を含む)
    assert any("配られた役職" in o for o in out)
    # 時間切れの次の要求では、遅れて届いた答えを捨てている(最初の要求では捨てない)
    # 1 判断につき初回 + 再要求の 2 回読むので、捨てる回数は読んだ回数 - 1
    assert never.discarded == 2 * len(skipped) - 1


def test_human_request_text_has_no_tool_names() -> None:
    config = make_config("g", 1, 4, 1, human="p1")
    state = {
        "config": config,
        "initial_roles": {"p1": Role.SEER, "p2": Role.VILLAGER, "p3": Role.WEREWOLF, "p4": Role.THIEF},
        "utterances": [],
        "delivered_upto": {},
    }
    for request in (
        NightRequest(role=Role.SEER, candidates=["p2", "p3", "p4", CENTER], hint=CENTER),
        SpeakRequest(turn=Turn(speaker="p1", round=1, pass_allowed=True), others=["p2", "p3", "p4"]),
        VoteRequest(candidates=["p2", "p3", "p4"]),
    ):
        text = build_message(state, "p1", request, human=True)
        assert not any(tool in text for tool in ("seer_look", "speak で", "pass_turn", "vote で", "/notes.md"))
    assert "pass でパス" in build_message(state, "p1", SpeakRequest(turn=Turn(speaker="p1", round=1, pass_allowed=True), others=["p2"]), human=True)


async def test_stdin_reader_keeps_typed_ahead_lines_and_can_discard(monkeypatch) -> None:
    import io

    from onenightagents.players.human import StdinReader

    monkeypatch.setattr("sys.stdin", io.StringIO("先に打った行\n答え\n古い答え\n"))
    reader = StdinReader()
    assert await reader.read() == "先に打った行"
    await asyncio.sleep(0.05)  # 残りの行がキューに入るのを待つ
    assert await reader.read() == "答え"  # 先に打った行は残る
    reader.discard()
    with pytest.raises(asyncio.TimeoutError):  # 捨てた後は入力が尽きるので待ち続ける
        await asyncio.wait_for(reader.read(), 0.1)


async def test_reveal_game_prints_roles_and_notes(tmp_path: Path, capsys) -> None:
    from onenightagents.cli import reveal_game

    config = make_config("g", 5, 4, 1)
    players = {p.id: RandomPlayer(p.id, random.Random(i)) for i, p in enumerate(config.players)}
    recorder = Recorder(tmp_path)
    ctx = GameContext(players=players, recorder=recorder, rng=random.Random(5))
    await build_graph().ainvoke({"config": config}, config={"recursion_limit": RECURSION_LIMIT}, context=ctx)
    recorder.emit("notes_updated", "secret", player="p2", notes="p1 が怪しい")
    reveal_game(tmp_path)
    out = capsys.readouterr().out
    assert "種明かし" in out and "投票 p1(" in out and "p2 の最後の推理メモ" in out and "p1 が怪しい" in out
