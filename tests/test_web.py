"""WebHumanPlayer のテスト: 要求の構造をブラウザに送り、1 行の入力を HumanPlayer と同じ書式で解析する。"""

from onenightagents.game.models import Role, SpeakRequest, Turn, VoteRequest
from onenightagents.web import WebHumanPlayer, WebUI


async def test_request_then_done_and_parse() -> None:
    ui = WebUI("p3")
    player = WebHumanPlayer("p3", ui)
    ui._inbox.put_nowait("@p1 CO:占い師 p1 は人狼でした")
    turn = Turn(speaker="p3", round=0, pass_allowed=False)
    action = await player.speak([], SpeakRequest(turn=turn, others=["p1", "p2"]))
    assert (action.addressed_to, action.claimed_role, action.content) == ("p1", Role.SEER, "p1 は人狼でした")
    assert [m["type"] for m in ui._history] == ["request", "done"]
    assert ui._history[0]["others"] == ["p1", "p2"]

    ui._inbox.put_nowait("p9")
    ui._inbox.put_nowait("p2 怪しい")
    vote = await player.vote([], VoteRequest(candidates=["p1", "p2"]))
    assert (vote.target, vote.reason) == ("p2", "怪しい")
    # 候補外の入力はブラウザにエラーを返し、入力欄は閉じない
    assert [m["type"] for m in ui._history[2:]] == ["request", "gm", "done"]


async def test_vote_skips_later_free_rounds() -> None:
    ui = WebUI("p3")
    player = WebHumanPlayer("p3", ui)
    ui._inbox.put_nowait("vote")
    free = SpeakRequest(turn=Turn(speaker="p3", round=1, pass_allowed=True), others=["p1"])
    assert (await player.speak([], free)).is_pass
    assert (await player.speak([], free)).is_pass
    # 2 回目は要求を送らずにパスする
    assert [m["type"] for m in ui._history] == ["request", "done"]
