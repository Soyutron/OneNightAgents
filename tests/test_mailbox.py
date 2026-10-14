"""行動ツールの検証(フェーズ外・自分投票・無効対象)と取り出し。"""

from onenightagents.game.models import CENTER, Role
from onenightagents.players.base import NightAction, SpeechAction, VoteAction
from onenightagents.players.mailbox import ACCEPTED, ALREADY, ActionMailbox, build_tools

OTHERS = ["p2", "p3", "p4"]


def setup():
    mailbox = ActionMailbox("p1")
    tools = {t.name: t for t in build_tools(mailbox)}
    return mailbox, tools


def test_speak_accepts_once_and_is_idempotent() -> None:
    mailbox, tools = setup()
    mailbox.expect("speak", OTHERS, pass_allowed=True)
    assert tools["speak"].invoke({"content": "こんにちは", "addressed_to": "p2", "claimed_role": "村人"}) == ACCEPTED
    assert tools["speak"].invoke({"content": "二回目"}) == ALREADY
    assert tools["pass_turn"].invoke({}) == ALREADY
    action = mailbox.take()
    assert action == SpeechAction(content="こんにちは", addressed_to="p2", claimed_role=Role.VILLAGER)
    assert mailbox.take() is None


def test_speak_rejects_bad_target_and_role() -> None:
    mailbox, tools = setup()
    mailbox.expect("speak", OTHERS, pass_allowed=True)
    assert "無効" in tools["speak"].invoke({"content": "x", "addressed_to": "p1"})
    assert "無効" in tools["speak"].invoke({"content": "x", "claimed_role": "神"})
    assert mailbox.take() is None


def test_pass_turn_respects_pass_allowed() -> None:
    mailbox, tools = setup()
    mailbox.expect("speak", OTHERS, pass_allowed=False)
    assert "パスできません" in tools["pass_turn"].invoke({})
    assert mailbox.take() is None
    mailbox.expect("speak", OTHERS, pass_allowed=True)
    assert tools["pass_turn"].invoke({}) == ACCEPTED
    assert mailbox.take() == SpeechAction(is_pass=True)


def test_phase_mismatch_is_rejected() -> None:
    mailbox, tools = setup()
    mailbox.expect("vote", OTHERS)
    assert "いまは vote" in tools["speak"].invoke({"content": "x"})
    assert "いまは vote" in tools["seer_look"].invoke({"target": "p2"})
    assert tools["vote"].invoke({"target": "p3", "reason": "怪しい"}) == ACCEPTED
    assert mailbox.take() == VoteAction(target="p3", reason="怪しい")


def test_vote_rejects_self_and_unknown() -> None:
    mailbox, tools = setup()
    mailbox.expect("vote", OTHERS)
    assert "無効" in tools["vote"].invoke({"target": "p1"})
    assert "無効" in tools["vote"].invoke({"target": "p9"})
    assert mailbox.take() is None


def test_night_tools() -> None:
    mailbox, tools = setup()
    mailbox.expect("seer_look", [*OTHERS, CENTER])
    assert tools["seer_look"].invoke({"target": CENTER}) == ACCEPTED
    assert mailbox.take() == NightAction(target=CENTER)
    mailbox.expect("thief_swap", OTHERS)
    assert "無効" in tools["thief_swap"].invoke({"target": CENTER})
    assert tools["thief_swap"].invoke({}) == ACCEPTED
    assert mailbox.take() == NightAction(target=None)


def test_expect_clears_previous_action() -> None:
    mailbox, tools = setup()
    mailbox.expect("vote", OTHERS)
    tools["vote"].invoke({"target": "p2"})
    mailbox.expect("speak", OTHERS, pass_allowed=True)
    assert mailbox.take() is None
