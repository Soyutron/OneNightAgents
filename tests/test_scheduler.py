"""スケジューラのテスト: ラウンド・応答権・連鎖上限・終了条件。"""

from onenightagents.game import scheduler
from onenightagents.game.models import GameConfig, PlayerConfig, Turn, TurnCursor, Utterance


def config(n: int = 4, max_free_rounds: int = 2, reply_chain_limit: int = 2) -> GameConfig:
    players = [PlayerConfig(id=f"p{i + 1}", display_name=f"P{i + 1}") for i in range(n)]
    return GameConfig(
        game_id="g", seed=0, players=players, max_free_rounds=max_free_rounds, reply_chain_limit=reply_chain_limit
    )


class Sim:
    """next_turn → 発言 → advance を繰り返す補助。"""

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg
        self.cursor = TurnCursor()
        self.utterances: list[Utterance] = []

    def next(self) -> Turn | None:
        return scheduler.next_turn(self.cursor, self.cfg)

    def say(self, addressed_to: str | None = None, is_pass: bool = False) -> Turn:
        turn = self.next()
        assert turn is not None
        u = Utterance(
            turn=len(self.utterances),
            round=turn.round,
            speaker=turn.speaker,
            content="" if is_pass else "...",
            addressed_to=addressed_to,
            is_pass=is_pass,
            reply_to_turn=turn.reply_to_turn,
        )
        self.utterances.append(u)
        self.cursor = scheduler.advance(self.cursor, turn, u, self.cfg)
        return turn


def test_seat_order_rotates_per_round() -> None:
    ids = ["p1", "p2", "p3", "p4"]
    assert scheduler.seat_order(ids, 0) == ids
    assert scheduler.seat_order(ids, 1) == ["p2", "p3", "p4", "p1"]
    assert scheduler.seat_order(ids, 4) == ids


def test_opening_round_no_pass_then_free_rounds_allow_pass() -> None:
    sim = Sim(config())
    for pid in ["p1", "p2", "p3", "p4"]:
        turn = sim.say()
        assert (turn.speaker, turn.round, turn.pass_allowed) == (pid, 0, False)
    turn = sim.next()
    assert turn is not None
    assert (turn.speaker, turn.round, turn.pass_allowed) == ("p2", 1, True)


def test_reply_turn_is_inserted_and_chain_is_limited() -> None:
    sim = Sim(config())
    sim.say(addressed_to="p3")  # p1 → p3 (chain 0)
    reply = sim.say(addressed_to="p1")  # p3 の応答 (chain 1)
    assert (reply.speaker, reply.reply_to_turn, reply.chain) == ("p3", 0, 1)
    reply2 = sim.say(addressed_to="p3")  # p1 の応答 (chain 2)。これ以上は連鎖しない
    assert (reply2.speaker, reply2.reply_to_turn, reply2.chain) == ("p1", 1, 2)
    nxt = sim.next()
    assert nxt is not None
    assert (nxt.speaker, nxt.reply_to_turn) == ("p2", None)


def test_one_reply_turn_per_player_per_round() -> None:
    sim = Sim(config())
    sim.say(addressed_to="p4")  # p1 → p4
    sim.say()  # p4 の応答
    sim.say(addressed_to="p4")  # p2 → p4: p4 はこのラウンドで応答済み
    nxt = sim.next()
    assert nxt is not None
    assert (nxt.speaker, nxt.reply_to_turn) == ("p3", None)


def test_pass_does_not_trigger_reply() -> None:
    sim = Sim(config())
    for _ in range(4):
        sim.say()
    sim.say(addressed_to="p1", is_pass=True)
    nxt = sim.next()
    assert nxt is not None and nxt.reply_to_turn is None


def test_ends_after_max_free_rounds() -> None:
    sim = Sim(config(n=3, max_free_rounds=1))
    for _ in range(3 + 3):
        sim.say()
    assert sim.next() is None


def test_ends_when_everyone_passes_in_a_free_round() -> None:
    sim = Sim(config(n=3, max_free_rounds=5))
    for _ in range(3):
        sim.say()
    for _ in range(3):
        sim.say(is_pass=True)
    assert sim.next() is None


def test_partial_pass_continues() -> None:
    sim = Sim(config(n=3, max_free_rounds=5))
    for _ in range(3):
        sim.say()
    sim.say(is_pass=True)
    sim.say()
    sim.say(is_pass=True)
    nxt = sim.next()
    assert nxt is not None and nxt.round == 2


def test_pending_reply_is_served_before_round_rollover() -> None:
    sim = Sim(config(n=3, max_free_rounds=1))
    sim.say()
    sim.say()
    sim.say(addressed_to="p1")  # ラウンド末尾の発言に宛先
    reply = sim.next()
    assert reply is not None and (reply.speaker, reply.round, reply.reply_to_turn) == ("p1", 0, 2)
    sim.say()
    nxt = sim.next()
    assert nxt is not None and (nxt.speaker, nxt.round) == ("p2", 1)
