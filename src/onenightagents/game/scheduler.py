"""次の発言者の決定(docs/game-design.md 5.3 のラウンド制 + パス + 応答権)。

入札方式などに差し替えるときはこのモジュールだけを置き換える。
"""

from __future__ import annotations

from onenightagents.game.models import GameConfig, Turn, TurnCursor, Utterance


def seat_order(player_ids: list[str], round: int) -> list[str]:
    """ラウンドごとに開始位置をずらした席順。"""
    k = round % len(player_ids)
    return player_ids[k:] + player_ids[:k]


def next_turn(cursor: TurnCursor, config: GameConfig) -> Turn | None:
    """次の発言機会。議論が終わっていれば None。"""
    if cursor.finished:
        return None
    if cursor.pending_reply is not None:
        return cursor.pending_reply
    speaker = seat_order(config.player_ids, cursor.round)[cursor.seat_index]
    return Turn(speaker=speaker, round=cursor.round, pass_allowed=cursor.round > 0)


def advance(cursor: TurnCursor, turn: Turn, utterance: Utterance, config: GameConfig) -> TurnCursor:
    """発言を受けて進行位置を進める。応答権の挿入、ラウンド送り、終了判定を行う。"""
    c = cursor.model_copy(deep=True)
    c.pending_reply = None
    if turn.reply_to_turn is None:
        c.seat_index += 1
        if utterance.is_pass:
            c.passes_this_round += 1

    target = utterance.addressed_to
    if (
        not utterance.is_pass
        and target is not None
        and turn.chain < config.reply_chain_limit
        and target not in c.replied_this_round
    ):
        c.pending_reply = Turn(
            speaker=target,
            round=c.round,
            pass_allowed=c.round > 0,
            reply_to_turn=utterance.turn,
            chain=turn.chain + 1,
        )
        c.replied_this_round.append(target)

    n = len(config.players)
    if c.pending_reply is None and c.seat_index >= n:
        all_passed = c.round > 0 and c.passes_this_round == n
        if all_passed or c.round >= config.max_free_rounds:
            c.finished = True
        else:
            c.round += 1
            c.seat_index = 0
            c.replied_this_round = []
            c.passes_this_round = 0
    return c
