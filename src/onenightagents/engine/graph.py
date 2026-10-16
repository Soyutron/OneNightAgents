"""ゲーム進行の StateGraph(docs/architecture.md 4.3)。

各ノードは「メッセージを作る → プレイヤーを呼ぶ → 検証・適用 → イベント emit」の定型。
無効な行動は 1 回だけ再要求し、再び無効なら決定的フォールバック(docs/game-design.md 3.3)。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from onenightagents.engine.context import GameContext
from onenightagents.game import rules, scheduler
from onenightagents.game.messages import Request, build_message
from onenightagents.game.models import (
    CENTER,
    GameState,
    NightRecord,
    NightRequest,
    NotesRequest,
    Role,
    SpeakRequest,
    TurnCursor,
    Utterance,
    Vote,
    VoteRequest,
)
from onenightagents.players.base import ActionNotSubmitted, NightAction, SpeechAction, VoteAction

A = TypeVar("A", NightAction, SpeechAction, VoteAction)


def _player_with_initial_role(state: GameState, role: Role) -> str | None:
    return next((pid for pid, r in state["initial_roles"].items() if r is role), None)


def _others(state: GameState, pid: str) -> list[str]:
    return [p for p in state["config"].player_ids if p != pid]


def _delivered(state: GameState, pid: str, upto: int) -> dict[str, int]:
    return {**state.get("delivered_upto", {}), pid: upto}


async def _request_action(
    state: GameState,
    runtime: Runtime[GameContext],
    pid: str,
    request: Request,
    call: Callable[[list[HumanMessage]], Awaitable[A]],
    validate: Callable[[A], str | None],
    fallback: Callable[[], A],
) -> tuple[A, bool]:
    """プレイヤーに行動を求める。戻り値は (行動, フォールバックしたか)。"""
    ctx = runtime.context
    timeout = state["config"].llm_timeout_sec
    text = build_message(state, pid, request)
    error: str | None = None
    for attempt in range(2):
        if attempt == 1:
            text = f"前回の行動は無効でした: {error}。もう一度、正しい行動を提出してください。"
        try:
            action = await asyncio.wait_for(call([HumanMessage(text)]), timeout)
        except TimeoutError:
            error = "制限時間内に行動が提出されませんでした"
            continue
        except ActionNotSubmitted:
            error = "ツールで行動が提出されませんでした。必ず対応するツールを呼んで提出してください"
            continue
        error = validate(action)
        if error is None:
            return action, False
    ctx.recorder.emit("turn_skipped", "public", player=pid, reason=error)
    return fallback(), True


# --- ノード -----------------------------------------------------------------------


async def setup(state: GameState, runtime: Runtime[GameContext]) -> dict:
    config = state["config"]
    ctx = runtime.context
    ctx.recorder.write_config(config)
    ctx.recorder.emit("game_started", "public", config=config.model_dump())
    roles, center = rules.assign_roles(config.player_ids, ctx.rng, werewolf=config.liar_seer)
    ctx.recorder.emit("roles_assigned", "secret", initial_roles=roles, center=center)
    return {
        "phase": "night",
        "initial_roles": roles,
        "current_roles": dict(roles),
        "center": center,
        "night": NightRecord(werewolves=rules.werewolves(roles)),
        "delivered_upto": {},
        "votes": [],
        "result": None,
        "cursor": TurnCursor(),
    }


async def night_seer(state: GameState, runtime: Runtime[GameContext]) -> dict:
    pid = _player_with_initial_role(state, Role.SEER)
    if pid is None:
        return {}
    ids = state["config"].player_ids
    request = NightRequest(role=Role.SEER, candidates=[*_others(state, pid), CENTER])
    action, fallback = await _request_action(
        state,
        runtime,
        pid,
        request,
        lambda msgs: runtime.context.players[pid].act_at_night(msgs, request),
        lambda a: "対象が指定されていません" if a.target is None else rules.validate_seer_target(pid, a.target, ids),
        lambda: NightAction(target=rules.fallback_seer_target()),
    )
    result = rules.resolve_seer(action.target, state["current_roles"], state["center"])
    runtime.context.recorder.emit(
        "night_action", "secret", actor=pid, action="seer_look", target=action.target, result=result, fallback=fallback
    )
    night = state["night"].model_copy(update={"seer_target": action.target, "seer_result": result})
    return {"night": night, "delivered_upto": _delivered(state, pid, 0)}


async def night_werewolf(state: GameState, runtime: Runtime[GameContext]) -> dict:
    # 人狼は互いを確認するだけなので呼び出し不要。結果は夜明けの通知で伝える
    wolves = state["night"].werewolves
    if wolves:
        runtime.context.recorder.emit("night_action", "secret", actor=wolves, action="werewolf_check", result=wolves)
    return {}


async def night_thief(state: GameState, runtime: Runtime[GameContext]) -> dict:
    pid = _player_with_initial_role(state, Role.THIEF)
    if pid is None:
        return {}
    ids = state["config"].player_ids
    request = NightRequest(role=Role.THIEF, candidates=_others(state, pid))
    action, fallback = await _request_action(
        state,
        runtime,
        pid,
        request,
        lambda msgs: runtime.context.players[pid].act_at_night(msgs, request),
        lambda a: rules.validate_thief_target(pid, a.target, ids),
        lambda: NightAction(target=rules.fallback_thief_target()),
    )
    roles, new_role = rules.resolve_thief(pid, action.target, state["current_roles"])
    runtime.context.recorder.emit(
        "night_action", "secret", actor=pid, action="thief_swap", target=action.target, result=new_role, fallback=fallback
    )
    night = state["night"].model_copy(update={"thief_target": action.target, "thief_new_role": new_role})
    return {"night": night, "current_roles": roles, "delivered_upto": _delivered(state, pid, 0)}


async def dawn(state: GameState, runtime: Runtime[GameContext]) -> dict:
    ctx = runtime.context
    timeout = state["config"].llm_timeout_sec
    delivered = dict(state["delivered_upto"])
    for pid in state["config"].player_ids:
        text = build_message(state, pid, NotesRequest())
        try:
            notes = await asyncio.wait_for(ctx.players[pid].update_notes([HumanMessage(text)]), timeout)
        except TimeoutError:
            notes = None
        if notes is not None:
            ctx.recorder.emit("notes_updated", "secret", player=pid, notes=notes)
        delivered[pid] = 0
    return {"phase": "day", "delivered_upto": delivered}


async def discussion_turn(state: GameState, runtime: Runtime[GameContext]) -> dict:
    config = state["config"]
    turn = scheduler.next_turn(state["cursor"], config)
    assert turn is not None
    pid = turn.speaker
    request = SpeakRequest(turn=turn, others=_others(state, pid))
    action, fallback = await _request_action(
        state,
        runtime,
        pid,
        request,
        lambda msgs: runtime.context.players[pid].speak(msgs, request),
        lambda a: rules.validate_speech(pid, a.content, a.addressed_to, a.is_pass, config.player_ids, turn.pass_allowed),
        lambda: SpeechAction(is_pass=True),
    )
    number = len(state["utterances"])
    utterance = Utterance(
        turn=number,
        round=turn.round,
        speaker=pid,
        content="" if action.is_pass else action.content,
        addressed_to=None if action.is_pass else action.addressed_to,
        claimed_role=None if action.is_pass else action.claimed_role,
        is_pass=action.is_pass,
        reply_to_turn=turn.reply_to_turn,
    )
    runtime.context.recorder.emit("utterance", "public", **utterance.model_dump(), fallback=fallback)
    return {
        "utterances": [utterance],
        "cursor": scheduler.advance(state["cursor"], turn, utterance, config),
        # 自分の発言は次回の差分に含めない
        "delivered_upto": _delivered(state, pid, number + 1),
    }


def after_discussion_turn(state: GameState) -> Literal["discussion_turn", "vote"]:
    return "vote" if scheduler.next_turn(state["cursor"], state["config"]) is None else "discussion_turn"


async def vote(state: GameState, runtime: Runtime[GameContext]) -> dict:
    config = state["config"]
    ctx = runtime.context
    votes: list[Vote] = []
    delivered = dict(state["delivered_upto"])
    for pid in config.player_ids:
        request = VoteRequest(candidates=_others(state, pid))
        action, fallback = await _request_action(
            state,
            runtime,
            pid,
            request,
            lambda msgs, pid=pid, request=request: ctx.players[pid].vote(msgs, request),
            lambda a, pid=pid: rules.validate_vote(pid, a.target, config.player_ids),
            lambda pid=pid: VoteAction(target=rules.fallback_vote(pid, config.player_ids, ctx.rng), reason="フォールバック"),
        )
        v = Vote(voter=pid, target=action.target, reason=action.reason)
        ctx.recorder.emit("vote_cast", "secret", **v.model_dump(), fallback=fallback)
        votes.append(v)
        delivered[pid] = len(state["utterances"])
    return {"phase": "resolve", "votes": votes, "delivered_upto": delivered}


async def resolve(state: GameState, runtime: Runtime[GameContext]) -> dict:
    executed = rules.tally(state["votes"])
    result = rules.judge(executed, state["current_roles"], state["center"])
    runtime.context.recorder.emit(
        "game_ended",
        "public",
        result=result.model_dump(),
        votes=[v.model_dump() for v in state["votes"]],
        initial_roles=state["initial_roles"],
        night=state["night"].model_dump(),
    )
    return {"phase": "end", "result": result}


def build_graph():
    return (
        StateGraph(GameState, context_schema=GameContext)
        .add_node("setup", setup)
        .add_node("night_seer", night_seer)
        .add_node("night_werewolf", night_werewolf)
        .add_node("night_thief", night_thief)
        .add_node("dawn", dawn)
        .add_node("discussion_turn", discussion_turn)
        .add_node("vote", vote)
        .add_node("resolve", resolve)
        .add_edge(START, "setup")
        .add_edge("setup", "night_seer")
        .add_edge("night_seer", "night_werewolf")
        .add_edge("night_werewolf", "night_thief")
        .add_edge("night_thief", "dawn")
        .add_edge("dawn", "discussion_turn")
        .add_conditional_edges("discussion_turn", after_discussion_turn, ["discussion_turn", "vote"])
        .add_edge("vote", "resolve")
        .add_edge("resolve", END)
        .compile()
    )
