"""エンジンのテスト: RandomPlayer で全ゲーム、seed 固定の再現性、再要求とフォールバック。"""

import json
import random
from pathlib import Path

import pytest
from langsmith import get_tracing_context

from onenightagents.cli import RECURSION_LIMIT, make_config
from onenightagents.engine.context import GameContext
from onenightagents.engine.graph import build_graph
from onenightagents.engine.recorder import Recorder
from onenightagents.game import rules
from onenightagents.game.models import NightRequest, SpeakRequest, VoteRequest
from onenightagents.players.base import NightAction, SpeechAction, VoteAction
from onenightagents.players.scripted import LiarSeerPlayer, RandomPlayer


def read_events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]


async def run_game(run_dir: Path, seed: int, players=None, num_players: int = 5, liar_seer: str | None = None) -> dict:
    config = make_config(f"g{seed}", seed, num_players, 2, liar_seer=liar_seer)
    if players is None:
        players = {p.id: RandomPlayer(p.id, random.Random(seed * 100 + i)) for i, p in enumerate(config.players)}
    ctx = GameContext(players=players, recorder=Recorder(run_dir), rng=random.Random(seed))
    return await build_graph().ainvoke({"config": config}, config={"recursion_limit": RECURSION_LIMIT}, context=ctx)


@pytest.mark.asyncio
async def test_full_game_with_random_players(tmp_path: Path) -> None:
    final = await run_game(tmp_path / "a", seed=3)
    assert final["phase"] == "end"
    result = final["result"]
    assert result == rules.judge(rules.tally(final["votes"]), final["current_roles"], final["center"])
    assert len(final["votes"]) == 5
    assert all(u.round == 0 for u in final["utterances"][:5]) or any(u.reply_to_turn is not None for u in final["utterances"][:5])

    events = read_events(tmp_path / "a")
    names = [e["event"] for e in events]
    assert names[0] == "game_started" and names[1] == "roles_assigned" and names[-1] == "game_ended"
    assert names.count("vote_cast") == 5
    assert (tmp_path / "a" / "config.json").exists()


@pytest.mark.asyncio
async def test_same_seed_reproduces_events(tmp_path: Path) -> None:
    await run_game(tmp_path / "a", seed=7)
    await run_game(tmp_path / "b", seed=7)
    strip = lambda evs: [{k: v for k, v in e.items() if k != "time"} for e in evs]
    assert strip(read_events(tmp_path / "a")) == strip(read_events(tmp_path / "b"))


class StubbornPlayer:
    """常に無効な行動を返す。再要求 → フォールバックの経路を踏ませる。"""

    def __init__(self, id: str) -> None:
        self.id = id
        self.calls = 0

    async def act_at_night(self, messages, request: NightRequest) -> NightAction:
        self.calls += 1
        return NightAction(target=self.id)

    async def speak(self, messages, request: SpeakRequest) -> SpeechAction:
        self.calls += 1
        return SpeechAction(content="", addressed_to=self.id)

    async def vote(self, messages, request: VoteRequest) -> VoteAction:
        self.calls += 1
        return VoteAction(target=self.id)

    async def update_notes(self, messages) -> str | None:
        return None


@pytest.mark.asyncio
async def test_invalid_actions_are_retried_once_then_fallback(tmp_path: Path) -> None:
    players = {f"p{i}": StubbornPlayer(f"p{i}") for i in range(1, 6)}
    final = await run_game(tmp_path / "a", seed=1, players=players)
    assert final["phase"] == "end"
    assert all(u.is_pass for u in final["utterances"])
    assert all(v.voter != v.target for v in final["votes"])
    events = read_events(tmp_path / "a")
    skipped = [e for e in events if e["event"] == "turn_skipped"]
    # 発言 5(オープニング)+ 自由 1 ラウンド(全員パスで終了)+ 投票 5 + 夜 2
    assert len(skipped) == 5 + 5 + 5 + 2
    retry_messages = [e for e in events if e["event"] == "night_action" and e["data"].get("fallback")]
    assert len(retry_messages) == 2
    # 各判断で 2 回呼ばれている(初回 + 再要求)
    assert sum(p.calls for p in players.values()) == 2 * len(skipped)


class SleepyPlayer(RandomPlayer):
    """投票だけ制限時間を超える。"""

    async def vote(self, messages, request: VoteRequest) -> VoteAction:
        import asyncio

        await asyncio.sleep(1)
        return await super().vote(messages, request)


@pytest.mark.asyncio
async def test_timeout_falls_back(tmp_path: Path) -> None:
    config = make_config("g", 2, 3, 1, llm_timeout_sec=0.01)
    players = {p.id: SleepyPlayer(p.id, random.Random(i)) for i, p in enumerate(config.players)}
    ctx = GameContext(players=players, recorder=Recorder(tmp_path), rng=random.Random(2))
    final = await build_graph().ainvoke({"config": config}, config={"recursion_limit": RECURSION_LIMIT}, context=ctx)
    assert len(final["votes"]) == 3
    skipped = [e for e in read_events(tmp_path) if e["event"] == "turn_skipped"]
    assert len(skipped) == 3 and all("制限時間" in e["data"]["reason"] for e in skipped)


@pytest.mark.asyncio
async def test_liar_seer_claims_seer_and_votes_target(tmp_path: Path) -> None:
    for seed in range(1, 11):
        liar = LiarSeerPlayer("p3", random.Random(seed))
        players = {f"p{i}": RandomPlayer(f"p{i}", random.Random(seed * 100 + i)) for i in range(1, 6)}
        players["p3"] = liar
        final = await run_game(tmp_path / f"s{seed}", seed, players, liar_seer="p3")
        assert final["initial_roles"]["p3"] is rules.Role.WEREWOLF
        partners = [w for w in final["night"].werewolves if w != "p3"]
        assert liar.partner == (partners[0] if partners else None)
        assert liar.target is not None and liar.target != liar.partner and liar.target != "p3"
        mine = [u for u in final["utterances"] if u.speaker == "p3"]
        assert mine[0].round == 0 and mine[0].claimed_role is rules.Role.SEER and liar.target in mine[0].content
        assert all(u.claimed_role is rules.Role.SEER and not u.is_pass for u in mine)
        assert next(v for v in final["votes"] if v.voter == "p3").target == liar.target


class RecordingPlayer(RandomPlayer):
    """呼び出し中の tracing context のメタデータを目的別に記録する。"""

    seen: dict[str, list[dict]] = {}

    def _record(self, purpose: str) -> None:
        self.seen.setdefault(purpose, []).append(get_tracing_context()["metadata"] or {})

    async def act_at_night(self, messages, request: NightRequest) -> NightAction:
        self._record("night")
        return await super().act_at_night(messages, request)

    async def speak(self, messages, request: SpeakRequest) -> SpeechAction:
        self._record("speak")
        return await super().speak(messages, request)

    async def vote(self, messages, request: VoteRequest) -> VoteAction:
        self._record("vote")
        return await super().vote(messages, request)

    async def update_notes(self, messages) -> str | None:
        self._record("notes")
        return None


@pytest.mark.asyncio
async def test_player_calls_carry_trace_metadata(tmp_path: Path) -> None:
    """GM がプレイヤーの判断に付けるメタデータ(docs/architecture.md 10.1)が、呼び出し中の tracing context に入る。"""
    RecordingPlayer.seen = {}
    players = {f"p{i}": RecordingPlayer(f"p{i}", random.Random(i)) for i in range(1, 6)}
    final = await run_game(tmp_path / "a", seed=1, players=players)
    seen = RecordingPlayer.seen
    roles = {pid: r.value for pid, r in final["initial_roles"].items()}
    assert set(seen) == {"night", "notes", "speak", "vote"}
    assert {m["initial_role"] for ms in seen.values() for m in ms} <= set(roles.values())
    assert len(seen["notes"]) == 5 and len(seen["vote"]) == 5
    assert [m["turn"] for m in seen["speak"]] == list(range(len(final["utterances"])))
    assert [m["round"] for m in seen["speak"]] == [u.round for u in final["utterances"]]
    assert all("turn" not in m and "round" not in m for m in seen["vote"] + seen["notes"] + seen["night"])
    assert {m["initial_role"] for m in seen["night"]} == {"seer", "thief"} & set(roles.values())


class CandidateRecorder(RandomPlayer):
    """夜の要求(候補と GM の提示)を記録する。"""

    seen: dict[str, NightRequest] = {}

    async def act_at_night(self, messages, request: NightRequest) -> NightAction:
        self.seen[request.role.value] = request
        return await super().act_at_night(messages, request)


@pytest.mark.asyncio
async def test_night_request_hint_is_a_candidate_chosen_by_seed(tmp_path: Path) -> None:
    """夜の要求の hint は候補の 1 つで、seed で決まる(025)。"""
    hints: list[dict[str, str]] = []
    for seed in (1, 2, 3, 1):
        CandidateRecorder.seen = {}
        players = {f"p{i}": CandidateRecorder(f"p{i}", random.Random(i)) for i in range(1, 6)}
        final = await run_game(tmp_path / f"s{seed}-{len(hints)}", seed=seed, players=players)
        for role, req in CandidateRecorder.seen.items():
            pid = next(p for p, r in final["initial_roles"].items() if r.value == role)
            assert req.candidates == [p for p in final["config"].player_ids if p != pid] + (["center"] if role == "seer" else [])
            assert req.hint in req.candidates
        hints.append({role: req.hint for role, req in CandidateRecorder.seen.items()})
    assert hints[0] == hints[3]
    events = read_events(tmp_path / "s1-0")
    night = [e["data"] for e in events if e["event"] == "night_action" and e["data"]["action"] != "werewolf_check"]
    assert night and all(x["hint"] in x["candidates"] for x in night)
