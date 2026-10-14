"""エンジンのテスト: RandomPlayer で全ゲーム、seed 固定の再現性、再要求とフォールバック。"""

import json
import random
from pathlib import Path

import pytest

from onenightagents.cli import RECURSION_LIMIT, make_config
from onenightagents.engine.context import GameContext
from onenightagents.engine.graph import build_graph
from onenightagents.engine.recorder import Recorder
from onenightagents.game import rules
from onenightagents.game.models import NightRequest, SpeakRequest, VoteRequest
from onenightagents.players.base import NightAction, SpeechAction, VoteAction
from onenightagents.players.scripted import RandomPlayer


def read_events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]


async def run_game(run_dir: Path, seed: int, players=None, num_players: int = 5) -> dict:
    config = make_config(f"g{seed}", seed, num_players, 2)
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
