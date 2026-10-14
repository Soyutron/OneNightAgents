"""CLI: play(1 ゲームを観戦)/ bench(複数 seed を連続実行)/ summarize(runs/ の集計)/ replay(役職付きの発言ログ)。"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml

from onenightagents.analysis.summarize import find_game_dirs, format_report, summarize_game
from onenightagents.engine.context import GameContext
from onenightagents.engine.graph import build_graph
from onenightagents.engine.recorder import Event, Recorder
from onenightagents.game.messages import ROLE_NAMES
from onenightagents.game.models import GameConfig, PlayerConfig
from onenightagents.llm.ollama import make_llm
from onenightagents.players.base import Player
from onenightagents.players.deep import DeepAgentPlayer
from onenightagents.players.prompts import render_persona
from onenightagents.players.scripted import RandomPlayer

DISPLAY_NAMES = ["太郎", "花子", "次郎", "美咲", "健太", "さくら", "大輔"]

# 議論ターンがノード 1 回ずつなので既定の 25 では足りない
RECURSION_LIMIT = 500


def make_config(
    game_id: str,
    seed: int,
    num_players: int,
    max_free_rounds: int,
    kind: str = "scripted",
    personas: list[str] | None = None,
    **overrides,
) -> GameConfig:
    # 席とペルソナの対応を固定すると同じ席が毎回標的になる(002)ので、seed で並べ替える
    if personas:
        order = random.Random(seed).sample(personas, len(personas))
    players = [
        PlayerConfig(
            id=f"p{i + 1}",
            display_name=DISPLAY_NAMES[i],
            kind=kind,
            persona=order[i % len(order)] if personas else None,
        )
        for i in range(num_players)
    ]
    return GameConfig(game_id=game_id, seed=seed, players=players, max_free_rounds=max_free_rounds, **overrides)


def load_personas(directory: str | None) -> list[str] | None:
    """ペルソナ YAML をファイル名順に読み、system prompt 用の文章にする。"""
    if not directory:
        return None
    files = sorted(Path(directory).glob("*.yaml"))
    return [render_persona(yaml.safe_load(f.read_text(encoding="utf-8"))) for f in files] or None


def make_players(config: GameConfig, recorder: Recorder, args: argparse.Namespace) -> dict[str, Player]:
    if args.kind == "random":
        return {p.id: RandomPlayer(p.id, random.Random(config.seed * 100 + i)) for i, p in enumerate(config.players)}
    llm = make_llm(
        args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        reasoning=args.reasoning,
        temperature=args.temperature,
        seed=config.seed,
    )

    def on_step(step: dict) -> None:
        notes = step.pop("notes")
        recorder.emit("agent_step", "secret", **step)
        if notes is not None:
            recorder.emit("notes_updated", "secret", player=step["player"], notes=notes)

    return {
        p.id: DeepAgentPlayer(
            id=p.id,
            display_name=p.display_name,
            game_id=config.game_id,
            llm=llm,
            persona=p.persona,
            model_call_limit=config.model_call_limit,
            on_step=on_step,
        )
        for p in config.players
    }


def _role_names(value: object) -> object:
    """夜の結果(役職・役職の辞書・役職のリスト)を日本語表記にする。"""
    if isinstance(value, dict):
        return {k: ROLE_NAMES[v] for k, v in value.items()}
    if isinstance(value, list):
        return [ROLE_NAMES[v] if v in ROLE_NAMES else v for v in value]
    return ROLE_NAMES.get(value, value)


def print_event(ev: Event, reveal: bool) -> None:
    if ev.visibility == "secret" and not reveal:
        return
    d = ev.data
    match ev.event:
        case "game_started":
            print(f"=== ゲーム開始 {d['config']['game_id']} (seed={d['config']['seed']}) ===")
        case "roles_assigned":
            roles = ", ".join(f"{p}={ROLE_NAMES[r]}" for p, r in d["initial_roles"].items())
            print(f"[配役] {roles} / 中央: {', '.join(ROLE_NAMES[r] for r in d['center'])}")
        case "night_action":
            print(f"[夜] {d['actor']} {d['action']} target={d.get('target')} result={_role_names(d.get('result'))}")
        case "utterance":
            if d["is_pass"]:
                print(f"#{d['turn']} R{d['round']} {d['speaker']}: (パス)")
            else:
                meta = []
                if d["addressed_to"]:
                    meta.append(f"→{d['addressed_to']}")
                if d["claimed_role"]:
                    meta.append(f"CO:{ROLE_NAMES[d['claimed_role']]}")
                print(f"#{d['turn']} R{d['round']} {d['speaker']}: 「{d['content']}」 {' '.join(meta)}")
        case "turn_skipped":
            print(f"[フォールバック] {d['player']}: {d['reason']}")
        case "notes_updated":
            print(f"[メモ {d['player']}]\n{d['notes']}")
        case "agent_step":
            errors = f" errors={d['errors']}" if d["errors"] else ""
            print(
                f"[step {d['player']} {d['purpose']}] calls={d['model_calls']} tools={d['tool_calls']} "
                f"{d['latency_ms'] / 1000:.1f}s ctx={d['last_prompt_tokens']}{errors}"
            )
        case "vote_cast":
            print(f"[投票] {d['voter']} → {d['target']} ({d['reason']})")
        case "game_ended":
            r = d["result"]
            votes = ", ".join(f"{v['voter']}→{v['target']}" for v in d["votes"])
            roles = ", ".join(f"{p}={ROLE_NAMES[x]}" for p, x in r["final_roles"].items())
            print(f"=== 投票: {votes}")
            print(f"=== 処刑: {r['executed'] or 'なし'} / 勝者: {r['winner']}")
            print(f"=== 最終役職: {roles} / 中央: {', '.join(ROLE_NAMES[x] for x in r['center'])}")


async def run_game(
    game_id: str, seed: int, run_dir: Path, args: argparse.Namespace, on_event: Callable[[Event], None] | None
) -> dict:
    kind = "scripted" if args.kind == "random" else "deep"
    config = make_config(
        game_id, seed, args.players, args.max_rounds, kind, load_personas(args.personas), llm_timeout_sec=args.timeout
    )
    recorder = Recorder(run_dir, on_event)
    ctx = GameContext(players=make_players(config, recorder, args), recorder=recorder, rng=random.Random(seed))
    # LangSmith でゲーム単位に絞り込むためのメタデータ(docs/architecture.md 10.1)
    run_config = {
        "recursion_limit": RECURSION_LIMIT,
        "run_name": game_id,
        "metadata": {"game_id": game_id, "seed": seed, "model": args.model, "kind": kind},
        "tags": ["onenightagents"],
    }
    return await build_graph().ainvoke({"config": config}, config=run_config, context=ctx)


async def play(args: argparse.Namespace) -> None:
    game_id = f"{datetime.now():%Y%m%d-%H%M%S}-seed{args.seed}"
    run_dir = Path(args.run_dir) / game_id
    await run_game(game_id, args.seed, run_dir, args, lambda ev: print_event(ev, args.reveal))
    print(f"ログ: {run_dir}")


async def bench(args: argparse.Namespace) -> None:
    """seed を変えて連続実行し、最後に集計を出す。"""
    bench_id = args.name or f"{datetime.now():%Y%m%d-%H%M%S}-bench"
    bench_dir = Path(args.run_dir) / bench_id
    for seed in range(args.seed, args.seed + args.games):
        game_id = f"{bench_id}/seed{seed}"
        t0 = time.perf_counter()
        final = await run_game(game_id, seed, bench_dir / f"seed{seed}", args, None)
        r = final["result"]
        print(f"seed={seed} winner={r.winner} executed={r.executed} {time.perf_counter() - t0:.0f}s", flush=True)
    print()
    print(format_report([summarize_game(d) for d in find_game_dirs([bench_dir])]))
    print(f"ログ: {bench_dir}")


def summarize(args: argparse.Namespace) -> None:
    dirs = find_game_dirs([Path(p) for p in args.paths])
    print(format_report([summarize_game(d) for d in dirs]))


def replay(args: argparse.Namespace) -> None:
    """終了したゲームの発言・投票を、真の役職を添えて表示する(観察用)。"""
    for run_dir in find_game_dirs([Path(p) for p in args.paths]):
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        roles = next(e["data"]["initial_roles"] for e in events if e["event"] == "roles_assigned")
        ended = next(e["data"] for e in events if e["event"] == "game_ended")
        final = ended["result"]["final_roles"]
        night = ended["night"]

        def tag(pid: str) -> str:
            r0, r1 = ROLE_NAMES[roles[pid]], ROLE_NAMES[final[pid]]
            return f"{pid}({r0}→{r1})" if r0 != r1 else f"{pid}({r0})"

        print(f"\n===== {run_dir}")
        print(
            f"中央: {', '.join(ROLE_NAMES[r] for r in ended['result']['center'])} / "
            f"占い: {night['seer_target']} → {_role_names(night['seer_result'])} / "
            f"怪盗: {night['thief_target']} → {_role_names(night['thief_new_role'])}"
        )
        for e in events:
            d = e["data"]
            if e["event"] == "utterance":
                if d["is_pass"]:
                    print(f"#{d['turn']} R{d['round']} {tag(d['speaker'])}: (パス)")
                    continue
                meta = (f" →{d['addressed_to']}" if d["addressed_to"] else "") + (
                    f" CO:{ROLE_NAMES[d['claimed_role']]}" if d["claimed_role"] else ""
                )
                print(f"#{d['turn']} R{d['round']} {tag(d['speaker'])}:{meta} {d['content']}")
            elif e["event"] == "turn_skipped":
                print(f"  [フォールバック] {d['player']}: {d['reason']}")
        for v in ended["votes"]:
            print(f"  投票 {tag(v['voter'])} → {v['target']}: {v['reason']}")
        print(f"  処刑: {ended['result']['executed'] or 'なし'} / 勝者: {ended['result']['winner']}")


def add_game_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--players", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--kind", choices=["random", "deep"], default="deep")
    p.add_argument("--model", default="gemma4:26b")
    p.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=60.0, help="1 判断の制限時間(秒)")
    p.add_argument("--personas", default=None, help="ペルソナ YAML のディレクトリ(例: configs/personas)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="onenightagents")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play", help="1 ゲームを実行して観戦する")
    add_game_options(p)
    p.add_argument("--reveal", action="store_true", help="秘密情報(配役・夜・投票・メモ)も表示する")
    b = sub.add_parser("bench", help="seed を変えて複数ゲームを実行し集計する")
    add_game_options(b)
    b.add_argument("--games", type=int, default=5)
    b.add_argument("--name", default=None, help="runs/ 配下のディレクトリ名")
    z = sub.add_parser("summarize", help="runs/ のログを集計する")
    z.add_argument("paths", nargs="+")
    r = sub.add_parser("replay", help="終了したゲームを役職付きで表示する")
    r.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.command == "play":
        asyncio.run(play(args))
    elif args.command == "bench":
        asyncio.run(bench(args))
    elif args.command == "summarize":
        summarize(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
