"""runs/ のイベントログを集計する(docs/agent-design.md 6.3)。"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

SUBMIT_TOOLS = {"speak", "pass_turn", "vote", "seer_look", "thief_swap"}
# 文字 bigram の Jaccard 類似度がこれ以上なら「他者の発言のなぞり」とみなす
REPEAT_THRESHOLD = 0.4


def find_game_dirs(paths: list[Path]) -> list[Path]:
    """events.jsonl を持つディレクトリを再帰的に集める。"""
    found: list[Path] = []
    for p in paths:
        if (p / "events.jsonl").exists():
            found.append(p)
        else:
            found.extend(sorted(d.parent for d in p.rglob("events.jsonl")))
    return found


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _is_repeat(content: str, earlier: list[str]) -> bool:
    bg = _bigrams(content)
    if len(bg) < 10:
        return False
    for prev in earlier:
        pbg = _bigrams(prev)
        if pbg and len(bg & pbg) / len(bg | pbg) >= REPEAT_THRESHOLD:
            return True
    return False


def summarize_game(run_dir: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    t0 = datetime.fromisoformat(events[0]["time"])
    t1 = datetime.fromisoformat(events[-1]["time"])
    ended = next(e["data"] for e in events if e["event"] == "game_ended")
    final_roles = ended["result"]["final_roles"]
    utterances = [e["data"] for e in events if e["event"] == "utterance"]
    steps = [e["data"] for e in events if e["event"] == "agent_step"]
    decisions = [s for s in steps if s["purpose"] != "notes"]

    claims = [u for u in utterances if u["claimed_role"]]
    lies = [u for u in claims if u["claimed_role"] != final_roles[u["speaker"]]]
    confessions = [u for u in claims if u["claimed_role"] == "werewolf" and final_roles[u["speaker"]] == "werewolf"]
    seen: list[str] = []
    repeats = 0
    for u in utterances:
        if u["is_pass"]:
            continue
        others = [c for c, s in zip(seen, [x["speaker"] for x in utterances if not x["is_pass"]]) if s != u["speaker"]]
        if _is_repeat(u["content"], others):
            repeats += 1
        seen.append(u["content"])

    return {
        "run_dir": str(run_dir),
        "seed": events[0]["data"]["config"]["seed"],
        "winner": ended["result"]["winner"],
        "executed": ended["result"]["executed"],
        "seconds": (t1 - t0).total_seconds(),
        "utterances": len([u for u in utterances if not u["is_pass"]]),
        "passes": len([u for u in utterances if u["is_pass"]]),
        "decisions": len(decisions),
        "model_calls": sum(s["model_calls"] for s in steps),
        "latencies": [s["latency_ms"] / 1000 for s in steps],
        "no_submission": sum(1 for s in decisions if not SUBMIT_TOOLS & set(s["tool_calls"])),
        "fallbacks": sum(1 for e in events if e["event"] == "turn_skipped"),
        "max_ctx": max((s["last_prompt_tokens"] or 0 for s in steps), default=0),
        "claims": len(claims),
        "lies": len(lies),
        "confessions": len(confessions),
        "repeats": repeats,
        # 終了時に占い師だった人が処刑されたか(真占い師を信じない構図の指標)
        "seer_executed": any(final_roles[pid] == "seer" for pid in ended["result"]["executed"]),
    }


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(games)
    lat = sorted(x for g in games for x in g["latencies"])
    decisions = sum(g["decisions"] for g in games)
    claims = sum(g["claims"] for g in games)
    utterances = sum(g["utterances"] for g in games)
    return {
        "games": n,
        "village": sum(g["winner"] == "village" for g in games),
        "werewolf": sum(g["winner"] == "werewolf" for g in games),
        "none": sum(g["winner"] == "none" for g in games),
        "seconds_mean": statistics.mean(g["seconds"] for g in games),
        "calls_per_decision": sum(g["model_calls"] for g in games) / decisions if decisions else 0.0,
        "latency_p50": statistics.median(lat) if lat else 0.0,
        "latency_p95": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else 0.0,
        "no_submission_rate": sum(g["no_submission"] for g in games) / decisions if decisions else 0.0,
        "fallback_rate": sum(g["fallbacks"] for g in games) / decisions if decisions else 0.0,
        "max_ctx": max(g["max_ctx"] for g in games),
        "lie_rate": sum(g["lies"] for g in games) / claims if claims else 0.0,
        "confessions": sum(g["confessions"] for g in games),
        "repeat_rate": sum(g["repeats"] for g in games) / utterances if utterances else 0.0,
        "seer_executed": sum(g["seer_executed"] for g in games),
    }


def format_report(games: list[dict[str, Any]]) -> str:
    s = summarize_games(games)
    lines = [
        "| seed | 勝者 | 処刑 | 占処刑 | 秒 | 判断 | 呼び出し | 未提出 | FB | 嘘CO/CO | なぞり |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in games:
        lines.append(
            f"| {g['seed']} | {g['winner']} | {','.join(g['executed']) or '-'} | {'o' if g['seer_executed'] else '-'} "
            f"| {g['seconds']:.0f} | {g['decisions']} "
            f"| {g['model_calls']} | {g['no_submission']} | {g['fallbacks']} | {g['lies']}/{g['claims']} | {g['repeats']}/{g['utterances']} |"
        )
    lines += [
        "",
        f"ゲーム数 {s['games']}: 村 {s['village']} / 人狼 {s['werewolf']} / なし {s['none']}",
        f"総時間 平均 {s['seconds_mean']:.0f} 秒、1 判断の呼び出し {s['calls_per_decision']:.2f}、"
        f"レイテンシ p50 {s['latency_p50']:.1f} / p95 {s['latency_p95']:.1f} 秒、最大コンテキスト {s['max_ctx']}",
        f"未提出率 {s['no_submission_rate']:.1%}、フォールバック率 {s['fallback_rate']:.1%}、"
        f"嘘 CO 率 {s['lie_rate']:.1%}、人狼の自白 {s['confessions']}、なぞり率 {s['repeat_rate']:.1%}、"
        f"占い師の処刑 {s['seer_executed']} ゲーム",
    ]
    return "\n".join(lines)
