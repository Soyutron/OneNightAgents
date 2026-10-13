"""検証 3: create_deep_agent + ChatOllama の最小構成で「/notes.md にメモを書いてから speak」を実行し、ステップ数・レイテンシ・成功率を測る。

使い方: uv run python 02_deep_agent.py <model> [--runs 10] [--reasoning] [--run-limit 4]
"""

import argparse
import json
import sys
import time
from typing import Literal

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from common import ModelCallRecorder, make_llm, percentiles, write_jsonl


class SpeakArgs(BaseModel):
    content: str = Field(max_length=200, description="発言本文。1〜3文")
    addressed_to: str | None = Field(default=None, description="質問や反論の宛先プレイヤーID(p1〜p5 のうち 1 つ)。全員向けなら null")
    claimed_role: Literal["villager", "werewolf", "seer", "thief"] | None = Field(default=None, description="この発言で主張する自分の役職。嘘でもよい。値は villager / werewolf / seer / thief のいずれか(日本語不可)。主張しないなら null")


# agent-design.md 2.3 の ActionMailbox 相当。invoke ごとに読み出す
submitted: list[dict] = []


@tool(args_schema=SpeakArgs)
def speak(content: str, addressed_to: str | None = None, claimed_role: str | None = None) -> str:
    """議論で発言する。ゲームマスターに発言を提出する。"""
    submitted.append({"content": content, "addressed_to": addressed_to, "claimed_role": claimed_role})
    return "受理しました"


SYSTEM = """あなたは「ワンナイト人狼」のプレイヤー p3(ユウキ)です。参加者は p1 アオイ, p2 ハルト, p3 ユウキ(あなた), p4 ミナ, p5 レン。
## ルール要約
村人・人狼・占い師・怪盗。夜に占い師は 1 人を占い、怪盗は誰かと役職を交換できる。朝に議論し、全員で投票。最多票が処刑される。人狼が処刑されれば村の勝ち。
## このゲームの注意
- 役職は夜の怪盗によって交換されている可能性があります。
- 嘘をついてもかまいません。役職を偽って主張してもかまいません。
- 発言は短く、1〜3 文の日本語にしてください。
## 行動の提出方法
- ゲームマスターから求められたら、必ず speak ツールを 1 回だけ呼んで発言を提出してください。
- 推理メモは /notes.md に保存できます。必要なときに read_file / write_file / edit_file を使ってください。
## あなたの性格
慎重で論理的。根拠のない疑いはかけない。"""

SCENARIOS = [
    """ゲーム開始。あなたの配布役職は村人です。
夜が明けました。これまでの発言:
p1 アオイ: 私は占い師です。p4 を占ったら人狼でした。
p4 ミナ: 嘘です。私は村人です。p1 が人狼だと思います。
p2 ハルト: どちらかが嘘をついている。p5 はどう思う?
p5 レン: まだ分からない。p3 の意見を聞きたい。

まず /notes.md に推理メモ(誰が何を主張したか、誰を疑うか)を書いてください。そのうえで、あなたの発言ターンです。speak で発言を提出してください。""",
    """ゲーム開始。あなたの配布役職は占い師で、夜に p2 を占ったところ「人狼」でした。
夜が明けました。これまでの発言:
p2 ハルト: 私は村人です。みんな落ち着いて話そう。
p1 アオイ: 占い師はいる? 名乗ってほしい。

まず /notes.md に推理メモを書いてください。そのうえで、あなたの発言ターンです。speak で発言を提出してください(占い師を名乗るかは自由)。""",
    """ゲーム開始。あなたの配布役職は人狼です。もう 1 人の人狼は p5 レンです。
夜が明けました。これまでの発言:
p1 アオイ: 私は占い師です。p5 を占ったら人狼でした。
p5 レン: 違います。私は村人です。
p4 ミナ: p1 を信じる。p5 が怪しい。

まず /notes.md に推理メモ(自分の立場と、誰に疑いを向けるか)を書いてください。そのうえで、あなたの発言ターンです。speak で発言を提出してください。""",
    """ゲーム開始。あなたの配布役職は怪盗で、夜に p4 と交換し、あなたは今「占い師」です(p4 は怪盗になりました)。
夜が明けました。これまでの発言:
p4 ミナ: 私は怪盗です。誰とも交換していません。
p2 ハルト: 怪盗の CO は珍しいね。

まず /notes.md に推理メモを書いてください。そのうえで、あなたの発言ターンです。speak で発言を提出してください。""",
    """ゲーム開始。あなたの配布役職は村人です。
夜が明けました。これまでの発言:
p2 ハルト: 私は占い師。中央を見たら人狼が 1 枚あった。
p5 レン: 私も占い師です。p2 は人狼。
p1 アオイ: 占い師が 2 人? どちらかが嘘だ。

まず /notes.md に推理メモを書いてください。そのうえで、あなたの発言ターンです。p2 か p5 のどちらかに質問する形で speak してください。""",
]


def build_agent(model: str, reasoning: bool, run_limit: int, seed: int, num_predict: int | None):
    # ChatOllama には既定ハーネスプロファイルがないので、general-purpose サブエージェント(task ツール)を無効化する
    register_harness_profile("ollama", HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))
    backend = StateBackend()
    return create_deep_agent(
        model=make_llm(model, reasoning=reasoning, seed=seed, num_predict=num_predict),
        system_prompt=SYSTEM,
        tools=[speak],
        backend=backend,
        middleware=[
            FilesystemMiddleware(backend=backend, tools=["ls", "read_file", "write_file", "edit_file"]),
            ModelCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"),
        ],
        checkpointer=InMemorySaver(),
        name="player-p3",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--num-predict", type=int, default=None)
    ap.add_argument("--run-limit", type=int, default=4)
    args = ap.parse_args()
    reasoning = True if args.reasoning else False

    rows = []
    first_prompt = None
    for i in range(args.runs):
        # 同じシナリオを seed 固定で回すと結果が完全に重複するので、実行ごとに seed を変える
        agent = build_agent(args.model, reasoning, args.run_limit, seed=1000 + i, num_predict=args.num_predict)
        scenario = SCENARIOS[i % len(SCENARIOS)]
        submitted.clear()
        rec = ModelCallRecorder()
        config = {"configurable": {"thread_id": f"spike:{i}"}, "callbacks": [rec]}
        t0 = time.perf_counter()
        err = None
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": scenario}]}, config=config)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            result = None
        wall = time.perf_counter() - t0
        if first_prompt is None and rec.first_prompt:
            first_prompt = rec.first_prompt
        files = agent.get_state(config).values.get("files", {}) or {}
        notes = files.get("/notes.md")
        # StateBackend のファイルは {"content": str, ...} の dict(旧版は行リスト)
        content = notes.get("content", "") if isinstance(notes, dict) else (notes or "")
        notes_text = "\n".join(content) if isinstance(content, list) else content
        tool_seq = [n for c in rec.calls for n in c.get("tool_calls", [])]
        final_text = ""
        if result is not None:
            last = result["messages"][-1]
            final_text = last.content if isinstance(last.content, str) else str(last.content)
        row = {
            "i": i,
            "scenario": i % len(SCENARIOS),
            "wall_sec": round(wall, 2),
            "model_calls": len(rec.calls),
            "tool_seq": tool_seq,
            "notes_written": bool(notes_text),
            "notes_chars": len(notes_text),
            "notes_text": notes_text[:600],
            "speak_count": len(submitted),
            "speak": submitted[0] if submitted else None,
            "ok": bool(notes_text) and len(submitted) == 1,
            "final_text": final_text[:200],
            "calls": rec.calls,
            "error": err,
        }
        rows.append(row)
        print(
            f"[{i:02d}] ok={row['ok']} calls={row['model_calls']} seq={tool_seq} {wall:.1f}s notes={row['notes_chars']}c speak={(row['speak'] or {}).get('content', '')[:50]!r}",
            file=sys.stderr,
        )

    n_ok = sum(r["ok"] for r in rows)
    summary = {
        "model": args.model,
        "reasoning": reasoning,
        "run_limit": args.run_limit,
        "num_predict": args.num_predict,
        "n": len(rows),
        "success": n_ok,
        "success_rate": round(n_ok / len(rows), 3),
        "latency": percentiles([r["wall_sec"] for r in rows]),
        "model_calls": percentiles([r["model_calls"] for r in rows]),
        "notes_written_rate": round(sum(r["notes_written"] for r in rows) / len(rows), 2),
        "speak_called_rate": round(sum(r["speak_count"] >= 1 for r in rows) / len(rows), 2),
        "prompt_eval_count_first_call": rows[0]["calls"][0].get("prompt_eval_count") if rows[0]["calls"] else None,
        "first_prompt": {k: v for k, v in (first_prompt or {}).items() if k != "system_text"},
    }
    tag = f"{args.model.replace(':', '_')}_{'think' if reasoning else 'nothink'}_lim{args.run_limit}" + (f"_np{args.num_predict}" if args.num_predict else "")
    path = write_jsonl(f"02_deep_agent_{tag}.jsonl", [summary, *rows])
    if first_prompt:
        (path.parent / f"02_system_prompt_{tag}.txt").write_text(first_prompt["system_text"], encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", path)


if __name__ == "__main__":
    main()
