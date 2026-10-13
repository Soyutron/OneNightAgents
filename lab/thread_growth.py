"""多ターンのスレッドで履歴がどう伸び、要約が発火するか、序盤の事実を終盤に思い出せるかを測る。

使い方: uv run --env-file ../.env python thread_growth.py gemma4:26b [--turns 16] [--num-ctx 16384] [--profile]
  --profile を付けると ChatOllama に profile={"max_input_tokens": num_ctx} を渡し、Deep Agents の要約既定を
  「170K トークンで発火」から「num_ctx の 85% で発火」に切り替える。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

TOPICS = ["ログ基盤", "採用計画", "オフィス移転", "セキュリティ研修", "四半期の売上", "顧客対応", "CI の高速化", "社内 Wiki", "監査対応", "新人研修", "API の改版", "障害訓練", "費用削減", "表彰制度", "備品管理", "年末の棚卸し"]


class Rec(BaseCallbackHandler):
    def __init__(self):
        self.calls = []
        self._t = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._t[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kw):
        meta = response.generations[0][0].message.response_metadata or {}
        self.calls.append({"wall": round(time.perf_counter() - self._t.pop(run_id, 0), 2), "prompt": meta.get("prompt_eval_count"), "out": meta.get("eval_count")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--reasoning", default="true", choices=["true", "false"])
    args = ap.parse_args()

    register_harness_profile("ollama", HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))
    llm_kwargs = dict(model=args.model, num_ctx=args.num_ctx, num_predict=2048, temperature=0.3, seed=7, reasoning=args.reasoning == "true", keep_alive="30m")
    if args.profile:
        llm_kwargs["profile"] = {"max_input_tokens": args.num_ctx}
    llm = ChatOllama(**llm_kwargs)
    backend = StateBackend()
    agent = create_deep_agent(
        model=llm,
        system_prompt="あなたは業務アシスタントです。指示されたファイル操作を行い、最後に 1 文で報告してください。",
        backend=backend,
        middleware=[FilesystemMiddleware(backend=backend, tools=["ls", "read_file", "write_file", "edit_file"]), ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")],
        checkpointer=InMemorySaver(),
        name="thread-growth",
    )
    config = {"configurable": {"thread_id": f"growth:{args.model}:{int(time.time())}"}}

    def turn(i: int, text: str) -> dict:
        rec = Rec()
        t0 = time.perf_counter()
        res = agent.invoke({"messages": [{"role": "user", "content": text}]}, config={**config, "callbacks": [rec]})
        final = res["messages"][-1].content
        n_msgs = len(res["messages"])
        row = {
            "turn": i,
            "wall": round(time.perf_counter() - t0, 1),
            "calls": len(rec.calls),
            "prompt_first": rec.calls[0]["prompt"] if rec.calls else None,
            "prompt_last": rec.calls[-1]["prompt"] if rec.calls else None,
            "messages_in_state": n_msgs,
            "final": (final if isinstance(final, str) else str(final))[:160],
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    rows = []
    rows.append(turn(0, "覚えておいてください: プロジェクトのコードネームは ORCHID-7、リリース日は 2026-12-03 です。この 2 つを /facts.md に書いてください。"))
    for i in range(1, args.turns + 1):
        topic = TOPICS[(i - 1) % len(TOPICS)]
        rows.append(turn(i, f"/journal.md に「{i} 日目: {topic}について打ち合わせをした。結論は来週に持ち越し。」という 1 行を追記してください(ファイルがなければ作成)。"))
    # 履歴から思い出せるか(ファイルを読ませない)
    rows.append(turn(args.turns + 1, "ファイルは読まずに答えてください: 最初に伝えたプロジェクトのコードネームとリリース日は何でしたか?"))
    recall = rows[-1]["final"]
    rows.append(turn(args.turns + 2, "/journal.md を読んで、何日分の記録があるか数字だけ答えてください。"))

    state = agent.get_state(config).values
    journal = (state.get("files") or {}).get("/journal.md", {}).get("content", "")
    summary = {
        "model": args.model,
        "num_ctx": args.num_ctx,
        "profile": args.profile,
        "reasoning": args.reasoning,
        "turns": args.turns,
        "recall_ok": ("ORCHID-7" in recall) and ("2026-12-03" in recall or "12 月 3 日" in recall or "12月3日" in recall),
        "journal_lines": len([ln for ln in journal.splitlines() if ln.strip()]),
        "max_prompt_tokens": max((r["prompt_last"] or 0) for r in rows),
        "messages_in_state_end": rows[-1]["messages_in_state"],
        "total_wall": round(sum(r["wall"] for r in rows), 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = Path(__file__).parent / "results" / f"thread_growth-{args.model.replace(':', '_')}-ctx{args.num_ctx}-{'profile' if args.profile else 'default'}-{args.reasoning}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [summary, *rows]) + "\n", encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
