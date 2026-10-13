"""検証 2: ChatOllama.bind_tools でダミー行動ツール 3 個を渡し、20 プロンプトでツール呼び出しの成功率を測る。

使い方: uv run python 01_tool_calls.py <model> [--reasoning]
"""

import argparse
import json
import sys
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from common import ModelCallRecorder, make_llm, percentiles, write_jsonl


class SpeakArgs(BaseModel):
    content: str = Field(max_length=200, description="発言本文。1〜3文")
    addressed_to: str | None = Field(default=None, description="質問や反論の宛先プレイヤーID(p1〜p5 のうち 1 つ)。全員向けなら null")
    claimed_role: Literal["villager", "werewolf", "seer", "thief"] | None = Field(default=None, description="この発言で主張する自分の役職。嘘でもよい。値は villager / werewolf / seer / thief のいずれか(日本語不可)。主張しないなら null")


class VoteArgs(BaseModel):
    target: str = Field(description="処刑したいプレイヤーID(p1〜p5)。自分は不可")
    reason: str = Field(max_length=200, description="理由の要約。公開されない")


class PassArgs(BaseModel):
    pass


@tool(args_schema=SpeakArgs)
def speak(content: str, addressed_to: str | None = None, claimed_role: str | None = None) -> str:
    """議論で発言する。"""
    return "受理しました"


@tool(args_schema=VoteArgs)
def vote(target: str, reason: str) -> str:
    """投票先を提出する。"""
    return "受理しました"


@tool(args_schema=PassArgs)
def pass_turn() -> str:
    """このターンは発言せずパスする。"""
    return "受理しました"


SYSTEM = """あなたは「ワンナイト人狼」のプレイヤー p3(ユウキ)です。参加者は p1 アオイ, p2 ハルト, p3 ユウキ(あなた), p4 ミナ, p5 レン。
あなたの配布役職は村人です。役職は夜の怪盗によって交換されている可能性があります。嘘をついてもかまいません。
発言は短く、1〜3文の日本語にしてください。
ゲームマスターから求められたら、必ず対応するツール(speak / pass_turn / vote)を 1 回だけ呼んで行動を提出してください。質問に答えるだけのときはツールを呼ばないでください。"""

LOG = """これまでの発言:
p1 アオイ: 私は占い師です。p4 を占ったら人狼でした。
p4 ミナ: 嘘です。私は村人です。p1 が人狼だと思います。
p2 ハルト: どちらかが嘘をついている。p5 はどう思う?
p5 レン: まだ分からない。p3 の意見を聞きたい。"""

# (プロンプト, 期待するツール名 or None)
PROMPTS: list[tuple[str, str | None]] = [
    (f"{LOG}\n\n発言ターンです。speak か pass_turn で提出してください。", "speak"),
    (f"{LOG}\n\np5 レンから意見を求められています。speak か pass_turn で提出してください。", "speak"),
    (f"{LOG}\n\n発言ターンです。p1 に質問するか、pass_turn してください。", "speak"),
    (f"{LOG}\n\n投票の時間です。vote で投票先を提出してください。", "vote"),
    (f"{LOG}\n\n発言ターンです。特に言うことがなければ pass_turn してください。", None),  # どちらでも可(speak/pass_turn)
    (f"{LOG}\n\n発言ターンです。自分が占い師だと偽って主張してください。speak で提出。", "speak"),
    (f"{LOG}\n\n投票の時間です。p1 か p4 のどちらかに vote してください。", "vote"),
    (f"{LOG}\n\n発言ターンです。p4 に反論してください。", "speak"),
    (f"{LOG}\n\n質問です(ツールは呼ばないでください): 今、誰が占い師を名乗っていますか? 一文で答えてください。", "NONE"),
    (f"{LOG}\n\n質問です(ツールは呼ばないでください): あなたの配布役職は何ですか?", "NONE"),
    (f"{LOG}\n\n発言ターンです。speak か pass_turn で提出してください。", "speak"),
    (f"{LOG}\n\np2 ハルトから「p3 は誰を疑っている?」と聞かれています。speak で答えてください。", "speak"),
    (f"{LOG}\n\n投票の時間です。vote で投票先と理由を提出してください。", "vote"),
    (f"{LOG}\n\n発言ターンです。今回はパスしてください。", "pass_turn"),
    (f"{LOG}\n\n発言ターンです。p5 に質問してください(addressed_to を指定)。", "speak"),
    (f"{LOG}\n\n投票の時間です。自分以外の誰かに vote してください。", "vote"),
    (f"{LOG}\n\n発言ターンです。村人であることを主張してください(claimed_role を指定)。", "speak"),
    (f"{LOG}\n\n質問です(ツールは呼ばないでください): 参加者は何人ですか?", "NONE"),
    (f"{LOG}\n\n発言ターンです。speak か pass_turn で提出してください。", "speak"),
    (f"{LOG}\n\n最終投票です。vote で提出してください。", "vote"),
]

SCHEMAS = {"speak": SpeakArgs, "vote": VoteArgs, "pass_turn": PassArgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--num-predict", type=int, default=None)
    args = ap.parse_args()
    reasoning = True if args.reasoning else False

    llm = make_llm(args.model, reasoning=reasoning, num_predict=args.num_predict).bind_tools([speak, vote, pass_turn])
    rows = []
    for i, (prompt, expected) in enumerate(PROMPTS):
        rec = ModelCallRecorder()
        t0 = time.perf_counter()
        try:
            msg = llm.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)], config={"callbacks": [rec]})
            err = None
        except Exception as e:  # noqa: BLE001
            msg, err = None, repr(e)
        wall = time.perf_counter() - t0
        tool_calls = msg.tool_calls if msg is not None else []
        names = [tc["name"] for tc in tool_calls]
        # 引数妥当性: 各呼び出しを Pydantic で検証
        arg_ok = True
        for tc in tool_calls:
            schema = SCHEMAS.get(tc["name"])
            try:
                if schema is None:
                    arg_ok = False
                else:
                    schema.model_validate(tc["args"])
                    if tc["name"] == "vote" and tc["args"].get("target") not in {"p1", "p2", "p4", "p5"}:
                        arg_ok = False
            except ValidationError:
                arg_ok = False
        if expected == "NONE":
            ok_lenient = len(tool_calls) == 0
        elif expected is None:
            ok_lenient = len(tool_calls) == 1 and names[0] in {"speak", "pass_turn"}
        else:
            ok_lenient = len(tool_calls) == 1 and names[0] == expected
        ok = ok_lenient and arg_ok
        rows.append(
            {
                "i": i,
                "expected": expected,
                "tool_calls": names,
                "args": [tc["args"] for tc in tool_calls],
                "content": (msg.content if msg is not None else "")[:200],
                "arg_ok": arg_ok,
                "ok": ok,
                "ok_lenient": ok_lenient,
                "wall_sec": round(wall, 2),
                "error": err,
                "meta": rec.calls[0] if rec.calls else None,
            }
        )
        print(f"[{i:02d}] ok={ok} lenient={ok_lenient} expected={expected} got={names} {wall:.1f}s content={rows[-1]['content'][:60]!r}", file=sys.stderr)

    n_ok = sum(r["ok"] for r in rows)
    summary = {
        "model": args.model,
        "reasoning": reasoning,
        "num_predict": args.num_predict,
        "n": len(rows),
        "success": n_ok,
        "success_rate": round(n_ok / len(rows), 3),
        "success_rate_lenient": round(sum(r["ok_lenient"] for r in rows) / len(rows), 3),
        "latency": percentiles([r["wall_sec"] for r in rows]),
        "prompt_eval_count_first": rows[0]["meta"] and rows[0]["meta"].get("prompt_eval_count"),
        "eval_count": percentiles([r["meta"]["eval_count"] for r in rows if r["meta"] and r["meta"].get("eval_count")]),
        "load_duration_first_sec": rows[0]["meta"] and rows[0]["meta"].get("load_duration_sec"),
    }
    tag = f"{args.model.replace(':', '_')}_{'think' if reasoning else 'nothink'}" + (f"_np{args.num_predict}" if args.num_predict else "")
    path = write_jsonl(f"01_tool_calls_{tag}.jsonl", [summary, *rows])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", path)


if __name__ == "__main__":
    main()
