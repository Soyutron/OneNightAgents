"""検証 4: 日本語の短文生成の自然さ(目視評価用サンプル)と reasoning=False の効き目。

使い方: uv run python 03_japanese.py <model>
reasoning=False / True / None の 3 条件で同じ 5 プロンプトを投げ、出力・eval_count・時間・reasoning_content の有無を記録する。
"""

import argparse
import json
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage

from common import ModelCallRecorder, make_llm, write_jsonl

SYSTEM = """あなたは「ワンナイト人狼」のプレイヤー ユウキです。発言は短く、1〜3 文の日本語で、口語で自然に。説明や前置きは不要です。"""

PROMPTS = [
    "アオイが「p4 を占ったら人狼だった」と言い、ミナが「嘘だ、アオイが人狼だ」と反論しています。あなたの意見を一言。",
    "あなたは占い師で、ハルトを占ったら人狼でした。占い師だと名乗って結果を伝えてください。",
    "あなたは人狼です。疑いをそらすため、村人のふりをしてレンを疑ってください。",
    "レンから「ユウキは誰を疑っているの?」と聞かれました。答えてください。",
    "投票の直前です。誰に投票するか、理由と一緒に宣言してください(候補: アオイ、ハルト、ミナ、レン)。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    rows = []
    for reasoning in (False, True, None):
        llm = make_llm(args.model, reasoning=reasoning)
        for i, p in enumerate(PROMPTS):
            rec = ModelCallRecorder()
            t0 = time.perf_counter()
            msg = llm.invoke([SystemMessage(SYSTEM), HumanMessage(p)], config={"callbacks": [rec]})
            wall = time.perf_counter() - t0
            rc = msg.additional_kwargs.get("reasoning_content") or ""
            row = {
                "reasoning": reasoning,
                "i": i,
                "content": msg.content,
                "content_chars": len(msg.content),
                "has_think_tag": "<think>" in msg.content,
                "reasoning_chars": len(rc),
                "wall_sec": round(wall, 2),
                "eval_count": rec.calls[0].get("eval_count") if rec.calls else None,
            }
            rows.append(row)
            print(f"[reasoning={reasoning}][{i}] {wall:.1f}s eval={row['eval_count']} rc={len(rc)}c :: {msg.content[:80]!r}", file=sys.stderr)

    path = write_jsonl(f"03_japanese_{args.model.replace(':', '_')}.jsonl", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print("saved:", path)


if __name__ == "__main__":
    main()
