"""results/ の JSONL を読んで実験ノート用の Markdown 表を出す。"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    print("## 01 ツール呼び出し(20 プロンプト)\n")
    print("| モデル | reasoning | 厳密成功 | 緩い成功 | p50 秒 | p95 秒 | 初回 prompt tok | eval tok p50 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(RESULTS.glob("01_tool_calls_*.jsonl")):
        if "v0-" in p.name:
            continue
        s = load(p)[0]
        print(
            f"| {s['model']} | {s['reasoning']} | {s['success']}/{s['n']} ({s['success_rate']:.0%}) | {s.get('success_rate_lenient', float('nan')):.0%} | "
            f"{s['latency']['p50']} | {s['latency']['p95']} | {s['prompt_eval_count_first']} | {s['eval_count'].get('p50')} |"
        )

    print("\n### 01 失敗の内訳\n")
    for p in sorted(RESULTS.glob("01_tool_calls_*.jsonl")):
        if "v0-" in p.name:
            continue
        rows = load(p)
        fails = [r for r in rows[1:] if not r["ok"]]
        if not fails:
            continue
        print(f"- {rows[0]['model']} (reasoning={rows[0]['reasoning']}):")
        for r in fails:
            kind = "ツール未呼び出し(文章で回答)" if not r["tool_calls"] and r["expected"] not in ("NONE",) else (
                "不要な呼び出し" if r["expected"] == "NONE" else ("ツール名違い" if r["ok_lenient"] is False else "引数不正")
            )
            print(f"  - #{r['i']} 期待={r['expected']} 実際={r['tool_calls']} → {kind}; args={json.dumps(r['args'], ensure_ascii=False)[:120]}")

    print("\n## 02 Deep Agent 最小構成(メモ → speak、run_limit=4)\n")
    print("| モデル | reasoning | 成功 | メモ書込率 | speak 率 | 呼び出し数 p50/max | p50 秒 | p95 秒 | max 秒 | 初回 prompt tok |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in sorted(RESULTS.glob("02_deep_agent_*.jsonl")):
        s = load(p)[0]
        print(
            f"| {s['model']} | {s['reasoning']} | {s['success']}/{s['n']} ({s['success_rate']:.0%}) | {s['notes_written_rate']:.0%} | {s['speak_called_rate']:.0%} | "
            f"{s['model_calls']['p50']}/{s['model_calls']['max']} | {s['latency']['p50']} | {s['latency']['p95']} | {s['latency']['max']} | {s['prompt_eval_count_first_call']} |"
        )
    print("\n### 02 ツール呼び出し列と失敗\n")
    for p in sorted(RESULTS.glob("02_deep_agent_*.jsonl")):
        rows = load(p)
        seqs: dict[str, int] = {}
        for r in rows[1:]:
            k = " → ".join(r["tool_seq"]) or "(なし)"
            seqs[k] = seqs.get(k, 0) + 1
        print(f"- {rows[0]['model']}: " + ", ".join(f"`{k}` ×{v}" for k, v in sorted(seqs.items(), key=lambda kv: -kv[1])))
        for r in rows[1:]:
            if not r["ok"]:
                print(f"  - #{r['i']} 失敗: calls={r['model_calls']} seq={r['tool_seq']} notes={r['notes_chars']}c speak={r['speak_count']} final={r['final_text'][:80]!r} err={r['error']}")

    print("\n## 03 日本語短文(reasoning 条件別)\n")
    print("| モデル | reasoning | 平均秒 | 平均 eval tok | reasoning_content あり | <think> 混入 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for p in sorted(RESULTS.glob("03_japanese_*.jsonl")):
        rows = load(p)
        model = p.stem.replace("03_japanese_", "")
        for cond in (False, True, None):
            rs = [r for r in rows if r["reasoning"] is cond]
            if not rs:
                continue
            print(
                f"| {model} | {cond} | {sum(r['wall_sec'] for r in rs)/len(rs):.1f} | {sum(r['eval_count'] or 0 for r in rs)/len(rs):.0f} | "
                f"{sum(r['reasoning_chars']>0 for r in rs)}/{len(rs)} | {sum(r['has_think_tag'] for r in rs)}/{len(rs)} |"
            )
    print("\n### 03 サンプル出力(reasoning=False)\n")
    for p in sorted(RESULTS.glob("03_japanese_*.jsonl")):
        rows = load(p)
        print(f"- **{p.stem.replace('03_japanese_', '')}**")
        for r in rows:
            if r["reasoning"] is False:
                print(f"  - [{r['i']}] {r['content'].strip().replace(chr(10), ' / ')[:160]}")


if __name__ == "__main__":
    main()
