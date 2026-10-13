"""results/*.jsonl を集計して Markdown 表を出す(LangSmith なしでも読める控え)。

設定ごとに最新のログファイルを使う。同じ設定を複数回回した場合は --all で全ファイルを合算する。
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
CATEGORIES = ["tool_use", "filesystem", "planning", "subagent", "long_context", "structured", "skills"]


def load_runs(use_all: bool, after: str = "", before: str = "~") -> dict[str, list[dict]]:
    """設定ラベル → 行。ファイル名は <label>-<YYYYmmdd>-<HHMMSS>.jsonl。after / before はその時刻で絞る。"""
    by_label: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(RESULTS.glob("*.jsonl")):
        if p.name.startswith("thread_growth"):
            continue
        label, d, t = p.stem.rsplit("-", 2)
        if not (after <= f"{d}-{t}" <= before):
            continue
        by_label[label].append(p)
    runs = {}
    for label, paths in by_label.items():
        chosen = paths if use_all else paths[-1:]
        rows = []
        for p in chosen:
            rows += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        runs[label] = rows
    return runs


def pct(rows: list[dict]) -> str:
    if not rows:
        return "-"
    ok = sum(r["success"] for r in rows)
    return f"{ok}/{len(rows)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--after", default="", help="この時刻以降のファイルだけ(例: 20261014-070210)")
    ap.add_argument("--before", default="~", help="この時刻以前のファイルだけ")
    args = ap.parse_args()
    runs = load_runs(args.all, args.after, args.before)
    labels = sorted(runs)

    print("## 成功数(タスク数)— カテゴリ × 設定\n")
    print("| カテゴリ | " + " | ".join(labels) + " |")
    print("| --- |" + " --- |" * len(labels))
    for cat in CATEGORIES:
        cells = [pct([r for r in runs[l] if r["category"] == cat]) for l in labels]
        print(f"| {cat} | " + " | ".join(cells) + " |")
    for lang in ("ja", "en"):
        cells = [pct([r for r in runs[l] if r["lang"] == lang]) for l in labels]
        print(f"| 言語 {lang} | " + " | ".join(cells) + " |")
    cells = [pct(runs[l]) for l in labels]
    print(f"| **合計** | " + " | ".join(f"**{c}**" for c in cells) + " |")

    print("\n## 速度と安定性 — 設定ごと\n")
    print("| 設定 | 成功率 | レイテンシ p50 / p95 / max 秒 | 呼び出し数 平均 | 暴走 | 上限到達 | エラー | 出力 tok 平均 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for l in labels:
        rows = runs[l]
        lat = sorted(r["latency_s"] for r in rows)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(
            f"| {l} | {sum(r['success'] for r in rows) / len(rows):.0%} | {statistics.median(lat):.1f} / {p95:.1f} / {lat[-1]:.1f} | "
            f"{statistics.mean(r['model_calls'] for r in rows):.1f} | {sum(r['runaway'] for r in rows)} | {sum(r['limit_hit'] for r in rows)} | "
            f"{sum(1 for r in rows if r['error'])} | {statistics.mean(r['output_tokens'] for r in rows):.0f} |"
        )

    print("\n## タスク別の成否(○=成功)\n")
    task_ids = []
    for l in labels:
        for r in runs[l]:
            if r["task_id"] not in task_ids:
                task_ids.append(r["task_id"])
    print("| タスク | " + " | ".join(labels) + " |")
    print("| --- |" + " --- |" * len(labels))
    for tid in task_ids:
        cells = []
        for l in labels:
            rs = [r for r in runs[l] if r["task_id"] == tid]
            cells.append("-" if not rs else ("○" if all(r["success"] for r in rs) else ("△" if any(r["success"] for r in rs) else "×")))
        print(f"| {tid} | " + " | ".join(cells) + " |")

    print("\n## 失敗の内訳\n")
    for l in labels:
        fails = [r for r in runs[l] if not r["success"]]
        if not fails:
            continue
        print(f"- **{l}**")
        for r in fails:
            why = r["error"] or ("暴走" if r["runaway"] else ("上限到達" if r["limit_hit"] else "判定不合格"))
            print(f"  - {r['task_id']}: {why}; calls={r['model_calls']} seq={r['tool_seq'][:8]} final={r['final'][:80]!r}")


if __name__ == "__main__":
    main()
