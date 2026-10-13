# Step 0 技術スパイク(使い捨て)

結果と結論は [docs/experiments/000-spike.md](../docs/experiments/000-spike.md)。本番コードでは再利用しない。

```bash
cd spike && uv sync
uv run python 01_tool_calls.py gemma4:12b --reasoning        # ツール呼び出し 20 プロンプト
uv run python 02_deep_agent.py gemma4:12b --reasoning --runs 10  # Deep Agent 最小構成
uv run python 03_japanese.py gemma4:12b                      # 日本語短文と reasoning 比較
./04_vram.sh                                                 # VRAM 単独計測
uv run python summarize.py                                   # results/ を Markdown 表に
```

`run_all.sh` が小型 4 モデルの一括計測、`run_followup.sh` がその後の追加計測、`run_large.sh` が 27B 級 4 モデルの計測。生データは `results/*.jsonl`(先頭行がサマリ)。
