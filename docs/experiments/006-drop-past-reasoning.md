# 006: 過去の思考を履歴から外すと、コンテキスト肥大と未提出の連鎖が消えるか

- 日付: 2026-10-15 / 対象ステップ: Step 4 観察項目 1・4

## 仮説

005 で、langchain-ollama が履歴中の AIMessage の `reasoning_content` を `thinking` として毎回送り返すため、暴走した思考(2K〜4K トークン)がゲーム終了まで固定費になり、qwen3.8:27b では `num_ctx` 到達 → ツール不能 → フォールバックの連鎖が起きていた。送信時に「最後の HumanMessage より前」の思考を外す `drop_past_reasoning` を入れると、両モデルで最大コンテキストが下がり、qwen の未提出(文章回答)とフォールバックが消える。勝率や騙りは変わらない。

## 設定

- 005 と同一設定(`reasoning=True`、`num_predict=4096`、ペルソナなし、seed 1〜5)に `drop_past_reasoning` を追加
- A: gemma4:26b `runs/np4096-5` → B: `runs/gemma-drop-5`
- C: qwen3.8:27b `runs/qwen-5` → D: `runs/qwen-drop-5`

## 観察

(記入待ち)

## 結論

(記入待ち)
