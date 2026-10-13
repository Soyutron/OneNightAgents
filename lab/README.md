# deepagents-local-lab

ローカル LLM(Ollama)× Deep Agents の用途別ベンチマーク。結果は LangSmith の実験として記録し、モデル・設定ごとに比較する。
知見のまとめは [docs/research/local-deepagents.md](../docs/research/local-deepagents.md)。

## 構成

| ファイル | 役割 |
| --- | --- |
| `tasks.py` / `tasks_extra.py` | タスク定義(7 カテゴリ × 日英、v1 19 件 + v2 10 件)と決定的な判定関数。追加はここ |
| `agent_factory.py` | モデル設定とタスクの features から `create_deep_agent` を組む。`coerce_tool_args`(引数の寛容な変換)もここ |
| `run_eval.py` | タスクを LangSmith データセットへ同期し、`evaluate()` で 1 設定 = 1 実験を回す。`results/*.jsonl` にも残す |
| `report.py` | `results/` をローカル集計して Markdown 表を出す(`--after` / `--before` で期間を絞る) |
| `rescore.py` | 判定関数を直した後、保存済み結果を再採点する |
| `thread_growth.py` | 多ターンでの履歴の伸びと要約の発火、序盤の事実の想起を測る |
| `structured_alt.py` | 構造化出力の代替(`with_structured_output(json_schema)`)を試す |
| `run_matrix.sh` ほか `run_*.sh` | 複数設定の逐次実行(GPU は 1 つ) |

カテゴリ: `tool_use`(ツール選択・引数・不要呼び出し・エラー回復・連鎖)、`filesystem`(仮想 FS の read/edit/grep/write、横断 rename)、`planning`(write_todos)、`subagent`(task ツール)、`long_context`(grep と長文読解)、`structured`(response_format)、`skills`(SKILL.md の読み込みと選択)。

## 使い方

```bash
cd lab && uv sync
uv run --env-file ../.env python run_eval.py gemma4:26b --reasoning true --coerce   # 全タスク(推奨構成)
uv run --env-file ../.env python run_eval.py qwen3.5:9b --category tool_use,skills  # 絞り込み
uv run --env-file ../.env python run_eval.py gemma4:26b --coerce --repeat 3         # 分散を見る
./run_matrix.sh "gemma4:26b true" "qwen3.5:9b false"                               # 複数設定
uv run python report.py --after 20261014-070210                                     # ローカル集計
uv run --env-file ../.env python thread_growth.py gemma4:26b --num-ctx 8192 --profile
```

LangSmith では、データセット `deepagents-local-lab-<hash>` の Experiments タブで実験を並べて比較できる(feedback: `success` `latency_s` `model_calls` `runaway` `limit_hit` `error`)。実験メタデータに `model_name` `reasoning` `num_ctx` `coerce` `deepagents` を付けているので、フィルタとグルーピングに使える。失敗の原因は各 run のトレースでツール run の inputs / error を見る。

## タスクを増やすとき

1. `tasks_extra.py`(または新しいモジュール)に `Task(...)` を追加する。判定関数は outputs の `final` / `files` / `custom_calls` / `tool_seq` / `structured` / `read_paths` / `todos_used` を見る。
2. プロンプトやファイルを変えるとデータセット名のハッシュが変わり、次回実行時に新しいデータセットが作られる(古い実験は古いデータセットに残る)。
3. 判定を直したら `rescore.py` でローカル結果を再採点する。LangSmith 側は `evaluate(experiment=...)` で評価器を追加する。

## 計測の注意

- `model_calls` と `tool_seq` にはサブエージェント内の呼び出しも含まれる(コールバックが伝播する)。
- `runaway` は 1 回のモデル呼び出しが `num_predict`(既定 2048)に達したことを指す。
- 判定は決定的で、LLM ジャッジは使わない。自然さなどの主観評価は LangSmith の UI で目視する。
- seed 固定なので同一プロンプトの再実行は同じ結果になる。分散を見るには `--repeat` と seed の変更を組み合わせる。
