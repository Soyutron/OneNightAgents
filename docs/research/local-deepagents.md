# ローカル LLM × Deep Agents の実用性調査

- 日付: 2026-10-14(初版)
- 目的: クラウド API に依存せず、ローカル LLM(Ollama)で Deep Agents の能力をどこまで引き出せるかを、用途別に実測して把握する。ワンナイト人狼に限らず、今後のプロジェクトで「どの用途に・どのモデルと構成で・どんな制約付きなら使えるか」を判断できる知見を蓄積する。
- 計測の仕組み: [lab/](../../lab/README.md)(LangSmith の実験として記録。モデル・設定ごとに比較できる)
- 先行調査: [experiments/000-spike.md](../experiments/000-spike.md)(人狼向けのツール呼び出し・テンポ検証)

---

## 1. 要約

- **ローカル LLM で Deep Agents は実用になる。** 32GB GPU 1 枚で、gemma4:26b(reasoning=True + 引数変換ミドルウェア)は 7 用途 29 タスクを全問正解、1 タスク p50 1.8 秒 / p95 6.7 秒。qwen3.8:27b、qwen3.6:27b、gemma4:12b(VRAM 10GB)も 28/29。計画(`write_todos`)、サブエージェント委譲(`task`)、スキルの必要時読み込み、12K トークン文書の grep 検索、3 ファイル横断の編集まで安定して動く。
- **ハーネス既定のままでは使えない箇所が 4 つある。** いずれもモデルを変えても残る、ローカル固有の落とし穴で、対策は 5.1 の 10 行の設定に収まる。
  1. 要約が発火しない(`ChatOllama` にプロファイルがなく閾値 170K)。`profile={"max_input_tokens": num_ctx}` で解決。放置するとコンテキスト溢れが「応答が空になる」無言の劣化として現れる(3.4 で再現)。
  2. JSON を書くときに `content` へ dict を渡し、エラーを返しても直らない(全モデル)。`wrap_tool_call` で変換すれば解決。
  3. 暴走(`num_ctx` まで生成)は `num_predict` でしか止められない。reasoning=True なら 29 タスク × 6 モデルで 1 件。
  4. ツール成功後に同じ呼び出しを繰り返すモデルがある。ツールの冪等化と `ModelCallLimitMiddleware`。
- **弱い用途は構造化出力。** `response_format=ToolStrategy` はモデルが自発的にツールを呼ぶ前提で、本文で答えてしまうと空になる(1/3〜3/3 でばらつく)。Ollama の制約付き生成(`with_structured_output(json_schema)`)を使う(3.3)。
- **reasoning=True が既定。** False は 2 倍速いが、スキルを読まない・判断を誤るで 2〜4 タスク落ちる。gpt-oss:20b は思考を切れず、ハーネス下では壊れたツール JSON を出して 23/29 と最下位。単発のツール呼び出し精度(000-spike では 20/20)と、ハーネス下の成績は別物。
- **比較の仕組み**: `lab/` のタスクを LangSmith データセットにし、1 設定 = 1 実験で `success` `latency_s` `model_calls` `runaway` `limit_hit` `error` を feedback として記録。新しいモデルやミドルウェアは 1 コマンドで同じ表に並び、失敗の原因はトレースのツール run で読める(本調査の原因特定はすべてこれで行った)。

---

## 2. 調査の設計

### 2.1 環境

| 項目 | 値 |
| --- | --- |
| GPU / RAM | RTX 5090 32GB / 60GB |
| Ollama | 0.33.3 |
| deepagents / langchain / langchain-ollama / langsmith | 0.7.24 / 1.4.5 / 1.1.0 / 0.14.8 |
| LangSmith | APAC リージョン、プロジェクト `onenightagents`、データセット `deepagents-local-lab-<hash>` |

### 2.2 比較するモデル(Ollama ライブラリの `tools` タグ付き、16K コンテキスト込みで 32GB に収まるもの)

| モデル | VRAM(num_ctx=16384) | 系統 |
| --- | --- | --- |
| gemma4:26b | 19.7GB | Google、dense |
| gemma4:12b | 9.7GB | 同上の小型 |
| qwen3.8:27b | 19.4GB | Alibaba、ライブラリで最多 pull |
| qwen3.6:27b | 19.3GB | 同系列の 1 つ前 |
| qwen3.5:9b | 7.3GB | 同系列の小型 |
| gpt-oss:20b | 13.3GB | OpenAI、MoE、常時思考 |

### 2.3 用途(カテゴリ)とタスク

Deep Agents の機能ごとに「その機能を使わないと解けない」タスクを日英で用意し、判定は決定的(LLM ジャッジなし)。

| カテゴリ | 検証する機能 | タスク(ja/en) | 成功の定義 |
| --- | --- | --- | --- |
| tool_use | カスタムツールの選択・引数・列挙値、ツール不要の判断、ツールエラーからの回復 | 天気 (en)、通貨換算 (ja)、天気→リマインダーの連鎖 (ja)、ツール不要 (en)、非対応通貨 (ja) | 期待ツールを 1 回だけ、引数がスキーマどおり、最終回答にツール結果 |
| filesystem | 仮想 FS(StateBackend)の read / edit / grep / write | バグ修正を edit_file で (en)、3 メモの要約 (ja)、grep で TODO 列挙 (ja)、JSON と README 作成 (en) | ファイル内容の検査 |
| planning | `write_todos`(TodoListMiddleware、オプトイン) | 3 ファイルの小プロジェクト構築 (en)、仕様 → テスト観点 3 ファイル + INDEX (ja) | 全ファイルが正しく、todo を使った |
| subagent | `task` ツールでカスタムサブエージェントに委譲 | 用語集ツールを持つ translator (en)、/docs/ を読む researcher (ja) | task が呼ばれ、結果が最終回答 / ファイルに反映 |
| long_context | 長いファイルの grep と読解 | 400 行ログの ERROR 集計 (en、約 9K トークン)、議事録の担当者 (ja) | 数値・固有名詞の一致 |
| structured | `response_format=ToolStrategy(Pydantic)` | 請求書抽出 (en)、イベント抽出 (ja) | `structured_response` のフィールド一致 |
| skills | `skills=["/skills/"]` の SKILL.md を必要時に読む | 報告書フォーマット (en)、俳句の返答形式 (ja) | SKILL.md を read_file し、規則に従った |

共通設定: `num_ctx=16384`、`num_predict=2048`、`temperature=0.3`、`seed` 固定、`ModelCallLimitMiddleware(run_limit=タスク別 3〜14)`、FS ツールは `ls read_file write_file edit_file glob grep`(`execute` は StateBackend では使えないので外す)、general-purpose サブエージェントは無効。

### 2.4 LangSmith での比較・改善の仕組み

- `lab/tasks.py` のタスクを LangSmith データセットに同期(内容のハッシュを名前に含め、タスクを変えると新データセット)。
- `evaluate()` で 1 設定 = 1 実験。feedback は `success` `latency_s` `model_calls` `runaway` `limit_hit` `error`。実験メタデータに `model_name` `reasoning` `num_ctx` `deepagents` を付ける。
- データセットの Experiments タブで実験を並べ、平均 success とレイテンシを比較。各 run のトレースから、失敗したツール呼び出しの引数・エラーをその場で読める(本調査の失敗分析はこれで行った)。
- ローカルにも `lab/results/*.jsonl` を残し、`report.py` で表にする(LangSmith なしでも読める控え)。
- 改善サイクル: 失敗したタスクのトレースを読む → プロンプト / ツール説明 / ミドルウェアを直す → 同じデータセットで再実験 → Experiments タブで前後比較。

---

## 3. 結果

### 3.1 用途別ベンチマーク v1(19 タスク)

LangSmith データセット `deepagents-local-lab-3598593e`。`coerce` は 4.2 / 4.3 の引数変換ミドルウェアあり。

| 設定 | 合計 | tool_use | filesystem | planning | subagent | long_ctx | structured | skills | ja / en | p50 / p95 秒 | 呼出 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **qwen3.8:27b think + coerce** | **19/19** | 5/5 | 4/4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 10/10, 9/9 | 2.1 / 16.0 | 3.0 |
| qwen3.8:27b think | 18/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 10/10, 8/9 | 2.8 / 18.7 | 3.2 |
| qwen3.6:27b think | 18/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 10/10, 8/9 | 2.9 / 22.0 | 3.2 |
| gemma4:26b think + coerce | 18/19 | 5/5 | 4/4 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 9/10, 9/9 | 1.8 / 12.9 | 3.9 |
| gemma4:12b think(coerce 有無同じ) | 18/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 10/10, 8/9 | 2.4 / 14.3 | 3.8 |
| gemma4:26b think | 17/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 9/10, 8/9 | 1.8 / 8.3 | 4.1 |
| gpt-oss:20b | 17/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 10/10, 7/9 | 1.2 / 25.1 | 4.1 |
| gemma4:26b nothink | 16/19 | 4/5 | 3/4 | 2/2 | 2/2 | 2/2 | 2/2 | 1/2 | 8/10, 8/9 | 0.8 / 19.1 | 3.9 |
| qwen3.5:9b think | 16/19 | 5/5 | 2/4 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 9/10, 7/9 | 2.0 / 12.4 | 3.7 |
| qwen3.5:9b nothink | 14/19 | 5/5 | 3/4 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 | 8/10, 6/9 | 0.9 / 7.3 | 3.2 |

読み方:

- **planning / subagent / long_context / skills(reasoning=True)は全モデルが全問正解。** `write_todos` による計画、`task` によるサブエージェント委譲(サブエージェントは親の仮想 FS を読めた)、9K トークンのログを grep で集計、SKILL.md の必要時読み込みは、9B〜27B のローカルモデルで問題なく動く。
- 失敗は 3 種類に集中する。(1) JSON ファイルを書くときに `content` へ dict を渡す(全モデル。→ 4.2、coerce で解消)、(2) 日本語の構造化出力で `ToolStrategy` のツールを呼ばず本文で答える(→ 4.6)、(3) reasoning=False でスキルを読まない・ツール不要の判断を誤る。
- 暴走(`num_predict` 到達)は reasoning=True の全設定でゼロ。000-spike で見た暴走は reasoning=False の現象。
- 1 タスクの p50 は 1〜3 秒、p95 は 8〜25 秒(planning の 10〜12 呼び出しが上限側)。27B 級と 12B 級の速度差は小さく、gpt-oss:20b(MoE)が最速だが p95 は長い。
- v1 は多くの設定が 17〜19/19 で天井に当たった。差を出すために v2(3.2)で難度を上げた。

### 3.2 用途別ベンチマーク v2(29 タスク、全設定 reasoning=True + coerce)

v1 に 10 タスクを追加(`lab/tasks_extra.py`): 2 都市の天気比較、2 段階の通貨換算、3 ファイル横断の関数 rename、JSON の読み書き更新、日本語の exact-match 編集、5 ステップのデータパイプライン(todo 必須)、2 種のサブエージェントの使い分け、約 12K トークンの文書からの検索、ネストしたスキーマの構造化出力、2 つのスキルからの選択。データセット `deepagents-local-lab-cc89fb8a`。

| 設定 | 合計 | tool_use | filesystem | planning | subagent | long_ctx | structured | skills | ja / en | p50 / p95 秒 | 呼出 平均 | 暴走 | エラー |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **gemma4:26b** | **29/29** | 7/7 | 7/7 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 16/16, 13/13 | 1.8 / 6.7 | 4.4 | 0 | 0 |
| gemma4:12b | 28/29 | 7/7 | 6/7 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 15/16, 13/13 | 2.7 / 9.8 | 4.3 | 1 | 0 |
| qwen3.8:27b | 28/29 | 7/7 | 7/7 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 15/16, 13/13 | 2.1 / 8.5 | 3.3 | 0 | 0 |
| qwen3.6:27b | 28/29 | 7/7 | 7/7 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 16/16, 12/13 | 3.2 / 12.8 | 3.2 | 0 | 0 |
| qwen3.5:9b | 27/29 | 7/7 | 6/7 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 15/16, 12/13 | 2.0 / 8.8 | 4.1 | 0 | 0 |
| gpt-oss:20b | 23/29 | 7/7 | 5/7 | 3/3 | 2/3 | 2/3 | 1/3 | 3/3 | 13/16, 10/13 | 1.4 / 9.2 | 4.0 | 0 | **2** |
| nemotron-3-nano:30b(2026-10-15 追記) | 23/29 | 6/7 | 6/7 | **1/3** | 3/3 | 3/3 | 2/3 | 2/3 | 13/16, 10/13 | 3.2 / 15.7 | 3.3 | **5** | 0 |
| **nemotron-3.5-lightning:30b**(2026-10-15 追記) | **28/29** | 6/7 | 7/7 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 15/16, 13/13 | **1.8 / 5.8** | 4.3 | 0 | 0 |

(2 種サブエージェントの課題は、判定が結論文に含まれない語を要求していたため、判定を修正して `lab/rescore.py` で再採点した。LangSmith 上の当該 3 件の feedback は修正前のまま)

読み方:

- **難度を上げても gemma4:26b は全問正解、27B 級と gemma4:12b は 28/29。** 3 ファイル横断の rename(grep → edit_file × 3)、5 ステップのパイプライン(todo で計画、CSV の数値フィルタ、平均の計算)、12K トークン文書からの grep 検索、2 つのスキルからの正しい選択は、ローカルモデルで安定して解ける。
- **残る失敗は構造化出力(ToolStrategy)に集中する。** 英語でも日本語でも、モデルがスキーマ名のツールを呼ばずに Markdown の表や JSON コードブロックで答えると `structured_response` が空になる。gpt-oss:20b は日本語キー(「タイトル」「日付」)の JSON を返した。→ 4.6 と 3.3 の代替方式。
- **gpt-oss:20b はハーネス下では最も弱い。** 単発のツール呼び出し(000-spike)では 20/20 だったが、`ls` の引数で壊れた JSON(`{"path":"/pkg","}`)を出して Ollama のパーサが `ResponseError` を投げ(エージェントのループ内で回復できない)、サブエージェントに委譲せず自分で読む、長文で grep を繰り返して上限に達する、など 6 失敗。思考を切れないことも含め、本用途には勧めない。
- gemma4:12b の暴走 1 件(grep の後に空応答のまま `num_predict` 到達)は 12B 級に残るリスク。`num_predict` と上限で 10 秒に抑えられている。
- **nemotron-3-nano:30b(2026-10-15 追記)は gpt-oss:20b と同じ 23/29 だが、失敗の内訳が違う。** 6 失敗のうち 3 件は planning / skills で最初の呼び出しの思考が `num_predict`(2048)に達して空応答(暴走 5 件はすべて思考)、残りは構造化出力を JSON コードブロックで答える(4.6)、「対応外通貨」をツールを呼ばずに説明で済ませる、`edit_file` で `return a - b` を一意でないまま置換して隣の関数まで書き換える、の各 1 件。エラー(パーサ例外)は 0 で、ツール呼び出しの形式自体は崩れない。生成は速く(2048 トークンの思考が 5.9 秒)、`num_predict` を 4096 にした再評価で planning 3 件は通った(26/29 相当)。詳細は [nemotron-deepagents.md](nemotron-deepagents.md)。
- **nemotron-3.5-lightning:30b(2026-10-15 追記)は 28/29 で、p95 5.8 秒は全設定で最速、暴走 0。** 唯一の失敗は「都市名だけ答えて」に `Osaka` と英語で答えたもので(判定は「大阪」を要求)、判断の誤りではない。Nano 3 で落ちた planning・structured・skills が全部通り、思考の長さが用途に見合う(出力 845 トークン/タスク)。ローカルの Nemotron を使うなら Nano 3 ではなくこちら。
- 1 タスク p50 は 1.8〜3.2 秒、p95 は 7〜13 秒。planning の 12〜16 呼び出しでも 15 秒以内。

### 3.3 構造化出力の代替: `ChatOllama.with_structured_output(method="json_schema")`(`lab/structured_alt.py`)

structured の 3 タスクを、Deep Agent を介さず Ollama の制約付き生成で 3 回ずつ。

| モデル | 請求書 (en) | 注文・ネスト (en) | イベント (ja) | 1 回の時間 |
| --- | --- | --- | --- | --- |
| gemma4:26b(reasoning=True) | 3/3 | 3/3 | 0/3 ※ | 1.4〜6.6 秒 |
| gemma4:12b(True) | 3/3 | 3/3 | 0/3 ※ | 1.5〜3.5 秒 |
| gpt-oss:20b | 3/3 | 3/3 | 3/3 | 0.8〜6.2 秒 |
| qwen3.5:9b(True) | 3/3 | 3/3 | 0/3 ※ | **85〜105 秒** |
| qwen3.5:9b(False) | — | — | 日付は正規化 | 2.2 秒 |

※ 抽出内容は正しく、日付が「2026年10月21日」のままで判定の `YYYY-MM-DD` に合わなかっただけ。

分かったこと:

- **形(キーと型とネスト)は JSON Schema の制約付き生成で 100% 保証される。** ToolStrategy のように「ツールを呼ばず本文で答える」ことはない。
- **書式(日付の形式など)は保証されない。** Pydantic の `Field(pattern=...)` は Ollama のグラマーに反映されず、生成後の Pydantic 検証で落ちる(gemma4 で 6/6 失敗)。プロンプトに「日付は YYYY-MM-DD 形式で」と書くと 6/6 正規化された。書式はプロンプトかコード側の正規化で担保する。
- **qwen3.5:9b は reasoning=True と制約付き生成の相性が悪い**(思考が 100 秒近く続く)。構造化出力では reasoning=False にする。gemma4 は True でも数秒。
- 推奨パターン: エージェント本体は Deep Agent、最終成果物の型付けは `with_structured_output(json_schema)` の 1 回呼び出しに分ける。

### 3.4 多ターンのスレッド: 履歴の成長と要約(`lab/thread_growth.py`)

1 スレッドで「事実を /facts.md に書く → 16 ターンにわたり /journal.md に 1 行ずつ追記 → ファイルを読まずに最初の事実を答える → journal の行数を数える」を実行。gemma4:26b、reasoning=True。

| 設定 | 最大プロンプト tok | 要約の発火 | journal 行数 | 事実の想起 | 症状 |
| --- | --- | --- | --- | --- | --- |
| num_ctx=16384、profile なし(既定) | 9,560 | なし | 16/16 | ○ | 16 ターンでは収まる。1 ターン +600 tok で伸び続ける(毎回ファイルを読み直すため) |
| num_ctx=16384、profile あり | 9,434 | なし(閾値 13.9K 未満) | 16/16 | ○ | 同上 |
| **num_ctx=8192、profile なし** | 8,187 | **なし** | **15/16** | **×** | 16 ターン目でプロンプトが 8.1K を超えると Ollama が先頭を黙って切り詰め、**以降の応答が空になる**。エラーも警告もない |
| **num_ctx=8192、profile あり** | 6,599 | **14 ターン目**(6,660 → 2,633 tok) | 16/16 | ○ | 要約後も作業を継続し、コードネームとリリース日は要約に残った |

結論:

- **`ChatOllama(profile={"max_input_tokens": num_ctx})` は必須。** これがないと Deep Agents の要約は 170K トークンまで発火せず、ローカルモデルではコンテキスト溢れが「無言の劣化」として現れる。原因が分かりにくい最悪の失敗型。
- 要約 1 回のコストは約 3 秒(gemma4:26b)。要約後に事実が残るかは要約プロンプト次第で、本テストでは残った。
- 履歴は 1 ターンあたり 500〜700 トークン伸びる(ツールの往復 + ファイル内容のエコー)。`num_ctx=16384` では、要約なしで約 25 ターンが上限の目安。ワンナイト人狼(1 プレイヤー 20〜30 判断)はまさにこの境界にあり、profile 指定は初期版から入れる。

---

## 4. 発見した課題と対策

計測中に特定した、ローカル LLM × Deep Agents 固有の落とし穴。モデルを変えても残るものが多い。

### 4.1 要約(SummarizationMiddleware)が実質的に発火しない

`ChatOllama` にはモデルプロファイル(`max_input_tokens`)がないため、deepagents 0.7.24 の既定は **170,000 トークンで発火、直近 6 メッセージを保持** になる。`num_ctx=16384` では要約より先に Ollama がプロンプト先頭(system prompt を含む)を黙って切り詰める。

対策: `ChatOllama(..., profile={"max_input_tokens": num_ctx})` を渡す。これだけで既定が `trigger=("fraction", 0.85)`、`keep=("fraction", 0.10)` に切り替わる(`compute_summarization_defaults` で確認)。多ターンでの効果は 3.4 を参照。

### 4.2 `write_file` の `content` に JSON オブジェクトを渡してしまう

JSON ファイルを書かせると、gemma4:26b は `content` に文字列ではなく JSON オブジェクト(dict)を渡す。Ollama のツール呼び出しパーサがネストした JSON をオブジェクトとして返すため、`WriteFileSchema` の Pydantic 検証で `Input should be a valid string` になる。**エラー文を返しても同じ誤りを 5 回繰り返し、上限で終了した。** エラーからの自己修正は期待できない。

対策: ツール呼び出しの引数を受け取る側で寛容に変換する(`wrap_tool_call` ミドルウェアで `content` が dict / list なら `json.dumps` する)。文字列型の引数に構造化データが入る可能性があるツールは、すべて同様の前処理が要る。

### 4.3 列挙値(Literal)が守られない

000-spike で確認。`Literal["villager", ...]` だけでは日本語の値(「村人」)や文字列の `"None"` が来る。description に許容値を明記すると改善するが、`"None"` → `None` のような正規化はツール側で持つ。

### 4.4 暴走(num_ctx まで生成し続ける)

gemma4:12b(reasoning=False)、qwen3:8b、qwen3.5:9b(False)、qwen3.6:27b(True)で観測。ツール呼び出しの構文を崩したまま生成が止まらず、`num_ctx` が埋まるまで 80〜100 秒。`num_predict` で上限を切る以外に防ぐ手段がない(ハーネスは関与できない。`ModelRetryMiddleware` の対象にもならない)。

対策: `num_predict`(本調査は 2048)+ 呼び出しタイムアウト。暴走率を feedback `runaway` として実験ごとに監視する。

### 4.5 reasoning の既定

gemma4 / qwen3.x は `reasoning=True` でツール呼び出しの安定性が上がり、暴走も減る(コストは 1 呼び出し +1〜3 秒)。gpt-oss は `reasoning=False` を指定しても思考する(効かない)。`reasoning=None` は思考するが `<think>` は本文に出ない(Ollama が分離)。

### 4.6 構造化出力は「ツールを呼ぶかどうか」次第

`response_format=ToolStrategy(...)` は、モデルがスキーマ名のツールを自発的に呼ぶことで成立する。v2 では 6 設定中 4 設定が 1 件以上、Markdown の表や JSON コードブロックで答えて `structured_response` が空になった(gpt-oss:20b は日本語キーの JSON)。確実に取りたい場合は `ChatOllama.with_structured_output(method="json_schema")`(Ollama の制約付き生成)を使う。形は保証されるが書式はプロンプトで指示する(3.3)。

### 4.7 ツールが成功した後も同じ呼び出しを繰り返す

gemma4:12b は coerce で `write_file` が成功する(「Updated file /app/config.json」が返る)ようになっても、同じ内容の `write_file` を 5 回繰り返して README に進まず、上限で終了した。000-spike の qwen3:8b(「受理しました」の後も speak を繰り返す)と同じ型で、**ツールの成功応答を「完了」と認識できない**。モデル固有(gemma4:26b、qwen3.8:27b では起きない)。

対策: 副作用のあるツールは冪等にし、2 回目以降は「すでに完了しています。次の手順に進んでください」と返す。`ModelCallLimitMiddleware` は最後の砦として必ず入れる。

### 4.8 reasoning=False は速いが「読む・判断する」系で落ちる

gemma4:26b は reasoning=False で p50 0.8 秒(True の半分以下)だが、スキルを読まずに俳句だけ返す、ツール不要の判断で本文回答に失敗、など 17 → 16/19。qwen3.5:9b は 16 → 14/19 で、スキル 0/2、構造化出力 0/2。**ローカルの小型モデルで Deep Agents を使うなら reasoning=True を既定にし、速度が要る用途だけ個別に False を試す。**

### 4.9 サブエージェントの呼び出しはコールバックで親に見える

`task` で委譲したサブエージェントのモデル呼び出し・ツール呼び出しは、親に渡したコールバック(と LangSmith トレース)に全部現れる。本ラボの `model_calls` / `tool_seq` にはサブエージェント分が含まれる。StateBackend の仮想 FS はサブエージェントからも読めた(researcher が `/docs/` を読んで親に報告できた)。

---

### 4.10 Ollama のツール呼び出しパーサが例外を投げると、エージェントのループでは回復できない

gpt-oss:20b が `{"path":"/pkg","}` のような壊れた JSON を出すと、Ollama 側のパースエラーが `ResponseError` として `invoke()` から飛ぶ。ツールの引数エラー(Pydantic)はツールメッセージとしてモデルに返るので再試行の余地があるが、この例外はモデル呼び出しそのものの失敗なので、`ModelRetryMiddleware` で再試行するか、呼び出し側で catch して再要求するしかない。

### 4.12 seed 固定でも出力はリクエスト履歴に依存する

Ollama(0.33.3、`keep_alive` でモデル常駐)は、`seed` を固定しても「モデル読み込み後に処理したリクエストの履歴」が違うと同じプロンプトに違う出力を返す(前方一致する KV キャッシュの再利用による数値差、と考えるのが整合的)。読み込み直後から同じ履歴をたどると完全に再現する(3 回確認)。**1 本の run の A/B は履歴の違いによる差を含む**ので、比較は条件ごとに履歴違いの複製を 3 本以上取り、変更なしの複製で出る幅(ノイズフロア)に収まる差は効果と読まない(OneNightAgents の docs/experiments/013)。

### 4.11 LangSmith での失敗分析が効く箇所

本調査で特定した 4.2(dict content)、4.7(成功後の繰り返し)、4.10(パースエラー)は、いずれも **LangSmith のツール run の inputs / error を見て 1 分で原因が分かった**。ローカルの JSONL は成否と呼び出し列しか持たないので、原因調査はトレース、集計はデータセットの Experiments タブ、という分担が実用的。

---

## 5. 用途別の推奨構成

### 5.1 共通の既定(どの用途でも入れる)

```python
llm = ChatOllama(
    model="gemma4:26b",
    num_ctx=16384,
    num_predict=2048,            # 暴走の上限(4.4)
    reasoning=True,              # 小型モデルほど効く(4.5)
    profile={"max_input_tokens": 16384},  # これがないと要約が発火しない(4.1)
    keep_alive="30m",
    validate_model_on_init=True,
)
agent = create_deep_agent(
    model=llm,
    middleware=[
        FilesystemMiddleware(backend=backend, tools=[...必要なものだけ...]),
        ModelCallLimitMiddleware(run_limit=N, exit_behavior="end"),   # 4.7 の砦
        coerce_tool_args,                                            # 4.2 / 4.3(lab/agent_factory.py)
        ModelRetryMiddleware(max_retries=1),                         # 4.10
    ],
    ...
)
```

- 行動を提出するツールは冪等にし、2 回目以降は「完了済み」を返す(4.7)。
- 列挙値は description に許容値を書く(4.3)。
- general-purpose サブエージェントは、使わない用途では `HarnessProfile` で無効化してツール定義のトークンを節約する。

### 5.2 用途別

| 用途 | 推奨モデル | 構成の要点 | 期待できる水準(本計測) |
| --- | --- | --- | --- |
| ツール呼び出し中心のアシスタント(API 連携、業務ツール) | gemma4:26b / gemma4:12b(VRAM 10GB で足りる) | reasoning=True、coerce、ツール説明に列挙値 | 2 段階の依存関係、比較、エラー回復まで 7/7。1 判断 1〜3 秒 |
| ファイル編集・コード修正(仮想 FS / FilesystemBackend) | gemma4:26b、qwen3.8:27b | `edit_file` を使わせる指示、JSON は coerce | 3 ファイル横断の rename、JSON 更新、日本語 exact-match 編集まで 7/7 |
| 多段の作業(計画 → 実行 → 検証) | gemma4:26b、qwen3.x 27B | `TodoListMiddleware` をオプトイン、`run_limit` 15〜20 | 5 ステップのパイプラインを 12〜16 呼び出し、15 秒以内で完走 |
| サブエージェント委譲 | gemma4:26b、qwen3.8:27b | サブエージェントは StateBackend を共有できる。`run_limit` はサブエージェント分も消費するので余裕を持つ | 2 種の使い分けまで 3/3(gpt-oss は自分で読んでしまう) |
| 長文の検索・読解(ログ、議事録、文書) | 全モデル可(gpt-oss 以外) | 「全文を読まず grep」を指示。12K トークンの文書は `num_ctx=16384` に全文は入らない | 9K〜12K トークンの文書から 3/3 |
| 構造化出力(抽出・分類) | **Deep Agent の `response_format` ではなく `ChatOllama.with_structured_output(method="json_schema")`** | Ollama の制約付き生成で確実に JSON を得る(3.3)。エージェントの最終回答を別途この方式で整形する 2 段構成が堅い | ToolStrategy は 1/3〜3/3 でばらつく |
| スキル(手順書の必要時読み込み) | 全モデル可(reasoning=True) | SKILL.md の description を具体的に書く | 2 つから正しい方を選ぶまで 3/3。reasoning=False では読まないことがある |
| 多ターンの長い対話(1 スレッドで 20 ターン以上) | gemma4:26b | **profile 必須**。要約後に残したい事実は system prompt かメモファイルに置く | 8K コンテキストでも要約で継続(3.4) |
| 会話ゲーム・ロールプレイ(人狼など) | gemma4:26b(000-spike) | 役職秘匿を明示、行動ツールは最初の 1 回だけ受理 | 1 判断 p95 6.5 秒、役職を漏らさない |

### 5.3 VRAM 別の選択肢

| VRAM | 第一候補 | 備考 |
| --- | --- | --- |
| 32GB | gemma4:26b(20GB) | 本調査の最良。2 モデル同時常駐は不可 |
| 16〜24GB | gemma4:12b(10GB)、qwen3.5:9b(7GB) | 12b は 28/29。9b は 27/29 だが役職秘匿などロールプレイの質は落ちる(000-spike) |
| 8〜12GB | gemma4:12b を `num_ctx=8192` + profile で | 要約が頻繁に走る前提。未計測 |

---

## 6. 限界と未検証

- **判定は決定的なものだけ。** 文章の質(要約の良さ、説明の分かりやすさ)は測っていない。LLM ジャッジを足すならクラウド API が要るので、ローカル縛りの範囲では人手評価になる。
- **タスクは各カテゴリ 3 件、試行 1 回、seed 固定。** 成功率の差 1〜2 件はノイズの範囲。LangSmith の `num_repetitions` で 3 回回せば分散が見える(未実施)。
- **MCP ツール、`execute`(サンドボックス)、`FilesystemBackend` の実ディスク操作、`StoreBackend` のゲーム横断メモリ、human-in-the-loop は未検証。** 特に MCP はツール定義が長くなりがちで、固定コスト(1 呼び出し 1.5K トークン)がさらに増える。
- **並列実行は未計測。** 1 GPU・逐次が前提。`OLLAMA_NUM_PARALLEL` でのスループットは別途。
- **英語プロンプト + 日本語出力の構成は試していない。** 日英で成功率に大差はなかった(ja 15〜16/16、en 12〜13/13)ので、優先度は低い。
- 量子化は Ollama 既定(Q4 系)のみ。Q8 版(例: qwen3.8:27b-q8_0 29GB)は 16K コンテキストを含めると 32GB に収まらない。
- モデルは 2026-10 時点のもの。Ollama ライブラリの更新が速いので、新モデルは `lab/run_eval.py <model>` を 1 回回せば同じ表に並ぶ。

## 7. 次にやること

1. 人狼(OneNightAgents)の Step 1〜3 に 5.1 の既定を組み込む。特に `profile` と coerce と冪等な行動ツール。
2. `num_repetitions=3` で主要 3 モデルを回し、成功率の分散を見る。
3. 構造化出力は `with_structured_output(json_schema)` を Deep Agent の最終整形に組み合わせる実装パターンを作り、ラボに追加する。
4. MCP ツールを 1 つ(例: ファイル検索か Web 取得)つないだカテゴリを追加し、ツール定義のトークン増とローカルモデルの選択精度を見る。
