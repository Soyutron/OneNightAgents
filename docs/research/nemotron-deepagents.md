# Deep Agents × NVIDIA Nemotron の互換性調査

- 日付: 2026-10-15
- 対象: deepagents 0.7.25 / langchain 1.4.5 / langchain-ollama 1.1.0 / Ollama 0.33.3 / `nemotron-3-nano:30b`(Q4_K_M、24GB)
- 目的: Deep Agents が Nemotron を正式に扱えるか、ツール呼び出し・構造化出力・長期タスクで何が起きるか、本リポジトリ(ローカル Ollama)で使うなら何を変えるべきかを、公式情報と推測に分けて整理する

## 1. 要約

1. **Deep Agents は Nemotron 3 Ultra(550B、クラウド API)だけを正式サポートしている。** 0.7.0 で組み込みのハーネスプロファイルが入り(PR #4192)、LangChain 公式ドキュメントにページがある。Nano / Super / Ollama 経由についての記述は、公式ドキュメントにも GitHub にもない。
2. **プロファイルは `provider:model` のキーで引かれるため、Ollama の `nemotron-3-nano:30b` には何も適用されない**(`ollama:nemotron-3-nano:30b` は未登録)。OpenAI 互換エンドポイント経由でも同じ理由で適用されない(Issue #4555)。
3. **`nemotron-3-nano:30b` は langchain-ollama 経由でツール呼び出し・構造化出力(`format=json_schema`)とも動く**(本リポジトリで確認)。Ollama は Go 実装のレンダラー/パーサー(`nemotron3nano`)を持ち、過去ターンの思考を自動で切り落とすので、006 の `drop_past_reasoning` は Nemotron では no-op になる(gemma4 と同じ)。
4. **既知の問題は「推論モデルの出力が content に入らない」「思考が途中で切れるとツール呼び出しが壊れる」「存在しないツール名を呼ぶ」の 3 系統。** 前者と後者は langchain-ollama + Deep Agents の構成では出ない。**残るのは「思考が `num_predict` で切れる」で、これが本リポジトリでの実質的な制約**: 人狼 bench では呼び出しの 12〜16% が 4096 トークンの思考で切れ、未提出率 28〜32%(gemma4 6%、qwen3.8 0.6%)。lab の短いタスクは `num_predict` を 4096 にすれば 26/29 相当。
5. NVIDIA の公式推奨サンプリング(ツール呼び出し用途で temperature 0.6 / top_p 0.95)を試したが、人狼では思考の長さも未提出率も改善しなかった(007)。**ハーネス側に Nemotron 固有の修正を入れる必要はなく、人狼の既定モデルにもしない。**
6. **2026-08 公開の `nemotron-3.5-lightning:30b` では思考の暴走が消えた**(008)。用途別 29 タスクで 28/29・p95 5.8 秒(全設定で最速)、人狼でも未提出 2〜4%・1 ゲーム 2.5 分と gemma4 並み。ローカルで Nemotron を使うならこちらを **temperature 1.0(公式推奨)** で(009: 0.7 ではなぞり 12.8%・村 0 勝、1.0 でなぞり 5.9%・村 2 勝)。ただし怪盗が交換後の役職と方針を口にする、英語が混じるなど発言の癖があり、人狼の既定モデルは据え置き。

## 2. 公式情報

### 2.1 Deep Agents の Nemotron 対応

| 項目 | 内容 | 出典 |
| --- | --- | --- |
| 組み込みプロファイル | `deepagents>=0.7.0` に Nemotron 3 Ultra 用の `HarnessProfile` が同梱。`create_deep_agent(model=ChatNVIDIA("nvidia/nemotron-3-ultra-550b-a55b"))` だけで自動適用 | [docs: NVIDIA middleware](https://docs.langchain.com/oss/python/integrations/middleware/nvidia)、[PR #4192](https://github.com/langchain-ai/deepagents/pull/4192) |
| 登録キー | `NVIDIA:` / `nvidia:` / `baseten:` / `fireworks:` / `openrouter:` / `nebius:` / `together:` の Ultra の ID のみ。**モデル単位の登録で、プロバイダ全体には及ばない** | `deepagents/profiles/harness/_nvidia_nemotron_3_ultra.py` |
| プロファイルの中身 | system prompt の末尾に `<approach>` `<grounding>` `<loop_control>` `<tool_selection>` `<state_changes>` `<final_answer_completeness>` `<followup_defaults>` `<context_compaction>` の 8 節を追加、`read_file` の説明を差し替え、ミドルウェア 12 個(下表) | 同上 |
| 開発方法 | 「モデルではなくハーネスを調整する」。評価スイートの失敗トラジェクトリから、プロンプト・ツール説明・ミドルウェアを直す。ベンチマークはストキャスティックなので複数回実行で検証 | [NVIDIA 技術ブログ](https://developer.nvidia.com/blog/create-a-langchain-deep-agents-harness-profile-for-nvidia-nemotron-3-ultra-to-improve-performance) |
| 公表値 | LangChain の Deep Agents 評価で 0.86、1 run あたり $4.48(ベンダー発表、第三者検証なし) | [NVIDIA ブログ](https://blogs.nvidia.com/blog/nemotron-langchain-agents-open-stack/)、[LangChain ブログ](https://www.langchain.com/blog/langchain-and-nvidia-launch-the-nemoclaw-deep-agents-blueprint) |
| 他モデル | Nemotron 3.5 Super VL のプロファイルは Issue #6598 / #6599 と draft PR #6596 があるが未マージ(#6598 は bot がクローズ)。deepagents CLI の NVIDIA 既定モデルは `nemotron-3-super-120b`(PR #1801) | [#6598](https://github.com/langchain-ai/deepagents/issues/6598)、[#6599](https://github.com/langchain-ai/deepagents/issues/6599) |
| OpenAI 互換経由 | `ChatOpenAI` で Ultra を指していてもキーが `openai:...` になるので適用されない。例外は出ず、空のプロファイルに黙って落ちる(Issue #4555、bot クローズで未修正) | [#4555](https://github.com/langchain-ai/deepagents/issues/4555) |

Ultra プロファイルのミドルウェア(`_build_extra_middleware`):

| ミドルウェア | 役割 | Nano / Ollama への転用 |
| --- | --- | --- |
| `NemotronTextToolCallParser` | ツール呼び出しを本文に `<tool_call><function=…>` や `{"tool": …}` の文字列で出したとき、構造化された `tool_calls` に直す。存在しないツール名は捨てる | 転用可能。ただし Ollama のパーサーが先に `<tool_call>` を解釈するので、Ollama 経由で本文に残るのは壊れた形式だけ |
| `NemotronReasoningTagCleanupMiddleware` | 本文に残った `<think>…</think>` を `reasoning_content` に移す | Ollama は `think` を分離して返すので通常は不要 |
| `NemotronToolCallShim` | `path` → `file_path` の引数名修正、`read_file` に `limit` を補う、空のツール結果に文字列を入れる | 本リポジトリの `coerce_tool_args` と同種 |
| `ReadFileContinuationNoticeMiddleware` | `read_file` が上限行数ちょうどで終わったら「続きがある」通知を足す | 人狼では不要(メモは短い) |
| `ToolRetryMiddleware` / `ModelRateLimitRetryMiddleware` | FS ツール失敗と 429 の再試行 | ローカルでは 429 は起きない |
| `ChatNVIDIAMessageCompatibilityMiddleware` | ChatNVIDIA 固有のメッセージ正規化 | 不要 |
| `NemotronProgressBudgetMiddleware` / `NemotronPolicyNudgeMiddleware` / `FollowupDisciplineMiddleware` / `EntityResolutionGuardMiddleware` / `FinalAnswerGuardMiddleware` | 進捗予算(モデル呼び出し 16 回・ツール結果 48 件)、業務タスク向けの英語の nudge(最終回答に具体値を入れろ、確認質問を減らせ等) | 英語の業務タスク(予約・レポート・ファイル操作)向けの正規表現で動くので、日本語の人狼には適用されない |

プロファイルはすべて `deepagents.profiles.harness._nvidia_nemotron_3_ultra` の非公開モジュールにあり、公開 API で部分的に取り出す手段はない(`register_harness_profile` に自前の `HarnessProfile` を渡す形になる)。

### 2.2 Nemotron 3 Nano のモデル仕様

| 項目 | 内容 | 出典 |
| --- | --- | --- |
| 構成 | 30B 総パラメータ / 3.5B アクティブ、Mamba-2 + MoE のハイブリッド(Attention 6 層)。対応言語に日本語を含む | [Ollama library](https://ollama.com/library/nemotron-3-nano)、[HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| コンテキスト | 最大 1M。HF 既定は 256K(VRAM の都合)。Ollama は `num_ctx` で指定 | 同上 |
| 推奨サンプリング | **推論タスク: temperature 1.0 / top_p 1.0。ツール呼び出し: temperature 0.6 / top_p 0.95**。Ollama の Modelfile 既定は 1.0 / 1.0 | HF model card、[Unsloth guide](https://unsloth.ai/docs/models/nemotron-3) |
| 思考の切り替え | チャットテンプレートの `enable_thinking`(既定 True)。Ollama では `think` パラメータ(langchain-ollama の `reasoning`)。`/think` `/no_think` を user / system に書く方式もレンダラーが解釈する | HF model card、Ollama レンダラー |
| 思考の予算 | model card に「reasoning budget」の 2 段呼び出しクライアント例があるが、Ollama には相当機能がない(PR #17566 が open) | HF model card、[ollama PR #17566](https://github.com/ollama/ollama/pull/17566) |
| ツール呼び出し形式 | XML 風(`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`)。vLLM / SGLang / TRT-LLM では `qwen3_coder` パーサーを流用 | HF model card、[技術レポート](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) |
| 構造化出力 | model card に JSON モードの記述なし(学習データに「structured outputs」を含むとだけ) | HF model card |
| 思考を切った場合 | 「難しいプロンプトでわずかに精度が落ちる」と明記 | Ollama library |

### 2.3 Ollama の実装(`model/renderers/nemotron3nano.go`、`model/parsers/nemotron3nano.go`)

Ollama 0.33.3 の `nemotron-3-nano:30b` は Modelfile の TEMPLATE が `{{ .Prompt }}` だけで、`RENDERER nemotron-3-nano` / `PARSER nemotron-3-nano` の Go 実装がプロンプト構築と出力解釈を担う。読んで分かったこと:

- **過去ターンの思考は描画されない。** 最後の user メッセージより前の assistant メッセージは `<think></think>` に潰される(`shouldTruncate = i < lastUserIdx`)。本リポジトリの `drop_past_reasoning` と同じ境界なので、Nemotron では no-op(§4 で確認)。
- **ツール結果は tool 名なしで `<tool_response>` ブロックとして user ロールに描画される。** langchain-ollama が `tool_name` を送らない問題([langchain #41085](https://github.com/langchain-ai/langchain/issues/41085)、gemma4 では `response:unknown` になる)は Nemotron には影響しない。
- **ツール一覧の末尾に、固定の例示文が入る**: `<function=example_function_name><parameter=example_parameter_1>…`。
- 思考が有効なときは生成プロンプトが `<think>\n` で終わり、無効なときは `<think></think>` を前置する。
- パーサーは思考中に `</think>` が来なくても `<tool_call>` を見つけたら思考を閉じてツール呼び出しとして解釈する(「モデルが `</think>` を飛ばす」ケースへの対処)。ツール呼び出し本体の解釈は `Qwen3CoderParser` を流用。
- レンダラーには `v35`(Nemotron 3.5 のプロンプト形式、`{reasoning effort: efficient}`)の分岐があり、`nemotron-3.5-lightning` もこの実装で動く。
- **ハイブリッド構成(`nemotron_h_moe`)は `OLLAMA_NUM_PARALLEL` が 1 に固定される**([ollama #18938](https://github.com/ollama/ollama/issues/18938)、#4165 の重複として closed)。スロットが 1 つなのでプロンプトキャッシュも 1 つ。本リポジトリは 5 プレイヤーを逐次に呼ぶので、他モデルと条件は変わらない。

### 2.4 langchain-ollama 側

- `reasoning=True/False/None` がそのまま Ollama の `think` になる(`str` で effort 指定も可)。応答の `thinking` は `additional_kwargs["reasoning_content"]` に入る。
- 履歴の AIMessage に `reasoning_content` があれば `thinking` として毎回送り返す(006 と同じ)。Nemotron ではレンダラーが過去分を捨てるので害はない。
- `ChatNVIDIA` 側には Nemotron 用の思考タグ対応(PR #221)と多ターンでの reasoning 保持(Issue #236 → PR #260)があるが、Ollama 経由では関係ない。

### 2.5 既知の問題・制約(Issue / Discussion)

| 出典 | 症状 | 状態 | 本リポジトリへの影響 |
| --- | --- | --- | --- |
| [NemoClaw #246](https://github.com/NVIDIA/NemoClaw/issues/246) / [#247](https://github.com/NVIDIA/NemoClaw/issues/247) | Ollama の OpenAI 互換 `/v1` で `nemotron-3-nano:30b` の本文が `reasoning` に入り `content` が空 → UI が空白。`reasoning_effort: "none"` か `think: false` で回避 | #247 は 2026-06 に「現行版で再現せず」で closed。修正 PR の紐付けなし | なし。langchain-ollama は native `/api/chat` を使い、`thinking` と `content` を分けて受け取る |
| [HF discussion #3](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/3) | vLLM で思考パーサーとツール呼び出しが干渉。NVIDIA の回答は「**completion の token 予算が足りず途中で切れるとツール解析が壊れる**」。`str_replace_editor` など存在しないツール名を頻繁に幻覚 | vLLM nightly で改善、以後も散発報告 | `num_predict=4096` で思考が切れた呼び出しは「未提出」になる(001 と同じ扱い)。存在しないツール名は Deep Agents がエラーの ToolMessage で返し、再呼び出しされる |
| [ollama #17785](https://github.com/ollama/ollama/issues/17785) | `chat_template_kwargs.enable_thinking` / `low_effort` / `reasoning_budget` は Ollama では無視される。`/v1` では `reasoning_effort`、native では `think` を使う | 2026-09 closed | `reasoning=` を使っている限り問題ない |
| [ollama #18110](https://github.com/ollama/ollama/issues/18110) | `nemotron-3-nano:4b` が自分を Qwen と名乗る | closed(詳細なし) | なし(自己紹介はしない) |
| [ollama #15260](https://github.com/ollama/ollama/issues/15260) | `think=false` + `format`(JSON schema)で schema が無視される(gemma4、qwen3.x) | gemma4 は 0.21.1 で修正、qwen は #14645 | Nemotron は未報告。本リポジトリの確認では `reasoning=True/False` とも `json_schema` が通った(§4) |
| [crewAI #4117](https://github.com/crewAIInc/crewAI/issues/4117)、[opencode #6327](https://github.com/sst/opencode/issues/6327) | system prompt を無視する、ツール呼び出しが壊れる | 未解決 / not planned | ハーネス固有のプロンプト形式との相性。Deep Agents での再現報告はない |
| [ai-dynamo PR #381](https://github.com/ai-dynamo/frontend-crates/pull/381) | `</think>` を閉じる前に `<tool_call>` を出す | 各推論サーバーが個別対応 | Ollama のパーサーは対応済み(§2.3) |
| [deepagents #4572](https://github.com/langchain-ai/deepagents/pull/4572)(closed PR)/ [#5982](https://github.com/langchain-ai/deepagents/issues/5982) | Ultra が `execute` のプレースホルダ引数を出す / nudge ミドルウェアの誤判定 | Ultra プロファイル内の話 | Ultra プロファイルを使わないので無関係 |

### 2.6 Ollama 上の新しい Nemotron(2026-10-15 時点)

| モデル | サイズ | 能力 | 備考 |
| --- | --- | --- | --- |
| `nemotron-3-nano:30b` | 24GB(Q4_K_M)、1M ctx | tools, thinking | 本リポジトリで取得済み。2025-12 公開 |
| `nemotron-3.5-lightning:30b` | 25GB、1M ctx | tools, thinking | 30B-A3B。2026-08-11 公開、「常駐エージェントの実行層向け」。Ollama のレンダラーは `v35` 分岐で対応、`think="medium"` で `{reasoning effort: efficient}` を付ける。langchain-nvidia にも登録済み(PR #359)。詳細は §2.7 |
| `nemotron3:33b` | 28GB、128K ctx | vision, tools, thinking | Nano Omni(動画・音声・画像)。32GB には 16K ctx 込みで厳しい |
| `nemotron-3-super:120b` | 87GB | tools, thinking | 120B-A12B。**ローカルには載らない**(`:cloud` あり) |
| `nemotron-3-ultra` | cloud のみ | tools, thinking | Deep Agents の正式対応モデル |

### 2.7 Nemotron 3.5 Lightning 30B-A3B(2026-08-11 公開)

| 項目 | 内容 | 出典 |
| --- | --- | --- |
| 構成 | Nano 3 と同じ Mamba-2 + MoE + Attention のハイブリッド、30B(3B アクティブ)。MTP(Multi-Token Prediction)層を同梱し、Ollama の Modelfile にも `draft_num_predict 2` が入っている(投機的デコード) | [HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)、`ollama show --modelfile` |
| コンテキスト・言語 | 最大 1M(H100 1 枚では 256K)。英語・スペイン語・フランス語・ドイツ語・イタリア語・日本語 | HF model card |
| 推奨サンプリング | **temperature 1.0 / top_p 0.95**(Ollama の Modelfile 既定も同じ)。Unsloth は思考モード 0.6 / 0.95 と書くが、NVIDIA の card には 0.6 の記載がない | HF model card、[Unsloth guide](https://unsloth.ai/docs/models/nemotron-3.5) |
| 思考の切り替え | `enable_thinking`(既定 True)。effort の段階や予算は card に記載なし。Ollama のレンダラーは `think="medium"` のときだけ最後の user に `{reasoning effort: efficient}` を付け、`low` / `high` は `True` と同じ扱い | HF model card、Ollama レンダラー(`v35`) |
| ツール呼び出し | Nano 3 と同じ XML 形式。vLLM は `--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`。coding agent 向けに `chat_template_kwargs` の `force_nonempty_content` の記載あり | HF model card |
| 位置づけ | card は「カスタマイズと post-training の起点」とし、Ollama の説明は「常駐エージェントの実行層向け(OpenClaw、Hermes Agent)」 | HF model card、Ollama library |

本リポジトリでのプローブ(§4.1 と同じ手順、seed=1): ツール呼び出しは `reasoning=True/False` とも構造化された `tool_calls` で返り、同じ発言プロンプトの思考は 361 文字(Nano 3 は 1,670)。過去ターンの思考を送る / 外すで `prompt_eval_count` は 589 / 589 で一致(レンダラーが捨てる)。`json_schema` の構造化出力も両方通った。`reasoning="medium"` は出力が変わる(eval 435 → 505)が短くはならず、`"low"` / `"high"` は `True` と同一の出力だった。

## 3. 推測・解釈(公式情報ではないもの)

- **000 で観測した `example_function_name` の呼び出しは、Ollama レンダラーがツール一覧の末尾に入れる例示文をそのまま写したもの**だと考えられる(§2.3)。モデル側ではなくプロンプト側の文言が原因なので、ツール説明を厚くしても消えず、「存在しないツール名はエラーで返して再呼び出しさせる」運用が妥当。
- **Ultra プロファイルの 8 節の prompt suffix と nudge 群は Nano には転用しない方がよい。** 英語の業務タスク(予約・レポート・ファイル操作)向けの正規表現と文言で、日本語の人狼では発火しないか誤発火する。転用するなら `NemotronTextToolCallParser` 相当の「本文に出たツール呼び出しの救済」だけで、それも Ollama のパーサーが先に処理するので出番は少ない。
- **temperature 0.6 / top_p 0.95 は「ツール呼び出しの安定性」向けの推奨**で、議論の多様性が落ちる可能性がある。人狼では「発言の質(推理)」と「提出の確実さ」の両方が要るので、既定 0.7 との比較は別の実験に分ける。
- 「思考を切ると精度が落ちる」は公式の注意だが、003 で `reasoning=False` は怪盗の交換や人狼の騙りが消えたのと同じ系統の劣化が予想される。Nemotron で `reasoning=False` を試す優先度は低い。
- Nemotron 3 Nano の思考は gemma4 より長く qwen3.x より短い傾向(000 の p50 6.3 秒)。`num_predict=4096` に当たる頻度は qwen 並みと見込む。

## 4. 本リポジトリでの確認

### 4.1 langchain-ollama 経由の挙動(プローブ、seed=1、temperature=0.6)

| 確認項目 | 結果 |
| --- | --- |
| `reasoning=True` + `bind_tools` | `speak(claimed_role='占い師', content=…)` を構造化された `tool_calls` で返す。思考 1,670 文字、eval 541 トークン |
| `reasoning=False` + `bind_tools` | 同じく `tool_calls` で返す。eval 69 トークン |
| 過去ターンの `reasoning_content` を送る / 外す | ラウンド 2 の `prompt_eval_count` が **577 / 577 で一致**。レンダラーが過去の思考を捨てている(§2.3 の読みと一致) |
| `with_structured_output(method="json_schema")` | `reasoning=True` / `False` とも Pydantic に収まる JSON を返した(True は英語で理由を書いた) |

### 4.2 lab の用途別ベンチマーク v2(29 タスク、`lab/run_eval.py nemotron-3-nano:30b --reasoning true --coerce`)

結果は [local-deepagents.md 3.2](local-deepagents.md) の表に並べた。**23/29**(gemma4:26b 29、qwen3.8:27b 28、gpt-oss:20b 23)。

| 指標 | 値 |
| --- | --- |
| tool_use / filesystem / planning / subagent / long_ctx / structured / skills | 6/7 / 6/7 / **1/3** / 3/3 / 3/3 / 2/3 / 2/3 |
| p50 / p95 秒 | 3.2 / 15.7 |
| 暴走(1 呼び出しが `num_predict`=2048 に到達) | **5**(すべて思考。うち 3 件は最初の呼び出しで空応答のまま終了) |
| エラー(Ollama パーサ例外など) | 0 |

失敗 6 件の内訳:

- planning 2 件 + skills 1 件: 最初の呼び出しの思考が 2048 トークンに達し、ツールを呼ばずに空応答。生成は速く 2048 トークンで 5.9 秒。
- `struct_order_en`: `ToolStrategy` のツールを呼ばず JSON コードブロックで答えた(他モデルでも出る 4.6 の症状)。
- `tool_error_ja`: 「対応外の通貨」をツールを呼ばずに説明で済ませた(ツール説明から判断)。
- `fs_fix_bug_en`: `edit_file` で `return a - b` を一意でないまま置換し、隣の `sub` まで `a + b` にした上で「修正した」と答えた。

HF discussion #3 の NVIDIA の回答(token 予算が足りないとツール解析が壊れる)と整合する。`num_predict` を 4096 にした再評価は §4.4。

### 4.3 人狼 5 seed bench(`runs/nemotron-5`)

詳細は [docs/experiments/007-nemotron.md](../experiments/007-nemotron.md)。005 / 006 と同一設定(`reasoning=True`、`num_predict=4096`、`drop_past_reasoning`、seed 1〜5)。

| 指標 | gemma4:26b | qwen3.8:27b | nemotron-3-nano:30b |
| --- | --- | --- | --- |
| 村の勝ち(うち推理で) | 1 / 5(0) | 2 / 5(2) | 2 / 5(0) |
| 総時間 平均 | 129 秒 | 271 秒 | **364 秒** |
| 未提出率 / フォールバック率 | 6.1% / 0% | 0.6% / 0% | **27.9% / 9.5%** |
| 思考が 4096 に達した呼び出し | 6 / 370 | 0 / 433 | **53 / 423** |
| 1 呼び出しの出力 p50(トークン) | 218 | 173 | **932** |

- 未提出 53 件のうち 48 件が思考の暴走(4096 トークン到達、投票に集中)。文章回答は 4 件。架空ツール名は 0 件。
- 推理の質は gemma4 以下(村人が占い師を騙る、真占い師の黒出しを「証拠不十分」で退けて処刑)。
- `drop_past_reasoning` は no-op のまま害なし。コンテキストは 1 判断 +250 の直線。

### 4.4 `num_predict` 4096 での lab 再評価(失敗した 6 タスクのみ)

| タスク | 2048 | 4096 |
| --- | --- | --- |
| plan_spec_tests_ja / plan_project_en / skill_haiku_ja(暴走) | × × × | **○ ○ ○**(planning は 11〜12 呼び出し、20〜25 秒) |
| struct_order_en / tool_error_ja / fs_fix_bug_en | × × × | × × ×(同じ失敗) |

思考に 2048〜4096 トークンを使えば planning は解ける。残り 3 件は思考の長さと無関係な判断の癖(JSON をコードブロックで答える、ツールを呼ばずに説明する、一意でない置換)。29 タスク換算では 26/29 相当。

### 4.5 nemotron-3.5-lightning:30b(lab + 人狼 5 seed、[008](../experiments/008-nemotron-lightning.md))

| 指標 | nemotron-3-nano:30b | nemotron-3.5-lightning:30b | gemma4:26b(参考) |
| --- | --- | --- | --- |
| lab 29 タスク | 23/29(暴走 5) | **28/29**(暴走 0、p95 5.8 秒で最速) | 29/29 |
| 人狼: 未提出率 / フォールバック率 | 27.9% / 9.5% | **2.1% / 0.7%** | 6.1% / 0% |
| 人狼: 思考が 4096 に達した呼び出し | 53 / 423 | **3 / 420** | 6 / 370 |
| 人狼: 総時間 平均 | 364 秒 | **143 秒** | 129 秒 |
| 人狼: 村の勝ち / 占い師の処刑 | 2 / 1 | 0 / 0 | 1 / 1 |
| 人狼: なぞり率 | 10.3% | 12.8% | 8.3% |

Nano 3 の問題(思考の暴走)は Lightning で解消した。人狼の推理の質は gemma4 以下(怪盗が自分を「originally werewolf」と誤って説明し村側が処刑される、直前の発言をなぞる)。

temperature を公式推奨の 1.0 にすると([009](../experiments/009-lightning-temperature.md))、なぞり率は 12.8% → 5.9% に半減し村が 2 勝(うち推理で 1)、未提出 3.8%・161 秒/ゲームのまま。ただし怪盗が交換後の役職と方針をそのまま口にする「怪盗の自白」と英語混じりの発言が残る。

## 5. 現在の実装への適用

| 項目 | 判断 | 理由 |
| --- | --- | --- |
| Ultra プロファイルの転用(`register_harness_profile("ollama:nemotron-3-nano:30b", …)`) | **見送り** | 非公開モジュールにしかなく、中身は英語の業務タスク向け。bench で本文へのツール呼び出し・架空ツール名・思考タグの漏れは 0 件で、救済対象がない |
| `drop_past_reasoning` | **そのまま** | Nemotron では no-op(レンダラーが同じ境界で切る)。qwen3.8 には必要 |
| `top_p` の指定(`make_llm(top_p=…)`、`--top-p`) | **追加した** | NVIDIA 推奨 0.6 / 0.95 の比較(007 B)に必要。既定は未指定のまま(モデルの既定) |
| lab `run_eval.py --num-predict` | **追加した** | 思考が 2048 で切れる planning タスクの再評価に必要(§4.4) |
| `num_predict` の引き上げ(8192) | **見送り** | 暴走 1 回が 12 → 24 秒になり、総時間が qwen の 2 倍を超える。投票の思考が 4096 で収まらない比率(7%)に対して高すぎる |
| 既定モデルの変更 | **しない** | 007〜009 の結論。gemma4:26b(速さ)/ qwen3.8:27b(推理)のまま。**Nemotron を使う場合の推奨は `--model nemotron-3.5-lightning:30b --temperature 1.0`**(取得済み、25GB。009 で 0.7 よりなぞりが半減、村 2 勝) |
| 投票だけ `reasoning=False` にする | 見送り | Lightning では暴走が消えたので不要 |
| `reasoning="medium"` を CLI から指定できるようにする | 見送り | プローブで思考が短くならず、暴走も消えたので必要がない |
| 怪盗向けの「交換後の役職と方針を明かさない」指示 | 次の仮説(009) | Lightning 固有ではなく 005 / 006 の仮説 3 と同根 |

## 参考リンク

- LangChain docs: [NVIDIA middleware(Nemotron 3 Ultra harness profile)](https://docs.langchain.com/oss/python/integrations/middleware/nvidia)、[NVIDIA provider](https://docs.langchain.com/oss/python/integrations/providers/nvidia)、[ChatNVIDIA](https://docs.langchain.com/oss/python/integrations/chat/nvidia_ai_endpoints)
- deepagents: [PR #4192 Nemotron 3 Ultra profile](https://github.com/langchain-ai/deepagents/pull/4192)、[#4555](https://github.com/langchain-ai/deepagents/issues/4555)、[#6598](https://github.com/langchain-ai/deepagents/issues/6598)、[#6599](https://github.com/langchain-ai/deepagents/issues/6599)
- NVIDIA: [技術ブログ(ハーネスプロファイルの作り方)](https://developer.nvidia.com/blog/create-a-langchain-deep-agents-harness-profile-for-nvidia-nemotron-3-ultra-to-improve-performance)、[NemoClaw ブループリント発表](https://blogs.nvidia.com/blog/nemotron-langchain-agents-open-stack/)、[HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)、[技術レポート](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- Ollama: [nemotron-3-nano](https://ollama.com/library/nemotron-3-nano)、[nemotron-3.5-lightning](https://ollama.com/library/nemotron-3.5-lightning)、[renderers/nemotron3nano.go](https://github.com/ollama/ollama/blob/main/model/renderers/nemotron3nano.go)、[parsers/nemotron3nano.go](https://github.com/ollama/ollama/blob/main/model/parsers/nemotron3nano.go)
