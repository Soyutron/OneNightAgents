# 000: 技術スパイク(Deep Agents + Ollama の動作検証とテンポの実測)

- 日付: 2026-10-14
- 対象ステップ: Step 0([implementation-plan.md](../implementation-plan.md) 3 節)
- スクリプト: `spike/`(使い捨て。`spike/results/` に生データの JSONL)

## 仮説

ローカルのツール対応モデル(8B〜20B 級)が Deep Agents ハーネス下で行動ツールを正しく呼び、`/notes.md` を扱え、1 判断(最大 4 ステップ)が p95 30 秒以内に収まる。

合格の目安(計画より): ツール呼び出し成功率 90% 以上、1 判断 p95 30 秒以下、日本語が破綻しない。

## 設定

### 環境

| 項目 | 値 |
| --- | --- |
| GPU | RTX 5090 32GB(ドライバ 580.178.04)、RAM 60GB |
| Ollama | 0.33.3 |
| Python / パッケージ | 3.13(uv)、deepagents 0.7.24、langchain 1.4.5、langchain-core 1.6.10、langchain-ollama 1.1.0、langgraph 1.2.15、langsmith 0.14.8 |
| モデル | qwen3.5:9b、gemma4:12b、qwen3:8b、gpt-oss:20b(いずれも Ollama ライブラリの `tools` タグ付き)。追加計測で gemma4:26b、qwen3.6:27b、qwen3.8:27b、nemotron-3-nano:30b |
| ChatOllama | `num_ctx=16384`, `keep_alive=30m`, `temperature=0.7`, `seed` 固定, `reasoning=False`(条件比較あり), `validate_model_on_init=True` |

### 検証項目とスクリプト

| # | 内容 | スクリプト | 計測 |
| --- | --- | --- | --- |
| 1 | モデル取得と `ollama ps` で 100% GPU の確認 | `04_vram.sh` | `ollama ps` の SIZE / PROCESSOR、nvidia-smi |
| 2 | `ChatOllama.bind_tools` にダミー行動ツール 3 個(`speak` / `vote` / `pass_turn`、Pydantic スキーマ付き)を渡し、日本語 20 プロンプトで呼び出し成功率を測る | `01_tool_calls.py` | 厳密成功(期待ツールを 1 回だけ呼び、引数が Pydantic 検証を通る)、緩い成功(ツール名のみ一致)、レイテンシ、`prompt_eval_count` / `eval_count` |
| 3 | `create_deep_agent(model=ChatOllama, tools=[speak], backend=StateBackend())` + `FilesystemMiddleware(tools=[ls, read_file, write_file, edit_file])` + `ModelCallLimitMiddleware(run_limit=4)` で、「/notes.md にメモを書いてから speak」を 5 シナリオ × 2 回 | `02_deep_agent.py` | モデル呼び出し回数、ツール列、1 判断のレイテンシ、成功(メモが書かれ、speak が 1 回だけ)、初回呼び出しの `prompt_eval_count`(固定コスト) |
| 4 | 日本語短文の自然さ(目視)と `reasoning=False / True / None` の差 | `03_japanese.py` | 出力、`eval_count`、時間、`reasoning_content` の有無 |

20 プロンプトのうち 3 つは「ツールを呼ばずに答える」質問(不要な呼び出しの検出用)、1 つは speak / pass_turn どちらでも可。

### LangSmith について

`LANGSMITH_API_KEY` がこの環境に未設定のため、LangSmith トレースの確認は **未実施**。代わりにコールバック(`on_chat_model_start` / `on_llm_end`)で、モデルに送った system prompt とツール定義 JSON の文字数、`prompt_eval_count`(実トークン数)を記録した。Step 2〜3 で `.env` に鍵を置いてから確認する。

## 観察

### 2. ツール呼び出し成功率(20 プロンプト)

| モデル | reasoning | 厳密成功 | p50 秒 | p95 秒 | 初回 prompt tok | 出力 tok p50 |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-oss:20b | False | **20/20** | 0.77 | 2.07 | 592 | 220 |
| gpt-oss:20b | True | **20/20** | 0.73 | 1.45 | 597 | 201 |
| qwen3:8b | False | **20/20** | 0.56 | 10.35 | 623 | 55 |
| qwen3:8b | True | 17/20 | 4.75 | 9.88 | 617 | 471 |
| qwen3.5:9b | False | 17/20 | 0.44 | 1.98 | 735 | 73 |
| qwen3.5:9b | True | 19/20 | 1.66 | 4.15 | 733 | 310 |
| gemma4:12b | False | 14/20 | 0.30 | **88.7** | 503 | 54 |
| gemma4:12b | True | **20/20** | 2.75 | 4.51 | 501 | 727 |

厳密成功と緩い成功は全条件で一致した(= 引数の JSON とスキーマ違反はゼロ。失敗はすべて「ツール未呼び出し」か「ツール名違い」)。

失敗の内訳:

- **qwen3.5:9b(False)**: 3 件とも、speak を呼ばずに発言内容を本文で返した(例: 「p5, p1 と p4 のどちらが人狼だと思いますか？」)。reasoning=True では 1 件に減る。
- **gemma4:12b(False)**: 6 件。うち 4 件は **暴走**: ツール呼び出しも本文もない空応答のまま `num_ctx` が埋まるまで生成し続け(`eval_count` 15,881 ≈ 16384 − prompt)、1 回 80〜89 秒。残り 2 件は本文で回答(1 件は `call:speak{content:<|"|>…}` という壊れたツール呼び出し構文を本文に出した)。reasoning=True では暴走せず 20/20。
- **qwen3:8b(True)**: 3 件、発言ターンで `vote` を呼んだ(ツール選択ミス)。False では 20/20。p95 10.4 秒は初回呼び出しのモデルロード(`load_duration` 10.1 秒)で、生成自体は 0.7 秒。

初回呼び出しのロード時間は gpt-oss:20b 10.6 秒、gemma4:12b 6.3 秒。ゲーム開始前のプリロード(`keep_alive` + 空リクエスト)は計画どおり必要。2 回目以降の `load_duration` は 0.001 秒。
- **gpt-oss:20b**: 初回実行(ツール説明に列挙値を書く前)では `claimed_role` に `"villager"` ではなく **`"村人"`** を入れて 20 件中 9 件がスキーマ違反だった(`spike/results/01_tool_calls_gpt-oss_20b_nothink_v0-before-enum-fix.jsonl`)。description に「値は villager / werewolf / seer / thief のいずれか(日本語不可)」と明記したら 20/20 になった。Literal 型だけでは列挙値が守られない。

### 3. Deep Agent 最小構成(メモ → speak、run_limit=4、10 回)

| モデル | reasoning | 成功 | メモ書込 | speak 呼出 | 呼び出し数 p50/max | p50 秒 | p95 秒 | 初回 prompt tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-oss:20b | False | **10/10** | 100% | 100% | 3/4 | 4.65 | 6.3 | 1434 |
| gemma4:12b | False | 9/10 | 100% | 90% | 3/4 | 1.83 | **76.4** | 1485 |
| qwen3.5:9b | False | 8/10 ※ | 100% | 80% | 3/3 | 1.30 | 1.89 | 1716 |
| qwen3:8b | False | 0/10 | 90% | 90% | 4/4 | 4.94 | 52.0 | 1524 |

※ qwen3.5:9b の初回計測は seed 固定で同一シナリオが完全に重複していたため実質 5 サンプル。seed を実行ごとに変えて再計測した結果は「追加計測」に記載。

ツール呼び出し列(典型): `write_file → speak → (最終応答)` の 3 呼び出し。gpt-oss:20b は 2 回 `ls` を挟んだ(`ls → write_file → speak`、`speak → write_file → ls → write_file`)。

失敗の型:

- **speak を呼ばずに終了**(qwen3.5:9b ×2、gemma4:12b ×1): `write_file` の後、発言内容を本文で返して終わった。単発テストの「ツール未呼び出し」と同じ型。
- **speak の繰り返し**(qwen3:8b 9/10): 「受理しました」の後も speak を呼び続け(1 応答で複数の並列ツール呼び出しも含む)、`run_limit=4` で打ち切られた。最初の speak の内容自体は妥当。
- **暴走**(gemma4:12b ×1 で 75 秒、qwen3:8b ×1 で 52 秒 → `ValueError: No data received from Ollama stream`)。

固定コスト: 初回呼び出しの `prompt_eval_count` は 1,434〜1,716 トークン。内訳は自前 system prompt 440 文字(≈ 300 tok)+ FS ツール 4 個 + speak のツール定義 JSON 4,623 文字(≈ 800〜1,000 tok)+ シナリオ本文。**deepagents 0.7.24 は ChatOllama に対してハーネスの base system prompt を付けない**(送られた system は自前の 440 文字のみ)。FS ツールの使い方は各ツールの description に入っており、これがトークンの大半を占める。

メモの内容(gemma4:12b、人狼シナリオ):

```
【自分の立場】人狼(p3 ユウキ)/ 共犯: p5 レン
【現状の把握】p1 アオイ: 占い師を自称。p5 を人狼と断定。…
【戦略】… 今は p5 を吊る方向へ誘導するのが村人側にとっても有利。私は慎重な立場をとりつつ、p1 の主張を支持する形で動く。
```

同じシナリオで gpt-oss:20b は「私は人狼です。p1 は正しいと信じ、p5 が人狼だと考えます」と書き、発言でも相方 p5 を疑った(自分の陣営を理解していない)。qwen3.5:9b の怪盗シナリオでは「私はいまだに怪盗ですが、先ほど p4 と役職を交換しました。私は現在『占い師』を名乗っていますが、実…」と、交換後の役職と秘密情報を自分から漏らした。**ツール呼び出しの成否とは別に、役職理解・情報秘匿の品質はモデルで大きく違う。**

### 4. 日本語短文と reasoning の効き目(5 プロンプト × 3 条件)

| モデル | reasoning=False 平均秒 / 出力 tok | True 平均秒 / 出力 tok | None 平均秒 / 出力 tok | 備考 |
| --- | --- | --- | --- | --- |
| qwen3.5:9b | 0.2 / 23 | 11.7 / 2,336 | 10.3 / 2,057 | **False が効く。** None は思考するが `<think>` は本文に出ない(Ollama 側で分離) |
| gemma4:12b | 0.2 / 20 | 1.5 / 399 | 1.5 / 399 | False が効く。思考は短い |
| qwen3:8b | 0.2 / 16 | 2.6 / 251 | 2.2 / 235 | False が効く |
| gpt-oss:20b | 1.4 / 383 | 1.1 / 296 | 1.1 / 318 | **False が効かない。** 出力 tok が 3 条件とも 300 前後で、短文でも常に思考している |

自然さの目視評価(5 段階。reasoning=False の 5 文、口語・短さ・文脈整合):

| モデル | 評価 | 所見 |
| --- | --- | --- |
| gemma4:12b | 4 | 「どっちも怪しいね。どっちかが嘘ついてると思う」など口語で自然。文脈も合う |
| qwen3:8b | 4 | 短く自然(「俺はレンを疑ってる」)。ややぶっきらぼう |
| qwen3.5:9b | 3 | 自然だが文脈ずれあり(「アオイは潔白だね」と直前の疑いと矛盾、「今すぐこの人を狩ってほしい」) |
| gpt-oss:20b | 3 | 文は成立するが、存在しない名前「カナ」、「Ren」とローマ字、「彼は何も動かなかった」と硬い。reasoning=True の方がやや自然 |

破綻(文字化け・言語混在・無限反復)はどのモデルにもなかった。

## 追加計測

### 5. VRAM(`num_ctx=16384`、モデル単独ロード)

| モデル | nvidia-smi の増分 | `ollama ps` SIZE | PROCESSOR |
| --- | --- | --- | --- |
| qwen3.5:9b | 約 7.3GB(after 8.0GB) | 6.1 GB | 100% GPU |
| gemma4:12b | 9.7GB | 1.4 GB(表示異常。nvidia-smi を正とする) | 100% GPU |
| qwen3:8b | 7.8GB | 7.5 GB | 100% GPU |
| gpt-oss:20b | 13.3GB | 12 GB | 100% GPU |

4 モデルとも 32GB に余裕で収まり、`OLLAMA_KV_CACHE_TYPE=q8_0` は不要。2 モデル同時常駐(gpt-oss:20b + qwen3.5:9b で 21GB)も可能。

### Deep Agent 構成の再計測(seed を実行ごとに変更、reasoning 比較)

| モデル | reasoning | 成功 | メモ書込 | speak 呼出 | 呼び出し数 p50/max | p50 秒 | p95 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4:12b | **True** | **10/10** | 100% | 100% | 3/4 | 4.87 | 10.4 |
| gemma4:12b | False | 9/10 | 100% | 90% | 3/4 | 1.83 | 76.4(暴走 1 件) |
| qwen3.5:9b | **True** | **10/10** | 100% | 100% | 3/3 | 3.82 | 6.25 |
| qwen3.5:9b | False(再計測) | 7/10 | 90% | 70% | 3/3 | 1.67 | 3.03 |
| gpt-oss:20b | False(= 思考あり) | **10/10** | 100% | 100% | 3/4 | 4.65 | 6.3 |

qwen3.5:9b(False)の失敗 3 件: speak を呼ばず「speak で発言を提出します。」と本文で返して終了 ×2、壊れたツール呼び出しを Ollama 側が拒否(`ResponseError: XML syntax error … element <parameter> closed by </function>`)×1。

### gemma4:12b の暴走に対する `num_predict=1024`

| 条件 | 厳密成功 | p95 秒 | 暴走時の `eval_count` |
| --- | --- | --- | --- |
| 上限なし | 14/20 | 88.7 | 15,881(num_ctx まで) |
| `num_predict=1024` | 14/20 | **2.95** | 1,024 |

成功率は変わらないが、1 回の失敗が 80 秒から 3 秒に縮む。暴走はモデル側の現象(seed 固定で同じプロンプトなら毎回再現)で、ハーネスでは防げない。

### 役職の秘匿と陣営理解(Deep Agent、人狼・怪盗シナリオの発言。各 2 回)

| モデル | 人狼シナリオ(相方 p5 が占い師 p1 に人狼と言われている) | 怪盗シナリオ(p4 と交換済み) |
| --- | --- | --- |
| gemma4:12b(True) | 「p1 自身が偽の占い師である可能性も考慮すべき」と相方を庇う。役職は明かさない | 明かさない |
| gemma4:12b(False) | 「p1 さんの占い結果を信じます。p5 さんの主張には一貫性が感じられません」と相方を売る。役職は明かさない | 明かさない。「私は占い師として協力します」(交換後の役職を主張) |
| gpt-oss:20b | 1 件は `claimed_role=werewolf` を付けて「p5 が人狼だとすべき」。メモにも「私は人狼です。…p5 が人狼だと考えます」 | 「あなたの役職は人狼ではありません」と文脈不明の発言 |
| qwen3.5:9b(True) | 「p5 も人狼の可能性が高いので、p4 を処刑しましょう」(claimed_role=werewolf) | 「私は実際には p4 と交換済みです」と秘密を漏らす |
| qwen3.5:9b(False) | **「私は人狼ですが」「私は人狼です」**(2 回とも自ら明かす) | 明かさない |

`claimed_role` の意味(「この発言で主張する役職」)を本当の役職と混同するモデルが多い。system prompt の「嘘をついてもかまいません」だけでは、人狼が自分の役職を隠すことも相方を守ることも伝わっていない。

### 32GB VRAM に収まる上位モデルの追加計測(reasoning=True、`spike/run_large.sh`)

Ollama ライブラリの `tools` タグ付きで 16K コンテキスト込みで 32GB に収まるものから、gemma4:26b、qwen3.6:27b、qwen3.8:27b(最多 pull)、nemotron-3-nano:30b(ローカル既存、MoE)を追加。前半の小型モデルと同じスクリプトで比較。

| モデル | VRAM | ツール呼び出し 20 件(ロード除く p50) | Deep Agent 10 件 | 1 判断 p50 / p95 | 人狼シナリオでの振る舞い | 怪盗シナリオ |
| --- | --- | --- | --- | --- | --- | --- |
| **gemma4:26b** | 19.7GB | **20/20**(2.0 秒) | **10/10** | **4.2 / 6.5 秒** | 相方を庇う(「p1 の占い結果は唐突」「鵜呑みは早計」)。役職は明かさない | 占い師を名乗り、p4 の矛盾を突く |
| gemma4:12b(再掲) | 9.7GB | 20/20(2.7 秒) | 10/10 | 4.9 / 10.4 秒 | 相方を庇う | 明かさない |
| qwen3.8:27b | 19.4GB | 19/20(2.3 秒。失敗は `claimed_role: "None"` の文字列) | 10/10 | 8.4 / 19.4 秒 | 「私は村人です。p1 の主張は検証不能」と隠して p1 を疑う | 交換の事実を自分から言う |
| qwen3.6:27b | 19.3GB | 19/20(7.7 秒。失敗は **暴走** 97 秒、思考中に 15.6K トークン) | 10/10 | 6.9 / 14.2 秒 | 相方を庇う | 交換の事実を自分から言う |
| qwen3.5:9b(再掲) | 7.3GB | 19/20(1.6 秒) | 10/10 | 3.8 / 6.3 秒 | 相方を疑う、claimed_role=werewolf | 秘密を漏らす |
| gpt-oss:20b(再掲) | 13.3GB | 20/20(0.8 秒) | 10/10 | 4.7 / 6.3 秒 | 相方を疑う | 文脈不明の発言 |
| nemotron-3-nano:30b | 23.8GB | 17/20(2.7 秒。`example_function_name` を呼ぶ、JSON を本文に出す) | 9/10 | 6.3 / 12.4 秒 | 相方を庇う | **「p4 が怪盗を名乗っていますが、実際は私です」**と漏らす |

所見:

- **gemma4:26b が全指標で最良。** 12b より 1 判断が速い(p95 6.5 秒。12b の 10.4 秒は外れ値 1 件による)のは、出力が簡潔で思考が短いため。VRAM は 20GB で、32GB では 2 モデル同時常駐は難しい。初回ロード 17 秒。
- qwen3.x の 27B 級は陣営理解は良いが、1 判断が 7〜8 秒と gemma4 の 2 倍で、qwen3.6 は reasoning=True でも暴走した。
- モデルサイズを上げても、Deep Agent 構成での成功率は小型の gemma4:12b / qwen3.5:9b(True)ですでに 10/10 なので、**サイズの効果はツール呼び出しよりも「役職を隠す・相方を庇う」の質に出る**。
- 大型でも `claimed_role` に `"None"` の文字列(qwen3.8)や、存在しないツール名(nemotron)は出る。ツール側の寛容な検証(`"None"`/`"null"` → None)と GM の再要求は引き続き必要。

## 結論

### 合格判定

| 基準 | 結果 |
| --- | --- |
| ツール呼び出し成功率 90% 以上 | **合格**: gpt-oss:20b(100%)、gemma4:12b(True: 100%)、qwen3.5:9b(True: 95%)、qwen3:8b(False: 100%)。Deep Agent 構成でも gemma4:12b(True)、qwen3.5:9b(True)、gpt-oss:20b が 10/10 |
| 1 判断(最大 4 ステップ)の p95 が 30 秒以下 | **合格**: 暴走を `num_predict` で抑えれば全モデル p95 10 秒台以下。gemma4:12b(True)10.4 秒、qwen3.5:9b(True)6.3 秒、gpt-oss:20b 6.3 秒 |
| 日本語が破綻しない | **合格**: 破綻なし。自然さは gemma4:12b と qwen3:8b が上、gpt-oss:20b は硬くて名前の誤りあり |

Deep Agents + Ollama の組み合わせは動く。1 判断は典型的に `write_file → speak → 最終応答` の 3 呼び出しで 2〜5 秒。ゲーム 1 回(5 人 × 約 6 判断 = 30 判断)は gemma4:12b(True)で約 2.5 分の見込みで、10 分の目標に対して余裕がある(ただし履歴累積の影響は未計測)。

### 仮決め

| 項目 | 値 | 根拠 |
| --- | --- | --- |
| 既定モデル | **gemma4:26b、`reasoning=True`** | ツール呼び出し 20/20、Deep Agent 10/10、1 判断 p95 6.5 秒、役職を漏らさず相方を庇う、日本語が自然。VRAM 19.7GB |
| 軽量代替 | **gemma4:12b、`reasoning=True`** | 同系列で挙動が近く、VRAM 9.7GB。bench の大量実行や、2 モデル同時常駐が必要なときに |
| 比較候補 | gpt-oss:20b(ツール最堅牢・陣営理解が弱い)、qwen3.8:27b(陣営理解は良いが 2 倍遅い) | Step 5 の A/B で同 seed 比較 |
| `num_predict` | 1024 | 暴走時の損失を 80 秒 → 3 秒に。正常時の最大出力(メモ書込 700 トークン)に余裕 |
| `llm_timeout_sec` | 30 | 1 呼び出しの p95 が 10 秒未満。`num_predict` と二重防御 |
| `model_call_limit` | 4 | 典型 3。4 回目に達するのは「speak の繰り返し」か「ls の寄り道」で、どちらも 4 で打ち切って問題ない |
| `num_ctx` | 16384 | 全モデル GPU に収まる。KV 量子化不要 |

### 今後の実装方針への影響

1. **ツール定義の列挙値は description に明記する。** `Literal` 型だけでは日本語の値(「村人」)を入れてくる(gpt-oss:20b)。`SpeakArgs.claimed_role` などの全列挙フィールドに許容値を書く。あるいは列挙値そのものを日本語にすることも検討。
2. **`num_predict` と呼び出しタイムアウトは初期版から必須。** 暴走は gemma4:12b(False)、qwen3:8b、qwen3.5:9b(False)で観測され、ハーネス側では防げない。`ModelRetryMiddleware` の再試行対象にもならない(エラーにならず長く走る)。
3. **「ツールを呼ばずに文章で答える」が小型モデルの主失敗型。** 発生率はモデルと reasoning で 0〜30%。GM の「1 回だけ再要求 → 決定的フォールバック」(計画どおり)で吸収する。観察項目 1 の指標はこれ。
4. **行動ツールは「最初の 1 回だけ受理」にする。** qwen3:8b は「受理しました」の後も speak を繰り返し上限まで走った。`ActionMailbox` は 2 回目以降を「提出済みです。これ以上ツールを呼ばず終了してください」と返す。1 応答に複数の並列ツール呼び出しが入ることもあるので、ツール側で弾く。
5. **`reasoning` の既定は True に変える(計画では False)。** gemma4:12b / qwen3.5:9b はツール呼び出しの安定性と陣営理解が True で大きく改善し、コストは 1 判断 +2〜3 秒。gpt-oss:20b は False を指定しても思考する(効かない)。
6. **役職秘匿を system prompt に明示する。** 「人狼は自分の役職を明かさず、相方を守るように振る舞う」「`claimed_role` は本当の役職ではなく、他人に信じさせたい役職」を入れ、嘘 CO 率・自白率を Step 4 の観察項目 5 で計測する。[ASCII.jp の事例](https://ascii.jp/elem/000/004/207/4207849/)の「自分から人狼だと明かす」は qwen3.5:9b(False)で実際に起きた。
7. **固定コストは約 1.5K トークン/呼び出し。** 自前 system prompt 300 トークン + ツール定義 800〜1,000 トークン(FS ツール 4 個の description が大半)。deepagents 0.7.24 は Ollama モデルにハーネスの base system prompt を付けない。`FilesystemMiddleware(tools=[...])` で `ls` を外す(gpt-oss:20b が寄り道した)か、`custom_tool_descriptions` で短くするのは観察項目 7 の候補。
8. **LangSmith は未検証。** Step 2 で鍵を設定し、Deep Agent のトレース階層とトークン表示を確認する。計測コールバック(`spike/common.py` の `ModelCallRecorder`)は `llm/metrics.py` のたたき台になる。
9. **seed 固定はテストの再現に有効だが、サンプリング評価では罠。** 同一プロンプト + 同一 seed は Ollama で完全に同じ出力になる。`bench` は seed をゲームごとに変える。

### 未検証(次の仮説)

- 仮説 3(履歴累積が `num_ctx=16384` に収まり、要約が発火しても推理が崩れない)は単発スレッドしか試していないため未検証。Step 3 で 1 ゲーム通してから `prompt_eval_count` の推移を見る。
- 5 人同時のテンポ(逐次実行で 1 ラウンド 5 × 5 秒 = 25 秒)と、`keep_alive` によるロード回避(`load_duration` は 2 回目以降 0.001 秒で問題なし)。
- 投票・夜の行動ツール(`vote` / `seer_look` / `thief_swap`)は単発テストのみ。Deep Agent 構成で `vote` の `target` 妥当性を見る。
