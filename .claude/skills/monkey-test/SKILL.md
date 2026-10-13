---
name: monkey-test
description: Claude Code が Playwright MCP でテスター人格になり、TaxForge のローカル環境(LLM はスタブ)を探索的に壊す Monkey テスト。「モンキーテスト」「人格で壊して」「/monkey-test <人格> [step数]」で使う。
---

# Monkey テスト(人格型)

Claude Code がテスター人格として agent-ui を操作し、状態不整合や例外を探す。実 LLM は呼ばない。
設計と背景は `docs/design/monkey-e2e.md` §9。人格は `personas/*.md`。

引数: `/monkey-test <人格名> [step 数]`。step 数の既定は 30。人格名が無ければ `personas/` を列挙して 1 つ選ぶ。

## 前提の確認(必ず最初に行う)

`make monkey-status` で次を確認し、満たないものがあれば `make monkey-up` で起こす(Supabase・スタブ・LangGraph・Next をまとめて起動し、準備できるまで待つ)。個別に見るなら以下。

1. スタブ: `curl -s http://127.0.0.1:8790/health` が `{"ok":true}`。
2. LangGraph: `ss -ltnp | grep ':2024 '` がある。プロセスのコマンド行に `langgraph.e2e.json` が含まれること
   (`ps -o cmd -p <pid>`)。`langgraph.dev.json` で動いている場合は実 LLM に向いているので送信してはいけない。
   対話中ならユーザーに確認してそのプロセスを PID で止め(`pkill -f` は使わない)`make langgraph-e2e` で起こし直す。
   無人実行ならその旨を出力して終了する。
3. Next: `ss -ltnp | grep ':3000 '`。
4. Supabase: `curl -s http://127.0.0.1:54321/auth/v1/health`。
5. 偽 R2(moto): `curl -s http://127.0.0.1:8791/` が応答する。成果物・添付・スレッド削除の後始末はここを通る。
   `make monkey-up` が起動とバケット作成を行う。中身はプロセスのメモリ上にしか無い。
6. スタブのログ先を決める: `OPENAI_STUB_LOG` が未設定なら、スタブ起動時に scratchpad のパスを渡す。

ブラウザは常に `http://localhost:3000` で開く(`127.0.0.1` は Next dev が静的ファイルを 403 にする)。

## 検証ユーザー(ローカル Supabase)

| 役割 | メール | パスワード | 組織 |
| --- | --- | --- | --- |
| owner | `soyu@example.test` | `E2e-Passw0rd!` | ローカル検証会計事務所 |
| member | `invitee@example.test` | `E2e-Passw0rd!` | 同上 |

ログイン後は `document.cookie` に `taxforge-org=` が現れるまで待つ。現れる前に `/organizations` へ行くと
読み込み中のまま止まる(これは既知の挙動で、再発見しても Issue にしない)。

## スタブのマーカー

送信するメッセージ本文に含めると、被テスト側の LLM 応答を決められる。人格の目的に合わせて選ぶ。

| マーカー | 応答 | 主な用途 |
| --- | --- | --- |
| (無し) | 短い文を 3 チャンク | 通常の送信 |
| `[e2e:stall-8]` | 1 チャンク後に 8 秒無応答 | 実行中の reload・中止・連打 |
| `[e2e:todos-8]` | Todo を全件完了で書いた後、8 秒停滞して終える | 「Todo 全完了 + 6 秒無更新」での固着解除判定(`use-run-status.ts`)。停滞中はサーバーが busy なので入力欄は開かないのが正 |
| `[e2e:long]` | 200 チャンクを 50ms 間隔(約 10 秒) | 受信中の操作、スクロール、中止 |
| `[e2e:ask]` | `ask_user` で質問(選択肢 2 つ) | interrupt カード、回答、無視して送信、reload 復元 |
| `[e2e:task]` | quick-researcher へ委譲 | サブエージェント表示、進捗、途中の中止 |
| `[e2e:error]` | 500 を返し続ける(SDK 再試行で 15〜30 秒後に失敗) | RunError と再実行 |
| `[e2e:rate-limit-stream-N]` | 受信途中のレート制限を N 回、その後成功(9 以上で再試行を使い切る) | 一時障害の最中と直後の表示 |
| `[e2e:truncated[-N]]` | 本文の途中で切断(N があれば N 回だけ) | 途中までの文が残らないか |
| `[e2e:empty]` / `[e2e:bad-args]` / `[e2e:billing]` / `[e2e:no-response-N]` | 空応答 / 壊れたツール引数 / 残高切れ / N 秒無応答 | 人格 `unlucky-operator` を参照 |

## セッションの進め方

1. 人格ファイルを読み、目的・癖・使うマーカーを把握する。
2. 操作履歴ファイルを作る(`monkey-<人格>-<日時>.md`)。置き場所は環境変数 `MONKEY_LOG_DIR` があればそこ、
   無ければ scratchpad。各 step を 1 行で追記する: `step / URL / 観測した状態 / 選んだ操作 / 結果(正常・異常)`。
3. ループ(step 数まで):
   - **観測**: `browser_snapshot` で状態を判定する(下表)。
   - **選択**: 人格の癖に沿って、いまの状態で可能な操作を 1 つ選ぶ。同じ操作を 3 回続けない。
   - **実行**: MCP ツールで操作する。50ms 以内の連打やページ内の並列操作は `browser_run_code_unsafe` で
     ページ内から直接行う(ツール呼び出しの連打では再現できない)。
   - **点検**: 5 step ごと、および画面遷移・エラーらしい表示のたびに `browser_console_messages` と
     `browser_network_requests` を見る。下の「異常の判定」に当たれば記録する。
4. 終了時: 異常の一覧を操作履歴の末尾にまとめ、再現を 1 回試みる。再現したものは GitHub Issue にする
   (`docs/agents/issue-tracker.md` の手順。ラベル `bug`)。起票の前に `gh issue list --state open --label bug --json number,title`
   で同じ現象が無いか見て、あれば新規に作らず該当 Issue へ観測(日時・手順・数値)をコメントする。
   再現しないものは「未再現」として履歴に残すだけにする。
5. 無人実行(`scripts/monkey-loop.sh` から `claude -p` で起動された場合)は、ユーザーに質問せず自分で判断し、
   最後に異常の一覧と Issue 番号を要約として出力して終える。

### 状態の判定

| 状態 | 見るもの |
| --- | --- |
| 未ログイン | URL が `/login` `/signup` `/reset-password` |
| 組織なし | URL が `/organizations/new` |
| チャット待機 | 送信ボタンが有効で中止ボタンが無い |
| 実行中 | 中止ボタンがある |
| 回答待ち | 質問カードがあり入力欄が閉じている |
| 失敗 | 「処理を完了できませんでした」または再実行ボタン |
| 切断 / 失効 | 「サーバーに接続できません」/「サインインが必要です」 |

### 異常の判定(これに当たったら記録)

- `browser_console_messages` の error(Next の hydration / HMR 警告は除く)。
- network の 5xx、失敗リクエスト、ログイン済みなのに `/api/` や Supabase が 401。
- 「問題が発生しました」(エラー境界)。
- 想定しない赤いトースト(権限が無い操作への「権限がありません」は想定内)。
- **UI は待機中なのに、サーバーは実行中**: 送信ボタンが有効な状態で
  `curl -s -b <cookie> http://localhost:3000/api/threads/<threadId>` の `status` が `busy`。
  URL の `threadId` と、ブラウザの cookie は `browser_evaluate` の `document.cookie` から取る
  (`sb-*` は HttpOnly ではないので読める)。
- 送信後に自分のメッセージが 2 件並ぶ、または増えない。
- reload 後にメッセージが減る、質問カードが消える。
- 「サーバーに接続できません」が、`curl http://localhost:2024/ok` は応答するのに 40 秒以上残る。
- 組織を切り替えた後、サイドバーに切替前の組織のスレッドが残る。

### してはいけないこと

- `.env` の変更、`config.toml` の変更、リモート Supabase への操作。
- リポジトリ直下に作業ファイルを作ること(cookie jar、一時 JSON など)。一時ファイルは操作履歴と同じディレクトリ
  (`MONKEY_LOG_DIR`、無ければ scratchpad)に置く。
- ローカル DB のユーザー・組織の削除(次の人格が使う)。招待は自分で出したものだけ取り消してよい。
- 実 LLM に向いた `langgraph dev`(`langgraph.dev.json`)に対して送信すること。前提の確認 2 を守る。
- `navigator.clipboard.readText()` を呼ぶこと。権限が `prompt` のままエラーも返さず Promise が保留され、
  `browser_evaluate` が返らないまま反復が打ち切りまで固まる。コピーの検証はコピー元の DOM を読む。
- 同一宛先へ 60 秒以内に確認コードを再送すること(`over_email_send_rate_limit` の 429)。ローカルの Auth メールに
  時間あたりの上限は無い(`config.toml` の `email_sent = 2` はカスタム SMTP 有効時だけ送られ、ローカルの実値は 360000/時)。
  招待メールはローカルでは送られない(`RESEND_API_KEY` 未設定で `delivered: false`)ので、招待リンクは画面のコピーで取る。
- `invitee@example.test` を除名したまま終えること。除名する人格は同じセッションで再招待して受理させ、ロール `member` で戻す。
- サインインの繰り返し(5 分に 30 回で制限される)。人格の切り替え時だけサインアウト・サインインする。
- 共有状態を変えたまま終えること。画面サイズ・ロール・組織名・パスワードを変える人格は、異常が出て中断する場合も
  人格ファイルの終了処理で元に戻す(ブラウザと検証ユーザーは次の人格と共有される)。
