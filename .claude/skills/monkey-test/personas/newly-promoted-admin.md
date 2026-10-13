# 昇格したばかりの admin

**ログイン**: 3 段階。(1) `soyu@example.test`(owner)で `invitee@example.test` を admin に上げてサインアウト。
(2) `invitee@example.test` でサインインし、admin として動く。(3) 終了時に owner で戻り、`member` に戻して
サインアウトし、最後に member で入って権限が縮んだことを確かめる。サインインは合計 4 回まで。

**性格**: 権限をもらったばかりで何ができるか分かっていない。全部の画面を開く。owner のことも触ろうとする。
連携の設定を試しに入れては消す。

**狙い**: admin ロール固有の画面と境界。owner に触れないこと、owner ロールの招待が拒否されること、
連携(Chatwork トークン)の登録・上書き・削除、監査画面、降格後の縮み方。

**よく使うマーカー**: 無し、`[e2e:task]`。

**守ること**: `soyu` のロールは変えない。「組織から抜ける」は押さない(admin でも抜けられる設計。抜けると次の人格が
使えない)。**終了時に必ず member へ戻す。**

**癖**:

1. admin でサインインした直後、アカウントメニューに「監査」があるか(JWT は昇格前に発行されていない。
   サインイン時点で DB の値が載るはず)。`/audit` で組織全体のスレッドが見えるか。
2. `/organizations`。owner の行のロール選択と除名ボタンが無効か。`browser_run_code_unsafe` から Supabase REST に
   (cookie の JWT を使って)owner を member にする `PATCH` を投げる → 0 行または例外になるか、トーストは出るか。
3. 自分の行。ロール選択が無効か。同じく REST から自分を owner に上げる `PATCH` → 拒否されるか。
4. 招待を **role=owner** で発行する(RLS の with check で拒否される設計)。拒否の文言。通ってしまったら即取り消して記録する。
   続けて role=member で `monkey-<日時>@example.test` へ発行し、一覧に載ったら取り消す(ローカルでは招待メールは送られず
   `delivered: false` になる)。
5. `/integrations`。空文字、空白だけ、5000 文字、先頭と末尾に空白のあるトークンを順に登録する。どれが通り、どう表示されるか。
   登録済みの状態で再登録(上書き)→ 削除 → reload。削除後にもう一度削除を押す。
6. `/integrations` で登録直後に reload せず削除、を 30ms 間隔で 2 回押す(`browser_run_code_unsafe`)。
7. `/audit` から owner のスレッドを開く。入力欄が書き込めるように見えるのは #61 で起票済み。それ以外(表示崩れ、
   サイドバーの所属、戻る導線)を見る。
8. `/templates` で既存テンプレートを 1 つ公開する(admin は公開できる設計)。通ったら終了時に取り下げる。
9. チャットで `[e2e:task]` を送り、完了後に `/audit` で自分の行の表示(名前、時刻)を見る。
10. 終了処理。owner で member に戻し、member で入り直して: アカウントメニューから「監査」が消えている、
    `/audit` 直接アクセスが拒否される、`/integrations` が読み取り表示になる、`fetch('/api/store/items', {method:'PUT'})`
    が 403 になる。

**やらないこと**: soyu のロール変更、組織から抜ける、`invitee@example.test` の除名、signup、テンプレートの削除。
