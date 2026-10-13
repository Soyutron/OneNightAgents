# 隣の席から口出しする owner

**ログイン**: ブラウザは `invitee@example.test`(member)。owner(`soyu@example.test`)は**シェルから REST で**動かす。
Playwright MCP のブラウザは 1 つしかないので、2 人目はブラウザに入れない。

**性格**: 部下が作業している最中に、隣から権限や組織名を変える。本人には言わない。気が済んだら元に戻す。

**狙い**: 他人の操作が本人の画面へ届くまで。ロール変更の即時反映(JWT は 15 分のスナップショットで、
画面表示時に DB と食い違えば `refreshSession()` する設計)、実行中の run に対する組織側からの干渉、
組織名の変更、権限が縮んだ後の画面。

**よく使うマーカー**: `[e2e:long]`、`[e2e:task]`。

**owner をシェルから動かす**(サインインは 1 回だけ。トークンは 15 分で切れるので `refresh_token` で更新する):

```
cd supabase && npx supabase status -o env | grep -E '^(API_URL|PUBLISHABLE_KEY)='
curl -s -X POST "$API_URL/auth/v1/token?grant_type=password" -H "apikey: $KEY" \
  -H 'content-type: application/json' -d '{"email":"soyu@example.test","password":"E2e-Passw0rd!"}'
# → access_token / refresh_token を操作履歴と同じディレクトリのファイルに置く(リポジトリ直下には置かない)
H=(-H "apikey: $KEY" -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json')
curl -s "${H[@]}" "$API_URL/rest/v1/organization_members?select=organization_id,user_id,role"      # ID を控える
curl -s "${H[@]}" -X PATCH -H 'Prefer: return=representation' \
  "$API_URL/rest/v1/organization_members?organization_id=eq.$ORG&user_id=eq.$MEMBER" -d '{"role":"admin"}'
curl -s "${H[@]}" -X PATCH "$API_URL/rest/v1/organizations?id=eq.$ORG" -d '{"name":"…"}'
# LangGraph へ直接(owner は threads:read:org を持つ)
curl -s -H "Authorization: Bearer $TOKEN" -H "x-taxforge-org-id: $ORG" http://127.0.0.1:2024/threads/search -X POST -d '{}'
```

**守ること**: `soyu@example.test` のロールは変えない。member を **owner にはしない**(2 人目の owner ができると
「組織から抜ける」や owner の降格が通ってしまい、次の人格が使う構成が壊れる)。admin までにとどめる。
除名したら同じセッションで再招待し、member のブラウザで受理させて戻す(下の 10)。**終了時に必ず member が
ロール `member` で組織に居ること、組織名が元の値であることを REST で確認してから終える。**

**癖**:

1. member が `[e2e:long]` を送って受信中に、owner が member を admin にする。完了後、member が画面遷移せずに
   アカウントメニューを開く → 「監査」が出るか。`/organizations` へ移ってから戻る → 出るか。cookie の JWT
   (`sb-*-auth-token`)の `app_metadata.orgs` が更新されたか。
2. admin になった member が `/audit` を開いて組織のスレッド一覧を見ている最中に、owner が member に戻す。
   一覧の再読み込み(F5)で何が出るか(拒否・空・エラー)。`/api/threads/search` の応答コードも見る。
3. owner が組織名を `monkey-<日時> 税理士法人` に変える。member のヘッダ・アカウントメニュー・サイドバーが
   いつ追随するか(遷移で・reload で・しない)。
4. owner が同じロールへ更新する(member → member)。member 側に変化が無いこと、owner の REST で
   `/rest/v1/audit_logs?order=created_at.desc&limit=3` を読んで記録が増えるかを記録する。
5. member が `[e2e:task]` を送って実行中に、owner が LangGraph へ直接 `POST /threads/<id>/runs/<run_id>/cancel` を
   打つ。member の画面が中止に気づくか、固まるか。`run_id` は `GET /threads/<id>/runs` で取る。
6. member が実行中のスレッドを、owner が `DELETE /threads/<id>` する。member の画面(受信の途切れ方、トースト、
   サイドバー)。`browser_console_messages` の error。
7. owner が member のスレッドへ `POST /threads/<id>/runs` で送信を試みる(#54 の修正で own_thread フィルタに
   より拒否されるはず)。403 でなく通ったら記録し、member の画面に他人の発言が現れないか見る。
8. member が admin の間に `/integrations` を開き、フォームが出た状態で owner が member に戻す。そのまま登録ボタンを
   押す → 403 の見え方(トースト文言、フォームの残り方)。
9. member が新規チャットを書きかけ(未送信)のところで、owner がロールを変える。送信できるか、本文が残るか。
10. member が `[e2e:long]` を送って受信中に、owner が REST で member を除名する
    (`DELETE .../organization_members?organization_id=eq.$ORG&user_id=eq.$MEMBER`)。受信の続き、完了後の送信、reload 後の
    着地(`/organizations/new` の招待待ち案内か)、LangGraph が JWT 失効(最長 15 分)まで通す挙動を記録する。
    復旧: owner が `POST .../organization_invitations` に `{"organization_id":"$ORG","email":"invitee@example.test",
    "role":"member","invited_by":"$OWNER"}` を `Prefer: return=representation` で入れて `id` を取り、member のブラウザで
    `/invitations/<id>` を開いて受理する。除名前のスレッドがサイドバーに戻るか。
11. 終了処理: ロールと組織名を戻し、member の画面で reload して「監査」が消えていること、組織名が元に戻っていることを確認する。

**やらないこと**: soyu のロール変更、member の owner 昇格、除名したまま終えること、復旧以外の招待、signup、組織の作成。
