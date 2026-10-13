.DEFAULT_GOAL := help

# ---------------------------
# Setup
# ---------------------------

install: ## 依存関係の同期 (tax-agents, agent-ui, Supabase CLI)
	cd tax-agents && uv sync
	cd agent-ui && pnpm install
	cd supabase && npm ci

# クローン直後に一度だけ実行する。既存の .env と署名鍵は残す(gen signing-key は
# 既定で上書きするため、作り直すと発行済みのローカルセッションが全て無効になる)。
# officecli と Playwright のブラウザはここでは入れない(tax-agents/README.md と
# agent-ui/README.md の「セットアップ」を参照)。
#
# 空の配列を先に置くのは、config.toml が signing_keys_path を宣言していると
# CLI(2.116.0)がファイルの読み取りに失敗して落ちるため(新規作成ができない)。
setup-local: install ## 初回セットアップ (.env の雛形と Supabase の署名鍵。既存ファイルは残す)
	test -f tax-agents/.env || cp tax-agents/.env.example tax-agents/.env
	test -f agent-ui/.env || cp agent-ui/.env.example agent-ui/.env
	test -f supabase/signing_keys.json || (cd supabase && echo '[]' > signing_keys.json && npx supabase gen signing-key --algorithm ES256 --yes)

clean: ## キャッシュ / ビルド成果物を削除
	rm -rf tax-agents/dist tax-agents/build tax-agents/.langgraph_api agent-ui/.next
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------
# Development
# ---------------------------

# --no-reload: 自動リロードのたびにinmemランタイムが .langgraph_api の
# スナップショットを読み直すが、そのスナップショットはチェックポイント本体と
# writes(メッセージ差分)がずれた状態で保存されることがあり、復元すると
# スレッドの会話が空になる。リロードを止めて状態をプロセス内に保つ。
langgraph-dev: ## LangGraph dev server を起動 (tax-agents, Studio有効)
	cd tax-agents && uv run langgraph dev --config langgraph.dev.json --no-reload

ui-dev: ## Next.js 開発サーバーを起動 (agent-ui)
	cd agent-ui && pnpm dev

ui-storybook: ## Storybook を起動 (agent-ui)
	cd agent-ui && pnpm storybook

# ---------------------------
# Code Quality
# ---------------------------

py-lint: ## Lint (tax-agents, Ruff)
	cd tax-agents && uv run ruff check

py-format: ## フォーマット (tax-agents, Ruff)
	cd tax-agents && uv run ruff format

py-typecheck: ## 型チェック (tax-agents, ty)
	cd tax-agents && uv run ty check

ui-lint: ## Lint (agent-ui, ESLint)
	cd agent-ui && pnpm lint

ui-format: ## フォーマット (agent-ui, Prettier)
	cd agent-ui && pnpm format

ui-typecheck: ## 型チェック (agent-ui, tsc)
	cd agent-ui && pnpm typecheck

py-test: ## テスト (tax-agents, pytest)
	cd tax-agents && uv run pytest -q

ui-test: ## テスト (agent-ui, Vitest の node プロジェクト。Storybook のブラウザテストは ui-test-storybook)
	cd agent-ui && pnpm exec vitest run --project node

ui-test-storybook: ## Storybook の play 関数を Vitest ブラウザモードで実行 (agent-ui。Chromium が必要で重い)
	cd agent-ui && pnpm exec vitest run --project storybook

check: py-lint py-typecheck py-test ui-lint ui-typecheck ui-test ## lint / typecheck / テストをまとめて実行 (CI と同じ静的チェック + pytest + Vitest(node))

# ---------------------------
# Evals (Harbor)
# ---------------------------

# 詳細は tax-agents/README.md「Harbor による評価」。Docker が必要。
# SUPABASE_URL は Settings が必須にしているだけで評価経路では使わないため、ダミーを渡す。
harbor-eval: ## Harbor で tax-agents の評価タスクを実行 (結果は LangSmith の実験にも記録)
	uvx --env-file tax-agents/.env --from 'harbor[langsmith]==0.22.0' harbor run \
		-p tax-agents/evals -o tax-agents/evals/jobs \
		--agent langgraph --model openai:gpt-5.6-luna \
		--ak project_path=tax-agents/evals/agent --ak graph=harbor_agent --ak 'python_version="3.13"' \
		--ae SUPABASE_URL=https://placeholder.supabase.co \
		--plugin langsmith --pk dataset_name=tax-agents-evals -y

harbor-view: ## Harbor の評価結果をブラウザで閲覧 (http://127.0.0.1:8080)
	uvx --from 'harbor[langsmith]==0.22.0' harbor view tax-agents/evals/jobs --port 8080

# ---------------------------
# E2E (LLM スタブ)
# ---------------------------

# 実 LLM を呼ばずにエージェントを動かす構成。詳細は docs/design/monkey-e2e.md。
# langgraph.e2e.json は .env を読まず、ダミーキーとスタブの URL を直接持つ。
llm-stub: ## OpenAI Responses API のスタブを起動 (127.0.0.1:8790。ログは OPENAI_STUB_LOG に JSONL)
	cd tax-agents && uv run python tests/e2e/openai_stub.py --port 8790

langgraph-e2e: ## LLM スタブ向け設定で LangGraph dev server を起動 (認証あり、状態は langgraph-dev と共有。先に llm-stub を起動)
	cd tax-agents && uv run langgraph dev --config langgraph.e2e.json --no-browser --no-reload

# 環境の起動・停止は scripts/monkey-services.sh に集約している(monkey-loop も同じものを使う)。
monkey-up: ## Monkey テストの環境を起動 (Supabase / LLM スタブ / LangGraph(e2e) / Next。ログは logs/monkey)
	scripts/monkey-services.sh up

monkey-down: ## Monkey テストの環境を停止 (Next / LangGraph / LLM スタブ。Supabase は残る)
	scripts/monkey-services.sh down

monkey-status: ## Monkey テストの環境の起動状態を表示 (2024 が e2e 設定かも見る)
	scripts/monkey-services.sh status

monkey-loop: ## 人格型 Monkey テストを終了時刻まで繰り返す (例: make monkey-loop MONKEY_UNTIL=22:00 MONKEY_MODEL=sonnet。ローカル LLM なら MONKEY_PROVIDER=ollama MONKEY_MODEL=<ollama のモデル>。詳細は scripts/monkey-loop.sh)
	MONKEY_UNTIL=$(MONKEY_UNTIL) MONKEY_MODEL=$(MONKEY_MODEL) MONKEY_EFFORT=$(MONKEY_EFFORT) MONKEY_STEPS=$(MONKEY_STEPS) MONKEY_PERSONAS="$(MONKEY_PERSONAS)" MONKEY_PROVIDER=$(MONKEY_PROVIDER) MONKEY_OLLAMA_HOST=$(MONKEY_OLLAMA_HOST) scripts/monkey-loop.sh

monkey-stop: ## 走っている monkey-loop を止める (いまの人格の反復が終わってから終了。即座に止めるなら Ctrl+C)
	scripts/monkey-loop.sh stop

# ---------------------------
# Issue agent (Opus が実装、Fable がレビュー)
# ---------------------------
#
# GitHub Issue を worktree で実装し、レビューを通して feature へ統合する。設計は docs/design/issue-agent.md。
# 環境変数は make の変数としてそのまま渡す(例: make issue-agent ISSUE=123 ISSUE_PLAN_ONLY=1)。

issue-agent: ## Issue を 1 件処理して feature へ統合し push (make issue-agent ISSUE=123 [ISSUE_PLAN_ONLY=1] [ISSUE_PUSH=0]。詳細は scripts/issue-agent.sh)
	@test -n "$(ISSUE)" || { echo "ISSUE=<番号> を指定してください(例: make issue-agent ISSUE=123)"; exit 1; }
	scripts/issue-agent.sh $(ISSUE)

issue-loop: ## ready-for-agent の Issue を古い順に処理し続ける (make issue-loop ISSUE_UNTIL=22:00。詳細は scripts/issue-loop.sh)
	scripts/issue-loop.sh

issue-stop: ## 走っている issue-loop を止める (いまの Issue が終わってから終了。即座に止めるなら Ctrl+C)
	scripts/issue-loop.sh stop

issue-status: ## 処理中の Issue、残っている worktree、ログの場所を表示
	scripts/issue-agent.sh status

issue-labels: ## issue-loop と triage が使うラベルを GitHub に作る (初回のみ。既存のラベルは説明と色を更新)
	gh label create needs-triage --color e4e669 --description "Maintainer needs to evaluate this issue" --force
	gh label create needs-info --color d876e3 --description "Waiting on reporter for more information" --force
	gh label create ready-for-agent --color 0e8a16 --description "Fully specified, ready for an AFK agent (issue-loop picks these up)" --force
	gh label create ready-for-human --color fbca04 --description "Requires human implementation" --force
	gh label create agent:in-progress --color 1d76db --description "issue-agent is working on this" --force
	gh label create agent:done --color 0052cc --description "Merged into feature by issue-agent; closes on deploy" --force
	gh label create agent:needs-human --color b60205 --description "issue-agent stopped; a human must take over" --force

# ---------------------------
# Supabase
# ---------------------------
#
# ターゲット名の2語目が適用先。db-local-* はローカルの Docker だけを触り、
# db-remote-* はホスト型プロジェクトを触る(ENV=staging|prod の明示が必須)。
# supabase/package.json のCLIを使う(グローバル版とのバージョン差を避ける)。
# 詳細は supabase/README.md 参照。

# config.toml の site_url / additional_redirect_urls はこの2つの環境変数を参照している。
# 既定値はローカル開発用。staging も同じ値を使う。
TAXFORGE_SITE_URL ?= http://127.0.0.1:3000
TAXFORGE_REDIRECT_URL ?= https://127.0.0.1:3000

# --- ローカル(Docker) ---

# ローカル開発で使わないコンテナを除外する(realtime / storage / edge functions は未使用、
# logflare は 1GB 前後のメモリを食う)。config.toml は変えずに済むので config push に影響しない。
# Studio(と依存する postgres-meta)は残す。さらに削るなら studio,postgres-meta を足す。
SUPABASE_EXCLUDE ?= realtime,storage-api,imgproxy,edge-runtime,logflare,vector,supavisor

# Supabase CLI は公開ポートを 0.0.0.0 に束縛し、束縛先を変える設定を持たない(supabase/cli#1397)。
# このホストは公開 IP を持ち、2026-09-05 に 0.0.0.0 で公開していた Postgres がランサムウェアに
# 侵入された。そこで公開ポートを 127.0.0.1 に束縛する Docker ネットワークを用意し、CLI には
# SUPABASE_NETWORK_ID(--network-id に対応する環境変数)でそれを使わせる。
SUPABASE_NETWORK_ID ?= supabase-local
export SUPABASE_NETWORK_ID

db-local-start: ## [local] ローカル Supabase を軽量構成で起動(Docker 必須。公開ポートは 127.0.0.1 のみ。Studio は 54323、Mailpit は 54324)
	docker network inspect $(SUPABASE_NETWORK_ID) >/dev/null 2>&1 || docker network create -o com.docker.network.bridge.host_binding_ipv4=127.0.0.1 $(SUPABASE_NETWORK_ID)
	cd supabase && TAXFORGE_SITE_URL=$(TAXFORGE_SITE_URL) TAXFORGE_REDIRECT_URL=$(TAXFORGE_REDIRECT_URL) npx supabase start -x $(SUPABASE_EXCLUDE)

db-local-stop: ## [local] ローカル Supabase を停止(データは保持される)
	cd supabase && npx supabase stop

# start はボリュームが新規のときしかマイグレーションを流さない。追加分はこれで当てる。
# --local を明示するのは、リモート(--linked)へ誤って流さないため。
db-local-migrate: ## [local] 未適用のマイグレーションをローカルに当てる(データは残る)
	cd supabase && npx supabase migration up --local

db-local-reset: ## [local] ボリュームを作り直して全マイグレーションを適用(ユーザー・組織も消える)
	cd supabase && npx supabase db reset --local

db-local-test: ## [local] supabase/tests の pgTAP テストをローカル DB で実行(全件ロールバックされる)
	cd supabase && npx supabase test db --local

# --- リモート(ホスト型プロジェクト。ENV=staging|prod が必須) ---

# push 系のターゲットは link 先へ暗黙に流さず、ENV で宛先を明示させる
# (link 先の取り違えで別プロジェクトへ適用するのを防ぐ)。
# ref は公開 API の URL(https://<ref>.supabase.co)に含まれる識別子で、秘密ではない。
# 開発はローカル Supabase で行う。staging は本番前にホスト型でマイグレーションと
# config push を確かめる場所で、Supabase 上のプロジェクト名は taxforge-dev のまま。
SUPABASE_REF_staging = zrcszhzybvdgzywbirfu
SUPABASE_REF_prod    = ovdqbthwmnywaiavsmfe
SUPABASE_REF = $(SUPABASE_REF_$(ENV))

db-remote-login: ## [remote] Supabase CLI にログイン(ブラウザが開く。リモート操作の前に一度だけ)
	cd supabase && npx supabase login

db-remote-target: ## [remote] 操作対象を表示(ENV=staging|prod が必須。他の db-remote-* の前提)
	@test -n "$(SUPABASE_REF)" || { echo "ENV=staging または ENV=prod を指定してください(例: make db-remote-status ENV=staging)"; exit 1; }
	@echo "==> $(ENV) ($(SUPABASE_REF))"

db-remote-status: db-remote-target ## [remote] マイグレーションのローカル/リモートの適用状況を表示
	cd supabase && npx supabase migration list --project-ref $(SUPABASE_REF)

db-remote-push: db-remote-target ## [remote] 未適用のマイグレーションを適用(dry-run で件数を見せてから適用)
	cd supabase && npx supabase db push --project-ref $(SUPABASE_REF) --dry-run
	cd supabase && npx supabase db push --project-ref $(SUPABASE_REF)

# 本番はサイト URL が公開ドメインになるため、開発用の既定値のままでは push させない。
db-remote-config-push: db-remote-target ## [remote] config.toml を反映(差分表示のあと確認なしで適用される)
	@if [ "$(ENV)" = "prod" ] && [ "$(TAXFORGE_SITE_URL)" = "http://127.0.0.1:3000" ]; then \
		echo "ENV=prod では TAXFORGE_SITE_URL と TAXFORGE_REDIRECT_URL に本番の URL を渡してください"; exit 1; fi
	cd supabase && TAXFORGE_SITE_URL=$(TAXFORGE_SITE_URL) TAXFORGE_REDIRECT_URL=$(TAXFORGE_REDIRECT_URL) npx supabase config push --project-ref $(SUPABASE_REF)

# ---------------------------
# Help
# ---------------------------

help: ## コマンド一覧を表示
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: install setup-local clean langgraph-dev ui-dev ui-storybook py-lint py-format py-typecheck \
	ui-lint ui-format ui-typecheck py-test ui-test ui-test-storybook check harbor-eval harbor-view llm-stub langgraph-e2e \
	monkey-up monkey-down monkey-status monkey-loop monkey-stop \
	issue-agent issue-loop issue-stop issue-status issue-labels \
	db-local-start db-local-stop db-local-migrate db-local-reset db-local-test \
	db-remote-login db-remote-target db-remote-status db-remote-push db-remote-config-push help
