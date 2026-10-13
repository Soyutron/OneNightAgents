"""難度を上げた追加タスク(v2)。tasks.py の末尾で TASKS に連結する。

v1 は多くのモデルが 16〜18/19 で天井に当たったため、複数ファイル横断・長い計画・複数サブエージェント・
12K トークン級の文書・ネストしたスキーマ・スキルの選択など、差が出る課題を足す。
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from tasks import Task, _file, _final, _seq, _tool_calls

# --- ツール: 比較と依存のある連鎖 --------------------------------------------------


def _check_warmer(o: dict) -> bool:
    cities = sorted(c["args"].get("city", "").lower() for c in _tool_calls(o, "get_weather"))
    return cities == ["osaka", "sapporo"] and "大阪" in _final(o) and "札幌" not in _final(o).split("大阪")[0][-6:]


def _check_two_hop(o: dict) -> bool:
    calls = _tool_calls(o, "convert_currency")
    if len(calls) != 2:
        return False
    a, b = calls[0]["args"], calls[1]["args"]
    first_ok = a == {"amount": 100, "from_currency": "USD", "to_currency": "JPY"}
    second_ok = b.get("from_currency") == "JPY" and b.get("to_currency") == "EUR" and abs(float(b.get("amount", 0)) - 15000) < 1
    return first_ok and second_ok and "90" in _final(o)


# --- 仮想 FS: 複数ファイル横断の rename、JSON の更新、日本語の exact-match 編集 -------

REFACTOR_FILES = {
    "/pkg/client.py": "import requests\n\n\ndef fetch(url: str) -> dict:\n    return requests.get(url).json()\n",
    "/pkg/service.py": "from pkg.client import fetch\n\n\ndef load_user(uid: int) -> dict:\n    return fetch(f'/users/{uid}')\n",
    "/pkg/cli.py": "from pkg.client import fetch\n\nif __name__ == '__main__':\n    print(fetch('/health'))\n",
}


def _check_refactor(o: dict) -> bool:
    files = o.get("files") or {}
    c, s, k = files.get("/pkg/client.py", ""), files.get("/pkg/service.py", ""), files.get("/pkg/cli.py", "")
    return (
        "def fetch_user(" in c
        and "def fetch(" not in c
        and "import fetch_user" in s
        and "fetch_user(f'/users/" in s
        and "import fetch_user" in k
        and "fetch_user('/health')" in k
        and "write_file" not in _seq(o)
    )


CONFIG_JSON = '{\n  "host": "localhost",\n  "port": 8080,\n  "workers": 4\n}\n'


def _check_json_update(o: dict) -> bool:
    try:
        d = json.loads(_file(o, "/config.json"))
    except json.JSONDecodeError:
        return False
    return d.get("port") == 9090 and d.get("debug") is True and d.get("host") == "localhost" and d.get("workers") == 4


JA_DOC = "# 運用手順\n\n1. 毎朝 9 時にバックアップの完了を確認する。\n2. 障害時は、まず担当者に電話で連絡する。\n3. 週次で監査ログを確認する。\n"


def _check_ja_edit(o: dict) -> bool:
    d = _file(o, "/ops.md")
    return "まず Slack の #incident に投稿する" in d and "電話で連絡" not in d and "毎朝 9 時" in d and "edit_file" in _seq(o)


# --- 計画: 6 ステップ以上 --------------------------------------------------------------


def _check_pipeline(o: dict) -> bool:
    files = o.get("files") or {}
    raw = files.get("/pipeline/raw.csv", "")
    clean = files.get("/pipeline/clean.csv", "")
    report = files.get("/pipeline/report.md", "")
    check = files.get("/pipeline/CHECKLIST.md", "")
    clean_rows = [ln for ln in clean.splitlines() if ln.strip() and not ln.lower().startswith("name")]
    return (
        raw.count("\n") >= 4
        and len(clean_rows) == 3
        and all(x not in clean for x in ("Dave",))
        and ("84" in report or "84.0" in report)
        and all(f in check for f in ("raw.csv", "clean.csv", "report.md"))
        and bool(o.get("todos_used"))
    )


# --- サブエージェント: 2 種類を使い分ける ----------------------------------------------------

DOC_FOR_SUB = {"/docs/policy.md": "# Policy\n\nAll external calls must use the retry middleware. Conclusion: adopt the checkpointer for every long job.\n"}


def _check_two_subagents(o: dict) -> bool:
    f = _final(o)
    task_calls = [n for n in _seq(o) if n == "task"]
    # 結論文は "adopt the checkpointer for every long job" なので、訳文に必要なのはチェックポインタだけ
    return len(task_calls) >= 2 and "チェックポインタ" in f and "lookup_glossary" in _seq(o) and "read_file" in _seq(o)


# --- 長文: 約 12K トークンの文書から 1 行を探す ----------------------------------------------


def _make_book() -> str:
    paras = []
    for i in range(260):
        paras.append(
            f"第 {i + 1} 節。この節では架空の港町ミナトガワの {['漁業', '造船', '観光', '教育', '祭り', '交通'][i % 6]}について述べる。"
            "町の人口はおよそ 3 万人で、年間の来訪者は 40 万人に達する。産業の中心は長く漁業であったが、近年は観光の比重が高まっている。"
        )
        if i == 171:
            paras.append("補足: 町の図書館の蔵書数は 18 万 4 千冊であり、開館時間は平日 9 時から 19 時までである。")
    return "\n\n".join(paras) + "\n"


BOOK = _make_book()


def _check_book(o: dict) -> bool:
    f = _final(o)
    return ("18 万 4 千" in f or "184,000" in f or "184000" in f or "18万4千" in f) and not o.get("runaway")


# --- 構造化出力: ネストしたスキーマ ------------------------------------------------------------


class LineItem(BaseModel):
    name: str
    qty: int
    unit_price: float


class Order(BaseModel):
    order_id: str
    customer: str
    items: list[LineItem]
    total: float = Field(description="sum of qty * unit_price")


ORDER_TEXT = "Order A-2291 for customer Hoshino Trading: 4 x steel bracket at $12.50 each, 10 x hex bolt at $0.80 each, 1 x torque wrench at $45.00. Grand total $103.00."


def _check_order(o: dict) -> bool:
    s = o.get("structured") or {}
    items = s.get("items") or []
    return s.get("order_id") == "A-2291" and len(items) == 3 and abs(float(s.get("total", 0) or 0) - 103.0) < 0.01 and any(it.get("qty") == 10 for it in items)


# --- スキル: 2 つから正しい方を選ぶ ---------------------------------------------------------

from tasks import SKILL_HAIKU, SKILL_REPORT  # noqa: E402


def _check_skill_choice(o: dict) -> bool:
    reads = o.get("read_paths", [])
    r = _file(o, "/reports/2026-10-15.md")
    return any("status-report/SKILL.md" in p for p in reads) and not any("haiku-reply/SKILL.md" in p for p in reads) and "## Risks" in r


EXTRA_TASKS: list[Task] = [
    Task("tool_compare_ja", "tool_use", "ja", "大阪と札幌では今どちらが暖かいですか? ツールで両方の天気を調べて、都市名だけ答えてください。", _check_warmer, features=frozenset({"tools"}), max_calls=5),
    Task("tool_two_hop_en", "tool_use", "en", "Convert 100 USD to JPY, then convert that JPY amount to EUR. The tool only supports direct pairs, so do it in two steps. Report the final EUR amount.", _check_two_hop, features=frozenset({"tools"}), max_calls=6),
    Task(
        "fs_refactor_en",
        "filesystem",
        "en",
        "Rename the function `fetch` to `fetch_user` everywhere under /pkg/ (its definition in client.py and every import and call site). Use grep to find all occurrences and edit_file for each change. Do not rewrite whole files.",
        _check_refactor,
        files=REFACTOR_FILES,
        max_calls=12,
    ),
    Task("fs_json_update_ja", "filesystem", "ja", "/config.json を読み、port を 9090 に変更し、新しいキー debug を true で追加して、同じファイルに書き戻してください。他のキーは変えないでください。", _check_json_update, files={"/config.json": CONFIG_JSON}, max_calls=6),
    Task("fs_edit_ja", "filesystem", "ja", "/ops.md の手順 2 を「障害時は、まず Slack の #incident に投稿する。」に edit_file で書き換えてください。他の行は変えないでください。", _check_ja_edit, files={"/ops.md": JA_DOC}, max_calls=6),
    Task(
        "plan_pipeline_en",
        "planning",
        "en",
        "Plan with the todo list, then execute: (1) write /pipeline/raw.csv with header name,score and rows Alice,90 / Bob,75 / Carol,87 / Dave,abc; (2) write /pipeline/clean.csv containing only the rows whose score is a number; (3) write /pipeline/report.md with the average score of clean.csv (one decimal) and the number of dropped rows; (4) write /pipeline/CHECKLIST.md listing the three files with a checkbox each; (5) read back report.md and confirm the average in your final message.",
        _check_pipeline,
        features=frozenset({"todos"}),
        max_calls=20,
    ),
    Task(
        "sub_two_kinds_ja",
        "subagent",
        "ja",
        "2 つのサブエージェントを使ってください。researcher に /docs/policy.md を読ませて結論(Conclusion)の 1 文を報告させ、translator にその文に含まれる技術用語 'middleware' と 'checkpointer' を用語集で日本語に引かせてください。最後に、結論の日本語訳を 1 文で報告してください。",
        _check_two_subagents,
        files=DOC_FOR_SUB,
        features=frozenset({"subagent"}),
        max_calls=10,
    ),
    Task("long_book_ja", "long_context", "ja", "/book.md は長い文書です。全文を読まずに grep で探して、町の図書館の蔵書数を答えてください。", _check_book, files={"/book.md": BOOK}, max_calls=6),
    Task("struct_order_en", "structured", "en", f"Extract the order as structured data:\n\n{ORDER_TEXT}", _check_order, features=frozenset({"structured"}), max_calls=4, response_format=Order),
    Task(
        "skill_choice_ja",
        "skills",
        "ja",
        "プロジェクト Borealis の 2026-10-15 付けのステータスレポートを書いてください。要約: 認証機能が完成。リスク: 負荷試験が未実施。次: 負荷試験を実施。",
        _check_skill_choice,
        files={"/skills/status-report/SKILL.md": SKILL_REPORT, "/skills/haiku-reply/SKILL.md": SKILL_HAIKU},
        features=frozenset({"skills"}),
        max_calls=6,
    ),
]
