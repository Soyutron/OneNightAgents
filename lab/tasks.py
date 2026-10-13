"""ベンチマークのタスク定義と判定関数。

タスクは「ハーネスのどの機能を使うか(features)」「仮想 FS の初期ファイル」「プロンプト」「判定」を持つ。
判定は決定的(LLM ジャッジは使わない)。outputs は run_eval.Outcome の dict 表現。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# ツール選択タスクで使うダミーツール(呼び出し履歴は run_eval 側で記録する)
# ---------------------------------------------------------------------------

WEATHER = {"osaka": "sunny, 24°C", "tokyo": "rainy, 17°C", "sapporo": "cloudy, 9°C"}
RATES = {("JPY", "EUR"): 0.006, ("USD", "JPY"): 150.0, ("EUR", "JPY"): 165.0, ("USD", "EUR"): 0.9}


class GetWeatherArgs(BaseModel):
    city: str = Field(description="都市名(英語の小文字または大文字。例: Osaka, Tokyo, Sapporo)")


class ConvertCurrencyArgs(BaseModel):
    amount: float = Field(description="金額")
    from_currency: Literal["JPY", "USD", "EUR"] = Field(description="元の通貨。JPY / USD / EUR のいずれか")
    to_currency: Literal["JPY", "USD", "EUR"] = Field(description="換算先の通貨。JPY / USD / EUR のいずれか")


class CreateReminderArgs(BaseModel):
    title: str = Field(description="リマインダーの件名")
    date: str = Field(description="日付。YYYY-MM-DD 形式")
    priority: Literal["low", "normal", "high"] = Field(default="normal", description="優先度。low / normal / high のいずれか")


class LookupGlossaryArgs(BaseModel):
    term: str = Field(description="英語の用語")


GLOSSARY = {"deep agent": "深層エージェント", "checkpointer": "チェックポインタ", "middleware": "ミドルウェア"}


# ---------------------------------------------------------------------------
# 構造化出力タスクのスキーマ
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    vendor: str
    total: float
    due_date: str = Field(description="YYYY-MM-DD")
    items: list[str]


class Event(BaseModel):
    title: str
    date: str = Field(description="YYYY-MM-DD")
    location: str
    attendees: list[str]


# ---------------------------------------------------------------------------
# タスク定義
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    category: str
    lang: str
    prompt: str
    check: Callable[[dict], bool]
    files: dict[str, str] = field(default_factory=dict)
    features: frozenset[str] = frozenset()  # "tools" | "todos" | "subagent" | "skills" | "structured"
    max_calls: int = 6
    response_format: type[BaseModel] | None = None


def _file(outputs: dict, path: str) -> str:
    return (outputs.get("files") or {}).get(path, "") or ""


def _final(outputs: dict) -> str:
    return outputs.get("final") or ""


def _tool_calls(outputs: dict, name: str) -> list[dict]:
    return [c for c in outputs.get("custom_calls", []) if c["name"] == name]


def _seq(outputs: dict) -> list[str]:
    return outputs.get("tool_seq") or []


# --- ツール選択 ---------------------------------------------------------------


def _check_weather_osaka(o: dict) -> bool:
    calls = _tool_calls(o, "get_weather")
    return len(calls) == 1 and calls[0]["args"].get("city", "").lower() == "osaka" and "24" in _final(o)


def _check_convert_jpy_eur(o: dict) -> bool:
    calls = _tool_calls(o, "convert_currency")
    ok_args = len(calls) == 1 and calls[0]["args"] == {"amount": 10000, "from_currency": "JPY", "to_currency": "EUR"}
    return ok_args and "60" in _final(o)


def _check_weather_then_reminder(o: dict) -> bool:
    w = _tool_calls(o, "get_weather")
    r = _tool_calls(o, "create_reminder")
    if len(w) != 1 or len(r) != 1:
        return False
    a = r[0]["args"]
    return "傘" in a.get("title", "") and a.get("date") == "2026-10-15" and a.get("priority") == "high"


def _check_no_tool(o: dict) -> bool:
    return not o.get("custom_calls") and "paris" in _final(o).lower()


def _check_unsupported_currency(o: dict) -> bool:
    # Literal 違反のツールエラーから回復し、ツールの乱発なしに「できない」と説明できるか
    f = _final(o)
    return o.get("model_calls", 99) <= 4 and (("GBP" in f) or ("ポンド" in f)) and not o.get("runaway")


# --- 仮想 FS -------------------------------------------------------------------

CALC_PY = '''"""小さな計算モジュール。"""


def add(a: int, b: int) -> int:
    return a - b


def sub(a: int, b: int) -> int:
    return a - b


def mul(a: int, b: int) -> int:
    return a * b
'''


def _check_fix_bug(o: dict) -> bool:
    src = _file(o, "/src/calc.py")
    return "return a + b" in src and src.count("return a - b") == 1 and "def mul" in src


NOTES = {
    "/notes/2026-10-01.md": "# 10/1 定例\n\n議論: ログ基盤の選定。\n\n決定事項: ログ基盤は Loki を採用する。\n",
    "/notes/2026-10-08.md": "# 10/8 定例\n\n議論: リリース日程。\n\n決定事項: v2 のリリースは 11 月 20 日とする。\n",
    "/notes/2026-10-13.md": "# 10/13 定例\n\n議論: サポート体制。\n\n決定事項: 問い合わせ対応の一次窓口は佐藤が担当する。\n",
}


def _check_summary_notes(o: dict) -> bool:
    s = _file(o, "/summary.md")
    return all(k in s for k in ("Loki", "11 月 20 日", "佐藤")) or all(k in s for k in ("Loki", "11月20日", "佐藤"))


TODO_FILES = {
    "/src/a.py": "def f():\n    # TODO: 入力検証を追加\n    return 1\n\n\ndef g():\n    # TODO: 例外処理\n    return 2\n",
    "/src/b.py": "def h():\n    return 3  # TODO: マジックナンバーを定数化\n",
    "/src/c.py": "class C:\n    # TODO: docstring を書く\n    def m(self):\n        # TODO: 実装\n        pass\n",
}


def _check_todos_md(o: dict) -> bool:
    t = _file(o, "/todos.md")
    lines = [ln for ln in t.splitlines() if "TODO" in ln or "todo" in ln.lower()]
    return len(lines) >= 5 and all(name in t for name in ("a.py", "b.py", "c.py")) and "grep" in _seq(o)


def _check_create_config(o: dict) -> bool:
    cfg = _file(o, "/app/config.json")
    try:
        d = json.loads(cfg)
    except json.JSONDecodeError:
        return False
    return d.get("host") == "localhost" and d.get("port") == 8080 and bool(_file(o, "/app/README.md").strip())


# --- 計画(write_todos) ------------------------------------------------------------


def _check_project_build(o: dict) -> bool:
    csv = _file(o, "/proj/data.csv")
    stats = _file(o, "/proj/stats.md")
    readme = _file(o, "/proj/README.md")
    rows = [ln for ln in csv.splitlines() if ln.strip() and not ln.lower().startswith("name")]
    return (
        len(rows) == 3
        and ("82.3" in stats or "82.33" in stats)
        and "Alice" in stats
        and "data.csv" in readme
        and "stats.md" in readme
        and bool(o.get("todos_used"))
    )


SPEC_MD = "# 仕様\n\n## 要件 1: ログイン\nメールとパスワードでログインできる。5 回失敗でロック。\n\n## 要件 2: 検索\n商品名の部分一致で検索できる。結果は 20 件ずつ。\n\n## 要件 3: 通知\n注文確定時にメールを送る。\n"


def _check_spec_tests(o: dict) -> bool:
    files = o.get("files") or {}
    tests = [p for p in files if p.startswith("/tests/test_")]
    index = _file(o, "/tests/INDEX.md")
    return len(tests) >= 3 and all(p.rsplit("/", 1)[-1] in index for p in tests) and bool(o.get("todos_used"))


# --- サブエージェント --------------------------------------------------------------


def _check_translator(o: dict) -> bool:
    f = _final(o)
    return "task" in _seq(o) and "深層エージェント" in f and "チェックポインタ" in f


DOCS = {
    "/docs/alpha.md": "# Alpha\n\n調査の結果、Alpha 方式はレイテンシが 30% 改善した。\n\n結論: Alpha を採用する。\n",
    "/docs/beta.md": "# Beta\n\nBeta 方式はメモリ使用量が 2 倍になった。\n\n結論: Beta は見送る。\n",
    "/docs/gamma.md": "# Gamma\n\nGamma 方式は追加検証が必要。\n\n結論: Gamma は来月再評価する。\n",
}


def _check_subagent_report(o: dict) -> bool:
    r = _file(o, "/report.md")
    return "task" in _seq(o) and "採用" in r and "見送" in r and "再評価" in r


# --- 長文コンテキスト ---------------------------------------------------------------


def _make_log() -> tuple[str, int, str]:
    """約 400 行のログ。ERROR 行数と最多コンポーネントを返す。"""
    comps = ["auth", "db", "cache", "api", "worker"]
    lines = []
    errors = {c: 0 for c in comps}
    for i in range(400):
        c = comps[i % 5]
        if i % 17 == 0 or (c == "db" and i % 9 == 0):
            level = "ERROR"
            errors[c] += 1
        elif i % 5 == 0:
            level = "WARN"
        else:
            level = "INFO"
        lines.append(f"2026-10-14T10:{i // 60:02d}:{i % 60:02d}Z {level} [{c}] request id={1000 + i} handled in {13 + (i * 7) % 90}ms")
    top = max(errors, key=errors.get)
    return "\n".join(lines) + "\n", sum(errors.values()), top


LOG_TEXT, LOG_ERRORS, LOG_TOP = _make_log()


def _check_log_errors(o: dict) -> bool:
    f = _final(o)
    return str(LOG_ERRORS) in f and LOG_TOP in f


def _make_transcript() -> str:
    filler = "".join(
        f"- {n} 分: {spk} が{topic}について説明した。質疑では特に異論はなかった。\n"
        for n, (spk, topic) in enumerate(
            [("鈴木", "予算の配分"), ("高橋", "採用計画"), ("伊藤", "オフィス移転"), ("渡辺", "セキュリティ研修"), ("山本", "四半期の売上"), ("中村", "顧客対応の改善")] * 12
        )
    )
    middle = "- 73 分: 議長が担当を割り当てた。田中さんは『社内 Wiki の移行』を担当する。小林さんは『ログ監査』を担当する。\n"
    return "# 全体会議 議事録(2026-10-10)\n\n" + filler[: len(filler) // 2] + middle + filler[len(filler) // 2 :]


TRANSCRIPT = _make_transcript()


def _check_transcript(o: dict) -> bool:
    return "Wiki" in _final(o)


# --- 構造化出力 -------------------------------------------------------------------

INVOICE_TEXT = "Invoice #4471 from Northwind Supplies. Items: 2x desk lamp, 1x monitor arm, 3x cable kit. Total due: $412.50. Payment due by 2026-11-05."


def _check_invoice(o: dict) -> bool:
    s = o.get("structured") or {}
    return (
        "Northwind" in str(s.get("vendor", ""))
        and abs(float(s.get("total", 0) or 0) - 412.5) < 0.01
        and s.get("due_date") == "2026-11-05"
        and len(s.get("items") or []) == 3
    )


EVENT_TEXT = "来週の 2026 年 10 月 21 日、品川オフィス 3F 会議室で『第 2 四半期レビュー』を開催します。参加者は佐藤、鈴木、田中の 3 名です。"


def _check_event(o: dict) -> bool:
    s = o.get("structured") or {}
    att = s.get("attendees") or []
    return "レビュー" in str(s.get("title", "")) and s.get("date") == "2026-10-21" and "品川" in str(s.get("location", "")) and len(att) == 3


# --- スキル ------------------------------------------------------------------------

SKILL_REPORT = """---
name: status-report
description: Write project status reports in the team's required format. Use whenever asked for a status report.
---

# Status report format

Write the report to `/reports/<YYYY-MM-DD>.md` with write_file. The file must have exactly these three level-2 headings, in this order:

## Summary
## Risks
## Next steps

Keep each section to 1-3 bullet points.
"""

SKILL_HAIKU = """---
name: haiku-reply
description: 俳句を求められたときの返答形式。俳句の依頼には必ずこのスキルを使う。
---

# 俳句の返答形式

- 5-7-5 の一句だけを返す。解説は付けない。
- 最後の行に必ず `-- 詠み人 AI` と書く。
"""


def _check_status_report(o: dict) -> bool:
    r = _file(o, "/reports/2026-10-14.md")
    return all(h in r for h in ("## Summary", "## Risks", "## Next steps")) and any("SKILL.md" in p for p in o.get("read_paths", []))


def _check_haiku(o: dict) -> bool:
    return "詠み人 AI" in _final(o) and any("SKILL.md" in p for p in o.get("read_paths", []))


# ---------------------------------------------------------------------------
# 一覧
# ---------------------------------------------------------------------------

TASKS: list[Task] = [
    # ツール選択
    Task("tool_weather_en", "tool_use", "en", "What's the weather in Osaka right now? Use the tool and answer in one sentence.", _check_weather_osaka, features=frozenset({"tools"}), max_calls=4),
    Task("tool_convert_ja", "tool_use", "ja", "1 万円をユーロに換算してください。ツールを使い、結果を 1 文で答えてください。", _check_convert_jpy_eur, features=frozenset({"tools"}), max_calls=4),
    Task(
        "tool_chain_ja",
        "tool_use",
        "ja",
        "東京の今の天気を調べて、雨なら明日(2026-10-15)に「傘を買う」というリマインダーを高優先度で作ってください。最後に何をしたか 1 文で報告してください。",
        _check_weather_then_reminder,
        features=frozenset({"tools"}),
        max_calls=5,
    ),
    Task("tool_none_en", "tool_use", "en", "What is the capital of France? Answer with one word. Do not use any tools.", _check_no_tool, features=frozenset({"tools"}), max_calls=3),
    Task(
        "tool_error_ja",
        "tool_use",
        "ja",
        "100 ドルをイギリスポンド(GBP)に換算してください。できない場合は、その理由を 1 文で説明してください。",
        _check_unsupported_currency,
        features=frozenset({"tools"}),
        max_calls=4,
    ),
    # 仮想 FS
    Task("fs_fix_bug_en", "filesystem", "en", "There is a bug in add() in /src/calc.py. Fix it with edit_file (do not rewrite the whole file) and briefly say what you changed.", _check_fix_bug, files={"/src/calc.py": CALC_PY}, max_calls=6),
    Task("fs_summary_ja", "filesystem", "ja", "/notes/ にある 3 つの議事録を読み、各メモの決定事項を 1 行ずつ箇条書きにして /summary.md に保存してください。", _check_summary_notes, files=NOTES, max_calls=8),
    Task("fs_grep_todos_ja", "filesystem", "ja", "/src/ 配下の全ファイルから TODO コメントを grep で探し、「ファイル名:行番号: 内容」の形式で 1 行ずつ /todos.md に列挙してください。", _check_todos_md, files=TODO_FILES, max_calls=6),
    Task("fs_create_en", "filesystem", "en", 'Create /app/config.json containing {"host": "localhost", "port": 8080} and /app/README.md with one line describing the config file.', _check_create_config, max_calls=5),
    # 計画
    Task(
        "plan_project_en",
        "planning",
        "en",
        "Build a tiny project step by step using the todo list to plan: (1) create /proj/data.csv with header name,score and rows Alice,90 / Bob,75 / Carol,82; (2) create /proj/stats.md stating the average score (2 decimals) and the top scorer; (3) create /proj/README.md listing the two files. Then confirm all files exist with ls.",
        _check_project_build,
        features=frozenset({"todos"}),
        max_calls=14,
    ),
    Task(
        "plan_spec_tests_ja",
        "planning",
        "ja",
        "todo リストで計画を立ててから作業してください。/docs/spec.md を読み、記載された 3 つの要件それぞれについて /tests/test_1.md、/tests/test_2.md、/tests/test_3.md にテスト観点を 3 つずつ書き、最後に /tests/INDEX.md に 3 ファイルの一覧を作ってください。",
        _check_spec_tests,
        files={"/docs/spec.md": SPEC_MD},
        features=frozenset({"todos"}),
        max_calls=14,
    ),
    # サブエージェント
    Task(
        "sub_translator_en",
        "subagent",
        "en",
        "Delegate to the translator subagent: translate the terms 'deep agent' and 'checkpointer' into Japanese using its glossary tool. Then report both translations to me.",
        _check_translator,
        features=frozenset({"subagent"}),
        max_calls=6,
    ),
    Task(
        "sub_report_ja",
        "subagent",
        "ja",
        "researcher サブエージェントに /docs/ の 3 つのファイルを読ませ、各ファイルの「結論」を 1 行ずつ報告させてください。その報告をそのまま /report.md に書いてください。",
        _check_subagent_report,
        files=DOCS,
        features=frozenset({"subagent"}),
        max_calls=8,
    ),
    # 長文
    Task("long_log_en", "long_context", "en", "Look at /logs/app.log. How many ERROR lines are there in total, and which component (the name in square brackets) has the most ERROR lines? Use grep rather than reading the whole file. Answer with the two facts.", _check_log_errors, files={"/logs/app.log": LOG_TEXT}, max_calls=8),
    Task("long_transcript_ja", "long_context", "ja", "/meeting.md の議事録を読んで、田中さんが担当になったタスクは何か答えてください。", _check_transcript, files={"/meeting.md": TRANSCRIPT}, max_calls=6),
    # 構造化出力
    Task("struct_invoice_en", "structured", "en", f"Extract the invoice data from this text:\n\n{INVOICE_TEXT}", _check_invoice, features=frozenset({"structured"}), max_calls=4, response_format=Invoice),
    Task("struct_event_ja", "structured", "ja", f"次の文からイベント情報を抽出してください。\n\n{EVENT_TEXT}", _check_event, features=frozenset({"structured"}), max_calls=4, response_format=Event),
    # スキル
    Task(
        "skill_report_en",
        "skills",
        "en",
        "Write a status report for project Apollo dated 2026-10-14. Summary: the API is finished. Risk: the DB migration is behind schedule. Next: start the UI work.",
        _check_status_report,
        files={"/skills/status-report/SKILL.md": SKILL_REPORT},
        features=frozenset({"skills"}),
        max_calls=6,
    ),
    Task("skill_haiku_ja", "skills", "ja", "秋の夜について俳句を一句お願いします。", _check_haiku, files={"/skills/haiku-reply/SKILL.md": SKILL_HAIKU}, features=frozenset({"skills"}), max_calls=5),
]

TASK_BY_ID = {t.id: t for t in TASKS}

# v2 追加タスク(難度高)。末尾で連結して TASK_BY_ID も更新する
from tasks_extra import EXTRA_TASKS  # noqa: E402

TASKS += EXTRA_TASKS
TASK_BY_ID.update({t.id: t for t in EXTRA_TASKS})
