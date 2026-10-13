"""設定(モデル・reasoning)とタスクの features から Deep Agent を組み立てる。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent, register_harness_profile
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware, wrap_tool_call
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from tasks import GLOSSARY, RATES, WEATHER, ConvertCurrencyArgs, CreateReminderArgs, GetWeatherArgs, LookupGlossaryArgs, Task


@dataclass(frozen=True)
class ModelConfig:
    model: str
    reasoning: bool | None = True
    num_ctx: int = 16384
    num_predict: int = 2048
    temperature: float = 0.3
    seed: int = 7
    coerce: bool = False  # ツール引数の寛容な変換(4.2 / 4.3 の対策)を入れるか

    @property
    def label(self) -> str:
        r = {True: "think", False: "nothink", None: "default"}[self.reasoning]
        return f"{self.model}-{r}" + ("-coerce" if self.coerce else "")


def make_llm(cfg: ModelConfig) -> ChatOllama:
    return ChatOllama(
        model=cfg.model,
        num_ctx=cfg.num_ctx,
        num_predict=cfg.num_predict,
        temperature=cfg.temperature,
        seed=cfg.seed,
        reasoning=cfg.reasoning,
        keep_alive="30m",
        validate_model_on_init=True,
        metadata={"ls_provider": "ollama", "ls_model_name": cfg.model},
    )


# カスタムツールの呼び出し履歴。evaluate は max_concurrency=1 で回すのでモジュール変数で足りる
custom_calls: list[dict] = []


@tool(args_schema=GetWeatherArgs)
def get_weather(city: str) -> str:
    """指定した都市の現在の天気を返す。"""
    custom_calls.append({"name": "get_weather", "args": {"city": city}})
    return WEATHER.get(city.lower(), f"unknown city: {city}")


@tool(args_schema=ConvertCurrencyArgs)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """通貨を換算する。対応通貨は JPY / USD / EUR のみ。"""
    custom_calls.append({"name": "convert_currency", "args": {"amount": amount, "from_currency": from_currency, "to_currency": to_currency}})
    rate = RATES.get((from_currency, to_currency))
    if rate is None:
        return f"error: unsupported pair {from_currency}->{to_currency}"
    return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"


@tool(args_schema=CreateReminderArgs)
def create_reminder(title: str, date: str, priority: str = "normal") -> str:
    """リマインダーを作成する。"""
    custom_calls.append({"name": "create_reminder", "args": {"title": title, "date": date, "priority": priority}})
    return f"reminder created: {title} on {date} ({priority})"


@tool(args_schema=LookupGlossaryArgs)
def lookup_glossary(term: str) -> str:
    """英語の用語を社内用語集で日本語に引く。"""
    custom_calls.append({"name": "lookup_glossary", "args": {"term": term}})
    return GLOSSARY.get(term.lower().strip(), f"not found: {term}")


FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]


@wrap_tool_call
def coerce_tool_args(request, handler):
    """ローカルモデルが崩しがちな引数を、スキーマ検証の前に直す。

    - str 型の引数に dict / list が来たら JSON 文字列にする(write_file の content に JSON オブジェクトを渡す癖)
    - 文字列の "None" / "null" は None にする(Optional な列挙値で頻出)
    """
    schema = getattr(request.tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None) if schema is not None else None
    if not fields:
        return handler(request)
    args = dict(request.tool_call["args"])
    changed = False
    for name, value in args.items():
        ann = str(fields[name].annotation) if name in fields else ""
        if isinstance(value, (dict, list)) and ann == "<class 'str'>":
            args[name] = json.dumps(value, ensure_ascii=False, indent=2)
            changed = True
        elif isinstance(value, str) and value.strip() in ("None", "null") and "None" in ann:
            args[name] = None
            changed = True
    if changed:
        request = request.override(tool_call={**request.tool_call, "args": args})
    return handler(request)

SYSTEM_PROMPT = "You are a careful assistant. Use tools when they help. Answer in the language of the request. Keep the final answer short."


def build_agent(cfg: ModelConfig, task: Task):
    """タスクが要求する機能だけを有効にした Deep Agent を返す。"""
    # general-purpose サブエージェント(task ツール)は、サブエージェントのタスク以外では出さない
    register_harness_profile("ollama", HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)))
    backend = StateBackend()
    middleware = [
        FilesystemMiddleware(backend=backend, tools=FS_TOOLS),
        ModelCallLimitMiddleware(run_limit=task.max_calls, exit_behavior="end"),
    ]
    if "todos" in task.features:
        middleware.append(TodoListMiddleware())
    if cfg.coerce:
        middleware.append(coerce_tool_args)

    tools = [get_weather, convert_currency, create_reminder] if "tools" in task.features else []

    subagents = None
    if "subagent" in task.features:
        subagents = [
            {
                "name": "translator",
                "description": "Translates English technical terms into Japanese using the team glossary tool.",
                "system_prompt": "Use lookup_glossary for every term you are asked about and return the translations as a short list.",
                "tools": [lookup_glossary],
            },
            {
                "name": "researcher",
                "description": "ファイルを読んで要点を報告する調査役。/docs/ などのファイルの読み取りと要約を任せる。",
                "system_prompt": "指示されたファイルを read_file で読み、求められた要点だけを短く報告してください。",
                "tools": [],
            },
        ]

    return create_deep_agent(
        model=make_llm(cfg),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        backend=backend,
        middleware=middleware,
        subagents=subagents,
        skills=["/skills/"] if "skills" in task.features else None,
        response_format=ToolStrategy(task.response_format) if task.response_format is not None else None,
        checkpointer=InMemorySaver(),
        name=f"lab-{task.category}",
    )
