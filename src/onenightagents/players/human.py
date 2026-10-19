"""HumanPlayer: 人間が端末で 1 席を担当する(docs/implementation-plan.md Step 8)。

GM のメッセージをそのまま表示し、1 行の入力を行動にする。制限時間と時間切れのフォールバックは GM 側(_request_action)。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import Protocol

from langchain_core.messages import BaseMessage

from onenightagents.game.messages import ROLE_NAMES
from onenightagents.game.models import NightRequest, Role, SpeakRequest, VoteRequest
from onenightagents.players.base import NightAction, SpeechAction, VoteAction

ROLE_BY_NAME = {name: role for role, name in ROLE_NAMES.items()} | {r.value: r for r in Role}
REMIND_BEFORE_SEC = 60


class Reader(Protocol):
    async def read(self) -> str: ...

    def discard(self) -> None:
        """読まれずに残っている行を捨てる(時間切れの後に届いた古い答えを次の要求に使わない)。"""
        ...


class StdinReader:
    """標準入力を 1 本のスレッドで読んで行をキューに入れる。先に打った行(AI の発言を待つ間の入力)は残す。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._thread: threading.Thread | None = None

    def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        for line in sys.stdin:
            loop.call_soon_threadsafe(self._queue.put_nowait, line.rstrip("\n"))

    async def read(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self._pump, args=(asyncio.get_running_loop(),), daemon=True)
            self._thread.start()
        return await self._queue.get()

    def discard(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()


def parse_speech(line: str, others: list[str]) -> tuple[SpeechAction | None, str | None]:
    """`[@宛先] [CO:役職] 本文` を解析する。戻り値は (行動, エラー文)。"""
    addressed_to: str | None = None
    claimed: Role | None = None
    tokens = line.split()
    while tokens:
        t = tokens[0]
        if t.startswith("@"):
            if t[1:] not in others:
                return None, f"宛先は {', '.join(others)} から選んでください"
            addressed_to = t[1:]
        elif t[:3].upper() == "CO:":
            claimed = ROLE_BY_NAME.get(t[3:])
            if claimed is None:
                return None, "主張する役職は 村人 / 人狼 / 占い師 / 怪盗 のいずれかです"
        else:
            break
        tokens.pop(0)
    content = " ".join(tokens)
    if not content:
        return None, "発言を入力してください"
    return SpeechAction(content=content, addressed_to=addressed_to, claimed_role=claimed), None


class HumanPlayer:
    def __init__(
        self,
        id: str,
        reader: Reader | None = None,
        write: Callable[[str], None] = print,
        timeout_sec: float = 300.0,
    ) -> None:
        self.id = id
        self._reader = reader or StdinReader()
        self._write = write
        self._timeout_sec = timeout_sec
        self._auto_pass = False  # vote と入力した後は自由ラウンドをパスし続ける
        self._timed_out = False  # 前の要求が時間切れなら、遅れて届いた答えを次の要求の前に捨てる

    def _show(self, messages: list[BaseMessage]) -> None:
        for m in messages:
            self._write(f"\n{m.content}")

    async def _ask(self) -> str:
        """1 行読む。制限時間の 60 秒前に 1 回催促する。"""
        if self._timed_out:
            self._reader.discard()
            self._timed_out = False
        self._write("> ")
        remind = None
        if self._timeout_sec > REMIND_BEFORE_SEC:
            remind = asyncio.get_running_loop().call_later(
                self._timeout_sec - REMIND_BEFORE_SEC,
                self._write,
                f"(残り {REMIND_BEFORE_SEC} 秒です。時間切れになると自動で行動します)",
            )
        try:
            return (await self._reader.read()).strip()
        except asyncio.CancelledError:
            # GM の制限時間(asyncio.wait_for)による打ち切り
            self._timed_out = True
            raise
        finally:
            if remind is not None:
                remind.cancel()

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction:
        self._show(messages)
        while True:
            answer = await self._ask()
            if request.role is Role.THIEF and answer == "none":
                return NightAction(target=None)
            if answer in request.candidates:
                return NightAction(target=answer)
            extra = " か none" if request.role is Role.THIEF else ""
            self._write(f"候補の中から入力してください: {', '.join(request.candidates)}{extra}")

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction:
        self._show(messages)
        pass_allowed = request.turn.pass_allowed
        if self._auto_pass and pass_allowed:
            self._write("(自動でパス)")
            return SpeechAction(is_pass=True)
        while True:
            line = await self._ask()
            if line in ("pass", "vote"):
                self._auto_pass = self._auto_pass or line == "vote"
                if pass_allowed:
                    return SpeechAction(is_pass=True)
                self._write("このターンはパスできません。発言を入力してください")
                continue
            action, error = parse_speech(line, request.others)
            if action is None:
                self._write(error or "")
                continue
            return action

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction:
        self._show(messages)
        while True:
            target, _, reason = (await self._ask()).partition(" ")
            if target in request.candidates:
                return VoteAction(target=target, reason=reason.strip())
            self._write(f"候補の中から入力してください: {', '.join(request.candidates)}")

    async def update_notes(self, messages: list[BaseMessage]) -> str | None:
        # 夜明けの結果を見せるだけ。メモは人間が自分で持つ
        self._show(messages)
        return None
