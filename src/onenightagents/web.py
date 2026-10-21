"""ブラウザで人間が 1 席を担当する(`onenightagents web`)。端末入力の HumanPlayer を WebSocket につなぐ。

ブラウザは公開イベント・人間宛ての GM メッセージ・行動の要求を受け取り、入力を HumanPlayer と同じ 1 行の書式で返す。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from onenightagents.engine.recorder import Event
from onenightagents.game.models import NightRequest, SpeakRequest, VoteRequest
from onenightagents.players.base import NightAction, SpeechAction, VoteAction
from onenightagents.players.human import HumanPlayer

INDEX_HTML = Path(__file__).with_name("web.html")

A = TypeVar("A")


class WebUI:
    """ブラウザとの接続。送った内容は履歴に残し、リロードしたら最初から送り直して画面を復元する。"""

    def __init__(self, human_id: str) -> None:
        self.human_id = human_id
        self.connected = asyncio.Event()
        self._history: list[dict[str, Any]] = []
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._notes: dict[str, str] = {}
        self.app = Starlette(
            routes=[
                Route("/", lambda _: HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))),
                WebSocketRoute("/ws", self._ws),
            ]
        )

    def send(self, message: dict[str, Any]) -> None:
        self._history.append(message)
        for q in self._queues:
            q.put_nowait(message)

    async def _ws(self, ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for m in self._history:
            queue.put_nowait(m)
        self._queues.add(queue)
        self.connected.set()

        async def pump() -> None:
            while True:
                await ws.send_json(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            while True:
                self._inbox.put_nowait((await ws.receive_json())["line"])
        except WebSocketDisconnect:
            pass
        finally:
            self._queues.discard(queue)
            sender.cancel()

    # HumanPlayer の Reader
    async def read(self) -> str:
        return await self._inbox.get()

    def discard(self) -> None:
        while not self._inbox.empty():
            self._inbox.get_nowait()

    def on_event(self, ev: Event) -> None:
        """ゲームのイベントのうち、人間に見せてよいものだけ送る。AI のメモは種明かしまで溜めておく。"""
        d = ev.data
        if ev.event == "notes_updated":
            self._notes[d["player"]] = d["notes"]
        if ev.visibility == "secret":
            return
        match ev.event:
            case "game_started":
                players = [{"id": p["id"], "name": p["display_name"]} for p in d["config"]["players"]]
                self.send({"type": "start", "players": players, "you": self.human_id})
            case "utterance" | "turn_skipped":
                self.send({"type": ev.event, **d})
            case "game_ended":
                self.send({"type": "reveal", **d, "notes": self._notes})


class WebHumanPlayer(HumanPlayer):
    """要求の種類と候補をブラウザに送ってから、HumanPlayer と同じ処理で 1 行を待つ。"""

    def __init__(self, id: str, ui: WebUI, timeout_sec: float = 300.0) -> None:
        super().__init__(id, reader=ui, write=self._write_text, timeout_sec=timeout_sec)
        self._ui = ui

    def _write_text(self, text: str) -> None:
        # 端末向けの入力プロンプトは不要
        if text != "> ":
            self._ui.send({"type": "gm", "text": text.strip()})

    async def _requested(self, request: dict[str, Any], action: Awaitable[A]) -> A:
        self._ui.send({"type": "request", **request})
        try:
            return await action
        finally:
            # 時間切れで打ち切られたときも入力欄を閉じる
            self._ui.send({"type": "done"})

    async def act_at_night(self, messages: list[BaseMessage], request: NightRequest) -> NightAction:
        req = {"kind": "night", "role": request.role, "candidates": request.candidates}
        return await self._requested(req, super().act_at_night(messages, request))

    async def speak(self, messages: list[BaseMessage], request: SpeakRequest) -> SpeechAction:
        # 新しい発言は公開イベントで表示済みなので、GM のメッセージ(差分 + 端末向けの入力案内)は見せない
        # 自動のパスは公開イベントの「パスしました」で分かるので、端末向けの「(自動でパス)」は出さない
        if self._auto_pass and request.turn.pass_allowed:
            return SpeechAction(is_pass=True)
        turn = request.turn
        req = {
            "kind": "speak",
            "round": turn.round,
            "pass_allowed": turn.pass_allowed,
            "reply_to_turn": turn.reply_to_turn,
            "others": request.others,
        }
        return await self._requested(req, super().speak([], request))

    async def vote(self, messages: list[BaseMessage], request: VoteRequest) -> VoteAction:
        return await self._requested({"kind": "vote", "candidates": request.candidates}, super().vote([], request))
