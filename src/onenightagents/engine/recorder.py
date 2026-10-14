"""ゲームイベントの JSONL ログ(docs/architecture.md 6.3 / 10.2)。"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from onenightagents.game.models import GameConfig

Visibility = Literal["public", "secret"]


class Event(BaseModel):
    seq: int
    time: str
    event: str
    visibility: Visibility
    data: dict[str, Any]


class Recorder:
    """runs/<game_id>/ に config.json と events.jsonl を書く。on_event は観戦表示用。"""

    def __init__(self, run_dir: Path, on_event: Callable[[Event], None] | None = None) -> None:
        self.run_dir = run_dir
        self._on_event = on_event
        self._seq = 0
        run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = run_dir / "events.jsonl"

    def write_config(self, config: GameConfig) -> None:
        (self.run_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def emit(self, event: str, visibility: Visibility, **data: Any) -> None:
        self._seq += 1
        ev = Event(seq=self._seq, time=datetime.now(UTC).isoformat(), event=event, visibility=visibility, data=data)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(ev.model_dump_json() + "\n")
        if self._on_event is not None:
            self._on_event(ev)
