"""グラフ実行時のランタイムコンテキスト。state に入れられないオブジェクトを渡す。"""

from __future__ import annotations

import random
from dataclasses import dataclass

from onenightagents.engine.recorder import Recorder
from onenightagents.players.base import Player


@dataclass
class GameContext:
    players: dict[str, Player]
    recorder: Recorder
    rng: random.Random  # 配役とフォールバックに使う。config.seed から作る
