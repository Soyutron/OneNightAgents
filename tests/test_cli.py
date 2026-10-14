"""bench / summarize と、ペルソナ読み込みのテスト(RandomPlayer)。"""

import argparse
from pathlib import Path

from onenightagents.analysis.summarize import find_game_dirs, format_report, summarize_game, summarize_games
from onenightagents.cli import bench, load_personas, make_config, replay
from onenightagents.players.prompts import build_system_prompt


def bench_args(run_dir: Path, games: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        players=4,
        seed=1,
        max_rounds=1,
        run_dir=str(run_dir),
        kind="random",
        model="none",
        reasoning=True,
        num_ctx=16384,
        num_predict=2048,
        temperature=0.7,
        timeout=60.0,
        personas=None,
        games=games,
        name="b",
    )


async def test_bench_runs_games_and_summarizes(tmp_path: Path, capsys) -> None:
    await bench(bench_args(tmp_path))
    dirs = find_game_dirs([tmp_path / "b"])
    assert [d.name for d in dirs] == ["seed1", "seed2"]
    games = [summarize_game(d) for d in dirs]
    assert [g["seed"] for g in games] == [1, 2]
    assert all(g["winner"] in ("village", "werewolf", "none") for g in games)
    # RandomPlayer は agent_step を出さないので判断数は 0
    assert all(g["decisions"] == 0 for g in games)
    s = summarize_games(games)
    assert s["games"] == 2 and s["village"] + s["werewolf"] + s["none"] == 2
    report = format_report(games)
    assert "| 1 |" in report and "ゲーム数 2" in report
    assert "seed=1 winner=" in capsys.readouterr().out


def test_load_personas_renders_in_file_order() -> None:
    personas = load_personas("configs/personas")
    assert personas is not None and len(personas) == 5
    assert personas[0].startswith("あなたは「演技派」タイプです。")  # actor.yaml が先頭
    config = make_config("g", 1, 5, 2, "deep", personas)
    assert sorted(p.persona for p in config.players) == sorted(personas)  # 全員に別のペルソナ
    assert config == make_config("g", 1, 5, 2, "deep", personas)  # seed で決定的
    assert [p.persona for p in make_config("g", 2, 5, 2, "deep", personas).players] != [p.persona for p in config.players]
    assert "タイプです" in build_system_prompt("太郎", "p1", config.players[0].persona)
    assert load_personas(None) is None


async def test_replay_prints_roles_and_votes(tmp_path: Path, capsys) -> None:
    await bench(bench_args(tmp_path, games=1))
    capsys.readouterr()
    replay(argparse.Namespace(paths=[str(tmp_path / "b")]))
    out = capsys.readouterr().out
    assert "中央:" in out and "投票 p1(" in out and "勝者:" in out
