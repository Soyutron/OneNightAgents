"""bench / summarize と、ペルソナ読み込みのテスト(RandomPlayer)。"""

import argparse
from pathlib import Path

from onenightagents.analysis.summarize import (
    INFO_KINDS,
    classify_info,
    find_game_dirs,
    format_report,
    summarize_by_info,
    summarize_game,
    summarize_games,
)
from onenightagents.cli import bench, load_personas, make_config, replay
from onenightagents.players.prompts import build_system_prompt


def bench_args(run_dir: Path, games: int = 2, warmup_seed: int | None = None) -> argparse.Namespace:
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
        warmup_seed=warmup_seed,
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
    assert all(g["wolf_votes"] + g["seer_votes"] <= g["village_votes"] <= 4 for g in games)
    assert all(g["info"] in INFO_KINDS for g in games)
    by_info = summarize_by_info(games)
    assert sum(b["games"] for b in by_info.values()) == 2
    report = format_report(games)
    assert "| 1 |" in report and "ゲーム数 2" in report and "村側の票" in report and "| 情報 |" in report
    assert "seed=1 winner=" in capsys.readouterr().out


async def test_bench_warmup_game_is_excluded_from_summary(tmp_path: Path, capsys) -> None:
    await bench(bench_args(tmp_path, games=1, warmup_seed=7))
    assert [d.name for d in find_game_dirs([tmp_path / "b"])] == ["seed1"]
    assert (tmp_path / "b-warmup" / "seed7" / "events.jsonl").exists()
    assert "ゲーム数 1" in capsys.readouterr().out


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


def test_classify_info() -> None:
    roles = {"p1": "seer", "p2": "villager", "p3": "werewolf", "p4": "thief", "p5": "werewolf"}
    night = {"seer_target": "p3", "thief_target": None, "thief_new_role": None}
    assert classify_info(roles, night, roles) == "占い"
    # 怪盗が占い師と交換すると両方の占い師 CO が正直になり、占い結果は決め手にならない(017)
    swapped = {**roles, "p1": "thief", "p4": "seer"}
    assert classify_info(roles, {**night, "thief_target": "p1", "thief_new_role": "seer"}, swapped) == "怪盗"
    # 怪盗が占い先の人狼カードを奪うと占い結果は終了時に誤りで、怪盗も人狼側
    stolen = {**roles, "p3": "thief", "p4": "werewolf"}
    assert classify_info(roles, {**night, "thief_target": "p3", "thief_new_role": "werewolf"}, stolen) == "なし"
    # 白の占い結果は決め手にならない
    assert classify_info(roles, {**night, "seer_target": "p2"}, roles) == "なし"
    # 人狼が 2 枚とも中央
    no_wolf = {**roles, "p3": "villager", "p5": "villager"}
    assert classify_info(no_wolf, {**night, "seer_target": "p2"}, no_wolf) == "人狼なし"
    # 占い師が中央
    no_seer = {**roles, "p1": "villager"}
    none_night = {"seer_target": None, "thief_target": None, "thief_new_role": None}
    assert classify_info(no_seer, none_night, no_seer) == "なし"
    traded = {**no_seer, "p2": "thief", "p4": "villager"}
    assert classify_info(no_seer, {**none_night, "thief_target": "p2", "thief_new_role": "villager"}, traded) == "怪盗"
