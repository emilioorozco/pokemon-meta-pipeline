"""Ingest I/O: source discovery, idempotent bronze writes, and the CLI end to end on a
synthetic source tree (two replay batches + card CSV) with a pre-landed metadata cache."""

import json
import sys
from functools import partial
from pathlib import Path

import pyarrow.dataset as ds
import pytest

from pipeline import enrich, ingest
from tests.test_enrich import land_chunk
from tests.test_ingest import make_replay


def read_bronze(bronze_dir: Path, name: str) -> list[dict]:
    dataset = ds.dataset(bronze_dir / name, format="parquet", partitioning="hive")
    return dataset.to_table().to_pylist()


def make_source(root: Path, corpus: list[dict], corpus2: list[dict]) -> None:
    for batch, replays in (("corpus", corpus), ("corpus2", corpus2)):
        folder = root / "data" / "replays" / batch
        folder.mkdir(parents=True)
        for i, replay in enumerate(replays, start=1):
            (folder / f"episode-{i}-replay.json").write_text(json.dumps(replay))
    (root / "data" / "EN_Card_Data.csv").write_text("id,name\n")


def test_replay_files_walks_both_batches_and_honours_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_source(tmp_path, [make_replay()] * 3, [make_replay()] * 2)
    monkeypatch.setenv("SOURCE_DATA_DIR", str(tmp_path))

    files = ingest.replay_files(None)
    assert [b for b, _ in files] == ["corpus"] * 3 + ["corpus2"] * 2
    assert all(p.name.startswith("episode-") for _, p in files)
    assert [b for b, _ in ingest.replay_files(1)] == ["corpus", "corpus2"]


def test_write_bronze_is_idempotent_per_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)
    game_row, _, _ = ingest.to_rows(ingest.extract_game(make_replay(), "corpus"), meta=None)

    ingest.write_bronze("games", [game_row], ingest.GAMES_SCHEMA)
    ingest.write_bronze("games", [game_row], ingest.GAMES_SCHEMA)  # re-run replaces, not appends

    assert (tmp_path / "games" / "play_date=unknown").is_dir()
    rows = read_bronze(tmp_path, "games")
    assert len(rows) == 1
    assert rows[0]["episode_id"] == 12345


def test_main_refuses_to_run_without_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SOURCE_DATA_DIR", str(tmp_path))  # exists, but holds no corpus
    monkeypatch.setattr(sys, "argv", ["ingest", "--no-fetch"])
    with pytest.raises(SystemExit, match="missing source data"):
        ingest.main()


def test_main_end_to_end_joins_meta_and_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, bronze, cache = tmp_path / "source", tmp_path / "bronze", tmp_path / "episode_meta"
    good = make_replay()
    good["info"]["EpisodeId"] = "10"
    tie = make_replay(rewards=(0, 0))  # violates the corpus contract -> quarantine
    make_source(source, [good], [tie])
    land_chunk(cache, 10)

    monkeypatch.setenv("SOURCE_DATA_DIR", str(source))
    monkeypatch.setattr(ingest, "BRONZE_DIR", bronze)
    monkeypatch.setattr(enrich, "load_meta", partial(enrich.load_meta, cache_dir=cache))
    monkeypatch.setattr(sys, "argv", ["ingest", "--no-fetch"])

    assert ingest.main() == 1  # non-zero because something was quarantined
    out = capsys.readouterr().out
    assert "metadata joined for 1/1 games" in out
    assert "QUARANTINED 1 games" in out
    assert "episode-1-replay.json: ValueError" in out

    games = read_bronze(bronze, "games")
    seats = read_bronze(bronze, "game_seats")
    events = read_bronze(bronze, "game_events")
    assert [g["episode_id"] for g in games] == [10]
    assert games[0]["play_date"] == "2026-07-03"  # partition comes from the joined metadata
    assert (bronze / "games" / "play_date=2026-07-03").is_dir()
    assert len(seats) == 2
    assert [s["submission_id"] for s in seats] == [111, 222]
    assert [s["is_winner"] for s in seats] == [False, True]
    assert len(events) == 2
