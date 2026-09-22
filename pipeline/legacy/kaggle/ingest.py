"""Ingest: raw Kaggle replay JSON -> bronze Parquet lake.

Reads each episode replay, extracts the analytically useful core (~1% of the
bytes: ids, teams, outcome, both decklists, per-turn engine events), joins the
recovered episode metadata from pipeline.legacy.kaggle.enrich, and writes three bronze tables
partitioned by play_date:

  bronze/games        one row per game
  bronze/game_seats   one row per (game, seat)  <- the future fact grain
  bronze/game_events  one row per engine event (card moved, turn passed, ...)

Games failing basic quality checks (missing decks, no winner, not DONE) are
quarantined: reported and skipped, never silently written.

Usage:
  python -m pipeline.legacy.kaggle.ingest --sample 20   # first N games per batch
  python -m pipeline.legacy.kaggle.ingest               # full corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from pipeline.config import BRONZE_DIR
from pipeline.legacy.kaggle import enrich
from pipeline.legacy.kaggle.config import replay_batches, validate_source

DECK_SIZE = 60


# --------------------------------------------------------------------------- extract


def extract_game(replay: dict, batch: str) -> dict:
    """Pull the useful core out of one replay dict. Raises ValueError on
    anything that violates the corpus contract established during discovery."""
    steps = replay["steps"]
    statuses = replay["statuses"]
    if statuses != ["DONE", "DONE"]:
        raise ValueError(f"statuses not DONE: {statuses}")

    rewards = replay["rewards"]
    if sorted(rewards) != [-1, 1]:
        raise ValueError(f"no unambiguous winner: rewards={rewards}")

    # Both 60-card decklists live in the first visualize frame's action.
    decks = steps[0][0]["visualize"][0]["action"]
    if len(decks) != 2 or any(len(d) != DECK_SIZE for d in decks):
        raise ValueError(f"bad decklists: {[len(d) for d in decks]}")

    # the engine reports firstPlayer=-1 until the opening coin flip resolves
    first_player = None
    for step in steps:
        current = step[0]["observation"].get("current")
        if current is not None and current["firstPlayer"] in (0, 1):
            first_player = current["firstPlayer"]
            break
    if first_player is None:
        raise ValueError("firstPlayer never resolved to a seat")

    events = []
    for step_idx, step in enumerate(steps):
        for event_idx, log in enumerate(step[0]["observation"].get("logs") or []):
            events.append(
                {
                    "step_idx": step_idx,
                    "event_idx": event_idx,
                    "player_index": log.get("playerIndex"),
                    "event_type": log.get("type"),
                    "card_id": log.get("cardId"),
                    "serial": log.get("serial"),
                    "from_area": log.get("fromArea"),
                    "to_area": log.get("toArea"),
                }
            )

    return {
        "episode_id": int(replay["info"]["EpisodeId"]),
        "batch": batch,
        "n_steps": len(steps),
        "first_player": first_player,
        "team_names": replay["info"]["TeamNames"],
        "rewards": rewards,
        "decks": decks,
        "events": events,
    }


def to_rows(game: dict, meta: dict | None) -> tuple[dict, list[dict], list[dict]]:
    """Split one extracted game into rows for the three bronze tables."""
    meta = meta or {}
    played_at = meta.get("played_at")
    play_date = played_at.date().isoformat() if played_at else "unknown"

    game_row = {
        "episode_id": game["episode_id"],
        "batch": game["batch"],
        "n_steps": game["n_steps"],
        "first_player": game["first_player"],
        "played_at": played_at,
        "ended_at": meta.get("ended_at"),
        "play_date": play_date,
    }
    seat_rows = []
    for seat in (0, 1):
        seat_rows.append(
            {
                "episode_id": game["episode_id"],
                "seat": seat,
                "team_name": game["team_names"][seat],
                "team_id": (meta.get("team_id") or [None, None])[seat],
                "submission_id": (meta.get("submission_id") or [None, None])[seat],
                "rating_before": (meta.get("rating_before") or [None, None])[seat],
                "rating_after": (meta.get("rating_after") or [None, None])[seat],
                "reward": game["rewards"][seat],
                "is_winner": game["rewards"][seat] == 1,
                "went_first": game["first_player"] == seat,
                "deck": game["decks"][seat],
                "play_date": play_date,
            }
        )
    event_rows = [
        {"episode_id": game["episode_id"], "play_date": play_date, **e} for e in game["events"]
    ]
    return game_row, seat_rows, event_rows


# --------------------------------------------------------------------------- write


def _schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    # Typed wrapper: lets mypy check each (name, type) pair instead of joining the list to object.
    return pa.schema(list(fields))


GAMES_SCHEMA = _schema(
    ("episode_id", pa.int64()),
    ("batch", pa.string()),
    ("n_steps", pa.int16()),
    ("first_player", pa.int8()),
    ("played_at", pa.timestamp("us", tz="UTC")),
    ("ended_at", pa.timestamp("us", tz="UTC")),
    ("play_date", pa.string()),
)
SEATS_SCHEMA = _schema(
    ("episode_id", pa.int64()),
    ("seat", pa.int8()),
    ("team_name", pa.string()),
    ("team_id", pa.int64()),
    ("submission_id", pa.int64()),
    ("rating_before", pa.float64()),
    ("rating_after", pa.float64()),
    ("reward", pa.int8()),
    ("is_winner", pa.bool_()),
    ("went_first", pa.bool_()),
    ("deck", pa.list_(pa.int32(), DECK_SIZE)),
    ("play_date", pa.string()),
)
EVENTS_SCHEMA = _schema(
    ("episode_id", pa.int64()),
    ("play_date", pa.string()),
    ("step_idx", pa.int16()),
    ("event_idx", pa.int16()),
    ("player_index", pa.int8()),
    ("event_type", pa.int16()),
    ("card_id", pa.int32()),
    ("serial", pa.int32()),
    ("from_area", pa.int8()),
    ("to_area", pa.int8()),
)


def write_bronze(name: str, rows: list[dict], schema: pa.Schema) -> None:
    """Idempotent write: partitions being rewritten are replaced, not appended."""
    table = pa.Table.from_pylist(rows, schema=schema)
    ds.write_dataset(
        table,
        BRONZE_DIR / name,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("play_date", pa.string())]), flavor="hive"),
        existing_data_behavior="delete_matching",
    )


# --------------------------------------------------------------------------- main


def replay_files(sample_per_batch: int | None) -> list[tuple[str, Path]]:
    files = []
    for batch, folder in replay_batches().items():
        batch_files = sorted(folder.glob("episode-*-replay.json"))
        if sample_per_batch:
            batch_files = batch_files[:sample_per_batch]
        files += [(batch, p) for p in batch_files]
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="first N games per batch")
    parser.add_argument("--no-fetch", action="store_true", help="skip the metadata API call")
    args = parser.parse_args()

    if missing := validate_source():
        sys.exit(f"missing source data: {missing}")

    files = replay_files(args.sample)
    print(f"ingesting {len(files)} replays ({'sample' if args.sample else 'full corpus'})")

    games, quarantined = [], []
    for batch, path in files:
        try:
            games.append(extract_game(json.loads(path.read_text()), batch))
        except (ValueError, KeyError, IndexError) as e:
            quarantined.append((path.name, repr(e)))

    ids = [g["episode_id"] for g in games]
    if not args.no_fetch:
        print("fetching episode metadata (cached ids skipped)...")
        enrich.fetch_missing(ids)
    meta = enrich.load_meta()
    n_meta = sum(1 for i in ids if i in meta)

    game_rows, seat_rows, event_rows = [], [], []
    for g in games:
        gr, srs, ers = to_rows(g, meta.get(g["episode_id"]))
        game_rows.append(gr)
        seat_rows += srs
        event_rows += ers

    write_bronze("games", game_rows, GAMES_SCHEMA)
    write_bronze("game_seats", seat_rows, SEATS_SCHEMA)
    write_bronze("game_events", event_rows, EVENTS_SCHEMA)

    print(f"\nbronze written to {BRONZE_DIR}")
    print(f"  games:       {len(game_rows):>7} rows")
    print(f"  game_seats:  {len(seat_rows):>7} rows")
    print(f"  game_events: {len(event_rows):>7} rows")
    print(f"  metadata joined for {n_meta}/{len(games)} games")
    if quarantined:
        print(f"  QUARANTINED {len(quarantined)} games:")
        for name, err in quarantined[:10]:
            print(f"    {name}: {err}")
    return 1 if quarantined else 0


if __name__ == "__main__":
    sys.exit(main())
