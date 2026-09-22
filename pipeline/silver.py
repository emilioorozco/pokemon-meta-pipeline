"""Silver stage: bronze Parquet in, four typed tables out, on PySpark.

Bronze is one nested row per game. Silver is the grain change: a game row, a
seat row, a turn row and a card row, each with plain columns a query can filter
on without walking a struct. Everything here is a projection of bronze, so the
stage is rerunnable from scratch and holds no state of its own.

The four tables, all partitioned by `play_date`:

- `games`: one row per game, the summary's identity, timing and outcome fields
  lifted out of the struct, with `winner_seat`, `went_first_seat`,
  `coin_toss_winner_seat` and `first_player` resolved from handles to seats.
- `game_sides`: two rows per game, one per seat, carrying that seat's player
  token, archetype, per-side counters and decklist facts, plus the uploader's
  own deck record on the uploader seat. This is the grain gold's fact table is
  built on.
- `turns`: one row per turn segment, with the action counters that need no
  `fields_json` parsing.
- `cards_seen`: one row per (game, seat, card), left joined to the card catalog.

Why Spark for a corpus this small: the explodes are where the data grows. A
game is two seat rows but forty card rows and twenty turn rows, and the same
code runs unchanged on a laptop and on a cluster.

What changes on a cluster: nothing in this file. The master URL comes from
`--master` or `PRA_SPARK_MASTER`, the directories become `s3a://` paths passed
to the same arguments, and `spark.sql.shuffle.partitions` (8 here, right for a
laptop and far too low for a real cluster) is raised. The transforms, the
schemas and the reconciliation are the same.

Four shapes of the real data drive decisions here, and each is worth stating
because the obvious implementation gets them wrong:

- A deck name is not an archetype. `summary.deckName` is whatever the player
  typed into the game client, and on the real corpus it is the client's default
  ("New Deck 54") on about half the games and a joke or a shorthand on most of
  the rest. So it is carried as `game_sides.deck_name`, a player-level fact on
  the uploader seat, and never coalesced into `archetype_name_raw`. An uploaded
  game gets an uploader archetype only when the application derived one or the
  user set one, which today is rare, so most uploader seats have none.

- `observedCards` never carries a `cardId`. Both the stock and the debug export
  derive it from the battle log, which prints card names, while `cardId` only
  appears in a decklist read from the debug preamble. So `cards_seen.card_id`
  is the card's identity as silver can resolve it: the reference's `cardId`
  when it has one, else its `baseCardId`, else its lowercased name. That key is
  never null, which is what makes (game, seat, card) a real grain, and the
  catalog join finds a row whenever the key is a client card id the catalog
  knows.
- A decklist reference is the mirror image: `cardId` and `baseCardId` set,
  `name` absent. Observed cards and decklist cards therefore live in disjoint
  key spaces, and `in_decklist` would be false everywhere if it compared them
  directly. The catalog is the bridge: a decklist entry contributes both its
  card id and, when the catalog knows that id, the lowercased catalog name. Run
  with no catalog and there is no bridge, so `in_decklist` falls back to
  id-to-id matching and reads false for every row. The column is only worth
  reading once `scripts/fetch_catalog.py` has run.
- A player token is not a member. `member_tokens` is the set of tokens that
  appear on an uploader seat anywhere in bronze; every other token belongs to a
  stranger who never consented to anything and is written as NULL, with
  `is_member` recording which is which (docs/data-handling.md).

Why the output schemas are pinned in `SILVER_SCHEMAS` rather than inferred, the
same reason bronze pins its own: a batch that happens to contain no manual game
would type a column differently from a batch that does, and a reader spanning
both partitions would see two incompatible schemas.
"""

import argparse
import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from pipeline.config import BRONZE_DIR, CATALOG_PATH, SILVER_DIR
from pipeline.contract import ActionKind, SideStats
from pipeline.observability import configure_logging, emit_summary, stage_run

logger = logging.getLogger(__name__)

STAGE: Final = "silver"
APP_NAME: Final = "pra-silver"
MASTER_VAR: Final = "PRA_SPARK_MASTER"
DEFAULT_MASTER: Final = "local[*]"
SHUFFLE_PARTITIONS: Final = "8"
PART_GLOB: Final = "*/part-0.parquet"
TABLES: Final = ("games", "game_sides", "turns", "cards_seen")
SIDES_PER_GAME: Final = 2

# ActionKind -> the `turns` counter it feeds. A kind that is not listed counts
# toward `n_entries` and nothing else. The draw and trainer buckets follow the
# producer's own `deriveStats` definitions (docs/schema.md section 5) so a turn
# counter summed over a game matches the side counter it came from:
# `cardsDrawn` is opening_hand, draw, draw_named and mulligan_draw, and
# `cardsPlayed` is play_card and play_stadium. `drawn_cards` is deliberately
# absent: it is the sub-entry that lists what a `draw` drew, so counting it
# would count the same draw twice.
ACTION_BUCKETS: Final[dict[ActionKind, str]] = {
    ActionKind.OPENING_HAND: "n_draw",
    ActionKind.DRAW: "n_draw",
    ActionKind.DRAW_NAMED: "n_draw",
    ActionKind.MULLIGAN_DRAW: "n_draw",
    ActionKind.ATTACH: "n_attach",
    ActionKind.ATTACK: "n_attack",
    ActionKind.PLAY_POKEMON: "n_play_pokemon",
    ActionKind.PLAY_CARD: "n_play_trainer",
    ActionKind.PLAY_STADIUM: "n_play_trainer",
    ActionKind.EVOLVE: "n_evolve",
    ActionKind.RETREAT: "n_retreat",
    ActionKind.KNOCKOUT: "n_knockout",
    ActionKind.PRIZE: "n_prize_taken",
}
TURN_COUNTERS: Final = (
    "n_draw",
    "n_attach",
    "n_attack",
    "n_play_pokemon",
    "n_play_trainer",
    "n_evolve",
    "n_retreat",
    "n_knockout",
    "n_prize_taken",
)

# The result the other seat gets when this one gets the key.
MIRRORED_RESULT: Final = {"win": "loss", "loss": "win", "tie": "tie", "unknown": "unknown"}

_STR = T.StringType()
_INT = T.IntegerType()
_BOOL = T.BooleanType()
_DATE = T.DateType()
_TS = T.TimestampType()
_STRS = T.ArrayType(T.StringType())


class ReconciliationError(RuntimeError):
    """A silver table does not agree with bronze; `failures` names every check that broke."""

    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures))
        self.failures = list(failures)


@dataclass
class SilverSummary:
    """What one run produced: rows per table, the reconciliation lines, and how long it took.

    `checks` holds one readable line per reconciliation check, filled in by
    `reconcile`. They are carried on the summary rather than printed where they
    are computed because `reconcile` is a library function and stdout belongs to
    the command line; the command prints the block it is handed.
    """

    games_in: int = 0
    rows: dict[str, int] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __str__(self) -> str:
        counts = " ".join(f"{table}={self.rows.get(table, 0)}" for table in TABLES)
        return "\n".join(
            [
                f"games_in: {self.games_in}",
                f"rows: {counts}",
                *self.checks,
                f"duration_s: {self.duration_s:.2f}",
            ]
        )


def _stats_columns() -> list[T.StructField]:
    """One `stats_*` column per `SideStats` field, typed from the contract model.

    Taken from the model rather than listed here so a counter added upstream
    becomes a silver column instead of being silently dropped.
    """
    fields: list[T.StructField] = []
    for name, info in SideStats.model_fields.items():
        spark_type = _STRS if info.annotation == list[str] else _INT
        fields.append(T.StructField(f"stats_{name}", spark_type, True))
    return fields


SILVER_SCHEMAS: Final[dict[str, T.StructType]] = {
    "games": T.StructType(
        [
            T.StructField("game_id", _STR, True),
            T.StructField("user_id", _STR, True),
            T.StructField("play_date", _DATE, True),
            T.StructField("played_at", _TS, True),
            T.StructField("played_at_source", _STR, True),
            T.StructField("export_variant", _STR, True),
            T.StructField("upload_source", _STR, True),
            T.StructField("turn_count", _INT, True),
            T.StructField("end_reason", _STR, True),
            T.StructField("result", _STR, True),
            T.StructField("winner_seat", _INT, True),
            T.StructField("went_first_seat", _INT, True),
            T.StructField("coin_toss_winner_seat", _INT, True),
            T.StructField("first_player", _INT, True),
            T.StructField("excluded_from_stats", _BOOL, True),
            T.StructField("has_full_decklists", _BOOL, True),
            T.StructField("my_side", _INT, True),
            T.StructField("season_id", _STR, True),
            T.StructField("season_name", _STR, True),
            T.StructField("parser_version", _INT, True),
            T.StructField("unparsed_count", _INT, True),
            T.StructField("contract_version", _INT, True),
            T.StructField("source_key", _STR, True),
            T.StructField("ingested_at", _TS, True),
        ]
    ),
    "game_sides": T.StructType(
        [
            T.StructField("game_id", _STR, True),
            T.StructField("play_date", _DATE, True),
            T.StructField("seat", _INT, True),
            T.StructField("is_uploader", _BOOL, True),
            T.StructField("player_token", _STR, True),
            T.StructField("is_member", _BOOL, True),
            T.StructField("archetype_id", _STR, True),
            T.StructField("archetype_name", _STR, True),
            T.StructField("archetype_name_raw", _STR, True),
            T.StructField("archetype_source", _STR, True),
            T.StructField("result_for_seat", _STR, True),
            T.StructField("went_first", _BOOL, True),
            *_stats_columns(),
            T.StructField("decklist_source", _STR, True),
            T.StructField("decklist_complete", _BOOL, True),
            T.StructField("decklist_card_count", _INT, True),
            T.StructField("deck_name", _STR, True),
            T.StructField("deck_id", _STR, True),
        ]
    ),
    "turns": T.StructType(
        [
            T.StructField("game_id", _STR, True),
            T.StructField("play_date", _DATE, True),
            T.StructField("turn_number", _INT, True),
            T.StructField("seat", _INT, True),
            T.StructField("n_entries", _INT, True),
            *[T.StructField(name, _INT, True) for name in TURN_COUNTERS],
            T.StructField("concession", _BOOL, True),
        ]
    ),
    "cards_seen": T.StructType(
        [
            T.StructField("game_id", _STR, True),
            T.StructField("play_date", _DATE, True),
            T.StructField("seat", _INT, True),
            T.StructField("card_id", _STR, True),
            T.StructField("base_card_id", _STR, True),
            T.StructField("card_name", _STR, True),
            T.StructField("set_code", _STR, True),
            T.StructField("number", _STR, True),
            T.StructField("count_seen", _INT, True),
            T.StructField("in_decklist", _BOOL, True),
            T.StructField("catalog_name", _STR, True),
            T.StructField("catalog_set", _STR, True),
            T.StructField("catalog_type", _STR, True),
            T.StructField("catalog_hp", _INT, True),
            T.StructField("catalog_reg", _STR, True),
        ]
    ),
}

CATALOG_SCHEMA: Final = T.StructType(
    [
        T.StructField("card_id", _STR, True),
        T.StructField("catalog_name", _STR, True),
        T.StructField("catalog_set", _STR, True),
        T.StructField("catalog_number", _STR, True),
        T.StructField("catalog_type", _STR, True),
        T.StructField("catalog_hp", _INT, True),
        T.StructField("catalog_reg", _STR, True),
    ]
)


def build_session(master: str) -> SparkSession:
    """A local or cluster session with the settings every silver run needs.

    UTC everywhere so a timestamp means the same thing in the lake and in a
    query, a small shuffle width because a laptop run with the default 200 wastes
    more time on empty tasks than on work, and dynamic partition overwrite so a
    rerun over one day replaces that day instead of the whole table.
    """
    builder = (
        SparkSession.builder.master(master)
        .appName(APP_NAME)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    )
    if master.startswith("local"):
        # A laptop whose hostname does not resolve (no network, a container)
        # cannot bind the driver to it. Loopback is always right in local mode
        # and always wrong on a cluster, so it is set only here.
        builder = builder.config("spark.driver.host", "127.0.0.1").config(
            "spark.driver.bindAddress", "127.0.0.1"
        )
    return builder.getOrCreate()


def read_bronze(spark: SparkSession, bronze_dir: Path) -> DataFrame:
    """Every bronze partition as one DataFrame, `play_date` typed from the directory.

    `basePath` is what makes Spark read `play_date=YYYY-MM-DD` as a partition
    column rather than as part of an opaque path; it arrives as a DATE and
    shadows the string copy of itself that bronze also writes into the file.
    Spark logs that shadowing once as a COLUMN_ALREADY_EXISTS warning and then
    uses the partition value, which is the one this stage wants: it is typed,
    and it is the value a partition filter can skip whole files on.
    """
    return spark.read.option("basePath", str(bronze_dir)).parquet(str(bronze_dir / PART_GLOB))


def load_catalog(spark: SparkSession, catalog_path: Path | None) -> DataFrame:
    """The card catalog as a DataFrame, or an empty one when the file is absent.

    The catalog is a 25k-entry export of the client's card database and is not
    committed here (`scripts/fetch_catalog.py` downloads it). A run without it
    is a real run: every `catalog_*` column is null and `in_decklist` falls back
    to matching card ids against card ids.
    """
    if catalog_path is None or not catalog_path.is_file():
        logger.warning(
            "no card catalog at %s: catalog columns will be null", catalog_path or "<unset>"
        )
        return spark.createDataFrame([], CATALOG_SCHEMA)
    with catalog_path.open(encoding="utf-8") as handle:
        entries: dict[str, dict[str, Any]] = json.load(handle)
    rows = [
        (
            card_id,
            entry.get("name"),
            entry.get("set"),
            entry.get("number"),
            entry.get("type"),
            _as_int(entry.get("hp")),
            entry.get("reg"),
        )
        for card_id, entry in entries.items()
    ]
    logger.info("card catalog: %d entries from %s", len(rows), catalog_path)
    return spark.createDataFrame(rows, CATALOG_SCHEMA)


def build_games(bronze: DataFrame) -> DataFrame:
    """One row per game: the summary's scalars, with handles resolved to seats."""
    summary = F.col("summary")
    my_side = summary["my_side"].cast(_INT)
    winner_seat = F.when(
        summary["winner"].isNotNull(),
        F.array_position(summary["players"], summary["winner"]).cast(_INT) - 1,
    )
    return _conform(
        bronze.select(
            F.col("game_id"),
            F.col("user_id"),
            F.col("play_date"),
            F.col("played_at"),
            summary["played_at_source"].alias("played_at_source"),
            summary["export_variant"].alias("export_variant"),
            summary["upload_source"].alias("upload_source"),
            summary["turn_count"].alias("turn_count"),
            summary["end_reason"].alias("end_reason"),
            summary["result"].alias("result"),
            F.when(winner_seat >= 0, winner_seat).alias("winner_seat"),
            _flag_seat(summary["went_first"], my_side).alias("went_first_seat"),
            _flag_seat(summary["won_coin_toss"], my_side).alias("coin_toss_winner_seat"),
            _first_turn_seat(summary).alias("first_player"),
            F.coalesce(summary["excluded_from_stats"], F.lit(False)).alias("excluded_from_stats"),
            summary["has_full_decklists"].alias("has_full_decklists"),
            my_side.alias("my_side"),
            summary["season_id"].alias("season_id"),
            summary["season_name"].alias("season_name"),
            summary["parser_version"].alias("parser_version"),
            summary["unparsed_count"].alias("unparsed_count"),
            F.col("contract_version"),
            F.col("source_key"),
            F.col("ingested_at"),
        ),
        "games",
    )


def archetype_aliases(bronze: DataFrame) -> DataFrame:
    """One canonical name per `archetype_id`: the name the newest ingested game gives it.

    An archetype is renamed upstream by editing the shared row, and the rename
    reaches every game the next time it is written, so older bronze rows keep
    the old label forever. The newest row therefore wins, and the older games
    inherit its name rather than splitting the archetype in two. Ties inside a
    run (every row of a backfill shares one `ingested_at`) break on the play
    date and then the game id, so the map is the same on every rerun.
    """
    summary = F.col("summary")
    labelled = [
        bronze.select(
            summary[f"{role}_archetype_id"].alias("archetype_id"),
            summary[f"{role}_archetype"].alias("archetype_name"),
            F.col("ingested_at"),
            F.col("played_at"),
            F.col("game_id"),
        )
        for role in ("my", "opponent")
    ]
    newest = Window.partitionBy("archetype_id").orderBy(
        F.col("ingested_at").desc(), F.col("played_at").desc(), F.col("game_id").desc()
    )
    return (
        labelled[0]
        .unionByName(labelled[1])
        .where(F.col("archetype_id").isNotNull() & F.col("archetype_name").isNotNull())
        .withColumn("rank", F.row_number().over(newest))
        .where(F.col("rank") == 1)
        .select("archetype_id", F.col("archetype_name").alias("canonical_name"))
    )


def member_tokens(bronze: DataFrame) -> DataFrame:
    """The distinct player tokens that hold an uploader seat somewhere in bronze.

    Every other token in the lake belongs to somebody who was matched against a
    member and never uploaded anything themselves, so they never saw the notice
    and never consented. One column, `player_token`, so it joins.
    """
    summary = F.col("summary")
    return (
        bronze.where(summary["my_side"].isNotNull())
        .select(F.get(summary["players"], summary["my_side"].cast(_INT)).alias("player_token"))
        .where(F.col("player_token").isNotNull())
        .distinct()
    )


def build_game_sides(bronze: DataFrame, aliases: DataFrame, members: DataFrame) -> DataFrame:
    """Two rows per game, one per seat, with the token, archetype, counters and decklist."""
    summary = F.col("summary")
    seat = F.col("seat")
    my_side = summary["my_side"].cast(_INT)
    is_uploader = F.coalesce(seat == my_side, F.lit(False))
    token = F.get(summary["players"], seat)
    manual = summary["export_variant"] == "manual"

    exploded = bronze.select(
        "game_id",
        "play_date",
        "summary",
        "stats_by_player",
        "my_decklist",
        "opponent_decklist",
        F.explode(F.array(F.lit(0), F.lit(1))).alias("seat"),
    )

    # The counters are keyed by handle, so the seat's own token is the lookup
    # key. Manual games carry no `statsByPlayer` at all and land all nulls.
    stats = F.get(F.filter(F.col("stats_by_player"), lambda row: row["handle"] == token), 0)[
        "stats"
    ]
    decklist = (
        F.when(my_side.isNull(), F.lit(None))
        .when(is_uploader, F.col("my_decklist"))
        .otherwise(F.col("opponent_decklist"))
    )

    sides = exploded.select(
        F.col("game_id"),
        F.col("play_date"),
        seat.alias("seat"),
        is_uploader.alias("is_uploader"),
        token.alias("player_token"),
        _archetype_id(summary, is_uploader).alias("archetype_id"),
        _archetype_name_raw(summary, is_uploader, manual, token).alias("archetype_name_raw"),
        _archetype_source(summary, is_uploader, manual).alias("archetype_source"),
        _result_for_seat(summary, my_side, is_uploader).alias("result_for_seat"),
        _went_first(summary, my_side, seat).alias("went_first"),
        *[
            stats[name].alias(f"stats_{name}")
            for name in SideStats.model_fields  # declaration order, same as the schema
        ],
        decklist["source"].alias("decklist_source"),
        decklist["complete"].alias("decklist_complete"),
        decklist["card_count"].alias("decklist_card_count"),
        _deck_name(summary, is_uploader).alias("deck_name"),
        F.when(is_uploader, summary["deck_id"]).alias("deck_id"),
    )

    resolved = sides.join(F.broadcast(aliases), on="archetype_id", how="left").join(
        F.broadcast(members.withColumn("is_member", F.lit(True))), on="player_token", how="left"
    )
    return _conform(
        resolved.withColumn("is_member", F.coalesce(F.col("is_member"), F.lit(False)))
        .withColumn(
            "archetype_name",
            F.coalesce(F.col("canonical_name"), F.col("archetype_name_raw")),
        )
        # The stranger rule, applied last so the joins above still see the token.
        .withColumn("player_token", F.when(F.col("is_member"), F.col("player_token"))),
        "game_sides",
    )


def build_turns(bronze: DataFrame) -> DataFrame:
    """One row per turn segment, counted by action kind.

    Every action line in the segment is counted, the top-level entries and the
    sub-entries under them alike, because the draw a Professor's Research causes
    is printed as a sub-entry of the line that played it. `n_entries` is that
    whole count, so a kind with no bucket still shows up in it.
    """
    segment = F.col("segment")
    kinds = F.flatten(
        F.array(
            F.transform(segment["entries"], lambda entry: entry["kind"]),
            F.flatten(
                F.transform(
                    segment["entries"],
                    lambda entry: F.transform(entry["subs"], lambda sub: sub["kind"]),
                )
            ),
        )
    )
    buckets = {
        bucket: [kind.value for kind, mapped in ACTION_BUCKETS.items() if mapped == bucket]
        for bucket in TURN_COUNTERS
    }
    # `kinds` is materialized once and every counter below reads that column.
    turns = (
        bronze.select("game_id", "play_date", "summary", F.explode("segments").alias("segment"))
        .where(segment["kind"] == "turn")
        .withColumn("kinds", kinds)
    )
    return _conform(
        turns.select(
            F.col("game_id"),
            F.col("play_date"),
            segment["turn_number"].alias("turn_number"),
            _seat_of(F.col("summary")["players"], segment["player"]).alias("seat"),
            F.size("kinds").alias("n_entries"),
            *[
                _count_kinds(F.col("kinds"), names).alias(bucket)
                for bucket, names in buckets.items()
            ],
            F.array_contains("kinds", ActionKind.CONCEDE.value).alias("concession"),
        ),
        "turns",
    )


def build_cards_seen(bronze: DataFrame, catalog: DataFrame) -> DataFrame:
    """One row per (game, seat, card) from `observedCards`, joined to the catalog.

    A game with no `mySide` produces no rows: the export says which cards the
    uploader saw and which the opponent did, and with no seat for the uploader
    there is nothing to attach either list to.
    """
    summary = F.col("summary")
    my_side = summary["my_side"].cast(_INT)
    sides = [
        bronze.where(my_side.isNotNull()).select(
            F.col("game_id"),
            F.col("play_date"),
            seat.alias("seat"),
            F.explode(F.coalesce(summary["observed_cards"][role], F.array())).alias("card"),
        )
        for role, seat in (("me", my_side), ("opponent", F.lit(1) - my_side))
    ]
    card = F.col("card")
    observed = (
        sides[0]
        .unionByName(sides[1])
        .select(
            F.col("game_id"),
            F.col("play_date"),
            F.col("seat"),
            _card_key(card).alias("card_id"),
            card["base_card_id"].alias("base_card_id"),
            card["name"].alias("card_name"),
            card["set"].alias("set_code"),
            card["number"].alias("number"),
            card["count"].alias("count_seen"),
        )
        # Two references can resolve to one card (a named one and an id one), so
        # the grain is enforced here rather than trusted.
        .groupBy("game_id", "play_date", "seat", "card_id")
        .agg(
            F.max("base_card_id").alias("base_card_id"),
            F.max("card_name").alias("card_name"),
            F.max("set_code").alias("set_code"),
            F.max("number").alias("number"),
            F.max("count_seen").alias("count_seen"),
        )
    )
    decklists = _decklist_keys(bronze, catalog)
    joined = observed.join(
        F.broadcast(catalog.drop("catalog_number")), on="card_id", how="left"
    ).join(decklists, on=["game_id", "seat"], how="left")
    return _conform(
        joined.withColumn(
            "in_decklist",
            F.when(
                F.col("decklist_keys").isNotNull(),
                F.array_contains(F.col("decklist_keys"), F.col("card_id")),
            ),
        ),
        "cards_seen",
    )


def run_silver(
    spark: SparkSession,
    bronze_dir: Path,
    silver_dir: Path,
    catalog_path: Path | None,
) -> SilverSummary:
    """Build the four tables, write them and reconcile them against bronze.

    Raises `ReconciliationError` after the write when a count does not add up:
    the tables are on disk either way, and a run that quietly dropped half the
    games is worse than one that wrote them and said so.
    """
    started = time.monotonic()
    bronze = read_bronze(spark, bronze_dir).cache()
    catalog = load_catalog(spark, catalog_path)
    aliases = archetype_aliases(bronze)

    tables = {
        "games": build_games(bronze),
        "game_sides": build_game_sides(bronze, aliases, member_tokens(bronze)),
        "turns": build_turns(bronze),
        "cards_seen": build_cards_seen(bronze, catalog),
    }

    summary = SilverSummary(games_in=bronze.count())
    written: dict[str, DataFrame] = {}
    for name in TABLES:
        frame = tables[name].cache()
        write_table(frame, silver_dir, name)
        written[name] = frame
        summary.rows[name] = frame.count()
        logger.info("silver.%s: %d row(s)", name, summary.rows[name])

    reconcile(summary, written["cards_seen"])
    summary.duration_s = time.monotonic() - started
    return summary


def write_table(frame: DataFrame, silver_dir: Path, name: str) -> None:
    """Write one table under `silver_dir/<name>/play_date=.../`, replacing touched days."""
    frame.write.mode("overwrite").partitionBy("play_date").parquet(str(silver_dir / name))


def reconcile(summary: SilverSummary, cards_seen: DataFrame) -> None:
    """Check the invariants a silver run must hold, raising with every failure at once."""
    failures: list[str] = []
    checks = [
        ("games_in == games_out", summary.games_in, summary.rows["games"]),
        ("game_sides == 2 * games", SIDES_PER_GAME * summary.games_in, summary.rows["game_sides"]),
    ]
    for label, expected, actual in checks:
        status = "ok" if expected == actual else "FAILED"
        logger.info(
            "reconciliation check",
            extra={"check": label, "status": status, "expected": expected, "actual": actual},
        )
        summary.checks.append(
            f"reconciliation {label}: {status} (expected {expected}, got {actual})"
        )
        if expected != actual:
            failures.append(f"{label}: expected {expected}, got {actual}")

    duplicates = (
        cards_seen.groupBy("game_id", "seat", "card_id").count().where(F.col("count") > 1).count()
    )
    status = "ok" if duplicates == 0 else "FAILED"
    logger.info(
        "reconciliation check",
        extra={"check": "cards_seen grain", "status": status, "duplicate_keys": duplicates},
    )
    summary.checks.append(
        f"reconciliation cards_seen grain: {status} ({duplicates} duplicate key(s))"
    )
    if duplicates:
        failures.append(f"cards_seen has {duplicates} duplicate (game_id, seat, card_id) key(s)")

    if failures:
        raise ReconciliationError(failures)


def _conform(frame: DataFrame, table: str) -> DataFrame:
    """Project a frame onto its pinned schema: same columns, same order, same types."""
    schema = SILVER_SCHEMAS[table]
    return frame.select(
        *[F.col(field.name).cast(field.dataType).alias(field.name) for field in schema.fields]
    )


def _card_key(card: Column) -> Column:
    """A card reference's identity; see the module docstring for why it is not just `cardId`."""
    return F.coalesce(card["card_id"], card["base_card_id"], F.lower(card["name"]))


def _decklist_keys(bronze: DataFrame, catalog: DataFrame) -> DataFrame:
    """Per (game, seat), every key that seat's decklist can be matched on.

    Null for a seat with no decklist, which is what makes `in_decklist` null
    rather than false there.
    """
    summary = F.col("summary")
    my_side = summary["my_side"].cast(_INT)
    per_seat = [
        bronze.where(my_side.isNotNull() & F.col(column).isNotNull()).select(
            F.col("game_id"),
            seat.alias("seat"),
            F.explode(F.col(column)["cards"]).alias("card"),
        )
        for column, seat in (("my_decklist", my_side), ("opponent_decklist", F.lit(1) - my_side))
    ]
    card = F.col("card")
    return (
        per_seat[0]
        .unionByName(per_seat[1])
        .withColumn("card_id", _card_key(card))
        .join(F.broadcast(catalog.select("card_id", "catalog_name")), on="card_id", how="left")
        .select(
            "game_id",
            "seat",
            F.explode(
                F.array_compact(F.array(F.col("card_id"), F.lower(F.col("catalog_name"))))
            ).alias("key"),
        )
        .groupBy("game_id", "seat")
        .agg(F.collect_set("key").alias("decklist_keys"))
    )


def _count_kinds(kinds: Column, names: Iterable[str]) -> Column:
    wanted = F.array(*[F.lit(name) for name in names])
    return F.size(F.filter(kinds, lambda kind: F.array_contains(wanted, kind)))


def _seat_of(players: Column, handle: Column) -> Column:
    """The seat a handle sits in, or null when it is absent from `players`."""
    position = F.array_position(players, handle).cast(_INT)
    return F.when(position > 0, position - 1)


def _first_turn_seat(summary: Column) -> Column:
    """The seat that took the first turn, read off the log rather than the summary.

    `went_first_seat` is the producer's answer to the same question, out of
    `summary.wentFirst`. The two agree on every logged game in the corpus today,
    and they are kept apart so that stays checkable: a seat resolution that
    silently flips is exactly the bug the legacy stage shipped once.
    """
    first = F.get(F.filter(F.col("segments"), lambda seg: seg["kind"] == "turn"), 0)["player"]
    return _seat_of(summary["players"], first)


def _flag_seat(flag: Column, my_side: Column) -> Column:
    """A summary boolean about the uploader ("I went first") turned into a seat index."""
    return F.when(
        flag.isNotNull() & my_side.isNotNull(), F.when(flag, my_side).otherwise(1 - my_side)
    )


def _went_first(summary: Column, my_side: Column, seat: Column) -> Column:
    return _flag_seat(summary["went_first"], my_side) == seat


def _result_for_seat(summary: Column, my_side: Column, is_uploader: Column) -> Column:
    """The uploader's result as recorded, mirrored for the other seat."""
    result = summary["result"]
    mirrored = F.create_map(*[F.lit(value) for pair in MIRRORED_RESULT.items() for value in pair])
    return (
        F.when(my_side.isNull(), F.lit("unknown"))
        .when(is_uploader, result)
        .otherwise(F.coalesce(mirrored[result], F.lit("unknown")))
    )


def _archetype_id(summary: Column, is_uploader: Column) -> Column:
    return F.when(is_uploader, summary["my_archetype_id"]).otherwise(
        summary["opponent_archetype_id"]
    )


def _archetype_name_raw(
    summary: Column, is_uploader: Column, manual: Column, token: Column
) -> Column:
    """The label this row carries, before the alias map has its say.

    The uploader's side is labelled by `myArchetype` and by nothing else. The
    deck record the upload was linked to is not a fallback: `deckName` is a
    nickname typed into the game client, so most of the corpus carries the
    client's default ("New Deck 54") or a private joke, and reading those as
    archetypes floods every matchup with labels that name no deck. An uploaded
    game therefore has an uploader archetype only when the application derived
    one or the user set one, and otherwise none at all. A manual game with no
    archetype row for the opponent falls back to the name the uploader typed,
    which bronze has already turned into a token, so the stranger rule applies
    to it like any other token.
    """
    uploader = summary["my_archetype"]
    opponent = F.coalesce(
        summary["opponent_archetype"],
        F.when(manual & summary["opponent_archetype_id"].isNull(), token),
    )
    return F.when(is_uploader, uploader).otherwise(opponent)


def _archetype_source(summary: Column, is_uploader: Column, manual: Column) -> Column:
    """Where the label came from: `auto` derived, `user` pinned, `manual` typed in.

    Null on an uploader seat that has no archetype, which is most of them: the
    deck name is not a source because it is not a label (see
    `_archetype_name_raw`).
    """
    uploader = F.when(manual, F.lit("manual")).otherwise(
        F.when(
            F.coalesce(summary["my_archetype_id"], summary["my_archetype"]).isNotNull(),
            F.lit("user"),
        )
    )
    opponent = F.coalesce(summary["opponent_archetype_source"], F.when(manual, F.lit("manual")))
    return F.when(is_uploader, uploader).otherwise(opponent)


def _deck_name(summary: Column, is_uploader: Column) -> Column:
    """The uploader's own name for the deck they brought, null on the other seat.

    A player-level fact and never an archetype: these are nicknames typed into
    the game client, mostly its default ("New Deck 54"). The opponent seat has
    no deck record to read, so it stays null rather than borrowing the
    uploader's.
    """
    return F.when(
        is_uploader, F.coalesce(summary["deck_name"], summary["my_deck_meta"]["deck_name"])
    )


def _as_int(value: Any) -> int | None:
    """A catalog number that may arrive as an int, a numeric string or nothing."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the silver stage from the command line and report what it wrote."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.silver",
        description="Build the silver tables from bronze Parquet with PySpark.",
    )
    parser.add_argument(
        "--bronze-dir", type=Path, default=BRONZE_DIR, metavar="PATH", help="bronze input"
    )
    parser.add_argument(
        "--silver-dir", type=Path, default=SILVER_DIR, metavar="PATH", help="silver output"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        metavar="PATH",
        help="card catalog JSON; the run continues without it, with null catalog columns",
    )
    parser.add_argument(
        "--master",
        default=os.environ.get(MASTER_VAR) or DEFAULT_MASTER,
        help=f"Spark master URL (default: ${MASTER_VAR} or {DEFAULT_MASTER})",
    )
    args = parser.parse_args(argv)
    configure_logging(STAGE)

    spark = build_session(args.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        with stage_run(STAGE) as metrics:
            try:
                summary = run_silver(spark, args.bronze_dir, args.silver_dir, args.catalog)
            except ReconciliationError as exc:
                # Counted as a failed run: the tables are on disk, and a silver
                # run whose invariants broke is not a run anything downstream
                # should read as a success.
                metrics.extra = {"failures": list(exc.failures)}
                raise
            metrics.rows_in = summary.games_in
            metrics.rows_out = summary.rows.get("games", 0)
            metrics.rows_quarantined = 0
            metrics.extra = {"rows": summary.rows, "checks": summary.checks}
    except ReconciliationError as exc:
        logger.error("reconciliation failed", extra={"failures": list(exc.failures)})
        return 1
    finally:
        spark.stop()

    emit_summary(
        logger,
        "silver summary",
        {
            "games_in": summary.games_in,
            "rows": summary.rows,
            "checks": summary.checks,
            "duration_s": round(summary.duration_s, 4),
        },
        text=str(summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
