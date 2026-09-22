"""Silver on a real local SparkSession: schemas, aliases, seats, grains, reconciliation.

Every test here starts a Java Virtual Machine (JVM), so the whole module is
marked `spark` and the default `uv run pytest` skips it; `uv run pytest -m spark`
runs it and needs Java on the path.

Most tests read the bronze lake the session fixture builds out of the committed
games, so they assert against real shapes rather than against a mock. The three
shapes the committed games do not contain (a renamed archetype, a manual game,
a token that is a member on one game and a stranger on another) are written as
bronze rows through the bronze writer itself, so they go through the same
schema and the same partition layout as everything else.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

from pipeline import silver
from pipeline.bronze import BronzeRecord, write_partitions
from pipeline.contract import (
    ActionKind,
    Entry,
    GameSummary,
    ParsedBlobV2,
    RoleStats,
    Segment,
    SideStats,
    SubEntry,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

pytestmark = pytest.mark.spark

CATALOG: Final = Path(__file__).parent / "catalog.json"
EARLIER: Final = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
LATER: Final = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)
ME: Final = "0000000000000001"
OPPONENT: Final = "00000000000000ff"
STRANGER: Final = "00000000000000aa"


def stats(**overrides: Any) -> SideStats:
    base: dict[str, Any] = {
        "cards_drawn": 7,
        "energy_attached": 2,
        "damage_dealt": 120,
        "knockouts": 1,
        "prizes_taken": 2,
        "mulligans": 0,
        "turns_taken": 1,
        "pokemon_played": ["Drakloak"],
        "cards_played": ["Ultra Ball"],
        "evolutions": [],
        "attacks": ["Phantom Dive"],
    }
    return SideStats(**{**base, **overrides})


def summary(game_id: str, played_at: str, **overrides: Any) -> GameSummary:
    base: dict[str, Any] = {
        "game_id": game_id,
        "user_id": "user-1",
        "uploaded_at": played_at,
        "played_at": played_at,
        "export_variant": "stock",
        "parser_version": 1,
        "unparsed_count": 0,
        "players": [ME, OPPONENT],
        "my_side": 0,
        "result": "win",
        "end_reason": "prizes",
        "winner": ME,
        "turn_count": 1,
        "stats": RoleStats(me=stats(), opponent=None),
        "has_full_decklists": False,
    }
    return GameSummary(**{**base, **overrides})


def turn_segment(player: str, *, concede: bool = False) -> Segment:
    """One turn: a draw entry with a nested draw, and optionally a concession."""
    entries = [
        Entry(
            line=10,
            text=f"{player} played Ultra Ball",
            kind=ActionKind.PLAY_CARD,
            actor=player,
            fields={},
            subs=[
                SubEntry(
                    line=11,
                    text=f"{player} drew 2 cards",
                    kind=ActionKind.DRAW,
                    actor=player,
                    fields={"n": 2},
                    details=[],
                )
            ],
        )
    ]
    if concede:
        entries.append(
            Entry(
                line=12,
                text=f"{player} conceded",
                kind=ActionKind.CONCEDE,
                actor=player,
                fields={},
                subs=[],
            )
        )
    return Segment(
        kind="turn", turn_number=1, player=player, title=f"{player}'s Turn", entries=entries
    )


def blob(game_id: str, played_at: str, *, manual: bool = False, **fields: Any) -> ParsedBlobV2:
    """A v2 blob in the shape the real corpus has, uploaded or manual."""
    if manual:
        return ParsedBlobV2(
            schema_version=2,
            summary=summary(
                game_id,
                played_at,
                export_variant="manual",
                parser_version=0,
                turn_count=0,
                winner=None,
                result="loss",
                stats=RoleStats(),
                **fields,
            ),
            segments=[],
            stats_by_player={},
            unparsed_lines=[],
            extras={},
        )
    return ParsedBlobV2(
        schema_version=2,
        summary=summary(game_id, played_at, **fields),
        segments=[turn_segment(ME), turn_segment(OPPONENT, concede=True)],
        stats_by_player={ME: stats(), OPPONENT: stats(knockouts=0)},
        unparsed_lines=[],
        extras={},
    )


def write_bronze(bronze_dir: Path, blobs: list[ParsedBlobV2], ingested_at: datetime) -> None:
    """Land the blobs through the real bronze writer, so silver reads a real partition."""
    write_partitions(
        [
            BronzeRecord(blob=item, source_key=f"parsed/user-1/{item.summary.game_id}.json")
            for item in blobs
        ],
        bronze_dir,
        ingested_at,
    )


def bronze_of(spark: "SparkSession", bronze_dir: Path) -> "DataFrame":
    return silver.read_bronze(spark, bronze_dir)


def sides_of(spark: "SparkSession", bronze_dir: Path) -> "DataFrame":
    bronze = bronze_of(spark, bronze_dir)
    return silver.build_game_sides(
        bronze, silver.archetype_aliases(bronze), silver.member_tokens(bronze)
    )


def typed(frame: "DataFrame") -> list[tuple[str, str]]:
    return [(field.name, field.dataType.simpleString()) for field in frame.schema.fields]


def expected(table: str) -> list[tuple[str, str]]:
    return [
        (field.name, field.dataType.simpleString()) for field in silver.SILVER_SCHEMAS[table].fields
    ]


def test_every_table_matches_its_pinned_schema(
    spark: "SparkSession", bronze_from_fixtures: Path
) -> None:
    bronze = bronze_of(spark, bronze_from_fixtures)
    catalog = silver.load_catalog(spark, CATALOG)
    built = {
        "games": silver.build_games(bronze),
        "game_sides": sides_of(spark, bronze_from_fixtures),
        "turns": silver.build_turns(bronze),
        "cards_seen": silver.build_cards_seen(bronze, catalog),
    }

    for table, frame in built.items():
        assert typed(frame) == expected(table), table

    # The counters come from the contract model, so the two cannot drift apart.
    names = [name for name, _ in expected("game_sides")]
    assert [f"stats_{field}" for field in SideStats.model_fields] == [
        name for name in names if name.startswith("stats_")
    ]


def test_a_renamed_archetype_resolves_to_the_newest_name(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """One archetype id under two names: the name the newest ingested game gave it wins."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [
            blob(
                "aaaa000000000001",
                "2026-08-01T12:00:00.000Z",
                opponent_archetype_id="arch-1",
                opponent_archetype="Old Name",
                opponent_archetype_source="auto",
            )
        ],
        EARLIER,
    )
    write_bronze(
        bronze_dir,
        [
            blob(
                "aaaa000000000002",
                "2026-08-02T12:00:00.000Z",
                opponent_archetype_id="arch-1",
                opponent_archetype="New Name",
                opponent_archetype_source="auto",
            )
        ],
        LATER,
    )

    rows = {row["game_id"]: row for row in sides_of(spark, bronze_dir).where("seat = 1").collect()}

    assert rows["aaaa000000000001"]["archetype_name"] == "New Name"
    assert rows["aaaa000000000002"]["archetype_name"] == "New Name"
    assert rows["aaaa000000000001"]["archetype_name_raw"] == "Old Name"
    assert rows["aaaa000000000002"]["archetype_name_raw"] == "New Name"


def test_a_conceded_game_still_produces_turn_rows(
    spark: "SparkSession", bronze_from_fixtures: Path
) -> None:
    """A concession ends the game inside a turn; that turn is still a row."""
    bronze = bronze_of(spark, bronze_from_fixtures)
    games = silver.build_games(bronze)
    turns = silver.build_turns(bronze)
    conceded = [
        row["game_id"]
        for row in games.where("end_reason in ('concede', 'opponent_concede')").collect()
    ]
    assert conceded, "the fixtures are supposed to include a conceded game"

    rows = turns.where(turns.game_id.isin(conceded)).collect()

    assert rows
    for game_id in conceded:
        for_game = [row for row in rows if row["game_id"] == game_id]
        assert for_game, game_id
        assert {row["seat"] for row in for_game} <= {0, 1}
        assert all(row["n_entries"] > 0 for row in for_game)
    assert any(row["concession"] for row in rows)


def test_turn_counters_only_count_the_kinds_they_are_mapped_to(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """Two action lines per turn: one play_card, one nested draw, and nothing else."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(bronze_dir, [blob("bbbb000000000001", "2026-08-03T12:00:00.000Z")], EARLIER)

    rows = sorted(silver.build_turns(bronze_of(spark, bronze_dir)).collect(), key=lambda r: r.seat)

    assert [row["seat"] for row in rows] == [0, 1]
    for row in rows:
        assert row["n_entries"] == (3 if row["seat"] == 1 else 2)
        assert (row["n_draw"], row["n_play_trainer"]) == (1, 1)
        assert row["n_attack"] == row["n_knockout"] == row["n_prize_taken"] == 0
    assert [row["concession"] for row in rows] == [False, True]


def test_cards_seen_has_one_row_per_game_seat_and_card(
    spark: "SparkSession", bronze_from_fixtures: Path
) -> None:
    bronze = bronze_of(spark, bronze_from_fixtures)
    cards = silver.build_cards_seen(bronze, silver.load_catalog(spark, CATALOG)).cache()

    duplicates = cards.groupBy("game_id", "seat", "card_id").count().where("count > 1").count()

    assert duplicates == 0
    assert cards.where("card_id is null").count() == 0
    assert cards.where("seat not in (0, 1)").count() == 0
    # The committed games are stock exports, so no seat has a decklist to check.
    assert cards.where("in_decklist is not null").count() == 0
    # The fixture catalog is keyed by the ids the fixtures actually resolve to.
    assert cards.where("catalog_name is not null").count() > 0


def test_a_run_without_a_catalog_leaves_the_catalog_columns_null(
    spark: "SparkSession", bronze_from_fixtures: Path, tmp_path: Path
) -> None:
    bronze = bronze_of(spark, bronze_from_fixtures)

    cards = silver.build_cards_seen(bronze, silver.load_catalog(spark, tmp_path / "absent.json"))

    assert cards.count() > 0
    assert cards.where("catalog_name is not null").count() == 0


def test_reconciliation_holds_on_the_fixtures(
    spark: "SparkSession", bronze_from_fixtures: Path, tmp_path: Path
) -> None:
    result = silver.run_silver(spark, bronze_from_fixtures, tmp_path / "silver", CATALOG)

    assert result.games_in == 10
    assert result.rows["games"] == result.games_in
    assert result.rows["game_sides"] == 2 * result.games_in
    assert result.rows["turns"] > 0
    assert result.rows["cards_seen"] > 0
    assert (tmp_path / "silver" / "games").is_dir()
    written = spark.read.parquet(str(tmp_path / "silver" / "games"))
    assert set(written.columns) == {name for name, _ in expected("games")}
    assert "games_in: 10" in str(result)


def test_reconciliation_failure_names_every_broken_check(spark: "SparkSession") -> None:
    """A run that lost a game and doubled a card says both, and raises."""
    cards = spark.createDataFrame(
        [("g1", 0, "budew"), ("g1", 0, "budew")], "game_id string, seat int, card_id string"
    )
    counted = silver.SilverSummary(games_in=10, rows={"games": 9, "game_sides": 19})

    with pytest.raises(silver.ReconciliationError) as raised:
        silver.reconcile(counted, cards)

    assert raised.value.failures == [
        "games_in == games_out: expected 10, got 9",
        "game_sides == 2 * games: expected 20, got 19",
        "cards_seen has 1 duplicate (game_id, seat, card_id) key(s)",
    ]


def test_a_stranger_token_is_null_and_a_member_token_survives(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """A token is a member because it uploaded something, not because it appeared."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [
            # The uploader of the first game is the opponent in the second one.
            blob("cccc000000000001", "2026-08-04T12:00:00.000Z"),
            blob(
                "cccc000000000002",
                "2026-08-05T12:00:00.000Z",
                players=[STRANGER, ME],
                my_side=1,
                winner=ME,
            ),
        ],
        EARLIER,
    )

    rows = {(row["game_id"], row["seat"]): row for row in sides_of(spark, bronze_dir).collect()}

    assert rows[("cccc000000000001", 0)]["player_token"] == ME
    assert rows[("cccc000000000001", 0)]["is_member"] is True
    assert rows[("cccc000000000001", 1)]["player_token"] is None
    assert rows[("cccc000000000001", 1)]["is_member"] is False
    # Same person, other seat: still a member, because membership is per token.
    assert rows[("cccc000000000002", 1)]["player_token"] == ME
    assert rows[("cccc000000000002", 0)]["player_token"] is None


def test_every_fixture_opponent_is_a_stranger(
    spark: "SparkSession", bronze_from_fixtures: Path
) -> None:
    rows = sides_of(spark, bronze_from_fixtures).collect()

    assert len(rows) == 20
    for row in rows:
        assert (row["player_token"] is not None) is row["is_member"]
        assert row["is_member"] is row["is_uploader"]


def test_a_manual_game_has_no_turns_and_two_side_rows(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """A manual game is a summary with no log: two seats, no turns, no cards."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [
            blob(
                "dddd000000000001",
                "2026-08-06T12:00:00.000Z",
                manual=True,
                my_archetype_id="arch-9",
                my_archetype="Dragapult control",
            )
        ],
        EARLIER,
    )
    bronze = bronze_of(spark, bronze_dir)

    sides = sorted(sides_of(spark, bronze_dir).collect(), key=lambda row: row.seat)

    assert silver.build_turns(bronze).count() == 0
    assert silver.build_cards_seen(bronze, silver.load_catalog(spark, CATALOG)).count() == 0
    assert len(sides) == 2
    assert sides[0]["archetype_name"] == "Dragapult control"
    assert sides[0]["archetype_source"] == "manual"
    assert sides[1]["archetype_source"] == "manual"
    # No statsByPlayer in a manual game, so every counter is null, not zero.
    assert sides[0]["stats_cards_drawn"] is None
    assert [row["result_for_seat"] for row in sides] == ["loss", "win"]


def test_a_manual_game_without_an_archetype_row_falls_back_to_the_typed_name(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """The fallback is a token, so the stranger rule blanks it like any other token."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [blob("eeee000000000001", "2026-08-07T12:00:00.000Z", manual=True)],
        EARLIER,
    )

    sides = sorted(sides_of(spark, bronze_dir).collect(), key=lambda row: row.seat)

    assert sides[1]["archetype_name_raw"] == OPPONENT
    assert sides[1]["player_token"] is None


def test_a_deck_name_is_carried_as_a_deck_name_and_never_as_an_archetype(
    spark: "SparkSession", tmp_path: Path
) -> None:
    """The client's default deck nickname labels the deck record, not the archetype."""
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [
            blob(
                "1111000000000001",
                "2026-08-09T12:00:00.000Z",
                deck_name="New Deck 54",
                deck_id="deck-54",
            )
        ],
        EARLIER,
    )

    sides = sorted(sides_of(spark, bronze_dir).collect(), key=lambda row: row.seat)

    assert sides[0]["is_uploader"] is True
    assert sides[0]["archetype_name"] is None
    assert sides[0]["archetype_name_raw"] is None
    assert sides[0]["archetype_source"] is None
    assert sides[0]["deck_name"] == "New Deck 54"
    assert sides[0]["deck_id"] == "deck-54"
    # The other seat has no deck record of its own to read.
    assert sides[1]["deck_name"] is None
    assert sides[1]["deck_id"] is None


def test_the_games_row_resolves_handles_to_seats(spark: "SparkSession", tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    write_bronze(
        bronze_dir,
        [
            blob(
                "ffff000000000001",
                "2026-08-08T12:00:00.000Z",
                my_side=1,
                players=[OPPONENT, ME],
                winner=ME,
                went_first=True,
                won_coin_toss=False,
            )
        ],
        EARLIER,
    )

    row = silver.build_games(bronze_of(spark, bronze_dir)).collect()[0]

    assert (row["winner_seat"], row["went_first_seat"], row["coin_toss_winner_seat"]) == (1, 1, 0)
    assert row["my_side"] == 1
    # The uploader took the first turn in the log, and the uploader sits in seat 1.
    assert row["first_player"] == 1
    assert row["excluded_from_stats"] is False


def test_the_catalog_reader_tolerates_a_hit_point_string() -> None:
    assert silver._as_int("60") == 60
    assert silver._as_int(320) == 320
    assert silver._as_int(None) is None
    assert silver._as_int("none") is None
