"""Bronze writer: idempotent partition replace, nested columns readable, no handle leaks.

Blobs are built through the contract models, so a row here is exactly what the
ingest stage will hand the writer. Handles are invented (PlayerA, PlayerB).
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.bronze import (
    BronzeLeakError,
    BronzeRecord,
    _arrow_type,
    bronze_schema,
    play_date_for,
    read_smoke,
    to_row,
    write_partitions,
)
from pipeline.contract import (
    CardRef,
    Decklist,
    Entry,
    GameSummary,
    ParsedBlobV2,
    RoleStats,
    Segment,
    SideStats,
    SubEntry,
)

INGESTED_AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
ME = "PlayerA"
OPPONENT = "PlayerB"


def side_stats(**overrides: Any) -> SideStats:
    base: dict[str, Any] = {
        "cards_drawn": 12,
        "energy_attached": 4,
        "damage_dealt": 300,
        "knockouts": 3,
        "prizes_taken": 6,
        "mulligans": 0,
        "turns_taken": 8,
        "pokemon_played": ["Pikachu"],
        "cards_played": ["Professor's Research"],
        "evolutions": [],
        "attacks": ["Thunder Shock"],
    }
    return SideStats(**{**base, **overrides})


def make_blob(
    game_id: str,
    played_at: str,
    *,
    me: str = ME,
    opponent: str = OPPONENT,
    manual: bool = False,
) -> ParsedBlobV2:
    """A valid v2 blob. `manual` gives the summary-only shape: no segments, no stats."""
    summary = GameSummary(
        game_id=game_id,
        user_id="user-1",
        uploaded_at=played_at,
        played_at=played_at,
        export_variant="manual" if manual else "stock",
        parser_version=0 if manual else 7,
        unparsed_count=0 if manual else 1,
        players=[] if manual else [me, opponent],
        my_side=None if manual else 0,
        result="win",
        end_reason="prizes",
        winner=None if manual else me,
        turn_count=0 if manual else 2,
        stats=RoleStats(me=None if manual else side_stats(), opponent=None),
        has_full_decklists=not manual,
    )
    if manual:
        return ParsedBlobV2(
            schema_version=2,
            summary=summary,
            segments=[],
            stats_by_player={},
            unparsed_lines=[],
            extras={},
        )
    entry = Entry(
        line=4,
        text=f"{me} used Thunder Shock for 60 damage",
        kind="attack",
        actor=me,
        # Heterogeneous on purpose: the values are str, int and bool.
        fields={"attack": "Thunder Shock", "damage": 60, "critical": False},
        subs=[
            SubEntry(
                line=5,
                text=f"{opponent}'s Pokemon took 60 damage",
                kind="took_damage",
                actor=opponent,
                fields={"n": 60},
                details=["60 damage counters placed"],
            )
        ],
    )
    decklist = Decklist(
        cards=[CardRef(card_id="sv1-1", name="Pikachu", set="SV1", number="1", count=4)],
        card_count=60,
        complete=True,
        source="debug",
    )
    return ParsedBlobV2(
        schema_version=2,
        summary=summary,
        segments=[
            Segment(kind="setup", title="Setup", entries=[]),
            Segment(kind="turn", turn_number=1, player=me, title=f"{me}'s Turn", entries=[entry]),
        ],
        stats_by_player={me: side_stats(), opponent: side_stats(knockouts=1)},
        unparsed_lines=["a line no pattern matched"],
        extras={"Note": ["a tagged trailer line"]},
        my_decklist=decklist,
        opponent_decklist=None,
    )


def records() -> list[BronzeRecord]:
    """Two games on 2026-09-01 (one of them manual-style) and one on 2026-09-02."""
    return [
        BronzeRecord(
            blob=make_blob("game-1", "2026-09-01T18:22:00.000Z"),
            source_key="parsed/user-1/game-1.json",
            source_version_id="v1",
            source_last_modified=datetime(2026, 9, 1, 19, 0, tzinfo=UTC),
        ),
        BronzeRecord(
            blob=make_blob("game-2", "2026-09-01T20:05:00.000Z", manual=True),
            source_key="parsed/user-1/game-2.json",
        ),
        BronzeRecord(
            blob=make_blob("game-3", "2026-09-02T09:10:00.000Z"),
            source_key="parsed/user-1/game-3.json",
        ),
    ]


def parquet_files(bronze_dir: Path) -> list[Path]:
    return sorted(bronze_dir.rglob("*.parquet"))


def query(bronze_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    source = f"read_parquet('{bronze_dir}/**/*.parquet', hive_partitioning=true)"
    with duckdb.connect() as con:
        return con.execute(sql.format(source=source)).fetchall()


def test_play_date_for_takes_the_date_part_of_played_at() -> None:
    blob = make_blob("game-1", "2026-09-01T18:22:00.000Z")
    assert play_date_for(blob) == "2026-09-01"


def test_play_date_for_rejects_a_non_iso_played_at() -> None:
    blob = make_blob("game-1", "yesterday")
    with pytest.raises(ValueError, match="not an ISO timestamp"):
        play_date_for(blob)


def test_writing_twice_is_a_replace_not_an_append(tmp_path: Path) -> None:
    first = write_partitions(records(), tmp_path, INGESTED_AT)
    counts_after_first = read_smoke(tmp_path)
    files_after_first = parquet_files(tmp_path)

    second = write_partitions(records(), tmp_path, INGESTED_AT)

    assert first == second == {"2026-09-01": 2, "2026-09-02": 1}
    assert counts_after_first == read_smoke(tmp_path) == [("2026-09-01", 2), ("2026-09-02", 1)]
    assert files_after_first == parquet_files(tmp_path)
    assert len(list((tmp_path / "play_date=2026-09-01").glob("*.parquet"))) == 1


def test_rewriting_one_date_leaves_the_other_partition_alone(tmp_path: Path) -> None:
    write_partitions(records(), tmp_path, INGESTED_AT)
    subset = [r for r in records() if r.source_key.endswith("game-1.json")]

    written = write_partitions(subset, tmp_path, INGESTED_AT)

    assert written == {"2026-09-01": 1}
    assert read_smoke(tmp_path) == [("2026-09-01", 1), ("2026-09-02", 1)]


def test_read_smoke_on_an_empty_directory(tmp_path: Path) -> None:
    assert read_smoke(tmp_path) == []
    assert write_partitions([], tmp_path, INGESTED_AT) == {}


def test_duckdb_reads_the_nested_columns(tmp_path: Path) -> None:
    write_partitions(records(), tmp_path, INGESTED_AT)

    rows = query(
        tmp_path,
        "SELECT summary.game_id, len(segments), summary.stats.me.knockouts "
        "FROM {source} ORDER BY 1",
    )
    assert rows == [("game-1", 2, 3), ("game-2", 0, None), ("game-3", 2, 3)]

    handles = query(
        tmp_path,
        "SELECT list_transform(stats_by_player, s -> s.handle) "
        "FROM {source} WHERE game_id = 'game-1'",
    )
    assert handles == [([ME, OPPONENT],)]

    nested = query(
        tmp_path,
        "SELECT segments[2].entries[1].fields_json, segments[2].entries[1].subs[1].details, "
        "extras[1].tag, unparsed_lines[1], my_decklist.card_count, opponent_decklist "
        "FROM {source} WHERE game_id = 'game-1'",
    )
    fields_json, details, tag, unparsed, card_count, opponent_decklist = nested[0]
    assert json.loads(fields_json) == {
        "attack": "Thunder Shock",
        "critical": False,
        "damage": 60,
    }
    assert details == ["60 damage counters placed"]
    assert (tag, unparsed, card_count, opponent_decklist) == (
        "Note",
        "a line no pattern matched",
        60,
        None,
    )


def test_metadata_columns(tmp_path: Path) -> None:
    write_partitions(records(), tmp_path, INGESTED_AT)

    # Timestamps are read back through AT TIME ZONE 'UTC': DuckDB hands an aware
    # value to Python only with pytz installed, and the naive value is the same instant.
    rows = query(
        tmp_path,
        "SELECT game_id, user_id, contract_version, source_key, source_version_id, "
        "source_last_modified AT TIME ZONE 'UTC', ingested_at AT TIME ZONE 'UTC', "
        "played_at AT TIME ZONE 'UTC', play_date, play_date_source, typeof(ingested_at) "
        "FROM {source} ORDER BY game_id",
    )
    assert rows[0] == (
        "game-1",
        "user-1",
        2,
        "parsed/user-1/game-1.json",
        "v1",
        datetime(2026, 9, 1, 19, 0),
        datetime(2026, 9, 3, 12, 0),
        datetime(2026, 9, 1, 18, 22),
        # hive_partitioning gives the directory value, typed DATE, over the file column
        date(2026, 9, 1),
        "summary",
        "TIMESTAMP WITH TIME ZONE",
    )
    assert rows[1][4:6] == (None, None)

    # The file is self-describing: play_date is a column in it, not only in the path.
    with duckdb.connect() as con:
        in_file = con.execute(
            f"SELECT DISTINCT play_date FROM read_parquet('{tmp_path}/**/*.parquet', "
            "hive_partitioning=false) ORDER BY 1"
        ).fetchall()
    assert in_file == [("2026-09-01",), ("2026-09-02",)]

    schema = bronze_schema()
    assert schema.field("contract_version").type == "int32"
    assert str(schema.field("ingested_at").type) == "timestamp[us, tz=UTC]"


def test_to_row_keeps_the_blob_content_alongside_the_lineage_columns() -> None:
    record = records()[0]
    row = to_row(record, INGESTED_AT)

    assert row["summary"]["game_id"] == "game-1"
    assert [extra["tag"] for extra in row["extras"]] == ["Note"]
    assert row["segments"][1]["entries"][0]["subs"][0]["fields_json"] == '{"n":60}'
    assert "fields" not in row["segments"][1]["entries"][0]


def test_a_leaking_blob_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(BronzeLeakError) as raised:
        write_partitions(records(), tmp_path, INGESTED_AT, real_handles={ME})

    assert any(path.startswith("parsed/user-1/game-1.json:") for path in raised.value.paths)
    assert parquet_files(tmp_path) == []


def test_an_anonymized_blob_passes_the_leak_guard(tmp_path: Path) -> None:
    anonymized = [
        BronzeRecord(
            blob=make_blob("game-1", "2026-09-01T18:22:00.000Z", me="abc123", opponent="def456"),
            source_key="parsed/user-1/game-1.json",
        )
    ]

    written = write_partitions(anonymized, tmp_path, INGESTED_AT, real_handles={ME, OPPONENT})

    assert written == {"2026-09-01": 1}
    assert read_smoke(tmp_path) == [("2026-09-01", 1)]


def test_the_schema_is_the_same_whatever_a_batch_contains(tmp_path: Path) -> None:
    """A batch of manual-only games must not produce a narrower or null-typed schema."""
    manual_only = [
        BronzeRecord(
            blob=make_blob("game-2", "2026-09-01T20:05:00.000Z", manual=True),
            source_key="parsed/user-1/game-2.json",
        )
    ]
    write_partitions(manual_only, tmp_path, INGESTED_AT)
    written = pq.read_schema(tmp_path / "play_date=2026-09-01" / "part-0.parquet")

    assert written.names == bronze_schema().names
    assert written.field("my_decklist").type == bronze_schema().field("my_decklist").type


def test_a_naive_ingested_at_is_read_as_utc() -> None:
    row = to_row(records()[0], datetime(2026, 9, 3, 12, 0))
    assert row["ingested_at"] == INGESTED_AT


def test_the_type_mapper_refuses_what_it_cannot_represent() -> None:
    """The guard that fires if the contract grows a type this writer cannot land."""
    assert _arrow_type(float) == pa.float64()
    with pytest.raises(TypeError, match="no Arrow type for contract annotation"):
        _arrow_type(complex)
    with pytest.raises(TypeError, match="no Arrow type for union"):
        _arrow_type(str | int)
