"""Bronze writer: validated, anonymized v2 blobs -> Parquet partitioned by play date.

One row per game. The blob's nested shape is kept as it arrives (`summary` as a
struct, `segments` as a list of structs, the decklists as structs) instead of
being flattened into seat and event tables here. Bronze stays a faithful,
queryable landing of the contract; the seat and event grains are a silver
concern, and keeping the nesting means a contract change shows up as a schema
difference rather than as a lossy flatten nobody notices.

Why partition replace: a run is identified by the days it touches, not by the
objects it read. Each `play_date` directory is deleted and rewritten whole, so
re-running the same games over the same day yields the same row count instead
of appending duplicates, and there is no dedupe step downstream. Writes are
atomic per partition: the file lands under a temporary name in the partition
directory and is moved into place with `os.replace`.

`write_partitions` is the batch write, for a caller that holds every game of a
day at once (the backfill). `upsert_records` and `delete_game` are the
single-game writes, for a caller that holds one game (the event consumer): they
read the partition, drop the rows for the game ids they are about to write or
remove, and rewrite the partition through the same atomic replace, so bronze has
one writer and one on-disk layout whichever command is running. Both are a
read-modify-write of a whole day, which at a few dozen rows per partition costs
less than the machinery to avoid it; `find_by_source_key` is the matching read,
a scan of every partition for one `source_key`. What changes at scale is in
`upsert_records`.


Why the schema is pinned from the contract models rather than inferred: with
inference, a batch where every game happens to lack `summary.elo` produces a
null-typed column while another batch produces a struct, and a reader spanning
both partitions then sees two incompatible schemas. `bronze_schema()` walks the
Pydantic models once, so every partition ever written has the same column
types, whatever a given batch happens to contain.

Why `fields_json`: `Action.fields` is an open record whose values are
`str | int | float | bool` and whose keys differ per `kind`. There is no stable
struct for it and Parquet has no heterogeneous map, so each entry and
sub-entry carries its fields as a compact JSON string in `fields_json`. Readers
that need a value use a JSON function on that column; silver promotes the few
numeric fields it cares about to real columns.

Why list-of-struct instead of map: `statsByPlayer` and `extras` arrive as
records keyed by handle and by tag. Parquet maps with struct values read back
awkwardly in DuckDB, so both become lists of structs: `stats_by_player` as
`{handle, stats}` and `extras` as `{tag, lines}`. Key order follows the blob.
`summary.elo.modeElos` stays a map because its values are plain integers.

v1 blobs are not accepted here: `play_date_for` needs `summary.playedAt`. The
backfill quarantines a v1 blob (with a hint to re-parse it upstream) before it
reaches this module.
"""

import json
import os
import shutil
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Final, Literal, Union, get_args, get_origin

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel

from pipeline.anonymize import assert_no_handles
from pipeline.contract import (
    CONTRACT_VERSION,
    Action,
    Decklist,
    GameSummary,
    ParsedBlobV2,
    Segment,
    SideStats,
)

TIMESTAMP: Final = pa.timestamp("us", tz="UTC")
PART_FILENAME: Final = "part-0.parquet"
COMPRESSION: Final = "zstd"
PLAY_DATE_SOURCE: Final = "summary"
DATE_LENGTH: Final = len("YYYY-MM-DD")
_LINES: Final = pa.list_(pa.string())


class BronzeLeakError(RuntimeError):
    """A handle survived anonymization; the batch was not written.

    `paths` locates every offending string as `<source_key>:<path inside the
    blob>`. The paths never echo a handle: a leaking dict key is reported as
    `<key>`.
    """

    def __init__(self, paths: Sequence[str]) -> None:
        super().__init__(f"{len(paths)} handle leak(s), nothing written: {list(paths[:5])}")
        self.paths = list(paths)


@dataclass(frozen=True)
class BronzeRecord:
    """One game on its way to bronze, with where it came from.

    `blob` is already validated against the contract and already anonymized by
    the caller; this module rewrites nothing, it only checks and writes.
    """

    blob: ParsedBlobV2
    source_key: str
    source_version_id: str | None = None
    source_last_modified: datetime | None = None


@dataclass(frozen=True)
class LandedGame:
    """A game that is in bronze: which game, and which partition holds it."""

    game_id: str
    play_date: str


def play_date_for(blob: ParsedBlobV2) -> str:
    """The partition date of a game: the date part of `summary.playedAt`.

    Only v2 blobs have a `playedAt`, which is why this takes `ParsedBlobV2`
    alone. A v1 blob is quarantined by the backfill before it gets here; there
    is no S3-last-modified fallback in this module.
    """
    played_at = blob.summary.played_at
    date = played_at[:DATE_LENGTH]
    if len(date) != DATE_LENGTH or date[4] != "-" or date[7] != "-":
        raise ValueError(f"playedAt is not an ISO timestamp: {played_at!r}")
    return date


def to_row(record: BronzeRecord, ingested_at: datetime) -> dict[str, Any]:
    """One bronze row: lineage columns, then the blob's own content nested."""
    blob = record.blob.model_dump(mode="json", by_alias=False)
    summary = blob["summary"]
    return {
        "game_id": summary["game_id"],
        "user_id": summary["user_id"],
        "play_date": play_date_for(record.blob),
        "play_date_source": PLAY_DATE_SOURCE,
        "played_at": _utc(summary["played_at"]),
        "contract_version": CONTRACT_VERSION,
        "source_key": record.source_key,
        "source_version_id": record.source_version_id,
        "source_last_modified": _utc(record.source_last_modified),
        "ingested_at": _utc(ingested_at),
        "summary": summary,
        "segments": [_segment_columns(segment) for segment in blob["segments"]],
        "stats_by_player": [
            {"handle": handle, "stats": stats} for handle, stats in blob["stats_by_player"].items()
        ],
        "unparsed_lines": blob["unparsed_lines"],
        "extras": [{"tag": tag, "lines": lines} for tag, lines in blob["extras"].items()],
        "my_decklist": blob["my_decklist"],
        "opponent_decklist": blob["opponent_decklist"],
    }


def write_partitions(
    records: Iterable[BronzeRecord],
    bronze_dir: Path,
    ingested_at: datetime,
    *,
    real_handles: set[str] | None = None,
) -> dict[str, int]:
    """Write the records under `bronze_dir`, one partition per play date.

    Every touched `play_date=YYYY-MM-DD` directory is replaced whole, so the
    same records written twice give the same rows. Partitions this batch does
    not mention are left alone. Returns rows written per date.

    `real_handles` are the handles known before anonymization. When given, the
    whole batch is scanned first and a single surviving handle raises
    `BronzeLeakError` with nothing written, so a leak never reaches the lake
    half-way through a batch.
    """
    batch = list(records)
    if real_handles:
        _check_no_leaks(batch, real_handles)

    by_date: dict[str, list[dict[str, Any]]] = {}
    for record in batch:
        row = to_row(record, ingested_at)
        by_date.setdefault(row["play_date"], []).append(row)

    written: dict[str, int] = {}
    for date in sorted(by_date):
        rows = by_date[date]
        _write_partition(bronze_dir / f"play_date={date}", rows)
        written[date] = len(rows)
    return written


def upsert_records(
    records: Iterable[BronzeRecord],
    bronze_dir: Path,
    ingested_at: datetime,
    *,
    real_handles: set[str] | None = None,
) -> dict[str, int]:
    """Land the records without dropping the games already in their partitions.

    The single-game counterpart of `write_partitions`: each touched partition is
    read, the rows whose `game_id` this call is about to write are dropped, the
    new rows are appended and the partition is written back through the same
    atomic replace. Landing the same game twice therefore leaves one row for it
    and leaves every other game of that day alone, which is what an event
    consumer needs and what a plain `write_partitions` of one game would destroy.
    Returns the rows this call landed per date, not the size of the partitions.

    At scale this read-modify-write is the wrong shape: a day with a million
    rows would be rewritten per event. The replacements are then an append-only
    file per event plus a periodic compaction, or a table format (Apache Iceberg,
    Delta Lake) that does the row-level upsert itself. At a few dozen rows per
    day, rewriting the day is cheaper than either.
    """
    batch = list(records)
    if real_handles:
        _check_no_leaks(batch, real_handles)

    by_date: dict[str, list[dict[str, Any]]] = {}
    for record in batch:
        row = to_row(record, ingested_at)
        by_date.setdefault(row["play_date"], []).append(row)

    written: dict[str, int] = {}
    for date in sorted(by_date):
        rows = by_date[date]
        partition_dir = bronze_dir / f"play_date={date}"
        fresh = pa.Table.from_pylist(rows, schema=bronze_schema())
        kept = _without_games(_partition_table(partition_dir), {row["game_id"] for row in rows})
        _write_partition_table(partition_dir, pa.concat_tables([kept, fresh]))
        written[date] = len(rows)
    return written


def delete_game(bronze_dir: Path, play_date: str, game_id: str) -> int:
    """Remove one game from its partition; returns the rows removed (0 or 1).

    The partition is rewritten without the game, and a partition left with no
    rows is removed entirely rather than kept as an empty file, so a day that
    has been fully deleted upstream disappears from the dataset instead of
    reading back as a day with no games.
    """
    partition_dir = bronze_dir / f"play_date={play_date}"
    existing = _partition_table(partition_dir)
    if existing is None:
        return 0
    kept = _without_games(existing, {game_id})
    removed = existing.num_rows - kept.num_rows
    if not removed:
        return 0
    if kept.num_rows:
        _write_partition_table(partition_dir, kept)
    else:
        shutil.rmtree(partition_dir, ignore_errors=True)
    return removed


def find_by_source_key(bronze_dir: Path, source_key: str) -> LandedGame | None:
    """The game landed from `source_key`, found by scanning every partition.

    The lookup a delete event needs: the object is already gone from S3, so its
    play date cannot be read off the blob and the only record of where the row
    went is bronze itself. Two columns of every partition file are read, which is
    cheap for a corpus this size and linear in partitions as it grows; the scale
    fix is a small `game_id -> play_date` index (a sidecar table, or the
    warehouse) written alongside each partition and read here instead.
    """
    for path in sorted(bronze_dir.glob(f"play_date=*/{PART_FILENAME}")):
        table = pq.read_table(path, columns=["game_id", "source_key", "play_date"])
        for row in table.to_pylist():
            if row["source_key"] == source_key:
                return LandedGame(game_id=str(row["game_id"]), play_date=str(row["play_date"]))
    return None


def read_smoke(bronze_dir: Path) -> list[tuple[str, int]]:
    """Games per play date, read back out of the written Parquet with DuckDB.

    Proves the output is readable as one hive-partitioned dataset rather than
    as a pile of files. An empty or absent directory reads as no rows.
    """
    if not any(bronze_dir.glob("play_date=*/*.parquet")):
        return []
    query = (
        "SELECT CAST(play_date AS VARCHAR), count(*) "
        f"FROM read_parquet('{bronze_dir}/**/*.parquet', hive_partitioning=true) "
        "GROUP BY 1 ORDER BY 1"
    )
    with duckdb.connect() as con:
        return [(str(date), int(count)) for date, count in con.execute(query).fetchall()]


def bronze_schema() -> pa.Schema:
    """The full bronze schema, scalars pinned here and nesting taken from the models."""
    stats_entry = pa.struct(
        [pa.field("handle", pa.string()), pa.field("stats", _struct_of(SideStats))]
    )
    extras_entry = pa.struct([pa.field("tag", pa.string()), pa.field("lines", _LINES)])
    # Annotated so the heterogeneous Field generics do not widen the list to object.
    columns: list[pa.Field[Any]] = [
        pa.field("game_id", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("play_date", pa.string()),
        pa.field("play_date_source", pa.string()),
        pa.field("played_at", TIMESTAMP),
        pa.field("contract_version", pa.int32()),
        pa.field("source_key", pa.string()),
        pa.field("source_version_id", pa.string()),
        pa.field("source_last_modified", TIMESTAMP),
        pa.field("ingested_at", TIMESTAMP),
        pa.field("summary", _struct_of(GameSummary)),
        pa.field("segments", pa.list_(_struct_of(Segment))),
        pa.field("stats_by_player", pa.list_(stats_entry)),
        pa.field("unparsed_lines", _LINES),
        pa.field("extras", pa.list_(extras_entry)),
        pa.field("my_decklist", _struct_of(Decklist)),
        pa.field("opponent_decklist", _struct_of(Decklist)),
    ]
    return pa.schema(columns)


def _write_partition(partition_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Replace one partition directory with a single Parquet file."""
    _write_partition_table(partition_dir, pa.Table.from_pylist(rows, schema=bronze_schema()))


def _partition_table(partition_dir: Path) -> pa.Table | None:
    """The partition as written, or None when the day holds nothing yet."""
    path = partition_dir / PART_FILENAME
    if not path.is_file():
        return None
    return pq.read_table(path)


def _without_games(table: pa.Table | None, game_ids: set[str]) -> pa.Table:
    """The partition minus those games; an absent partition reads as no rows.

    The rows that stay are kept as Arrow rather than as Python dicts, so a
    rewrite never re-derives a value: what was written is what is written back.
    """
    if table is None:
        return bronze_schema().empty_table()
    listed = pa.array(sorted(game_ids), pa.string())
    return table.filter(pc.invert(pc.is_in(table.column("game_id"), value_set=listed)))


def _write_partition_table(partition_dir: Path, table: pa.Table) -> None:
    """Replace one partition directory with a single Parquet file."""
    shutil.rmtree(partition_dir, ignore_errors=True)
    partition_dir.mkdir(parents=True, exist_ok=True)
    target = partition_dir / PART_FILENAME
    staged = partition_dir / f".{PART_FILENAME}.tmp"
    pq.write_table(table, staged, compression=COMPRESSION)
    os.replace(staged, target)


def _check_no_leaks(records: Sequence[BronzeRecord], real_handles: set[str]) -> None:
    """Raise if any record still carries one of the pre-anonymization handles."""
    paths: list[str] = []
    for record in records:
        blob = record.blob.model_dump(mode="json", by_alias=False)
        paths += [f"{record.source_key}:{path}" for path in assert_no_handles(blob, real_handles)]
    if paths:
        raise BronzeLeakError(paths)


def _segment_columns(segment: dict[str, Any]) -> dict[str, Any]:
    out = dict(segment)
    out["entries"] = [_entry_columns(entry) for entry in segment["entries"]]
    return out


def _entry_columns(entry: dict[str, Any]) -> dict[str, Any]:
    out = _action_columns(entry)
    out["subs"] = [_action_columns(sub) for sub in entry["subs"]]
    return out


def _action_columns(action: dict[str, Any]) -> dict[str, Any]:
    """Swap an action's open `fields` record for its JSON text; see the module docstring."""
    out = dict(action)
    out["fields_json"] = json.dumps(out.pop("fields"), sort_keys=True, separators=(",", ":"))
    return out


def _utc(value: str | datetime | None) -> datetime | None:
    """An aware UTC datetime from an ISO string or a datetime; naive input is read as UTC."""
    if value is None:
        return None
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _struct_of(model: type[BaseModel]) -> pa.DataType:
    return pa.struct(list(_struct_fields(model)))


def _struct_fields(model: type[BaseModel]) -> Iterator[pa.Field]:
    """One Arrow field per model field, in declaration order."""
    for name, info in model.model_fields.items():
        if name == "fields" and issubclass(model, Action):
            yield pa.field("fields_json", pa.string())
        else:
            yield pa.field(name, _arrow_type(info.annotation))


def _arrow_type(annotation: Any) -> pa.DataType:
    """The Arrow type for one contract annotation. Every column is nullable."""
    annotation = _without_none(annotation)
    origin = get_origin(annotation)
    if origin is list:
        (item,) = get_args(annotation)
        return pa.list_(_arrow_type(item))
    if origin is dict:
        _, value = get_args(annotation)
        return pa.map_(pa.string(), _arrow_type(value))
    if origin is Literal:
        # A Literal of strings (the contract's closed string sets) or of one int.
        return _arrow_type(type(get_args(annotation)[0]))
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return _struct_of(annotation)
        if issubclass(annotation, Enum) or annotation is str:
            return pa.string()
        if annotation is bool:  # bool before int: bool is an int subclass
            return pa.bool_()
        if annotation is int:
            return pa.int64()
        if annotation is float:
            return pa.float64()
    raise TypeError(f"no Arrow type for contract annotation {annotation!r}")


def _without_none(annotation: Any) -> Any:
    """`X | None` -> `X`; anything else unchanged. Unions of two real types are rejected."""
    if get_origin(annotation) not in (Union, UnionType):
        return annotation
    real = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(real) != 1:
        raise TypeError(f"no Arrow type for union {annotation!r}")
    return real[0]
