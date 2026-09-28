"""The stages against an `s3://` lake root, on moto's in-process S3.

The claim `pipeline.storage` makes is that a stage does not know which kind of
root it was handed. These tests are how that claim is checked end to end rather
than one helper at a time: the real backfill over the committed fixtures, the
real partition replace, the real single-game merge and the real run-metrics row,
all with `s3://` in place of a directory and nothing else changed.

What is not here, and why:

- Silver. Spark reads and writes through the `s3a://` connector, which is a
  Hadoop jar fetched from Maven at session start and a Java credential chain
  neither moto's botocore hook nor a fake key reaches. `test_silver.py` covers
  the transforms on a local root and `test_storage.py` covers the configuration
  this stage derives; the connector itself is a live check against a real
  bucket.
- Gold. dbt reads silver through DuckDB's `httpfs`, which speaks HTTP from C++
  and is therefore invisible to `mock_aws` for the same reason. `test_gold.py`
  covers the build on a local root, and the S3 path is covered here only where
  it is this repository's code: which locations the stage derives and what it
  uploads afterwards.

Both are named in `docs/stages.md` as the two things the live run has to prove.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline import backfill as backfill_module
from pipeline.backfill import run_backfill
from pipeline.bronze import (
    BronzeRecord,
    delete_game,
    find_by_source_key,
    read_smoke,
    upsert_records,
    write_partitions,
)
from pipeline.observability import RunMetrics, write_run_metrics
from pipeline.quarantine import INVALID_JSON, write_quarantine
from pipeline.settings import Settings
from pipeline.source import LocalSource
from pipeline.storage import Location
from tests.conftest import FIXTURES_DIR, INGESTED_AT, TEST_HMAC_KEY
from tests.test_bronze import make_blob

NOW = datetime(2026, 9, 25, 8, 0, tzinfo=UTC)


def record(game_id: str, played_at: str) -> BronzeRecord:
    """One bronze record, built through the contract models like every other test."""
    return BronzeRecord(
        blob=make_blob(game_id, played_at, me="abc123", opponent="def456"),
        source_key=f"parsed/user-1/{game_id}.json",
    )


# ---------------------------------------------------------------- bronze --


def test_a_partition_replace_on_s3_is_idempotent(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    batch = [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-02T09:10:00Z")]

    first = write_partitions(batch, bronze, INGESTED_AT)
    counts = read_smoke(bronze)
    second = write_partitions(batch, bronze, INGESTED_AT)

    assert first == second == {"2026-09-01": 1, "2026-09-02": 1}
    assert counts == read_smoke(bronze) == [("2026-09-01", 1), ("2026-09-02", 1)]
    # One object per partition after two runs: the replace overwrote the key it
    # wrote last time rather than adding a second file beside it.
    assert len(bronze.iter_files("part-0.parquet")) == 2


def test_a_partition_replace_on_s3_drops_what_the_new_batch_does_not_name(
    s3_lake: Location,
) -> None:
    bronze = s3_lake / "lake" / "bronze"
    partition = bronze / "play_date=2026-09-01"
    # A file from an older layout, which the replace has to remove: on a disk
    # the directory is rewritten, on S3 the key has to be deleted by name.
    (partition / "part-1.parquet").write_bytes(b"left over")

    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert [item.name for item in partition.iter_files()] == ["part-0.parquet"]


def test_a_single_game_merge_on_s3_keeps_the_rest_of_the_day(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions(
        [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-01T19:00:00Z")],
        bronze,
        INGESTED_AT,
    )

    landed = upsert_records([record("game-3", "2026-09-01T20:00:00Z")], bronze, NOW)

    assert landed == {"2026-09-01": 1}
    assert read_smoke(bronze) == [("2026-09-01", 3)]


def test_merging_the_same_game_twice_on_s3_leaves_one_row(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    upsert_records([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, NOW)
    upsert_records([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, NOW)

    assert read_smoke(bronze) == [("2026-09-01", 1)]


def test_a_game_can_be_found_and_deleted_on_s3(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions(
        [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-01T19:00:00Z")],
        bronze,
        INGESTED_AT,
    )

    found = find_by_source_key(bronze, "parsed/user-1/game-2.json")
    assert found is not None
    assert (found.game_id, found.play_date) == ("game-2", "2026-09-01")
    assert delete_game(bronze, found.play_date, found.game_id) == 1
    assert read_smoke(bronze) == [("2026-09-01", 1)]


def test_emptying_a_partition_on_s3_removes_it(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert delete_game(bronze, "2026-09-01", "game-1") == 1
    assert read_smoke(bronze) == []
    assert not (bronze / "play_date=2026-09-01").is_dir()


# ------------------------------------------------------------ quarantine --


def test_a_rejected_blob_and_its_sidecar_land_on_s3(s3_lake: Location) -> None:
    quarantine = s3_lake / "lake" / "quarantine"

    body = write_quarantine(
        quarantine, "parsed/user-1/game-9.json", b"not json", INVALID_JSON, "JSONDecodeError", NOW
    )

    assert body.read_bytes() == b"not json"
    sidecar = quarantine / INVALID_JSON / "parsed__user-1__game-9.meta.json"
    assert json.loads(sidecar.read_text())["reason"] == INVALID_JSON


# ----------------------------------------------------------- run metrics --


def test_a_run_metrics_row_lands_on_s3(s3_lake: Location) -> None:
    target = s3_lake / "lake" / "run_metrics"
    metrics = RunMetrics(stage="bronze_backfill", run_id="abc123", started_at=NOW)
    metrics.finished_at = NOW
    metrics.rows_in = 3
    metrics.rows_out = 3

    path = write_run_metrics(metrics, target)

    assert path.name == "abc123-bronze_backfill.parquet"
    row = path.read_table().to_pylist()[0]
    assert (row["run_id"], row["stage"], row["rows_out"], row["status"]) == (
        "abc123",
        "bronze_backfill",
        3,
        "ok",
    )


# -------------------------------------------------------------- backfill --


def test_the_backfill_lands_the_fixtures_in_an_s3_root(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    settings = Settings(bucket="", hmac_key=TEST_HMAC_KEY)

    summary = run_backfill(
        settings,
        bronze_dir=bronze,
        quarantine_dir=s3_lake / "lake" / "quarantine",
        source=LocalSource(FIXTURES_DIR),
        now=INGESTED_AT,
    )

    assert summary.read == summary.landed > 0
    assert summary.quarantined == {}
    assert sum(count for _, count in read_smoke(bronze)) == summary.landed
    assert len(bronze.iter_files("part-0.parquet")) == len(summary.partitions)


def test_the_backfill_command_takes_an_s3_bronze_directory(
    s3_lake: Location, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole command line, so the argument parser's type is exercised too."""
    monkeypatch.setenv("HANDLE_HMAC_KEY", TEST_HMAC_KEY.decode())
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(s3_lake))

    code = backfill_module.main(
        [
            "--source-dir",
            str(FIXTURES_DIR),
            "--bronze-dir",
            str(s3_lake / "lake" / "bronze"),
            "--quarantine-dir",
            str(s3_lake / "lake" / "quarantine"),
        ]
    )

    assert code == 0
    assert sum(count for _, count in read_smoke(s3_lake / "lake" / "bronze")) > 0
    # The stage wrote its own row into the same lake, because the run was
    # pointed at it and nothing had to be told twice.
    assert len((s3_lake / "lake" / "run_metrics").iter_files(".parquet")) == 1


def test_a_local_root_still_writes_where_it_always_did(tmp_path: Path) -> None:
    """The control: the same call with a directory, asserting the on-disk layout."""
    bronze = tmp_path / "bronze"
    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert (bronze / "play_date=2026-09-01" / "part-0.parquet").is_file()
    assert list((bronze / "play_date=2026-09-01").glob(".*")) == []
