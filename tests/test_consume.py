"""The event path end to end against moto: S3 plus SQS, a dead-letter queue, no credentials.

What these prove is the part the backfill's tests cannot: that an S3 event
notification turns into the right write, that a message is deleted exactly when
the work behind it is finished, and that the messages which are not finished
reach the dead-letter queue after three deliveries instead of disappearing.

Two things about the fake are worth knowing. moto does not deliver bucket
notifications to a queue dependably, so nothing here configures one: `notify`
enqueues the message the bucket would have sent, in the real S3 event shape
(`Records[].eventName`, `Records[].s3.bucket.name`, a URL-encoded
`Records[].s3.object.key`), and the consumer cannot tell the difference because
the queue is all it reads. And moto applies the redrive policy the way SQS does,
on receive: a message is moved to the dead-letter queue on the receive *after*
`maxReceiveCount` deliveries, so a `maxReceiveCount` of 3 means three drains
that see the message and a fourth that finds it gone.

Every run polls with `--wait-seconds 0`: long polling against a fake queue would
only make the suite wait.

The blob factories come from `test_backfill` on purpose. One contract-built blob
is what keeps both stages honest about the same contract; a second copy here
would drift.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote_plus

import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from pipeline import backfill, consume
from pipeline.bronze import read_smoke
from pipeline.consume import ConsumeSummary, run_consumer
from pipeline.settings import Settings, SettingsError
from tests.test_backfill import blob_v2, blob_with_unknown_kind

BUCKET: Final = "pra-test-bucket"
PREFIX: Final = "parsed/"
REGION: Final = "us-west-2"
KEY: Final = b"test-key-not-a-real-secret"
NOW: Final = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
MAX_RECEIVES: Final = 3

GAME_ONE: Final = "parsed/user-1/game-1.json"
GAME_TWO: Final = "parsed/user-1/game-2.json"
BROKEN: Final = "parsed/user-1/game-5.json"
OUTSIDE_PREFIX: Final = "raw/user-1/game-9.json"
PLAY_DATE: Final = "2026-09-01"

CREATED: Final = "ObjectCreated:Put"
REMOVED: Final = "ObjectRemoved:Delete"

TEST_EVENT_BODY: Final = {
    "Service": "Amazon S3",
    "Event": "s3:TestEvent",
    "Time": "2026-09-20T12:00:00.000Z",
    "Bucket": BUCKET,
    "RequestId": "0000000000000000",
    "HostId": "invented-host-id",
}


@dataclass(frozen=True)
class Cloud:
    """The fake account one test runs against: a bucket, a queue, its dead-letter queue."""

    s3: Any
    sqs: Any
    queue_url: str
    dlq_url: str


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials so a misconfigured run can never reach a real account."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def cloud(aws_credentials: None) -> Iterator[Cloud]:
    """An empty versioned bucket and a queue whose redrive policy points at a dead-letter queue."""
    import boto3

    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})

        sqs = boto3.client("sqs", region_name=REGION)
        dlq_url = sqs.create_queue(QueueName="parsed-games-dlq")["QueueUrl"]
        dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])[
            "Attributes"
        ]["QueueArn"]
        queue_url = sqs.create_queue(
            QueueName="parsed-games",
            Attributes={
                "RedrivePolicy": json.dumps(
                    {"deadLetterTargetArn": dlq_arn, "maxReceiveCount": MAX_RECEIVES}
                )
            },
        )["QueueUrl"]
        yield Cloud(s3=s3, sqs=sqs, queue_url=queue_url, dlq_url=dlq_url)


@pytest.fixture
def settings(cloud: Cloud) -> Settings:
    return Settings(
        bucket=BUCKET, hmac_key=KEY, prefix=PREFIX, region=REGION, queue_url=cloud.queue_url
    )


@pytest.fixture
def lake(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "bronze", tmp_path / "quarantine"


def put(cloud: Cloud, key: str, blob: dict[str, Any]) -> None:
    cloud.s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(blob).encode("utf-8"))


def notify(cloud: Cloud, key: str, event_name: str = CREATED) -> None:
    """Enqueue the message the bucket notification would have sent for `key`.

    moto does not fire bucket notifications into SQS dependably, so the event is
    put on the queue explicitly. The shape is the real one, including the
    URL-encoded key, which is what the consumer has to decode.
    """
    record = {
        "eventVersion": "2.1",
        "eventSource": "aws:s3",
        "awsRegion": REGION,
        "eventTime": "2026-09-20T12:00:00.000Z",
        "eventName": event_name,
        "s3": {
            "s3SchemaVersion": "1.0",
            "bucket": {"name": BUCKET, "arn": f"arn:aws:s3:::{BUCKET}"},
            "object": {"key": quote_plus(key, safe="/"), "size": 2048, "sequencer": "0"},
        },
    }
    send(cloud, {"Records": [record]})


def send(cloud: Cloud, body: Any) -> None:
    text = body if isinstance(body, str) else json.dumps(body)
    cloud.sqs.send_message(QueueUrl=cloud.queue_url, MessageBody=text)


def drain(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path], *, visibility_timeout: int = 60
) -> ConsumeSummary:
    """One receive batch, the way `--once` runs it."""
    return run_consumer(
        settings,
        cloud.sqs,
        cloud.s3,
        lake[0],
        lake[1],
        now=NOW,
        once=True,
        wait_seconds=0,
        visibility_timeout=visibility_timeout,
    )


def queued(cloud: Cloud, url: str) -> int:
    """Messages waiting on a queue, as SQS approximates it."""
    attributes = cloud.sqs.get_queue_attributes(
        QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages"]
    )["Attributes"]
    return int(attributes["ApproximateNumberOfMessages"])


def partition_rows(bronze_dir: Path, play_date: str) -> list[dict[str, Any]]:
    path = bronze_dir / f"play_date={play_date}" / "part-0.parquet"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return sorted(rows, key=lambda row: str(row["game_id"]))


def test_a_created_event_lands_the_game_and_the_message_is_deleted(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, quarantine_dir = lake
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)

    summary = drain(cloud, settings, lake)

    assert (summary.received, summary.landed) == (1, 1)
    assert (summary.quarantined, summary.deleted, summary.ignored) == ({}, 0, 0)
    assert summary.left_for_redelivery == 0
    assert read_smoke(bronze_dir) == [(PLAY_DATE, 1)]
    assert queued(cloud, cloud.queue_url) == 0
    assert queued(cloud, cloud.dlq_url) == 0
    assert not quarantine_dir.exists()

    row = partition_rows(bronze_dir, PLAY_DATE)[0]
    assert (row["game_id"], row["source_key"]) == ("game-1", GAME_ONE)
    assert row["source_version_id"]


def test_a_second_game_joins_the_partition_and_a_repeat_changes_nothing(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """The write is an upsert: the day keeps its other games, and a redelivery is a no-op."""
    bronze_dir, _ = lake
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)
    drain(cloud, settings, lake)
    first = partition_rows(bronze_dir, PLAY_DATE)[0]

    put(cloud, GAME_TWO, blob_v2("game-2", "2026-09-01T21:40:00.000Z"))
    notify(cloud, GAME_TWO)
    second = drain(cloud, settings, lake)

    assert second.landed == 1
    assert read_smoke(bronze_dir) == [(PLAY_DATE, 2)]
    rows = partition_rows(bronze_dir, PLAY_DATE)
    assert [row["game_id"] for row in rows] == ["game-1", "game-2"]
    assert rows[0] == first

    # The same key again, as an at-least-once queue will do sooner or later.
    notify(cloud, GAME_TWO)
    again = drain(cloud, settings, lake)

    assert again.landed == 1
    assert read_smoke(bronze_dir) == [(PLAY_DATE, 2)]
    assert partition_rows(bronze_dir, PLAY_DATE)[0] == first
    assert len(list(bronze_dir.rglob("*.parquet"))) == 1


def test_a_blob_that_fails_the_contract_is_quarantined_and_the_message_is_deleted(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """A bad blob is handled, not retried: the quarantine pair is the record of it."""
    bronze_dir, quarantine_dir = lake
    put(cloud, BROKEN, blob_with_unknown_kind())
    notify(cloud, BROKEN)

    summary = drain(cloud, settings, lake)

    assert summary.quarantined == {"contract_violation": 1}
    assert (summary.landed, summary.left_for_redelivery) == (0, 0)
    assert read_smoke(bronze_dir) == []
    assert queued(cloud, cloud.queue_url) == 0
    assert queued(cloud, cloud.dlq_url) == 0

    filed = quarantine_dir / "contract_violation"
    assert (filed / "parsed__user-1__game-5.json").is_file()
    sidecar = json.loads((filed / "parsed__user-1__game-5.meta.json").read_text())
    assert sidecar["source_key"] == BROKEN
    assert "entries.0.kind" in sidecar["detail"]


def test_an_unexpected_write_failure_is_redelivered_until_the_dead_letter_queue_has_it(
    cloud: Cloud,
    settings: Settings,
    lake: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure the blob is not to blame for keeps the message, and the queue does the rest."""
    bronze_dir, quarantine_dir = lake

    def no_disk(*args: Any, **kwargs: Any) -> dict[str, int]:
        raise OSError("no space left on device")

    monkeypatch.setattr(consume, "land_records", no_disk)
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)

    # Visibility timeout 0, so the message is available again immediately.
    for _ in range(MAX_RECEIVES):
        summary = drain(cloud, settings, lake, visibility_timeout=0)
        assert (summary.received, summary.left_for_redelivery) == (1, 1)
        assert (summary.landed, summary.quarantined) == (0, {})

    # The move happens on the receive after the third delivery, as in real SQS.
    after = drain(cloud, settings, lake, visibility_timeout=0)

    assert after.received == 0
    assert queued(cloud, cloud.dlq_url) == 1
    assert queued(cloud, cloud.queue_url) == 0
    assert read_smoke(bronze_dir) == []
    assert not quarantine_dir.exists()


def test_a_message_that_is_not_an_s3_event_is_left_for_the_dead_letter_queue(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    send(cloud, "this is not json, it is a log line")

    summary = drain(cloud, settings, lake, visibility_timeout=0)

    assert (summary.received, summary.left_for_redelivery) == (1, 1)
    assert summary.ignored == 0
    assert queued(cloud, cloud.queue_url) == 1


def test_a_removed_event_takes_the_game_out_of_bronze(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, _ = lake
    for key, game_id in ((GAME_ONE, "game-1"), (GAME_TWO, "game-2")):
        put(cloud, key, blob_v2(game_id, "2026-09-01T18:22:00.000Z"))
        notify(cloud, key)
    drain(cloud, settings, lake)
    assert read_smoke(bronze_dir) == [(PLAY_DATE, 2)]

    cloud.s3.delete_object(Bucket=BUCKET, Key=GAME_ONE)
    notify(cloud, GAME_ONE, REMOVED)
    summary = drain(cloud, settings, lake)

    assert (summary.deleted, summary.ignored, summary.left_for_redelivery) == (1, 0, 0)
    assert read_smoke(bronze_dir) == [(PLAY_DATE, 1)]
    assert [row["game_id"] for row in partition_rows(bronze_dir, PLAY_DATE)] == ["game-2"]
    assert queued(cloud, cloud.queue_url) == 0


def test_removing_the_last_game_of_a_day_removes_the_partition(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    bronze_dir, _ = lake
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)
    drain(cloud, settings, lake)

    notify(cloud, GAME_ONE, REMOVED)
    summary = drain(cloud, settings, lake)

    assert summary.deleted == 1
    assert read_smoke(bronze_dir) == []
    assert not (bronze_dir / f"play_date={PLAY_DATE}").exists()


def test_removing_a_key_bronze_never_landed_is_ignored(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """A delete for a quarantined or never-ingested object is a no-op, not a failure."""
    notify(cloud, GAME_TWO, REMOVED)

    summary = drain(cloud, settings, lake)

    assert (summary.deleted, summary.ignored, summary.left_for_redelivery) == (0, 1, 0)
    assert read_smoke(lake[0]) == []
    assert queued(cloud, cloud.queue_url) == 0


def test_a_test_event_and_a_key_outside_the_prefix_are_acknowledged_and_ignored(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    send(cloud, TEST_EVENT_BODY)
    notify(cloud, OUTSIDE_PREFIX)

    summary = drain(cloud, settings, lake)

    assert (summary.received, summary.ignored) == (2, 2)
    assert (summary.landed, summary.deleted, summary.left_for_redelivery) == (0, 0, 0)
    assert queued(cloud, cloud.queue_url) == 0
    assert queued(cloud, cloud.dlq_url) == 0
    assert read_smoke(lake[0]) == []


def test_an_event_that_is_neither_a_create_nor_a_delete_is_ignored(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    notify(cloud, GAME_ONE, "ObjectRestore:Post")

    summary = drain(cloud, settings, lake)

    assert (summary.ignored, summary.left_for_redelivery) == (1, 0)
    assert queued(cloud, cloud.queue_url) == 0


def test_an_s3_read_failure_leaves_the_message_on_the_queue(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """The object is announced but not there: infrastructure, so the queue retries it."""
    notify(cloud, GAME_ONE)

    summary = drain(cloud, settings, lake, visibility_timeout=0)

    assert (summary.received, summary.left_for_redelivery) == (1, 1)
    assert (summary.landed, summary.quarantined) == (0, {})
    assert queued(cloud, cloud.queue_url) == 1


def test_a_handle_that_survives_anonymization_is_quarantined_not_landed(
    cloud: Cloud,
    settings: Settings,
    lake: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leak is a property of the blob, so it is filed and the message is done with."""
    bronze_dir, quarantine_dir = lake
    monkeypatch.setattr(backfill, "anonymize", lambda blob, key: blob)
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)

    summary = drain(cloud, settings, lake)

    assert summary.quarantined == {"handle_leak_check_failed": 1}
    assert (summary.landed, summary.left_for_redelivery) == (0, 0)
    assert read_smoke(bronze_dir) == []
    assert queued(cloud, cloud.queue_url) == 0
    detail = json.loads(
        (
            quarantine_dir / "handle_leak_check_failed" / "parsed__user-1__game-1.meta.json"
        ).read_text()
    )["detail"]
    assert "path(s) still hold a handle" in detail


def test_a_record_naming_another_bucket_is_ignored(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    """The consumer reads one bucket: the configured one, whatever the message says."""
    send(
        cloud,
        {
            "Records": [
                {
                    "eventName": CREATED,
                    "s3": {
                        "bucket": {"name": "someone-elses-bucket"},
                        "object": {"key": GAME_ONE},
                    },
                }
            ]
        },
    )

    summary = drain(cloud, settings, lake)

    assert (summary.ignored, summary.landed, summary.left_for_redelivery) == (1, 0, 0)
    assert queued(cloud, cloud.queue_url) == 0


def test_an_s3_event_with_no_records_is_acknowledged(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    send(cloud, {"Service": "Amazon S3", "Records": []})

    summary = drain(cloud, settings, lake)

    assert (summary.ignored, summary.left_for_redelivery) == (1, 0)
    assert queued(cloud, cloud.queue_url) == 0


def test_a_json_body_that_is_not_an_event_object_is_left_for_redelivery(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    send(cloud, [1, 2, 3])

    summary = drain(cloud, settings, lake, visibility_timeout=0)

    assert summary.left_for_redelivery == 1
    assert queued(cloud, cloud.queue_url) == 1


def test_a_record_that_is_not_shaped_like_an_s3_event_is_left_for_redelivery(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    send(cloud, {"Records": [{"eventName": CREATED}]})

    summary = drain(cloud, settings, lake, visibility_timeout=0)

    assert (summary.left_for_redelivery, summary.ignored) == (1, 0)
    assert queued(cloud, cloud.queue_url) == 1


def test_the_summary_prints_one_line_per_field(
    cloud: Cloud, settings: Settings, lake: tuple[Path, Path]
) -> None:
    put(cloud, BROKEN, blob_with_unknown_kind())
    notify(cloud, BROKEN)

    summary = drain(cloud, settings, lake)

    lines = str(summary).splitlines()
    assert [line.split(":")[0] for line in lines] == [
        "received",
        "landed",
        "quarantined",
        "deleted",
        "ignored",
        "left_for_redelivery",
        "duration_s",
    ]
    assert lines[2] == "quarantined: contract_violation=1"
    assert str(ConsumeSummary()).splitlines()[2] == "quarantined: none"


def test_the_command_line_drains_once_and_prints_the_summary(
    cloud: Cloud,
    lake: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("PRA_PREFIX", PREFIX)
    monkeypatch.setenv("PRA_QUEUE_URL", cloud.queue_url)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "test-key-not-a-real-secret")
    monkeypatch.setattr(consume, "_default_sqs", lambda settings: cloud.sqs)
    monkeypatch.setattr(consume, "_default_s3", lambda settings: cloud.s3)
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    notify(cloud, GAME_ONE)

    exit_code = consume.main(
        [
            "--once",
            "--wait-seconds",
            "0",
            "--bronze-dir",
            str(lake[0]),
            "--quarantine-dir",
            str(lake[1]),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "received: 1" in out
    assert "landed: 1" in out
    assert read_smoke(lake[0]) == [(PLAY_DATE, 1)]


def test_the_consumer_names_the_queue_variable_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "test-key-not-a-real-secret")
    monkeypatch.delenv("PRA_QUEUE_URL", raising=False)

    with pytest.raises(SettingsError) as raised:
        consume.main(["--once"])

    assert raised.value.missing == ["PRA_QUEUE_URL"]
    assert "PRA_QUEUE_URL" in str(raised.value)


def test_the_backfill_still_runs_without_a_queue_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The queue is the consumer's requirement alone; a backfill machine needs no queue."""
    monkeypatch.setenv("PRA_BUCKET", BUCKET)
    monkeypatch.setenv("HANDLE_HMAC_KEY", "test-key-not-a-real-secret")
    monkeypatch.delenv("PRA_QUEUE_URL", raising=False)

    loaded = Settings.from_env()

    assert loaded.queue_url == ""
    assert loaded.bucket == BUCKET
