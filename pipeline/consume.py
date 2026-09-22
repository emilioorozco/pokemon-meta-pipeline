"""SQS consumer: one parsed blob per S3 event, landed by the backfill's own code.

The event path of bronze ingest. The producer's bucket notifies an SQS queue
(`parsed-games`, with a dead-letter queue at three deliveries) on every
`s3:ObjectCreated:*` and `s3:ObjectRemoved:*` under `parsed/`; this command
long-polls that queue and applies each event to the lake, so a game uploaded to
the application is queryable in bronze in seconds rather than at the next
backfill.

Nothing about a blob is decided here. `process_object` is the backfill's routing
(read, decode, contract, anonymize, re-validate, quarantine with a reason) and
`land_records` is the backfill's write, so an object that arrives by event and
the same object seen by a later backfill walk are handled by one implementation.
What this module owns is the queue: which messages to act on, what to write for a
delete, and when a message is deleted rather than redelivered.

Which records are acted on: `ObjectCreated*` and `ObjectRemoved*` for a key that
is under the configured prefix, ends in `.json` and names the configured bucket.
Everything else is counted as ignored and the message is deleted: the S3 test
event the bucket sends when the notification is configured, a key outside the
prefix, an event type that is neither a create nor a delete. Keys arrive
URL-encoded in an S3 event (a space is `+`), so every key is unquoted before it
is used.

When a message is deleted, and why this is not "leave it on any failure": the
message is deleted when every record in it was handled, where handled means
landed, deleted from bronze, ignored, or quarantined for a reason that is a
property of the blob (`invalid_json`, `contract_violation`, `v1_blob`,
`handle_leak_check_failed`). A blob that fails the contract fails it identically
on every redelivery, so leaving the message would replay the same failure until
the dead-letter queue swallowed it, and the operator would have three copies of
one bad blob in a queue instead of one named body and sidecar under
`quarantine/`. The quarantine folder is the record for a bad blob; the
dead-letter queue is the record for a bad run. So the message is left on the
queue, and only on the queue, for what a retry can fix: an S3 error, a
filesystem error, a body that is not JSON at all and therefore is not an S3
event. Those come back, and after three deliveries the queue moves them to the
dead-letter queue, which is where they should be looked for.

This is the one deliberate difference from the backfill's failure handling: the
backfill quarantines a failed write as `write_failed` because a batch has no
retry, while here the queue is the retry and quarantining would throw away the
redelivery.

How a single game is written: bronze partitions by play date, and the backfill's
write replaces each touched partition whole, which for one game would delete the
rest of its day. So a created object is written with `merge=True`
(`bronze.upsert_records`): the partition is read, any row with the same
`game_id` is dropped, the new row is appended, and the partition is written back
through the same atomic replace. Landing the same key twice therefore leaves one
row for that game and every other game of the day untouched, which is what an
at-least-once queue requires, since a redelivered message must be safe to apply
again.

A delete is the same rewrite without the row. The blob is gone from S3 by the
time the event arrives, so its play date cannot be read off it and the only
record of which partition holds the row is bronze: `find_by_source_key` scans the
partition files for the `source_key` column. At a few dozen rows per partition
that scan is cheap; at scale it is the thing to fix first, with a
`game_id -> play_date` index written beside the partitions (or the warehouse) and
read here instead of a scan. A deleted key that is not in bronze is not an error:
it was quarantined, or never ingested, and the event is counted as ignored.

At scale the rewrite goes too: a partition with a million rows cannot be rewritten
per event. The replacements are an append-only file per event plus a compaction
step, or a table format (Apache Iceberg, Delta Lake) that does the row-level
upsert and delete itself. The contract, the routing and the quarantine do not
change with either.

Nothing here logs a handle or any blob content: the logs carry keys, reasons,
counts and message ids.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import unquote_plus

from pipeline.backfill import LEAK_PATHS_SHOWN, Quarantiner, land_records, process_object
from pipeline.bronze import BronzeLeakError, delete_game, find_by_source_key
from pipeline.config import BRONZE_DIR, QUARANTINE_DIR
from pipeline.quarantine import (
    CONTRACT_VIOLATION,
    HANDLE_LEAK_CHECK_FAILED,
    INVALID_JSON,
    V1_BLOB,
)
from pipeline.settings import Settings
from pipeline.source import BLOB_SUFFIX, S3Source, Source

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient
    from mypy_boto3_sqs.type_defs import MessageTypeDef

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES: Final = 10
DEFAULT_WAIT_SECONDS: Final = 20
DEFAULT_VISIBILITY_TIMEOUT: Final = 60

TEST_EVENT: Final = "s3:TestEvent"
CREATED: Final = "ObjectCreated"
REMOVED: Final = "ObjectRemoved"

# Quarantine reasons that are a property of the blob and so will not come out
# differently on a redelivery; see the module docstring.
HANDLED_REASONS: Final = (INVALID_JSON, CONTRACT_VIOLATION, V1_BLOB, HANDLE_LEAK_CHECK_FAILED)


@dataclass
class ConsumeSummary:
    """What one drain did. `quarantined` maps reason to blobs filed under it."""

    received: int = 0
    landed: int = 0
    quarantined: dict[str, int] = field(default_factory=dict)
    deleted: int = 0
    ignored: int = 0
    left_for_redelivery: int = 0
    duration_s: float = 0.0

    def __str__(self) -> str:
        return "\n".join(
            [
                f"received: {self.received}",
                f"landed: {self.landed}",
                f"quarantined: {_counts(self.quarantined)}",
                f"deleted: {self.deleted}",
                f"ignored: {self.ignored}",
                f"left_for_redelivery: {self.left_for_redelivery}",
                f"duration_s: {self.duration_s:.2f}",
            ]
        )


def run_consumer(
    settings: Settings,
    sqs: "SQSClient | None" = None,
    s3: "S3Client | None" = None,
    bronze_dir: Path = BRONZE_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
    *,
    source: Source | None = None,
    now: datetime | None = None,
    once: bool = False,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    visibility_timeout: int = DEFAULT_VISIBILITY_TIMEOUT,
) -> ConsumeSummary:
    """Drain the queue in `settings` and return what was done.

    `once` handles one receive batch and returns, which is what a test and a
    demo need; without it the loop runs until it is interrupted. `source` is
    where blobs are read from, defaulting to the bucket in `settings` through
    `s3` or a default client. `now` pins the ingestion timestamp for a test; a
    real run stamps each message with the time it was handled.
    """
    started = time.monotonic()
    reading = source if source is not None else _s3_source(settings, s3)
    consumer = _Consumer(
        settings=settings,
        sqs=sqs if sqs is not None else _default_sqs(settings),
        source=reading,
        bronze_dir=bronze_dir,
        rejects=Quarantiner(quarantine_dir, now or datetime.now(UTC), dry_run=False),
        now=now,
        max_messages=max_messages,
        wait_seconds=wait_seconds,
        visibility_timeout=visibility_timeout,
    )
    logger.info("polling %s", settings.queue_url)
    summary = consumer.drain(once=once)
    summary.duration_s = time.monotonic() - started
    return summary


class _Consumer:
    """One drain: the queue, where blobs are read from, and where rows are written."""

    def __init__(
        self,
        *,
        settings: Settings,
        sqs: "SQSClient",
        source: Source,
        bronze_dir: Path,
        rejects: Quarantiner,
        now: datetime | None,
        max_messages: int,
        wait_seconds: int,
        visibility_timeout: int,
    ) -> None:
        self.settings = settings
        self.sqs = sqs
        self.source = source
        self.bronze_dir = bronze_dir
        self.rejects = rejects
        self.now = now
        self.max_messages = max_messages
        self.wait_seconds = wait_seconds
        self.visibility_timeout = visibility_timeout
        self.summary = ConsumeSummary()

    def drain(self, *, once: bool) -> ConsumeSummary:
        """Receive and handle messages until `once` is satisfied or the process is stopped."""
        try:
            while True:
                for message in self._receive():
                    self._handle(message)
                if once:
                    break
        except KeyboardInterrupt:
            logger.info("interrupted: finishing the current batch and reporting")
        self.summary.quarantined = dict(sorted(self.rejects.counts.items()))
        return self.summary

    def _receive(self) -> list["MessageTypeDef"]:
        """One long poll. An empty result is normal: the queue is idle."""
        response = self.sqs.receive_message(
            QueueUrl=self.settings.queue_url,
            MaxNumberOfMessages=self.max_messages,
            WaitTimeSeconds=self.wait_seconds,
            VisibilityTimeout=self.visibility_timeout,
        )
        return list(response.get("Messages", []))

    def _handle(self, message: "MessageTypeDef") -> None:
        """Apply one message, then delete it only if every record in it was handled."""
        self.summary.received += 1
        # A consumer runs for days, so each message is stamped and quarantined at
        # the time it was handled rather than at the time the process started.
        self.rejects.when = self.now or datetime.now(UTC)
        if self._records(message):
            self.sqs.delete_message(
                QueueUrl=self.settings.queue_url, ReceiptHandle=message["ReceiptHandle"]
            )
            return
        self.summary.left_for_redelivery += 1
        logger.warning(
            "message %s left on the queue for redelivery", message.get("MessageId", "unknown")
        )

    def _records(self, message: "MessageTypeDef") -> bool:
        """True when every record in the message was handled and it can be deleted."""
        try:
            event = json.loads(message.get("Body", ""))
        except json.JSONDecodeError:
            # Not JSON, so not an S3 event and not something to guess at: the
            # dead-letter queue is where this belongs, with the body intact.
            logger.error("message %s is not JSON", message.get("MessageId", "unknown"))
            return False
        if not isinstance(event, dict):
            logger.error("message %s is not an S3 event", message.get("MessageId", "unknown"))
            return False
        if event.get("Event") == TEST_EVENT:
            # The bucket sends this once when the notification is configured.
            logger.info("s3:TestEvent acknowledged")
            self.summary.ignored += 1
            return True
        records = event.get("Records") or []
        if not records:
            logger.info("message with no records; nothing to apply")
            self.summary.ignored += 1
            return True
        return all([self._record(record) for record in records])

    def _record(self, record: Any) -> bool:
        """Apply one S3 event record. True when it was handled."""
        try:
            name = str(record["eventName"])
            bucket = str(record["s3"]["bucket"]["name"])
            # S3 URL-encodes the key in the event; a space arrives as `+`.
            key = unquote_plus(str(record["s3"]["object"]["key"]))
        except (KeyError, TypeError):
            logger.error("record is not an S3 event notification record")
            return False

        if bucket != self.settings.bucket:
            logger.warning("record names another bucket; ignored")
            self.summary.ignored += 1
            return True
        if not key.startswith(self.settings.prefix) or not key.endswith(BLOB_SUFFIX):
            logger.info("key is not a parsed blob; ignored: %s", key)
            self.summary.ignored += 1
            return True

        try:
            if name.startswith(CREATED):
                return self._created(key)
            if name.startswith(REMOVED):
                return self._removed(key)
        except Exception:
            # Everything the blob itself can be wrong about is already a
            # quarantine reason, so what reaches here is infrastructure: an S3
            # error, a filesystem error. Those are what a redelivery is for.
            logger.exception("%s failed for %s; leaving it for redelivery", name, key)
            return False
        logger.info("event %s is neither a create nor a delete; ignored", name)
        self.summary.ignored += 1
        return True

    def _created(self, key: str) -> bool:
        """Land one new object, or record why it was quarantined."""
        outcome = process_object(self.source, key, self.settings, self.rejects)
        if outcome.prepared is None:
            return outcome.reason in HANDLED_REASONS
        try:
            written = land_records(
                [outcome.prepared],
                self.bronze_dir,
                self.rejects.when,
                outcome.prepared.handles,
                merge=True,
            )
        except BronzeLeakError as exc:
            # The paths are already masked: a leaking dict key reads as `<key>`.
            detail = f"{len(exc.paths)} path(s) still hold a handle: {exc.paths[:LEAK_PATHS_SHOWN]}"
            self.rejects.add(key, outcome.prepared.raw, HANDLE_LEAK_CHECK_FAILED, detail)
            return True
        self.summary.landed += sum(written.values())
        logger.info("landed %s into %s", key, _counts(written))
        return True

    def _removed(self, key: str) -> bool:
        """Drop the game that was landed from a now-deleted object."""
        game = find_by_source_key(self.bronze_dir, key)
        if game is None:
            logger.info("no bronze row for the deleted key; nothing to remove")
            self.summary.ignored += 1
            return True
        removed = delete_game(self.bronze_dir, game.play_date, game.game_id)
        if not removed:
            self.summary.ignored += 1
            return True
        self.summary.deleted += removed
        logger.info("removed the game landed from %s (play_date=%s)", key, game.play_date)
        return True


def _s3_source(settings: Settings, s3: "S3Client | None") -> S3Source:
    """The production source: the bucket and prefix the settings name."""
    client = s3 if s3 is not None else _default_s3(settings)
    return S3Source(client=client, bucket=settings.bucket, prefix=settings.prefix)


def _default_s3(settings: Settings) -> "S3Client":
    import boto3

    return boto3.client("s3", region_name=settings.region)


def _default_sqs(settings: Settings) -> "SQSClient":
    import boto3

    return boto3.client("sqs", region_name=settings.region)


def _counts(counts: dict[str, int]) -> str:
    return " ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "none"


def main(argv: list[str] | None = None) -> int:
    """Run the consumer from the command line and print the summary at exit."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.consume",
        description=(
            "Drain the S3 event queue into bronze: land created blobs, remove "
            "deleted ones, quarantine the ones that fail the contract."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="handle one receive batch and exit, instead of polling until stopped",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        metavar="N",
        help=f"messages per receive, 1 to 10 (default: {DEFAULT_MAX_MESSAGES})",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        metavar="N",
        help=f"long-poll wait per receive (default: {DEFAULT_WAIT_SECONDS})",
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="where the landed rows go (default: the configured bronze directory)",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="where rejected blobs go (default: the configured quarantine directory)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env(require_queue=True)
    summary = run_consumer(
        settings,
        bronze_dir=args.bronze_dir or BRONZE_DIR,
        quarantine_dir=args.quarantine_dir or QUARANTINE_DIR,
        once=args.once,
        max_messages=args.max_messages,
        wait_seconds=args.wait_seconds,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
