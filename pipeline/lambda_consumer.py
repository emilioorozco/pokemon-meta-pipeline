"""The event consumer as a Lambda: one SQS batch per invocation, no polling here.

The same ingest as `python -m pipeline.consume`, with the queue read by the
platform instead of by this code. An event source mapping on the
`parsed-games` queue does the long polling, batches up to ten messages, invokes
this function with them and, on a clean return, deletes the ones the function
did not report back. So there is no receive loop, no `delete_message` and no
visibility timeout in this module: what is left is the part that was always the
work, which is `MessageHandler.apply` from `pipeline.consume`, unchanged and
imported rather than copied.

Partial batch responses are why the return value has a shape at all. Without
them a Lambda that raises fails its whole batch, and SQS makes every message in
it visible again: nine landed games would be re-landed (harmlessly, because the
write is an upsert) and, worse, nine good messages would ride along with one
poison message all the way to the dead-letter queue. Returning
`{"batchItemFailures": [{"itemIdentifier": <messageId>}, ...]}` tells SQS to
delete the rest and redeliver only the named ones, so the receive count that
the redrive policy counts to three belongs to the message that is actually
failing. The event source mapping has to be created with
`ReportBatchItemFailures` for the field to be read at all; that is a line in the
application's stack, and a mapping without it silently ignores this return.

Which failures are named, and which are not, is `MessageHandler.apply`'s
answer, argued in `pipeline.consume`: a blob that fails the contract is
quarantined and acknowledged, because it will fail identically on every
redelivery and the quarantine pair is the record of it; an S3 error, a storage
error or a body that is not an S3 event at all is named, because a retry can
fix those and the dead-letter queue is where they belong after three of them.

Reserved concurrency is 1 on this function, and it is a correctness setting
rather than a throughput one. A bronze partition is written by reading the
day's Parquet file, dropping the rows this message replaces and writing the
file back; two invocations doing that to the same day at the same time would
each write a file built from what it read before the other wrote, and the
second write would silently drop the first one's game. One invocation at a time
makes that impossible. The ceiling it imposes is a batch of ten games every
second or so, which is orders of magnitude above what this application
produces; the way to lift it later is a table format with row-level writes
(Apache Iceberg, Delta Lake), not more concurrency over the same rewrite.

Configuration is the consumer's, with one addition. The bucket, the prefix, the
region and the lake root come from the same environment variables
`python -m pipeline.consume` reads, so a function and a laptop are configured
identically and there is no Lambda-only settings path to get wrong.
`PRA_QUEUE_URL` is deliberately not among them: the event source mapping owns
the queue, so the function never names it. The addition is the anonymization
key, which under Lambda comes from Secrets Manager: `HANDLE_HMAC_KEY_SECRET_ARN`
names the secret, its value is read once per container and kept, and
`HANDLE_HMAC_KEY` in the environment still wins so that this handler can be
invoked locally against a sample event with no AWS at all. The key is held as
bytes on a `Settings` that keeps it out of `repr`, and it is never logged.

What one invocation records: `configure_logging` installs the JSON formatter,
so every line is one object CloudWatch Logs can query with an Insights filter,
and the run identifier is the invocation's own `aws_request_id`, which is what
ties a log line to the request id CloudWatch already shows and to the
`run_metrics` row this invocation writes. The unit is the invocation, exactly as
the polling consumer's unit is the receive batch and for the same reason: a
consumer has no end, so the batch is the only thing that does. One
`stage_run("consume")` row per invocation goes to the lake root, which under
Lambda is the same `s3://` prefix bronze is written to, so the `ops` models see
the function's runs beside every other stage's with nothing to configure.

Nothing here logs a handle, a key or any blob content.
"""

import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from pipeline.backfill import Quarantiner
from pipeline.config import PIPELINE_DATA_DIR
from pipeline.consume import STAGE, MessageHandler
from pipeline.observability import configure_logging, stage_run
from pipeline.settings import (
    DATA_DIR_VAR,
    KEY_VAR,
    REGION_VAR,
    Settings,
    SettingsError,
    validate_data_root,
)
from pipeline.source import S3Source
from pipeline.storage import Location

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

KEY_SECRET_VAR: Final = "HANDLE_HMAC_KEY_SECRET_ARN"

# Held for the life of the execution environment, which is what a Lambda
# container is for: a function that handles a thousand batches must not resolve
# a secret and build a client a thousand times. `reset_container_state` is for
# the tests, which stand up one fake account after another in one process.
_hmac_key: bytes | None = None
_s3: "S3Client | None" = None


class SecretError(RuntimeError):
    """The named secret holds nothing this function can use as a key."""


def handler(event: dict[str, Any], context: Any) -> dict[str, list[dict[str, str]]]:
    """Apply one SQS batch to the lake and name the messages SQS should deliver again.

    The Lambda entry point, referenced by the image's `CMD`. `event` is an SQS
    event source payload (`Records[].messageId`, `Records[].body`) and `context`
    is the runtime's, read only for `aws_request_id`. Everything else this
    invocation needs comes from the environment, so the signature stays the one
    the platform calls.

    Never raises for a message: a record that cannot be handled comes back in
    `batchItemFailures` instead, which leaves the rest of the batch deleted and
    lets the redrive policy count this one message's third failure.
    """
    configure_logging(STAGE, run_id=_request_id(context), json_output=True)
    settings = lambda_settings()
    root = lake_root()
    records = list(event.get("Records") or [])
    messages = MessageHandler(
        settings=settings,
        source=S3Source(client=s3_client(settings), bucket=settings.bucket, prefix=settings.prefix),
        bronze_dir=root / "lake" / "bronze",
        rejects=Quarantiner(root / "lake" / "quarantine", datetime.now(UTC), dry_run=False),
    )

    failures: list[str] = []
    with stage_run(STAGE) as metrics:
        try:
            for record in records:
                if not _apply(messages, record):
                    failures.append(_message_id(record))
        finally:
            # In a `finally`, the same as the polling consumer's batch: an
            # invocation that dies partway still says how far it got beside the
            # `failed` status `stage_run` writes for it.
            metrics.rows_in = len(records)
            metrics.rows_out = messages.counts.landed
            metrics.rows_quarantined = messages.quarantined
            metrics.extra = {
                "deleted": messages.counts.deleted,
                "ignored": messages.counts.ignored,
                "failed": len(failures),
                "quarantined_by_reason": dict(sorted(messages.rejects.counts.items())),
            }
    return {"batchItemFailures": [{"itemIdentifier": failed} for failed in failures]}


def _apply(messages: MessageHandler, record: Any) -> bool:
    """One SQS record through the shared routine, with the last net under it.

    `MessageHandler.apply` already answers everything a blob can be wrong
    about and catches the infrastructure failures a single record can raise.
    What could still reach here is the invocation's own trouble, and catching
    it costs one record rather than the nine good ones beside it.
    """
    message_id = _message_id(record)
    try:
        return messages.apply(str(record.get("body", "")), message_id=message_id)
    except Exception:
        logger.exception("message %s failed outside the routine", message_id)
        return False


def _message_id(record: Any) -> str:
    """The identifier SQS deletes or redelivers by; empty is not one a mapping sends."""
    return str(record.get("messageId", "") or "unknown")


def _request_id(context: Any) -> str:
    """The invocation's request id as this run's identifier, or empty for a local call.

    Empty rather than a substitute, because `configure_logging` already resolves
    an empty one the way every other entry point does (`PRA_RUN_ID`, then a
    fresh identifier), and a local invocation with a hand-made context should
    not be made to invent a request id it does not have.
    """
    return str(getattr(context, "aws_request_id", "") or "").strip()


def lambda_settings() -> Settings:
    """The consumer's settings with the anonymization key resolved for this container.

    `require_queue` stays false: the event source mapping is what reads the
    queue, so a function that named one would be describing a resource it never
    talks to. `require_key` is false here and answered by `hmac_key` instead,
    which accepts either the environment variable or the secret.
    """
    settings = Settings.from_env(require_key=False)
    return replace(settings, hmac_key=hmac_key())


def hmac_key() -> bytes:
    """The anonymization key: the environment first, then the secret, fetched once.

    `HANDLE_HMAC_KEY` wins, which is what lets this handler be invoked on a
    laptop against a sample event with no AWS call at all, and is the same
    variable every other stage reads. Otherwise `HANDLE_HMAC_KEY_SECRET_ARN`
    names a Secrets Manager secret and its string value is the key.

    A secret rather than a plain Lambda environment variable because an
    environment variable is readable by anyone who can call `GetFunction`, and
    this key is the one thing standing between the lake and a reversible handle.
    Read once per execution environment and kept: the value does not change
    under a running container, and a rotation arrives as a new container, which
    a deployment or an idle timeout produces on its own.
    """
    global _hmac_key
    from_env = os.environ.get(KEY_VAR, "")
    if from_env:
        return from_env.encode("utf-8")
    if _hmac_key is not None:
        return _hmac_key
    arn = os.environ.get(KEY_SECRET_VAR, "").strip()
    if not arn:
        # Either one satisfies this function; naming both is what tells an
        # operator which of the two they meant to set.
        raise SettingsError([KEY_VAR, KEY_SECRET_VAR])
    _hmac_key = _read_secret(arn)
    return _hmac_key


def _read_secret(arn: str) -> bytes:
    """The secret's string value as bytes; the value itself is never logged."""
    import boto3

    # Untyped on purpose: the boto3 stubs this project installs cover S3, SQS
    # and DynamoDB, and adding a fourth service package for one call is a
    # dependency for a line of code.
    client: Any = boto3.client("secretsmanager", region_name=_region())
    value = client.get_secret_value(SecretId=arn).get("SecretString")
    if not isinstance(value, str) or not value:
        raise SecretError(f"{KEY_SECRET_VAR} names a secret with no string value")
    logger.info("anonymization key read from Secrets Manager")
    return value.encode("utf-8")


def s3_client(settings: Settings) -> "S3Client":
    """The client this container reads blobs with, built on first use and kept."""
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3", region_name=settings.region)
    return _s3


def lake_root() -> Location:
    """Where this invocation writes, read at call time rather than at import.

    `pipeline.config` resolves `PIPELINE_DATA_DIR` when the package is imported,
    which under Lambda happens once per container and is the right answer for a
    deployed function. It is the wrong answer for a test, which points the
    variable somewhere else afterwards, so the variable is read here too and
    validated the way every other entry point validates it: an `s3://` with no
    bucket is refused rather than treated as a relative directory.
    """
    override = os.environ.get(DATA_DIR_VAR, "").strip()
    return validate_data_root(override) if override else PIPELINE_DATA_DIR


def _region() -> str | None:
    """`AWS_REGION` when it is set, else None so boto3 resolves it the way it always does."""
    return os.environ.get(REGION_VAR) or None


def reset_container_state() -> None:
    """Forget the key and the client this container cached.

    For the tests: a Lambda keeps both on purpose, and a suite that stands up
    one fake account after another in one process must not hand the second the
    first one's client.
    """
    global _hmac_key, _s3
    _hmac_key = None
    _s3 = None
