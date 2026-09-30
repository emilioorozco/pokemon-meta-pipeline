"""The Lambda handler against moto: an SQS batch, an S3 lake root, a real secret.

What these prove that `test_consume.py` cannot is the half the queue used to
own and the platform now owns: that one invocation over a mixed batch answers
with exactly the message ids SQS should deliver again, that everything else is
finished with, and that the invocation leaves the same three artefacts on the
lake a polling batch leaves (the bronze row, the quarantine pair, the
`run_metrics` row). The routing itself is not re-tested here: it is
`MessageHandler.apply`, the same object `pipeline.consume` runs, and
`test_consume.py` covers it message by message.

The lake root is an `s3://` prefix on a moto bucket rather than a temporary
directory, because that is what production is and because the run-metrics row
and the quarantine sidecar are the two writes most likely to work on a disk and
not in a bucket. moto's `mock_aws` intercepts botocore in this process, so the
handler's own `boto3.client` calls reach the fake with no endpoint override and
no code in the handler knowing it is under test.

Nothing is monkeypatched onto the handler: the anonymization key really is read
out of a Secrets Manager secret, the blob really is read out of a bucket, and
the failing record really is a key that is not there. The blob factories come
from `test_backfill` for the same reason `test_consume` takes them from there:
one contract-built blob, no second copy to drift.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote_plus

import pytest

from pipeline import lambda_consumer
from pipeline.lambda_consumer import KEY_SECRET_VAR, handler
from pipeline.settings import SettingsError
from pipeline.storage import Location, reset_s3_client
from tests.test_backfill import blob_v2, blob_with_unknown_kind

SOURCE_BUCKET: Final = "parsed-blobs-under-test"
LAKE_BUCKET: Final = "lake-under-test"
LAKE_PREFIX: Final = "nightly"
PREFIX: Final = "parsed/"
REGION: Final = "us-west-2"
# Not a secret: the fixtures are already anonymized under a key nobody kept, so
# this one only has to be stable inside a test run.
HMAC_KEY: Final = "tests-only-key-not-a-real-secret"
SECRET_NAME: Final = "handle-hmac-key-under-test"

GAME_ONE: Final = "parsed/user-1/game-1.json"
BROKEN: Final = "parsed/user-1/game-5.json"
MISSING: Final = "parsed/user-1/game-7.json"
OUTSIDE_PREFIX: Final = "raw/user-1/game-9.json"
PLAY_DATE: Final = "2026-09-01"

CREATED: Final = "ObjectCreated:Put"
REQUEST_ID: Final = "7f3c9b2a-4d1e-4c8f-9a6b-2e5d8c1f0a3b"


@dataclass(frozen=True)
class Context:
    """The two fields of a Lambda context this handler reads, and no more."""

    aws_request_id: str = REQUEST_ID
    function_name: str = "parsed-games-consumer"


@dataclass(frozen=True)
class Cloud:
    """The fake account one invocation runs against: two buckets and one secret."""

    s3: Any
    secrets: Any
    secret_arn: str
    lake: Location


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> Iterator[Cloud]:
    """A source bucket, a lake bucket, a secret holding the key, and the function's environment.

    `HANDLE_HMAC_KEY` is deleted rather than left alone: it wins over the
    secret by design, so a developer with one exported would otherwise make
    every assertion about Secrets Manager pass without Secrets Manager.
    """
    import boto3
    from moto import mock_aws

    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    reset_s3_client()
    lambda_consumer.reset_container_state()
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        for bucket in (SOURCE_BUCKET, LAKE_BUCKET):
            s3.create_bucket(
                Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION}
            )
        s3.put_bucket_versioning(
            Bucket=SOURCE_BUCKET, VersioningConfiguration={"Status": "Enabled"}
        )
        secrets = boto3.client("secretsmanager", region_name=REGION)
        arn = str(secrets.create_secret(Name=SECRET_NAME, SecretString=HMAC_KEY)["ARN"])

        monkeypatch.setenv("PRA_BUCKET", SOURCE_BUCKET)
        monkeypatch.setenv("PRA_PREFIX", PREFIX)
        monkeypatch.setenv("PIPELINE_DATA_DIR", f"s3://{LAKE_BUCKET}/{LAKE_PREFIX}")
        monkeypatch.setenv(KEY_SECRET_VAR, arn)
        monkeypatch.delenv("HANDLE_HMAC_KEY", raising=False)
        monkeypatch.delenv("PRA_QUEUE_URL", raising=False)

        yield Cloud(
            s3=s3,
            secrets=secrets,
            secret_arn=arn,
            lake=Location(f"s3://{LAKE_BUCKET}/{LAKE_PREFIX}"),
        )
    reset_s3_client()
    lambda_consumer.reset_container_state()


def put(cloud: Cloud, key: str, blob: dict[str, Any]) -> None:
    cloud.s3.put_object(Bucket=SOURCE_BUCKET, Key=key, Body=json.dumps(blob).encode("utf-8"))


def sqs_record(message_id: str, body: Any) -> dict[str, Any]:
    """One record of an SQS event source payload, in the shape Lambda delivers it."""
    return {
        "messageId": message_id,
        "receiptHandle": f"receipt-{message_id}",
        "body": body if isinstance(body, str) else json.dumps(body),
        "eventSource": "aws:sqs",
        "awsRegion": REGION,
    }


def s3_event(key: str, event_name: str = CREATED) -> dict[str, Any]:
    """The message body the bucket notification puts on the queue for `key`."""
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": REGION,
                "eventTime": "2026-09-20T12:00:00.000Z",
                "eventName": event_name,
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "bucket": {"name": SOURCE_BUCKET},
                    "object": {"key": quote_plus(key, safe="/"), "size": 2048},
                },
            }
        ]
    }


def bronze_rows(cloud: Cloud, play_date: str) -> list[dict[str, Any]]:
    path = cloud.lake / "lake" / "bronze" / f"play_date={play_date}" / "part-0.parquet"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = path.read_table().to_pylist()
    return sorted(rows, key=lambda row: str(row["game_id"]))


def metrics_row(cloud: Cloud, run_id: str) -> dict[str, Any]:
    path = cloud.lake / "lake" / "run_metrics" / f"{run_id}-consume.parquet"
    rows: list[dict[str, Any]] = path.read_table().to_pylist()
    assert len(rows) == 1, rows
    return rows[0]


def test_one_batch_reports_only_the_records_a_retry_could_fix(cloud: Cloud) -> None:
    """The whole contract of the handler in one invocation, over one mixed batch.

    Five records, three of them finished with and two of them named: a landed
    game, a key outside the prefix, a blob that fails the contract, a body that
    is not JSON, and a key whose object is not in the bucket.
    """
    put(cloud, GAME_ONE, blob_v2("game-1", "2026-09-01T18:22:00.000Z"))
    put(cloud, BROKEN, blob_with_unknown_kind())
    event = {
        "Records": [
            sqs_record("m-landed", s3_event(GAME_ONE)),
            sqs_record("m-ignored", s3_event(OUTSIDE_PREFIX)),
            sqs_record("m-quarantined", s3_event(BROKEN)),
            sqs_record("m-not-json", "this is not json, it is a log line"),
            sqs_record("m-no-object", s3_event(MISSING)),
        ]
    }

    response = handler(event, Context())

    assert response == {
        "batchItemFailures": [
            {"itemIdentifier": "m-not-json"},
            {"itemIdentifier": "m-no-object"},
        ]
    }

    # The good game is a bronze row on the bucket, anonymized under the secret.
    rows = bronze_rows(cloud, PLAY_DATE)
    assert [(row["game_id"], row["source_key"]) for row in rows] == [("game-1", GAME_ONE)]

    # The bad blob is a quarantine pair, and the message that carried it is not
    # in the failures above: it will fail identically on every redelivery.
    filed = cloud.lake / "lake" / "quarantine" / "contract_violation"
    assert (filed / "parsed__user-1__game-5.json").is_file()
    sidecar = json.loads((filed / "parsed__user-1__game-5.meta.json").read_text())
    assert sidecar["source_key"] == BROKEN
    assert "entries.0.kind" in sidecar["detail"]

    # And the invocation is one run_metrics row under the invocation's own id.
    row = metrics_row(cloud, REQUEST_ID)
    assert (row["stage"], row["status"]) == ("consume", "ok")
    assert (row["rows_in"], row["rows_out"], row["rows_quarantined"]) == (5, 1, 1)
    assert json.loads(row["extra_json"]) == {
        "deleted": 0,
        "failed": 2,
        "ignored": 1,
        "quarantined_by_reason": {"contract_violation": 1},
    }


def test_an_empty_batch_is_an_empty_answer(cloud: Cloud) -> None:
    """A mapping can deliver a batch of nothing; it is a run that did nothing, not a failure."""
    response = handler({"Records": []}, Context())

    assert response == {"batchItemFailures": []}
    row = metrics_row(cloud, REQUEST_ID)
    assert (row["rows_in"], row["rows_out"], row["status"]) == (0, 0, "ok")


def test_the_key_comes_from_the_secret_and_is_read_once_per_container(cloud: Cloud) -> None:
    """The secret is the source of the key, and the container keeps it.

    The second call is proved to be cached rather than re-fetched by deleting
    the secret first: a handler that asked Secrets Manager again would raise.
    """
    assert lambda_consumer.hmac_key() == HMAC_KEY.encode("utf-8")

    cloud.secrets.delete_secret(SecretId=cloud.secret_arn, ForceDeleteWithoutRecovery=True)

    assert lambda_consumer.hmac_key() == HMAC_KEY.encode("utf-8")


def test_the_environment_key_wins_over_the_secret(
    cloud: Cloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local invocation sets the variable and never calls AWS for the key."""
    monkeypatch.setenv("HANDLE_HMAC_KEY", "from-the-environment")
    cloud.secrets.delete_secret(SecretId=cloud.secret_arn, ForceDeleteWithoutRecovery=True)

    assert lambda_consumer.hmac_key() == b"from-the-environment"


def test_a_secret_with_no_string_value_is_named(
    cloud: Cloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = cloud.secrets.create_secret(Name="binary-secret", SecretBinary=b"\x00\x01")
    monkeypatch.setenv(KEY_SECRET_VAR, str(binary["ARN"]))

    with pytest.raises(lambda_consumer.SecretError) as raised:
        lambda_consumer.hmac_key()

    assert KEY_SECRET_VAR in str(raised.value)


def test_neither_the_variable_nor_the_secret_names_both(
    cloud: Cloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_SECRET_VAR, raising=False)

    with pytest.raises(SettingsError) as raised:
        lambda_consumer.hmac_key()

    assert raised.value.missing == ["HANDLE_HMAC_KEY", KEY_SECRET_VAR]


def test_importing_the_handler_loads_no_heavy_framework() -> None:
    """The image's whole size argument, asserted rather than trusted.

    In a subprocess because the assertion is about what `sys.modules` holds
    after one import, and the test session has already imported most of the
    repository. DuckDB is in the list with the three frameworks: it is a
    dependency of the project but not of this path, and `Dockerfile.lambda`
    leaves it out of the image on the strength of that.
    """
    probe = (
        "import sys, pipeline.lambda_consumer;"
        "heavy = {'pyspark', 'mlflow', 'langchain', 'langchain_core', 'torch', 'duckdb'};"
        "print(','.join(sorted(n for n in sys.modules if n.split('.')[0] in heavy)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.stdout.strip() == ""
