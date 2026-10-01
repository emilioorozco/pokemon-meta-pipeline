"""The serving Lambda: the secret, the refresh rule, and one request end to end.

What these prove is the half that is not `pipeline.serve`, which
`tests/test_serve.py` already drives endpoint by endpoint. Three things are
this module's own and each of them fails quietly if it is wrong:

The keys. A value already in the environment has to win over the secret, or a
laptop under `.env.op` and this suite would both start talking to Secrets
Manager; the secret has to be read once per container rather than once per
question; and no value from it may ever reach a log. The read itself runs
against moto rather than a stub, so the JSON shape the application's stack
writes is the JSON shape this parses.

The refresh rule. The warehouse is downloaded once and kept, and the only
thing that makes a long-lived container notice a nightly is the ETag check.
It is driven here with a fake head and a fake clock, so the interval, the
first call, an unchanged object, a replaced one and a head that fails are five
assertions rather than a ten-minute wait.

The request. `handler` really goes through Mangum and really answers
`GET /health` from a function URL event. The model is
`pipeline.serve.StubPredictor`, reached the same way a demonstration reaches
it, so there is no registry, no MLflow and no network anywhere in it.
"""

import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest

from pipeline import lambda_serve
from pipeline.lambda_serve import (
    KEYS_SECRET_VAR,
    SECRET_KEYS,
    SecretError,
    WarehouseWatch,
    handler,
    load_keys,
    read_secret,
    refresh_warehouse,
)
from pipeline.serve import STUB_MODEL_VAR, STUB_VERSION
from pipeline.storage import Location

REGION: Final = "us-west-2"
SECRET_NAME: Final = "agent-keys-under-test"
# Not secrets: nothing accepts either of them, and they only have to be
# distinctive enough that a test can assert they are absent from a log.
PROVIDER_KEY: Final = "test-anthropic-value-not-a-real-key"
JUDGE_KEY: Final = "test-jev-value-not-a-real-key"
ALREADY_SET: Final = "from-the-environment-not-the-secret"

WAREHOUSE: Final = Location("s3://lake-under-test/nightly/warehouse/meta.duckdb")


def function_url_event(method: str, path: str) -> dict[str, Any]:
    """One Lambda function URL request, in the payload format a function URL sends.

    The version 2.0 shape, which is the API Gateway HTTP API event and is what
    Mangum reads. Written out here rather than imported from a fixture file so
    the one thing the handler's contract depends on, `requestContext.http`, is
    visible in the test that relies on it.
    """
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {
            "host": "localhost",
            "user-agent": "tests",
            "x-forwarded-proto": "https",
        },
        "requestContext": {
            "accountId": "anonymous",
            "apiId": "under-test",
            "domainName": "under-test.lambda-url.us-west-2.on.aws",
            "domainPrefix": "under-test",
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "tests",
            },
            "requestId": "7f3c9b2a-4d1e-4c8f-9a6b-2e5d8c1f0a3b",
            "stage": "$default",
            "time": "23/Sep/2026:09:00:00 +0000",
            "timeEpoch": 1790499600000,
        },
        "isBase64Encoded": False,
    }


@pytest.fixture(autouse=True)
def fresh_container(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A container that has cached nothing, before and after every test.

    The module keeps the adapter, the keys and the warehouse watch for the life
    of the execution environment on purpose, and a suite that builds one
    application after another in one process must not hand the second the
    first one's.
    """
    lambda_serve.reset_container_state()
    for name in (*SECRET_KEYS, KEYS_SECRET_VAR, STUB_MODEL_VAR):
        monkeypatch.delenv(name, raising=False)
    yield
    lambda_serve.reset_container_state()


@pytest.fixture
def secret_arn(aws_fake_credentials: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A Secrets Manager secret holding the two keys as the stack writes them."""
    import boto3
    from moto import mock_aws

    with mock_aws():
        secrets = boto3.client("secretsmanager", region_name=REGION)
        created = secrets.create_secret(
            Name=SECRET_NAME,
            SecretString=json.dumps({"ANTHROPIC_API_KEY": PROVIDER_KEY, "JEV_API_KEY": JUDGE_KEY}),
        )
        arn = str(created["ARN"])
        monkeypatch.setenv(KEYS_SECRET_VAR, arn)
        yield arn


# ------------------------------------------------------------- the secret --


def test_both_keys_come_out_of_the_secret(secret_arn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert sorted(load_keys()) == sorted(SECRET_KEYS)
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == PROVIDER_KEY
    assert os.environ["JEV_API_KEY"] == JUDGE_KEY


def test_a_key_already_in_the_environment_wins(
    secret_arn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env.op` on a laptop, and a key set by a test, are not overwritten."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", ALREADY_SET)
    import os

    assert load_keys() == ["JEV_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == ALREADY_SET
    assert os.environ["JEV_API_KEY"] == JUDGE_KEY


def test_no_secret_named_is_not_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """How the image runs on a laptop and under the runtime interface emulator."""
    with caplog.at_level(logging.INFO):
        assert load_keys() == []
    assert caplog.records[-1].variable == KEYS_SECRET_VAR  # type: ignore[attr-defined]


def test_the_secret_is_read_once_per_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thousand questions are one read; `reset_container_state` is the only way back."""
    reads: list[str] = []

    def counting(arn: str) -> dict[str, str]:
        reads.append(arn)
        return {"ANTHROPIC_API_KEY": PROVIDER_KEY}

    monkeypatch.setenv(KEYS_SECRET_VAR, "a-secret-under-test")
    monkeypatch.setattr(lambda_serve, "read_secret", counting)
    assert load_keys() == ["ANTHROPIC_API_KEY"]
    assert load_keys() == []
    assert load_keys() == []
    assert reads == ["a-secret-under-test"]


def test_no_key_value_is_ever_logged(
    secret_arn: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log line names which variables were filled, and nothing else about them."""
    with caplog.at_level(logging.DEBUG):
        load_keys()
    # This package's records only. botocore at DEBUG prints the whole wire
    # response, secret included, which is botocore's business and is off in
    # every process this ships in; what has to hold is that nothing under
    # `pipeline` writes a value down.
    ours = [record for record in caplog.records if record.name.startswith("pipeline")]
    assert ours
    # Both the message and the structured fields, because the JSON formatter
    # writes the fields and the console formatter writes the message.
    written = "".join(record.getMessage() + str(record.__dict__) for record in ours)
    assert PROVIDER_KEY not in written
    assert JUDGE_KEY not in written
    assert sorted(ours[-1].filled) == sorted(SECRET_KEYS)  # type: ignore[attr-defined]


def test_a_secret_that_is_not_json_is_refused(
    aws_fake_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain string is the shape the consumer's secret has, and it is not this one's."""
    import boto3
    from moto import mock_aws

    with mock_aws():
        secrets = boto3.client("secretsmanager", region_name=REGION)
        arn = str(secrets.create_secret(Name=SECRET_NAME, SecretString="not-json")["ARN"])
        with pytest.raises(SecretError, match="not JSON"):
            read_secret(arn)


def test_a_secret_with_no_string_value_is_refused(
    aws_fake_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import boto3
    from moto import mock_aws

    with mock_aws():
        secrets = boto3.client("secretsmanager", region_name=REGION)
        arn = str(secrets.create_secret(Name=SECRET_NAME, SecretBinary=b"\x00\x01")["ARN"])
        with pytest.raises(SecretError, match="no string value"):
            read_secret(arn)


# -------------------------------------------------------- the refresh rule --


class FakeObject:
    """One S3 object's ETag, and a count of how often it was asked for."""

    def __init__(self, etag: str) -> None:
        self.etag = etag
        self.heads = 0

    def head(self, target: Location) -> str | None:
        self.heads += 1
        return self.etag


class FakeClock:
    """A clock a test moves by hand, so an interval is an assertion not a wait."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def watch_over(obj: FakeObject, clock: FakeClock, *, interval: float = 600.0) -> WarehouseWatch:
    return WarehouseWatch(WAREHOUSE, interval=interval, head=obj.head, clock=clock)


def test_the_first_check_records_the_tag_and_reports_no_change() -> None:
    """The copy in hand came from this object, so the first answer is always False."""
    obj, clock = FakeObject('"one"'), FakeClock()
    watch = watch_over(obj, clock)
    assert watch.changed() is False
    assert obj.heads == 1


def test_the_object_is_not_checked_again_inside_the_interval() -> None:
    """A burst of questions costs one head, not one per question."""
    obj, clock = FakeObject('"one"'), FakeClock()
    watch = watch_over(obj, clock)
    watch.changed()
    obj.etag = '"two"'
    for _ in range(5):
        assert watch.changed() is False
    assert obj.heads == 1
    clock.now = 599.0
    assert watch.changed() is False
    assert obj.heads == 1


def test_a_replaced_object_is_noticed_once_the_interval_has_passed() -> None:
    obj, clock = FakeObject('"one"'), FakeClock()
    watch = watch_over(obj, clock)
    watch.changed()
    clock.now = 600.0
    assert watch.changed() is False, "the object has not changed"
    obj.etag = '"two"'
    clock.now = 1200.0
    assert watch.changed() is True
    assert obj.heads == 3
    # And only once: the new tag is now the one in hand.
    clock.now = 1800.0
    assert watch.changed() is False


def test_a_head_that_fails_keeps_the_copy_in_hand(caplog: pytest.LogCaptureFixture) -> None:
    """A transient S3 error must not throw away a working warehouse."""

    def broken(target: Location) -> str | None:
        raise RuntimeError("head refused")

    watch = WarehouseWatch(WAREHOUSE, head=broken, clock=FakeClock())
    with caplog.at_level(logging.WARNING):
        assert watch.changed() is False
    assert "head refused" in caplog.records[-1].error  # type: ignore[attr-defined]


def test_a_local_warehouse_is_never_checked(tmp_path: Path) -> None:
    """Nothing in this repository rewrites a local warehouse underneath a reader."""
    obj, clock = FakeObject('"one"'), FakeClock()
    watch = WarehouseWatch(tmp_path / "meta.duckdb", head=obj.head, clock=clock)
    assert watch.changed() is False
    assert obj.heads == 0


def test_a_replaced_warehouse_drops_the_agent_over_it(tmp_path: Path) -> None:
    """The agent holds an open connection to the old file, so it goes with the file."""

    class Holder:
        current: object | None = "an agent built over the old warehouse"

    class App:
        class state:  # noqa: N801 - mirrors Starlette's `app.state`
            agent = Holder()

    obj, clock = FakeObject('"one"'), FakeClock()
    watch = WarehouseWatch(tmp_path / "meta.duckdb", head=obj.head, clock=clock)
    assert refresh_warehouse(App, watch) is False
    assert App.state.agent.current is not None

    watch = watch_over(obj, clock)
    watch.changed()
    obj.etag = '"two"'
    clock.now = 601.0
    assert refresh_warehouse(App, watch) is True
    assert App.state.agent.current is None


# ---------------------------------------------------------- one invocation --


def test_the_handler_answers_health_for_a_function_url_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One request all the way through Mangum, the real application and back.

    `PRA_SERVE_STUB_MODEL` is the same switch a demonstration uses, so the
    model is `pipeline.serve.StubPredictor` and nothing here needs a registry,
    MLflow, LightGBM or a network. The agent is never built: it is built on the
    first question, and this asks none.
    """
    monkeypatch.setenv(STUB_MODEL_VAR, "1")
    response = handler(function_url_event("GET", "/health"), None)
    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok", "model_loaded": True}

    # And the model the stub loaded is named as a stub on `/model`, which is
    # the same application object answering a second invocation.
    described = handler(function_url_event("GET", "/model"), None)
    assert json.loads(described["body"])["version"] == STUB_VERSION


def test_a_secret_that_cannot_be_read_fails_the_invocation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One log line naming the failure, then out; never a 503 with no reason."""

    def broken(arn: str) -> dict[str, str]:
        raise SecretError("the secret is not readable")

    monkeypatch.setenv(KEYS_SECRET_VAR, "a-secret-under-test")
    monkeypatch.setattr(lambda_serve, "read_secret", broken)
    with caplog.at_level(logging.ERROR), pytest.raises(SecretError):
        handler(function_url_event("GET", "/health"), None)
    assert "could not start" in caplog.text


def test_importing_the_handler_loads_no_model_and_no_agent() -> None:
    """The import graph is the application's, and the heavy halves of it are lazy.

    torch is most of this image and MLflow is the rest; both are imported
    inside a function, so a container that only ever answers `/health` pays for
    neither. A subprocess because `sys.modules` in this one is already full of
    everything the suite has touched.
    """
    code = (
        "import pipeline.lambda_serve, sys; "
        "print(','.join(n for n in ('torch', 'sentence_transformers', 'langchain', "
        "'mlflow', 'pyspark', 'lightgbm') if n in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
