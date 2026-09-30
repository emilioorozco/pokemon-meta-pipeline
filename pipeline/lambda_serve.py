"""The serving application as a Lambda: the same FastAPI app behind a function URL.

`pipeline.serve.create_app` unchanged, wrapped by Mangum so that an HTTP event
from a Lambda function URL becomes an ASGI request and the response becomes an
event back. Every endpoint the uvicorn process answers is answered here:
`/health`, `/model`, `/reload`, `/predict` and `/ask`. Nothing in the
command-line path changes, and `python -m pipeline.serve` on a laptop under
`.env.op` behaves exactly as it did.

Why a Lambda at all, when `Dockerfile.serve` already containerizes the same
app: the application repository wants `/ask` without a laptop, and the traffic
is a handful of questions a day. A task that is up all month to answer them
costs a task all month; a function costs the seconds it runs. The price is a
cold start, which is measured in `docs/agent-service.md` rather than guessed
at, and the concurrency ceiling, which is deliberate and set on the function.

**What this module adds to the app, and nothing else.**

*The keys.* `ANTHROPIC_API_KEY` (the agent's provider) and `JEV_API_KEY` (the
SQL gate's judge) are not function environment variables, because anyone who
can call `GetFunction` reads those. `AGENT_KEYS_SECRET_ARN` names a Secrets
Manager secret whose string is one JSON object holding both, it is read once
per execution environment and kept, and each value is put into `os.environ`
only when that variable is not already set. A key already in the environment
therefore wins, which is what keeps a local run under `.env.op` and the tests
free of any AWS call. Neither value is ever logged; the log line names which
variables were filled and nothing more.

*The scratch root.* `/tmp` is the only writable path on Lambda, and the two
helpers that need one, `pipeline.storage.local_file` and `synced_dir`, both go
through `tempfile`, which honours `TMPDIR`. Lambda sets `TMPDIR=/tmp`, so the
warehouse copy and the synced MLflow store land there with nothing configured.
The one thing that does not go through `tempfile` is DuckDB, which installs
`httpfs` and `aws` under `$HOME/.duckdb` the first time it reads an `s3://`
view; `Dockerfile.agent` bakes both extensions into the image and sets
`PRA_DUCKDB_EXTENSION_CACHE`, and `prime_duckdb_extensions` copies them into
`$HOME` on the first invocation so a cold start never fetches an extension.

*The refresh rule.* `local_file` downloads the warehouse once per process and
keeps it, on the assumption that nothing rewrites the object underneath a
running process. That assumption holds for a command that ends and does not
hold for a container that may live for hours across a nightly rebuild. So the
warehouse is kept for the life of the container, and at most once every
`REFRESH_SECONDS` the object's ETag is checked with one `HeadObject`; when it
has changed the local copy is dropped and the agent is discarded, and the next
question rebuilds both over the new file.

The trade-off is deliberate and it is staleness against cost. Ten minutes means
a container that was warm when the nightly landed can answer from yesterday's
warehouse for up to ten more minutes, which for a metagame summary that moves
once a day is invisible. Checking on every invocation would cost a HeadObject
per question to detect a change that happens once a day; never checking would
mean a long-lived container answering from a warehouse that no longer exists.
A head that fails is logged and treated as no change: a transient S3 error
should not throw away a working warehouse, and the next check is ten minutes
away.

*Failing loudly.* A secret that cannot be read, or a lake that cannot be
reached while the app is being built, raises out of the handler with one
`exception` log line naming the stage. It is not caught and it is not retried
in place: an invocation that fails is visible in the function's error metric
and in the caller's response, and an invocation that hung would be visible only
as a timeout a minute later.

The app is built on the first invocation rather than at import. Lambda gives an
image's init phase about ten seconds and then re-runs the work inside the
invocation, and building this app can take longer than that on a cold container
because it pulls the MLflow store out of the lake. Doing it in the handler
means the whole 60 s function timeout is available for it and the failure, if
there is one, is reported against a request rather than against an init.
"""

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from pipeline.config import WAREHOUSE_PATH
from pipeline.observability import configure_logging
from pipeline.serve import STAGE, create_app, mlflow_loader, stub_loader, stub_requested
from pipeline.settings import REGION_VAR
from pipeline.storage import AnyLocation, Location, forget_download, location, s3_client

logger = logging.getLogger(__name__)

KEYS_SECRET_VAR: Final = "AGENT_KEYS_SECRET_ARN"
# The two variables the secret may fill, and the only two read out of it. Both
# are the names the code already reads: `pipeline.agent.API_KEY_VAR` and
# `pipeline.sql_gate.API_KEY_VAR`. Named as strings rather than imported, so
# this module does not import LangChain to learn a variable name.
SECRET_KEYS: Final[tuple[str, ...]] = ("ANTHROPIC_API_KEY", "JEV_API_KEY")
# Where `Dockerfile.agent` baked the DuckDB extensions, as a directory that is
# copied into `$HOME`. Unset outside the image, where DuckDB installs its own.
EXTENSION_CACHE_VAR: Final = "PRA_DUCKDB_EXTENSION_CACHE"
DUCKDB_HOME: Final = ".duckdb"
# How long a downloaded warehouse is trusted before its ETag is checked again.
REFRESH_SECONDS: Final = 600.0

# Held for the life of the execution environment, which is the whole point of
# one: a container that answers a thousand questions must not read the secret,
# build the application and pull the registry a thousand times.
# `reset_container_state` is for the tests.
_adapter: Any | None = None
_keys_loaded: bool = False
_watch: "WarehouseWatch | None" = None


class SecretError(RuntimeError):
    """The named secret holds nothing this function can use as provider keys."""


# ------------------------------------------------------------ the secret --


def load_keys() -> list[str]:
    """Fill the provider keys from Secrets Manager, once, without overwriting.

    Returns the variable names this call set, which is what the log line says
    and what the tests assert on. A variable already in the environment is left
    alone, so `.env.op` on a laptop and a monkeypatched key in a test both win
    over the secret and neither needs AWS.

    No secret named is not an error: it is how the image runs locally and under
    the runtime interface emulator. The function's own environment always names
    one.
    """
    global _keys_loaded
    if _keys_loaded:
        return []
    arn = os.environ.get(KEYS_SECRET_VAR, "").strip()
    if not arn:
        _keys_loaded = True
        logger.info(
            "no provider-key secret named; the environment is taken as it is",
            extra={"variable": KEYS_SECRET_VAR},
        )
        return []
    values = read_secret(arn)
    filled = [name for name in SECRET_KEYS if values.get(name) and not os.environ.get(name)]
    for name in filled:
        os.environ[name] = values[name]
    _keys_loaded = True
    # Names only, on both fields. A value from this secret is never written to
    # a log, a span, an error message or a response body.
    logger.info(
        "provider keys read from Secrets Manager",
        extra={"filled": filled, "offered": sorted(values)},
    )
    return filled


def read_secret(arn: str) -> dict[str, str]:
    """The secret as a flat mapping of variable name to value; values never logged.

    The contract with the application's stack is one JSON object,
    `{"ANTHROPIC_API_KEY": "...", "JEV_API_KEY": "..."}`. Anything else is
    refused here rather than turned into a missing key that only shows up as a
    503 from `/ask` an hour later. The failure messages name the variable and
    the shape, never the string that was read.
    """
    import boto3

    # Untyped for the reason `pipeline.lambda_consumer` gives: the boto3 stubs
    # this project installs cover S3, SQS and DynamoDB, and a fourth service
    # package for one call is a dependency for a line of code.
    client: Any = boto3.client("secretsmanager", region_name=os.environ.get(REGION_VAR) or None)
    raw = client.get_secret_value(SecretId=arn).get("SecretString")
    if not isinstance(raw, str) or not raw:
        raise SecretError(f"{KEYS_SECRET_VAR} names a secret with no string value")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as failure:
        raise SecretError(
            f"{KEYS_SECRET_VAR} names a secret whose string is not JSON; it has to be an "
            f"object of {', '.join(SECRET_KEYS)}"
        ) from failure
    if not isinstance(parsed, dict):
        raise SecretError(f"{KEYS_SECRET_VAR} names a secret that is not a JSON object")
    return {str(name): value for name, value in parsed.items() if isinstance(value, str) and value}


# ------------------------------------------------------- the warehouse copy --


class WarehouseWatch:
    """Whether the warehouse object has been replaced since it was downloaded.

    One `HeadObject` at most every `interval` seconds, and the answer is
    remembered in between, so a burst of questions costs one check rather than
    one per question. `head` is injectable so the tests can drive the whole rule
    with a fake object and no S3 at all; `clock` is injectable so they do not
    have to wait ten minutes to see the second check happen.

    A local warehouse never changes underneath this process by any path this
    repository has, so a local location answers False without asking anything.
    """

    def __init__(
        self,
        target: AnyLocation,
        *,
        interval: float = REFRESH_SECONDS,
        head: Callable[[Location], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target = location(target)
        self.interval = interval
        self._head = head or _head_etag
        self._clock = clock
        self._checked_at: float | None = None
        self._etag: str | None = None

    def changed(self) -> bool:
        """True when the object is not the one the local copy came from.

        False on the first call, which records the tag the copy was taken with,
        and False inside the interval, and False when the head itself failed.
        """
        if not self.target.is_s3:
            return False
        now = self._clock()
        if self._checked_at is not None and now - self._checked_at < self.interval:
            return False
        self._checked_at = now
        try:
            tag = self._head(self.target)
        except Exception as failure:
            # Kept rather than dropped: a working warehouse is worth more than
            # a fresh one, and the next check is one interval away.
            logger.warning(
                "the warehouse could not be checked for a new version; keeping the copy in hand",
                extra={"error": f"{type(failure).__name__}: {failure}"},
            )
            return False
        if tag is None or self._etag is None:
            self._etag = tag
            return False
        if tag == self._etag:
            return False
        self._etag = tag
        return True


def _head_etag(target: Location) -> str | None:
    """The object's ETag, or None when the head answered without one."""
    response = s3_client().head_object(Bucket=target.bucket, Key=target.key)
    tag = response.get("ETag")
    return str(tag) if tag else None


def refresh_warehouse(app: Any, watch: WarehouseWatch) -> bool:
    """Drop the downloaded warehouse and the agent over it when it has been replaced.

    The agent goes with the file because it holds an open DuckDB connection to
    it; `pipeline.serve.AgentHolder` builds a new one on the next question, over
    the copy the next `local_file` downloads. The old file is unlinked, so a
    container that sees a week of nightlies does not fill its 2 GB of `/tmp`
    with warehouses; a question that is mid-flight keeps reading the copy it
    already opened, because an unlinked file on Linux stays readable.
    """
    if not watch.changed():
        return False
    forget_download(watch.target)
    holder = getattr(app.state, "agent", None)
    if holder is not None:
        holder.current = None
    logger.info("the warehouse was replaced; the next question rebuilds over the new one")
    return True


# ------------------------------------------------------------ the container --


def prime_duckdb_extensions() -> None:
    """Put the image's baked DuckDB extensions where DuckDB looks for them.

    DuckDB installs into `$HOME/.duckdb` on first use of `httpfs`, which under
    Lambda means either an unwritable path or a download on the critical path of
    the first question. `Dockerfile.agent` installs them at build time into the
    directory `PRA_DUCKDB_EXTENSION_CACHE` names and sets `HOME=/tmp`; this
    copies the one into the other, once, and does nothing at all when the
    variable is unset, which is every run outside the image.
    """
    cache = os.environ.get(EXTENSION_CACHE_VAR, "").strip()
    home = os.environ.get("HOME", "").strip()
    if not cache or not home:
        return
    source = Path(cache)
    target = Path(home) / DUCKDB_HOME
    if target.exists() or not source.is_dir():
        return
    shutil.copytree(source, target)
    logger.info("duckdb extensions primed from the image", extra={"target": str(target)})


def build_app() -> Any:
    """The serving application, configured the way the command line configures it.

    The same two loaders `pipeline.serve.main` chooses between, so that
    `PRA_SERVE_STUB_MODEL` means the same thing in a function as it does on a
    laptop and a demonstration container needs no registry. Everything else the
    app needs it reads for itself out of `PIPELINE_DATA_DIR`.
    """
    loader = stub_loader() if stub_requested() else mlflow_loader()
    return create_app(loader)


def container() -> tuple[Any, WarehouseWatch]:
    """This execution environment's adapter and warehouse watch, built once.

    The order matters: logging first so everything after it is one JSON line
    per record, then the keys, so the agent's provider client finds them
    whenever it is built, then the application.
    """
    global _adapter, _watch
    if _adapter is None or _watch is None:
        configure_logging(STAGE)
        prime_duckdb_extensions()
        load_keys()
        app = build_app()
        _watch = WarehouseWatch(WAREHOUSE_PATH)
        _adapter = _mangum(app)
    return _adapter, _watch


def _mangum(app: Any) -> Any:
    """The ASGI adapter. Imported here so the module imports without Mangum.

    `lifespan="off"` because this application defines no startup or shutdown
    handlers: the model is loaded inside `create_app` and the agent is built on
    the first question, so running a lifespan would be a protocol exchange with
    nothing on the other end of it.
    """
    from mangum import Mangum

    return Mangum(app, lifespan="off")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """One function-URL request through the serving application.

    The Lambda entry point, referenced by the image's `CMD`. `event` is a
    function URL payload (the API Gateway HTTP API v2 shape) and `context` is
    the runtime's; Mangum reads both and neither is inspected here.
    """
    try:
        adapter, watch = container()
        refresh_warehouse(adapter.app, watch)
    except Exception:
        # One line, then out. A caller that gets a 500 and an operator reading
        # the function's errors both need the reason, and an invocation that
        # swallowed this would answer 503 from every endpoint for the life of
        # the container without ever saying why.
        logger.exception("the agent function could not start")
        raise
    result: dict[str, Any] = adapter(event, context)
    return result


def reset_container_state() -> None:
    """Forget the adapter, the keys and the warehouse watch this container cached.

    For the tests: a Lambda keeps all three on purpose, and a suite that builds
    one application after another in one process must not hand the second the
    first one's.
    """
    global _adapter, _keys_loaded, _watch
    _adapter = None
    _keys_loaded = False
    _watch = None
