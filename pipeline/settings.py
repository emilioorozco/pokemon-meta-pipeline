"""Settings for the stages that talk to the application: bucket, prefix, queue, key, table.

Separate from `config`, which holds only the output locations the whole pipeline
shares. Everything here is about the application's own AWS resources, so it
belongs to the stages that read and write them and is resolved once, at the
start of a run, rather than read from the environment deep inside the loop. Four
of the five are the read side, the private bucket bronze is built from; the
fifth, `PRA_INSIGHTS_TABLE`, is the write side, the one table
`python -m pipeline.publish` puts the marts back into.

`from_env` reports every missing variable in one message. A half-configured
environment is the normal failure (a new machine, a forgotten `op run`), and
fixing it one variable per run is three runs instead of one.

The HMAC key is held as bytes and kept out of `repr`, so a settings object in a
log line, a traceback or a debugger never shows it.

`AWS_PROFILE` is not read here on purpose: boto3 picks it up from the
environment itself, and duplicating it would mean two places that decide which
credentials a run uses.

`validate_data_root` is the one thing here that is about the pipeline's own
output rather than the application's resources. It sits with the other
environment checks because it answers the same question they do, "is this
machine configured to run", and because the failure it catches is the same
shape: a variable that was interpolated empty.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from pipeline.storage import Location, StorageError, location

load_dotenv()

DEFAULT_PREFIX: Final = "parsed/"
DEFAULT_REGION: Final = "us-west-2"

BUCKET_VAR: Final = "PRA_BUCKET"
PREFIX_VAR: Final = "PRA_PREFIX"
REGION_VAR: Final = "AWS_REGION"
KEY_VAR: Final = "HANDLE_HMAC_KEY"
QUEUE_VAR: Final = "PRA_QUEUE_URL"
INSIGHTS_TABLE_VAR: Final = "PRA_INSIGHTS_TABLE"


DATA_DIR_VAR: Final = "PIPELINE_DATA_DIR"


class DataRootError(RuntimeError):
    """`PIPELINE_DATA_DIR`, or a `--data-dir`, that cannot name a lake."""


def validate_data_root(value: "str | Path | Location") -> Location:
    """The lake root as a `Location`, refusing an `s3://` root with no bucket.

    Here rather than in `pipeline.storage` because this is the module that turns
    a badly configured environment into one readable message, and an `s3://`
    with no bucket is exactly that kind of mistake: `s3:///lake` and
    `s3://` are what a shell produces when a variable it interpolated was empty,
    and both would otherwise read as a relative directory called `s3:` and land
    a production lake in the working directory.

    Only the shape is checked. Whether the bucket exists, and whether this run
    may write to it, are questions with one honest answer each and both of them
    come from AWS on the first call, not from a guess here.
    """
    try:
        return location(value)
    except StorageError as bad:
        raise DataRootError(f"{DATA_DIR_VAR}: {bad}") from bad


class SettingsError(RuntimeError):
    """The environment does not describe a usable source; `missing` names every variable."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__(
            "missing or empty environment variable(s): "
            + ", ".join(missing)
            + " (see .env.example)"
        )
        self.missing = list(missing)


@dataclass(frozen=True)
class Settings:
    """Where the parsed blobs are, which queue announces them, and what anonymizes them."""

    bucket: str
    hmac_key: bytes = field(repr=False)
    prefix: str = DEFAULT_PREFIX
    region: str = DEFAULT_REGION
    queue_url: str = ""
    insights_table: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        require_bucket: bool = True,
        require_queue: bool = False,
        require_key: bool = True,
        require_insights_table: bool = False,
    ) -> "Settings":
        """Read the source settings, naming every missing variable at once.

        `require_bucket` is false only when the run reads a local directory
        instead of S3 (`--source-dir`): there is no bucket to name, and demanding
        one would make a run that never touches AWS depend on AWS configuration.

        `require_queue` is true only for the event consumer
        (`python -m pipeline.consume`), the one command that reads the queue. The
        backfill runs without it, so a machine that only ever backfills is not
        asked for a queue that may not exist yet.

        `require_key` is true for everything that writes bronze, whether it read
        S3 or a local directory, because a local run writes the same rows as any
        other and they are anonymized the same way. It is false for the one
        command downstream of bronze that needs settings at all
        (`python -m pipeline.publish`), which reads a warehouse whose handles
        were replaced with tokens several stages ago.

        `require_insights_table` is true only for that publish, and only when it
        was not given a table on the command line.
        """
        bucket = os.environ.get(BUCKET_VAR, "").strip()
        key = os.environ.get(KEY_VAR, "")
        queue_url = os.environ.get(QUEUE_VAR, "").strip()
        insights_table = os.environ.get(INSIGHTS_TABLE_VAR, "").strip()
        required = []
        if require_bucket:
            required.append((BUCKET_VAR, bucket))
        if require_key:
            required.append((KEY_VAR, key))
        if require_queue:
            required.append((QUEUE_VAR, queue_url))
        if require_insights_table:
            required.append((INSIGHTS_TABLE_VAR, insights_table))
        missing = [name for name, value in required if not value]
        if missing:
            raise SettingsError(missing)
        return cls(
            bucket=bucket,
            hmac_key=key.encode("utf-8"),
            prefix=os.environ.get(PREFIX_VAR) or DEFAULT_PREFIX,
            region=os.environ.get(REGION_VAR) or DEFAULT_REGION,
            queue_url=queue_url,
            insights_table=insights_table,
        )
