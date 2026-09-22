"""Source-side settings for bronze ingest: which bucket, which prefix, which queue, which key.

Separate from `config`, which holds only the output locations the whole pipeline
shares. Everything here is about reading the application's private bucket, so it
belongs to the stage that reads it and is resolved once, at the start of a run,
rather than read from the environment deep inside the loop.

`from_env` reports every missing variable in one message. A half-configured
environment is the normal failure (a new machine, a forgotten `op run`), and
fixing it one variable per run is three runs instead of one.

The HMAC key is held as bytes and kept out of `repr`, so a settings object in a
log line, a traceback or a debugger never shows it.

`AWS_PROFILE` is not read here on purpose: boto3 picks it up from the
environment itself, and duplicating it would mean two places that decide which
credentials a run uses.
"""

import os
from dataclasses import dataclass, field
from typing import Final

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PREFIX: Final = "parsed/"
DEFAULT_REGION: Final = "us-west-2"

BUCKET_VAR: Final = "PRA_BUCKET"
PREFIX_VAR: Final = "PRA_PREFIX"
REGION_VAR: Final = "AWS_REGION"
KEY_VAR: Final = "HANDLE_HMAC_KEY"
QUEUE_VAR: Final = "PRA_QUEUE_URL"


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

    @classmethod
    def from_env(cls, *, require_bucket: bool = True, require_queue: bool = False) -> "Settings":
        """Read the source settings, naming every missing variable at once.

        `require_bucket` is false only when the run reads a local directory
        instead of S3 (`--source-dir`): there is no bucket to name, and demanding
        one would make a run that never touches AWS depend on AWS configuration.
        The anonymization key is required either way, because a local run writes
        the same bronze rows as any other and they are anonymized the same way.

        `require_queue` is true only for the event consumer
        (`python -m pipeline.consume`), the one command that reads the queue. The
        backfill runs without it, so a machine that only ever backfills is not
        asked for a queue that may not exist yet.
        """
        bucket = os.environ.get(BUCKET_VAR, "").strip()
        key = os.environ.get(KEY_VAR, "")
        queue_url = os.environ.get(QUEUE_VAR, "").strip()
        required = [(KEY_VAR, key)]
        if require_bucket:
            required.insert(0, (BUCKET_VAR, bucket))
        if require_queue:
            required.append((QUEUE_VAR, queue_url))
        missing = [name for name, value in required if not value]
        if missing:
            raise SettingsError(missing)
        return cls(
            bucket=bucket,
            hmac_key=key.encode("utf-8"),
            prefix=os.environ.get(PREFIX_VAR) or DEFAULT_PREFIX,
            region=os.environ.get(REGION_VAR) or DEFAULT_REGION,
            queue_url=queue_url,
        )
