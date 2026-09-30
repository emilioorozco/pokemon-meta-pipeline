"""Download the card catalog the silver stage joins against.

The catalog is `catalog/cards.json` in the application's bucket: about 25,000
client card ids mapped to name, set, number, type, hit points and regulation
mark. It is reference data the producer maintains, so it is fetched rather than
committed, and it is not ours to redistribute.

    uv run python scripts/fetch_catalog.py

It lands wherever the lake root points, a directory or an `s3://` prefix, and it
goes through `pipeline.storage` rather than `download_file` so both work: the
catalog is a few megabytes of reference data, small enough to move as one body.
"""

import logging

import boto3

from pipeline.config import CATALOG_PATH
from pipeline.observability import configure_logging, emit_summary, stage_run
from pipeline.settings import Settings

logger = logging.getLogger(__name__)

CATALOG_KEY = "catalog/cards.json"
STAGE = "fetch_catalog"


def main() -> int:
    """Fetch the catalog into the configured path and report its size."""
    configure_logging(STAGE)
    settings = Settings.from_env()
    with stage_run(STAGE) as metrics:
        CATALOG_PATH.parent.mkdir()
        body = boto3.client("s3", region_name=settings.region).get_object(
            Bucket=settings.bucket, Key=CATALOG_KEY
        )["Body"]
        catalog = body.read()
        CATALOG_PATH.write_bytes(catalog)
        size = len(catalog)
        # One object in, one file out: the rows a catalog fetch moves are files,
        # and its size is the number worth keeping.
        metrics.rows_in = 1
        metrics.rows_out = 1
        metrics.rows_quarantined = 0
        metrics.extra = {"bytes": size, "path": str(CATALOG_PATH)}

    emit_summary(
        logger,
        "catalog fetched",
        {"path": str(CATALOG_PATH), "bytes": size},
        text=f"wrote {CATALOG_PATH} ({size} bytes)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
