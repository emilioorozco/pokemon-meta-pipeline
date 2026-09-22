"""Download the card catalog the silver stage joins against.

The catalog is `catalog/cards.json` in the application's bucket: about 25,000
client card ids mapped to name, set, number, type, hit points and regulation
mark. It is reference data the producer maintains, so it is fetched rather than
committed, and it is not ours to redistribute.

    uv run python scripts/fetch_catalog.py
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
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3", region_name=settings.region).download_file(
            settings.bucket, CATALOG_KEY, str(CATALOG_PATH)
        )
        size = CATALOG_PATH.stat().st_size
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
