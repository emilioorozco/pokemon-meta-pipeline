"""Download the card catalog the silver stage joins against.

The catalog is `catalog/cards.json` in the application's bucket: about 25,000
client card ids mapped to name, set, number, type, hit points and regulation
mark. It is reference data the producer maintains, so it is fetched rather than
committed, and it is not ours to redistribute.

    uv run python scripts/fetch_catalog.py
"""

import boto3

from pipeline.config import CATALOG_PATH
from pipeline.settings import Settings

CATALOG_KEY = "catalog/cards.json"


def main() -> int:
    """Fetch the catalog into the configured path and report its size."""
    settings = Settings.from_env()
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3", region_name=settings.region).download_file(
        settings.bucket, CATALOG_KEY, str(CATALOG_PATH)
    )
    print(f"wrote {CATALOG_PATH} ({CATALOG_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
