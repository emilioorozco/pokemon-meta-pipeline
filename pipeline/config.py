"""Central config: where the pipeline writes.

Only output locations live here. Source-side settings (the S3 bucket and prefix
of the parsed-game blobs, the anonymization key) belong to the stage that reads
them and are documented in .env.example. Everything is relative to the repo
unless PIPELINE_DATA_DIR overrides it, so a fresh clone runs with no setup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# pipeline outputs (relative to the repo unless overridden)
PIPELINE_DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", REPO_ROOT / "data"))
LAKE_DIR = PIPELINE_DATA_DIR / "lake"
BRONZE_DIR = LAKE_DIR / "bronze"
SILVER_DIR = LAKE_DIR / "silver"
# Rejected blobs are kept as received, so this directory stays local and gitignored.
QUARANTINE_DIR = LAKE_DIR / "quarantine"
# The card catalog is an export of the client's card database, downloaded by
# scripts/fetch_catalog.py. It is reference data, not lake output, but it is
# large and not ours to redistribute, so it lives under the gitignored data dir.
CATALOG_PATH = PIPELINE_DATA_DIR / "catalog" / "cards.json"
WAREHOUSE_PATH = PIPELINE_DATA_DIR / "warehouse" / "meta.duckdb"
