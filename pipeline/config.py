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
# One Parquet row per stage per run, written by `pipeline.observability.stage_run`
# and read by the `ops` dbt models. It lives in the lake rather than beside the
# logs because it is a table that gets queried, not a stream that gets tailed.
RUN_METRICS_DIR = LAKE_DIR / "run_metrics"
# The card catalog is an export of the client's card database, downloaded by
# scripts/fetch_catalog.py. It is reference data, not lake output, but it is
# large and not ours to redistribute, so it lives under the gitignored data dir.
CATALOG_PATH = PIPELINE_DATA_DIR / "catalog" / "cards.json"
# The retriever's corpus and its index. `scripts/fetch_card_text.py` writes the
# first from a public card API, `python -m pipeline.card_index build` writes the
# second from it. Both sit beside the catalog because both are reference data
# about cards rather than anything derived from a game.
CARD_TEXT_PATH = PIPELINE_DATA_DIR / "catalog" / "card_text.jsonl"
CARD_INDEX_DIR = PIPELINE_DATA_DIR / "catalog" / "card_index"
WAREHOUSE_PATH = PIPELINE_DATA_DIR / "warehouse" / "meta.duckdb"
# Where training runs and the model registry live when nothing says otherwise.
MLRUNS_DIR = PIPELINE_DATA_DIR / "mlruns"

# The registry vocabulary, shared by the three commands that use it: training
# registers a version, promotion moves an alias, serving loads by that alias.
# One name and two alias strings, in one place, because a typo in any of them
# is a service that loads nothing and says nothing about why.
REGISTERED_MODEL_NAME = "win-probability"
PRODUCTION_ALIAS = "production"
STAGING_ALIAS = "staging"


def default_tracking_uri() -> str:
    """`MLFLOW_TRACKING_URI` when it is set, else a local directory under the data dir.

    Read at call time rather than at import, because the tests and the command
    line both set the variable after this module is first imported.
    """
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    return configured if configured else f"file:{MLRUNS_DIR}"
