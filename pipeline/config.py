"""Central config: where the pipeline writes.

Only output locations live here. Source-side settings (the S3 bucket and prefix
of the parsed-game blobs, the anonymization key) belong to the stage that reads
them and are documented in .env.example. Everything is relative to the repo
unless PIPELINE_DATA_DIR overrides it, so a fresh clone runs with no setup.

PIPELINE_DATA_DIR takes an `s3://bucket/prefix` as readily as a directory, and
every path below is derived from it either way, so the layout under an S3 prefix
is the layout under `data/` and only the root moves. `pipeline.storage` is what
makes that true; its docstring has the reasoning, and `docs/stages.md` has the
operational half. The derived names are `Location` rather than `Path`, and a
local `Location` compares equal to the `Path` it wraps, so a caller that already
held a path keeps working unchanged.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.storage import Location, location

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# pipeline outputs (relative to the repo unless overridden)
PIPELINE_DATA_DIR: Location = location(os.environ.get("PIPELINE_DATA_DIR") or REPO_ROOT / "data")
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
# The regulation marks legal in the Standard format, which is the format this
# corpus is about: the card-text fetch keeps only printings carrying one of
# these, so the retriever holds a few thousand cards a player can actually meet
# rather than every printing since 2011. Rotation retires the oldest mark each
# spring; this tuple is the one place to bump when it does.
STANDARD_REGULATION_MARKS: tuple[str, ...] = ("H", "I", "J")
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
    """`MLFLOW_TRACKING_URI` when it is set, else the store under the data dir.

    Read at call time rather than at import, because the tests and the command
    line both set the variable after this module is first imported.

    A local data dir gives `file:<path>`, which is what it always gave. An S3
    data dir gives the `s3://` prefix itself, which MLflow cannot use directly;
    `pipeline.storage.tracking_store` turns it into a `file:` URI over a synced
    temporary directory for the length of the command, and says there why.
    """
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    return str(MLRUNS_DIR) if MLRUNS_DIR.is_s3 else f"file:{MLRUNS_DIR}"
