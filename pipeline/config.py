"""Central config: where the raw corpus lives and where the pipeline writes.

The raw corpus is external to this repo (it belongs to the original Kaggle
project). Everything reads it through SOURCE_DATA_DIR so the pipeline has a
single, explicit dependency on the source system — swap the path (or an S3
URI later) and nothing else changes.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATA_DIR = Path(
    os.environ.get("SOURCE_DATA_DIR", "/Users/emiloroz/workplace/pokemon-kaggle")
)
PIPELINE_DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", REPO_ROOT / "data"))

# source inputs (read-only)
REPLAY_BATCHES = {
    "corpus": SOURCE_DATA_DIR / "data" / "replays" / "corpus",
    "corpus2": SOURCE_DATA_DIR / "data" / "replays" / "corpus2",
}
CARD_DATA_CSV = SOURCE_DATA_DIR / "data" / "EN_Card_Data.csv"

# pipeline outputs
LAKE_DIR = PIPELINE_DATA_DIR / "lake"
WAREHOUSE_PATH = PIPELINE_DATA_DIR / "warehouse" / "meta.duckdb"


def validate_source() -> list[str]:
    """Return a list of missing source inputs (empty = all present)."""
    missing = []
    for name, path in {**REPLAY_BATCHES, "card_csv": CARD_DATA_CSV}.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    return missing
