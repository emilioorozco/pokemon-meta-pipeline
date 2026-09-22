"""Source config for the deprecated Kaggle corpus stage.

The corpus was external to the repo and reached through the SOURCE_DATA_DIR
environment variable. There is deliberately no default: an unset variable is an
error, never a silent fallback to someone's laptop layout. Paths are resolved
lazily so importing the package never requires the corpus to be present.
"""

import os
from pathlib import Path


class SourceNotConfigured(RuntimeError):
    """SOURCE_DATA_DIR is unset or empty."""


def source_data_dir() -> Path:
    """The root of the raw corpus. Raises SourceNotConfigured if unset."""
    value = os.environ.get("SOURCE_DATA_DIR", "").strip()
    if not value:
        raise SourceNotConfigured(
            "SOURCE_DATA_DIR is not set. Copy .env.example to .env and point it at the "
            "directory holding data/replays/corpus* and data/EN_Card_Data.csv."
        )
    return Path(value).expanduser()


def replay_batches() -> dict[str, Path]:
    """Source replay folders, keyed by batch name (read-only inputs)."""
    root = source_data_dir()
    return {
        "corpus": root / "data" / "replays" / "corpus",
        "corpus2": root / "data" / "replays" / "corpus2",
    }


def card_data_csv() -> Path:
    return source_data_dir() / "data" / "EN_Card_Data.csv"


def validate_source() -> list[str]:
    """Return a list of problems with the source inputs (empty = all present)."""
    try:
        inputs = {**replay_batches(), "card_csv": card_data_csv()}
    except SourceNotConfigured as e:
        return [str(e)]
    return [f"{name}: {path}" for name, path in inputs.items() if not path.exists()]
