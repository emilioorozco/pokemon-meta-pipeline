"""Output locations: repo-relative by default, PIPELINE_DATA_DIR overrides the root."""

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from pipeline import config


@pytest.fixture
def reloaded_config(tmp_path: Path) -> Iterator[Path]:
    """Re-import config with PIPELINE_DATA_DIR set, and restore the module afterwards."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PIPELINE_DATA_DIR", str(tmp_path))
        importlib.reload(config)
        yield tmp_path
    # env is restored once the context exits; re-import so later tests see the defaults
    importlib.reload(config)


def test_defaults_are_repo_relative() -> None:
    assert Path(__file__).resolve().parent.parent == config.REPO_ROOT
    assert (config.REPO_ROOT / "pyproject.toml").is_file()
    root = config.PIPELINE_DATA_DIR
    assert (
        root / "lake",
        root / "lake" / "bronze",
        root / "lake" / "silver",
        root / "catalog" / "cards.json",
        root / "warehouse" / "meta.duckdb",
    ) == (
        config.LAKE_DIR,
        config.BRONZE_DIR,
        config.SILVER_DIR,
        config.CATALOG_PATH,
        config.WAREHOUSE_PATH,
    )


def test_env_override_moves_every_derived_path(reloaded_config: Path) -> None:
    root = reloaded_config
    assert (
        root,
        root / "lake" / "bronze",
        root / "lake" / "silver",
        root / "catalog" / "cards.json",
        root / "warehouse" / "meta.duckdb",
    ) == (
        config.PIPELINE_DATA_DIR,
        config.BRONZE_DIR,
        config.SILVER_DIR,
        config.CATALOG_PATH,
        config.WAREHOUSE_PATH,
    )
