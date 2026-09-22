"""Shared fixtures for the tests that need a Java Virtual Machine or a real bronze lake.

Both are session scoped because both are expensive: a SparkSession costs a few
seconds of Java Virtual Machine (JVM) startup, and the bronze lake costs a full
backfill over the committed games. Nothing in the silver tests writes to either,
so one of each serves the whole run.

The bronze lake is built by running the real backfill over `tests/fixtures` with
`LocalSource`, the same code path `--source-dir` takes, rather than by writing
Parquet by hand. That way the silver tests read the bronze a real run produces,
and a bronze schema change reaches them.
"""

import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from pipeline.backfill import run_backfill
from pipeline.settings import Settings
from pipeline.source import LocalSource

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

FIXTURES_DIR: Final = Path(__file__).parent / "fixtures"
CATALOG_PATH: Final = Path(__file__).parent / "catalog.json"
# Not a secret: the fixtures are already anonymized under a key nobody kept, so
# this one only has to be stable inside a test run.
TEST_HMAC_KEY: Final = b"tests-only-key-not-a-real-secret"
INGESTED_AT: Final = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def spark() -> Iterator["SparkSession"]:
    """A local SparkSession for the whole test session.

    `local[2]` keeps two cores busy without oversubscribing a continuous
    integration runner, the web user interface is off because nothing looks at
    it, and the driver host is pinned to the loopback address so a machine whose
    hostname does not resolve (a container, a laptop off the network) still
    starts a session.
    """
    pyspark_sql = pytest.importorskip("pyspark.sql", reason="pyspark is not installed")
    session = (
        pyspark_sql.SparkSession.builder.master("local[2]")
        .appName("pra-silver-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def bronze_from_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The committed games run through the real backfill into a temporary bronze lake."""
    root = tmp_path_factory.mktemp("lake")
    bronze_dir = root / "bronze"
    settings = Settings(bucket="", hmac_key=TEST_HMAC_KEY)
    summary = run_backfill(
        settings,
        bronze_dir=bronze_dir,
        quarantine_dir=root / "quarantine",
        source=LocalSource(FIXTURES_DIR),
        now=INGESTED_AT,
    )
    assert summary.landed == summary.read, summary
    return bronze_dir


@pytest.fixture
def scratch_bronze(bronze_from_fixtures: Path, tmp_path: Path) -> Path:
    """A writable copy of the fixture bronze, for a test that adds a partition of its own."""
    copy = tmp_path / "bronze"
    shutil.copytree(bronze_from_fixtures, copy)
    return copy
