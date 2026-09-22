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


@pytest.fixture(scope="session", autouse=True)
def isolated_run_metrics(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point `PIPELINE_DATA_DIR` at a temporary directory for the whole test session.

    Autouse and unconditional, because any test that calls a stage's `main`
    writes a `run_metrics` row, and the default location for that row is the
    repository's own `data/` directory. One forgotten fixture would mean a test
    run quietly appending to a developer's lake.

    Session scoped rather than per test, and with its own `MonkeyPatch` because
    the built-in one is not: a module-scoped fixture such as `test_promote`'s
    registry is set up before any function-scoped fixture, and it trains a real
    model, so a per-test patch would come too late to catch it.

    Only the run-metrics path moves. `pipeline.config` resolved its other paths
    when it was imported, so the tests that pass explicit directories keep
    passing them, and `pipeline.observability.run_metrics_dir` is the one place
    that re-reads the variable.
    """
    root = tmp_path_factory.mktemp("pipeline-data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PIPELINE_DATA_DIR", str(root))
        yield root


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


@pytest.fixture(scope="session")
def silver_from_fixtures(
    spark: "SparkSession",
    bronze_from_fixtures: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A whole data directory with the fixture bronze and a silver run beside it.

    Returns the root, not the silver directory, because that root is what
    `PIPELINE_DATA_DIR` names: the dbt sources resolve
    `$PIPELINE_DATA_DIR/lake/silver/<table>/**/*.parquet`, so the layout here
    has to be the layout a real run writes. `run_silver` raises on a failed
    reconciliation, so a broken silver build fails the gold tests at setup
    rather than as a wrong number later.
    """
    from pipeline import silver

    root = tmp_path_factory.mktemp("datadir")
    lake = root / "lake"
    shutil.copytree(bronze_from_fixtures, lake / "bronze")
    silver.run_silver(spark, lake / "bronze", lake / "silver", CATALOG_PATH)
    return root


@pytest.fixture(scope="session")
def gold_from_fixtures(silver_from_fixtures: Path) -> Path:
    """One real `dbt run` plus `dbt test` over the fixture silver, once per session.

    Returns the warehouse file the gold tests read and the agent tests query
    through the SQL tool. Session scoped because a dbt build is the most
    expensive thing in the suite and two modules need the same one; the exit
    code is asserted here so a failed build is a setup error rather than a
    dozen confusing assertion failures spread over two files.
    """
    from pipeline.gold import run_gold

    assert run_gold(data_dir=silver_from_fixtures) == 0
    warehouse = silver_from_fixtures / "warehouse" / "meta.duckdb"
    assert warehouse.is_file(), warehouse
    return warehouse


@pytest.fixture
def scratch_bronze(bronze_from_fixtures: Path, tmp_path: Path) -> Path:
    """A writable copy of the fixture bronze, for a test that adds a partition of its own."""
    copy = tmp_path / "bronze"
    shutil.copytree(bronze_from_fixtures, copy)
    return copy
