"""Shared fixtures for the tests that need a Java Virtual Machine, a bronze lake or a bucket.

The first two are session scoped because both are expensive: a SparkSession
costs a few seconds of Java Virtual Machine (JVM) startup, and the bronze lake
costs a full backfill over the committed games. Nothing in the silver tests
writes to either, so one of each serves the whole run.

The bronze lake is built by running the real backfill over `tests/fixtures` with
`LocalSource`, the same code path `--source-dir` takes, rather than by writing
Parquet by hand. That way the silver tests read the bronze a real run produces,
and a bronze schema change reaches them.

There are two bronze lakes, not one. `bronze_from_fixtures` is the committed
games and nothing else, which is what the silver tests count against.
`bronze_with_duplicate_upload` is the same games laid out under their uploader
prefixes with one of them uploaded twice (`duplicate_upload`), which is the
shape PLA-175 broke on; silver and gold are built from that one, so the grain
the warehouse is tested at is the grain a real nightly run has to survive.

`s3_lake` is the third, and it is function scoped for the opposite reason: it is
cheap. moto's `mock_aws` serves a real boto3 client from an in-process fake, so
a test that wants an `s3://` lake root gets an empty bucket and a `Location`
pointing into it with no account, no network and no credentials beyond the fake
ones it sets. Every stage that goes through `pipeline.storage` works against it
unchanged, which is the point of the fixture: the S3 tests are the local tests
with one argument different.
"""

import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from pipeline.anonymize import token_for
from pipeline.backfill import run_backfill
from pipeline.settings import Settings
from pipeline.source import LocalSource
from pipeline.storage import Location, location, reset_s3_client

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

FIXTURES_DIR: Final = Path(__file__).parent / "fixtures"
CATALOG_PATH: Final = Path(__file__).parent / "catalog.json"
# Not a secret: the fixtures are already anonymized under a key nobody kept, so
# this one only has to be stable inside a test run.
TEST_HMAC_KEY: Final = b"tests-only-key-not-a-real-secret"
INGESTED_AT: Final = datetime(2026, 9, 23, 9, 0, tzinfo=UTC)
# The fake lake bucket and the prefix inside it. A prefix rather than the bucket
# root on purpose: a real deployment shares a bucket between environments, and a
# root that is not the bucket root is the case where a key is built wrongly.
LAKE_BUCKET: Final = "pra-lake-under-test"
LAKE_PREFIX: Final = "nightly"
TEST_REGION: Final = "us-west-2"
# When the two copies of the duplicated game were uploaded. A second apart,
# which is how far apart the real pair in the bucket was.
FIRST_UPLOAD_AT: Final = datetime(2026, 9, 23, 8, 0, tzinfo=UTC)
SECOND_UPLOAD_AT: Final = FIRST_UPLOAD_AT + timedelta(seconds=1)
UPLOADER_TOKEN_LENGTH: Final = 8


@pytest.fixture
def aws_fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials, so a misconfigured run can never reach a real account."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", TEST_REGION)
    monkeypatch.setenv("AWS_REGION", TEST_REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture(params=["local", "s3"])
def root(request: pytest.FixtureRequest, tmp_path: Path) -> Location:
    """A lake root of each kind, so a test that asks for it runs against both.

    The whole claim of `pipeline.storage` is that a stage cannot tell them
    apart, and a parameterized fixture is the cheapest way to keep asserting it:
    one test body, two roots, and a difference between them fails rather than
    going unnoticed.
    """
    if request.param == "local":
        return location(tmp_path / "lake")
    lake: Location = request.getfixturevalue("s3_lake")
    return lake


@pytest.fixture
def s3_lake(aws_fake_credentials: None) -> Iterator[Location]:
    """An empty bucket and a `Location` naming a lake root inside it.

    The cached client is dropped on both sides of the mock: it is built lazily
    and kept for the process, so a client made under one test's fake bucket must
    not be handed to the next test's.
    """
    import boto3
    from moto import mock_aws

    reset_s3_client()
    with mock_aws():
        boto3.client("s3", region_name=TEST_REGION).create_bucket(
            Bucket=LAKE_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": TEST_REGION},
        )
        yield Location(f"s3://{LAKE_BUCKET}/{LAKE_PREFIX}")
    reset_s3_client()


@pytest.fixture
def s3_lake_server(
    aws_fake_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Location]:
    """The same empty bucket, but served over HTTP so a subprocess can reach it.

    `mock_aws` patches botocore inside this interpreter, which is everything the
    in-process tests need and nothing a child process can see. `pipeline.run_all`
    runs every stage as a subprocess on purpose, so the only way to give those
    children a fake bucket is a real endpoint: moto's threaded server, with
    `AWS_ENDPOINT_URL_S3` in the environment they inherit. boto3 reads that
    variable itself, so no code here knows about it.

    Skipped rather than failed when the server extra is not installed: it
    arrives with MLflow's Flask today, and a suite that loses it should lose one
    test rather than report a broken pipeline.
    """
    pytest.importorskip("flask", reason="moto's threaded server needs flask")
    import boto3
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    reset_s3_client()
    try:
        boto3.client("s3", region_name=TEST_REGION, endpoint_url=endpoint).create_bucket(
            Bucket=LAKE_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": TEST_REGION},
        )
        yield Location(f"s3://{LAKE_BUCKET}/{LAKE_PREFIX}")
    finally:
        reset_s3_client()
        server.stop()


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


@dataclass(frozen=True)
class DuplicateUpload:
    """A source directory holding the committed games, one of them uploaded twice.

    `game_id` is the game that has two blobs; `first_key` and `second_key` are
    their keys, in upload order, so a test can name the row silver is supposed
    to keep and the one it is supposed to fold away. `blobs` is how many objects
    the directory holds in total, which is one more than the fixture count.
    """

    source_dir: Path
    game_id: str
    first_key: str
    second_key: str
    blobs: int


@pytest.fixture(scope="session")
def duplicate_upload(tmp_path_factory: pytest.TempPathFactory) -> DuplicateUpload:
    """The committed games under their uploader prefixes, with one game uploaded twice.

    The shape PLA-175 broke on: both players of a match upload their own log
    within a second of each other, so one game id arrives under two uploader
    prefixes and bronze lands two rows for it.

    Built here rather than committed as a twelfth fixture, for two reasons. The
    fixture set is produced only by `scripts/refresh_fixtures.py`, which deletes
    every `game-*.json` it finds and writes back only its own picks, so a
    hand-added file would vanish on the next refresh. And a parsed blob carries
    no upload time: `source_last_modified` is the object's, which is the local
    file's mtime here, so the "one second later" the collapse rule sorts on
    cannot live in a committed file at all.

    Nothing is invented. The second copy is the first fixture in sort order,
    byte for byte, under a second uploader prefix, with `summary.userId` set to
    a token of the same shape derived from the first one through the pipeline's
    own keyed hash. The real second upload would be that player's own view of
    the match, which the fixture set does not have and which is not ours to
    make up.
    """
    root = tmp_path_factory.mktemp("source")
    fixtures = sorted(FIXTURES_DIR.glob("game-*.json"))
    assert fixtures, FIXTURES_DIR
    for path in fixtures:
        blob = json.loads(path.read_text())
        _put(
            root / "parsed" / blob["summary"]["userId"] / path.name,
            path.read_text(),
            FIRST_UPLOAD_AT,
        )

    original = fixtures[0]
    blob = json.loads(original.read_text())
    uploader = blob["summary"]["userId"]
    other = "user-" + token_for(uploader, TEST_HMAC_KEY)[:UPLOADER_TOKEN_LENGTH]
    blob["summary"]["userId"] = other
    _put(
        root / "parsed" / other / original.name,
        json.dumps(blob, indent=2, sort_keys=True) + "\n",
        SECOND_UPLOAD_AT,
    )
    return DuplicateUpload(
        source_dir=root,
        game_id=str(json.loads(original.read_text())["summary"]["gameId"]),
        first_key=f"parsed/{uploader}/{original.name}",
        second_key=f"parsed/{other}/{original.name}",
        blobs=len(fixtures) + 1,
    )


def _put(path: Path, body: str, uploaded_at: datetime) -> None:
    """Write one source blob and give it the mtime `LocalSource` reports as its upload time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    os.utime(path, (uploaded_at.timestamp(), uploaded_at.timestamp()))


@pytest.fixture(scope="session")
def bronze_with_duplicate_upload(
    duplicate_upload: DuplicateUpload, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The duplicated source directory run through the real backfill: one row per blob.

    Bronze is raw, so the two uploads of one game are two rows here and the
    count is one above the fixture count. Collapsing them is silver's job.
    """
    root = tmp_path_factory.mktemp("lake-with-duplicate")
    bronze_dir = root / "bronze"
    settings = Settings(bucket="", hmac_key=TEST_HMAC_KEY)
    summary = run_backfill(
        settings,
        bronze_dir=bronze_dir,
        quarantine_dir=root / "quarantine",
        source=LocalSource(duplicate_upload.source_dir),
        now=INGESTED_AT,
    )
    assert summary.landed == summary.read == duplicate_upload.blobs, summary
    return bronze_dir


@pytest.fixture(scope="session")
def silver_from_fixtures(
    spark: "SparkSession",
    bronze_with_duplicate_upload: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A whole data directory with the fixture bronze and a silver run beside it.

    Returns the root, not the silver directory, because that root is what
    `PIPELINE_DATA_DIR` names: the dbt sources resolve
    `$PIPELINE_DATA_DIR/lake/silver/<table>/**/*.parquet`, so the layout here
    has to be the layout a real run writes. `run_silver` raises on a failed
    reconciliation, so a broken silver build fails the gold tests at setup
    rather than as a wrong number later.

    The bronze it reads is the fixtures plus one duplicated upload, so the gold
    build downstream is the one that used to fail PLA-175: every uniqueness
    test there is now a test that the collapse held.
    """
    from pipeline import silver

    root = tmp_path_factory.mktemp("datadir")
    lake = root / "lake"
    shutil.copytree(bronze_with_duplicate_upload, lake / "bronze")
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
