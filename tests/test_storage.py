"""The storage helper against both roots: the same assertions, twice.

Every behavioural test here is parameterized over a local root and an `s3://`
root and asserts the same thing of both, because that is the claim the module
makes: a stage does not know which one it was given. The S3 half runs against
moto's in-process fake, so it needs no account, no network and no credentials.

The tests that are not parameterized are the ones about the difference: what an
`s3://` root refuses, what a local root does byte for byte, and the URIs and
statements derived for Spark and DuckDB, which are strings this module owns and
neither engine is started here to check.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from pipeline import storage
from pipeline.settings import DATA_DIR_VAR, DataRootError, validate_data_root
from pipeline.storage import (
    LAKE_SECRET,
    SYNC_WORKERS,
    Location,
    StorageError,
    artifact_root,
    duckdb_connect,
    duckdb_s3_profile,
    local_file,
    location,
    spark_configuration,
    synced_dir,
    table_bytes,
    tracking_store,
)

TABLE = pa.table({"game_id": ["g1", "g2"], "rows": [1, 2]})


# ------------------------------------------------------------- the parsing --


def test_a_path_and_a_string_are_the_same_local_location(tmp_path: Path) -> None:
    assert location(tmp_path) == location(str(tmp_path)) == tmp_path


def test_an_s3_uri_splits_into_a_bucket_and_a_key() -> None:
    target = Location("s3://a-bucket/lake/bronze")
    assert (target.is_s3, target.bucket, target.key) == (True, "a-bucket", "lake/bronze")
    assert str(target) == "s3://a-bucket/lake/bronze"


def test_a_bucket_root_has_an_empty_key() -> None:
    target = Location("s3://a-bucket/")
    assert (target.bucket, target.key, str(target)) == ("a-bucket", "", "s3://a-bucket")
    assert str(target / "lake" / "bronze") == "s3://a-bucket/lake/bronze"


def test_an_s3_uri_with_no_bucket_is_refused() -> None:
    with pytest.raises(StorageError):
        Location("s3://")


def test_the_data_root_check_names_the_variable() -> None:
    with pytest.raises(DataRootError, match=DATA_DIR_VAR):
        validate_data_root("s3:///lake")


def test_the_data_root_check_passes_a_local_path_through(tmp_path: Path) -> None:
    assert validate_data_root(tmp_path) == tmp_path


def test_a_local_location_has_no_bucket(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        assert location(tmp_path).bucket


def test_an_s3_location_has_no_local_path() -> None:
    with pytest.raises(StorageError):
        assert Location("s3://a-bucket/lake").path


def test_joining_keeps_the_kind_and_the_layout(root: Location) -> None:
    assert (root / "lake" / "bronze").name == "bronze"
    assert (root / "lake" / "bronze").parent == root / "lake"
    assert str(root / "lake" / "bronze").endswith("lake/bronze")


# ------------------------------------------------------------ the contents --


def test_bytes_written_read_back(root: Location) -> None:
    target = root / "catalog" / "cards.json"
    target.write_bytes(b'{"a": 1}')
    assert target.read_bytes() == b'{"a": 1}'
    assert target.read_text() == '{"a": 1}'


def test_a_rewrite_replaces_rather_than_appends(root: Location) -> None:
    target = root / "catalog" / "cards.json"
    target.write_text("first")
    target.write_text("second")
    assert target.read_text() == "second"


def test_presence_before_and_after_a_write(root: Location) -> None:
    target = root / "lake" / "bronze" / "part-0.parquet"
    assert not target.exists()
    assert not target.is_file()
    assert not (root / "lake").is_dir()
    target.write_bytes(b"x")
    assert target.is_file()
    assert (root / "lake").is_dir()
    assert not (root / "lake" / "bronze").is_file()


def test_a_table_survives_the_round_trip(root: Location) -> None:
    target = root / "lake" / "run_metrics" / "run-stage.parquet"
    target.write_table(TABLE)
    assert target.read_table() == TABLE
    assert target.read_table(columns=["game_id"]).column_names == ["game_id"]


def test_listing_is_recursive_sorted_and_filtered(root: Location) -> None:
    for name in ("play_date=2026-09-02", "play_date=2026-09-01"):
        (root / "lake" / "bronze" / name / "part-0.parquet").write_bytes(b"x")
    (root / "lake" / "bronze" / "notes.txt").write_bytes(b"x")
    listed = (root / "lake" / "bronze").iter_files()
    assert [item.name for item in listed] == ["notes.txt", "part-0.parquet", "part-0.parquet"]
    parquet = (root / "lake" / "bronze").iter_files("part-0.parquet")
    assert [str(item) for item in parquet] == [
        str(root / "lake" / "bronze" / "play_date=2026-09-01" / "part-0.parquet"),
        str(root / "lake" / "bronze" / "play_date=2026-09-02" / "part-0.parquet"),
    ]


def test_listing_an_absent_prefix_is_empty_not_an_error(root: Location) -> None:
    assert (root / "nothing" / "here").iter_files() == []


# ------------------------------------------------------------- the replace --


def test_a_replace_leaves_exactly_the_named_files(root: Location) -> None:
    partition = root / "lake" / "bronze" / "play_date=2026-09-01"
    partition.replace_dir({"part-0.parquet": b"first", "stale.parquet": b"old"})
    partition.replace_dir({"part-0.parquet": b"second"})
    assert [item.name for item in partition.iter_files()] == ["part-0.parquet"]
    assert (partition / "part-0.parquet").read_bytes() == b"second"


def test_a_replace_does_not_touch_a_sibling_partition(root: Location) -> None:
    bronze = root / "lake" / "bronze"
    (bronze / "play_date=2026-09-01").replace_dir({"part-0.parquet": b"one"})
    (bronze / "play_date=2026-09-02").replace_dir({"part-0.parquet": b"two"})
    (bronze / "play_date=2026-09-01").replace_dir({"part-0.parquet": b"one again"})
    assert (bronze / "play_date=2026-09-02" / "part-0.parquet").read_bytes() == b"two"


def test_deleting_a_prefix_removes_everything_under_it(root: Location) -> None:
    bronze = root / "lake" / "bronze"
    (bronze / "play_date=2026-09-01" / "part-0.parquet").write_bytes(b"one")
    (bronze / "play_date=2026-09-02" / "part-0.parquet").write_bytes(b"two")
    (bronze / "play_date=2026-09-01").delete_prefix()
    assert [str(item) for item in bronze.iter_files()] == [
        str(bronze / "play_date=2026-09-02" / "part-0.parquet")
    ]


def test_deleting_an_absent_prefix_is_not_an_error(root: Location) -> None:
    (root / "nothing").delete_prefix()


# ------------------------------------------------------------- transfers --


def test_a_file_is_readable_locally_whichever_root_it_is_on(root: Location) -> None:
    target = root / "warehouse" / "meta.duckdb"
    target.write_bytes(b"not really a database")
    local = local_file(target)
    assert local.is_file()
    assert local.read_bytes() == b"not really a database"


def test_a_local_file_is_not_copied(tmp_path: Path) -> None:
    target = location(tmp_path) / "warehouse" / "meta.duckdb"
    target.write_bytes(b"x")
    assert local_file(target) == tmp_path / "warehouse" / "meta.duckdb"


def test_a_synced_directory_comes_back_with_what_was_written(root: Location) -> None:
    store = root / "mlruns"
    (store / "0" / "meta.yaml").write_text("experiment_id: 0\n")
    with synced_dir(store) as working:
        assert (working / "0" / "meta.yaml").read_text() == "experiment_id: 0\n"
        (working / "0" / "a-run" / "metrics").parent.mkdir(parents=True, exist_ok=True)
        (working / "0" / "a-run" / "metrics").write_text("logloss 0.5\n")
    assert (store / "0" / "a-run" / "metrics").read_text() == "logloss 0.5\n"


def test_a_synced_directory_keeps_a_directory_that_holds_nothing(root: Location) -> None:
    """An empty directory survives the round trip, which object storage does not do for free.

    MLflow's file store reads a run that has no `metrics`, `params` and
    `artifacts` subdirectories as a run that is not there, and with the
    artifacts on S3 the local `artifacts` directory of every run is empty. A
    sync that carried only files would therefore delete every run in the store
    from the next command's point of view.
    """
    store = root / "mlruns"
    with synced_dir(store) as working:
        (working / "0" / "a-run" / "artifacts").mkdir(parents=True)
        (working / "0" / "a-run" / "meta.yaml").write_text("run_id: a-run\n")

    with synced_dir(store, write_back=False) as working:
        assert (working / "0" / "a-run" / "artifacts").is_dir()
        assert (working / "0" / "a-run" / "meta.yaml").read_text() == "run_id: a-run\n"


def relative_files(root: Path) -> dict[str, bytes]:
    """Every file under a directory, keyed by its path relative to it."""
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def empty_directories(root: Path) -> list[str]:
    """Every directory under a root that holds nothing, relative to it."""
    return sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_dir() and not any(item.iterdir())
    )


def test_a_tree_wider_than_the_pool_round_trips_unchanged(
    s3_lake: Location, tmp_path: Path
) -> None:
    """The sync runs `SYNC_WORKERS` objects at a time, and that may change only the clock.

    One object at a time was the whole of a deployed cold start: a production
    MLflow store is several hundred files of a few hundred bytes each, and the
    cost of each one is a round trip rather than its bytes. The pool is the
    fix, and the only ways it could be a bad trade are a file landing in the
    wrong place, landing truncated, or not landing at all. So a store several
    times wider than the pool goes up, comes back down into a different
    directory, and is compared path by path and byte by byte, empty
    directories included: the markers are the half an object store does not
    carry for free and the half a concurrent walk would drop first.
    """
    written = tmp_path / "written"
    for experiment in range(3):
        for run in range(SYNC_WORKERS):
            run_dir = written / str(experiment) / f"run-{run:02d}"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "metrics" / "holdout_logloss").write_text(f"{experiment}.{run}\n")
            (run_dir / "meta.yaml").write_text(f"run_id: {experiment}-{run:02d}\n")
            (run_dir / "artifacts").mkdir()
    assert len(relative_files(written)) > SYNC_WORKERS

    store = s3_lake / "mlruns"
    store.upload_tree(written)
    read_back = store.download_tree(tmp_path / "read-back")

    assert relative_files(read_back) == relative_files(written)
    assert empty_directories(read_back) == empty_directories(written)


def test_an_empty_directory_marker_is_not_a_file(s3_lake: Location) -> None:
    """The markers are invisible to every reader but the sync that wrote them."""
    store = s3_lake / "mlruns"
    with synced_dir(store) as working:
        (working / "0" / "artifacts").mkdir(parents=True)
        (working / "0" / "meta.yaml").write_text("experiment_id: 0\n")

    assert [item.name for item in store.iter_files()] == ["meta.yaml"]


def test_a_reader_does_not_write_the_synced_store_back(s3_lake: Location) -> None:
    """`write_back=False` is what keeps a read-only command from being a second writer."""
    store = s3_lake / "mlruns"
    (store / "0" / "meta.yaml").write_text("experiment_id: 0\n")

    with synced_dir(store, write_back=False) as working:
        (working / "0" / "scratch").write_text("nothing anyone asked for\n")

    assert not (store / "0" / "scratch").is_file()


def test_a_failed_block_uploads_nothing(s3_lake: Location) -> None:
    store = s3_lake / "mlruns"
    (store / "0" / "meta.yaml").write_text("experiment_id: 0\n")

    with pytest.raises(RuntimeError), synced_dir(store) as working:
        (working / "0" / "half-written").write_text("a stage that died\n")
        raise RuntimeError("the stage failed")

    assert not (store / "0" / "half-written").is_file()


def test_the_tracking_store_yields_a_file_uri_for_an_s3_store(s3_lake: Location) -> None:
    store = s3_lake / "mlruns"
    (store / "0" / "meta.yaml").write_text("experiment_id: 0\n")
    with tracking_store(str(store)) as uri:
        assert uri.startswith("file:")
        assert (Path(uri[len("file:") :]) / "0" / "meta.yaml").is_file()


def test_the_tracking_store_names_an_artifact_prefix_beside_the_store(s3_lake: Location) -> None:
    """Artifacts go next to the store and not inside it, so the sync never walks them."""
    store = s3_lake / "mlruns"
    with tracking_store(str(store)) as uri:
        assert artifact_root(uri) == str(s3_lake / "mlruns-artifacts")
    assert artifact_root(uri) is None


def test_a_local_tracking_store_names_no_artifact_prefix(tmp_path: Path) -> None:
    """A laptop keeps the artifacts beside the runs, which is where `mlflow ui` looks."""
    with tracking_store(f"file:{tmp_path}") as uri:
        assert artifact_root(uri) is None


def test_the_tracking_store_passes_a_server_uri_through() -> None:
    with tracking_store("http://mlflow:5000") as uri:
        assert uri == "http://mlflow:5000"


def test_the_tracking_store_passes_a_local_file_uri_through(tmp_path: Path) -> None:
    with tracking_store(f"file:{tmp_path}") as uri:
        assert uri == f"file:{tmp_path}"


# ------------------------------------------------- what the engines are told --


def test_spark_is_told_nothing_extra_for_a_local_root(tmp_path: Path) -> None:
    assert spark_configuration(tmp_path, tmp_path / "silver") == {}


def test_spark_gets_the_connector_and_the_credential_chain_for_an_s3_root() -> None:
    settings = spark_configuration(Path("/tmp/bronze"), Location("s3://a-bucket/lake/silver"))
    assert "hadoop-aws" in settings["spark.jars.packages"]
    assert settings["spark.hadoop.fs.s3a.impl"].endswith("S3AFileSystem")
    assert "DefaultCredentialsProvider" in settings["spark.hadoop.fs.s3a.aws.credentials.provider"]


def test_spark_uris_use_the_s3a_scheme() -> None:
    assert Location("s3://a-bucket/lake/silver").spark_uri == "s3a://a-bucket/lake/silver"
    assert location("/tmp/silver").spark_uri == "/tmp/silver"


def test_duckdb_reaches_the_regional_endpoint_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

    assert duckdb_s3_profile() == {
        "endpoint": "s3.eu-west-1.amazonaws.com",
        "url_style": "vhost",
        "use_ssl": "true",
    }


def test_duckdb_uses_path_style_against_a_local_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://127.0.0.1:5555")

    assert duckdb_s3_profile() == {
        "endpoint": "127.0.0.1:5555",
        "url_style": "path",
        "use_ssl": "false",
    }


# ------------------------------------------- the lake secret, under threads --


class Recorder:
    """A stand-in for a DuckDB connection that remembers what it was asked to run.

    Enough of one for `duckdb_connect`, which opens the file and then executes
    two statements on it. `fails` makes the first `CREATE SECRET` raise the
    conflict a second connection to the same catalog really raises, so the
    retry is exercised without two threads and a timing window.
    """

    def __init__(self, *, fails: int = 0) -> None:
        self.statements: list[str] = []
        self.fails = fails

    def execute(self, sql: str) -> "Recorder":
        self.statements.append(sql)
        if sql.startswith("CREATE SECRET") and self.fails:
            self.fails -= 1
            raise duckdb.TransactionException(
                'TransactionContext Error: Catalog write-write conflict on create with "pra_lake"'
            )
        return self


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """`duckdb_connect` against a fake connection and a warehouse nobody downloads."""
    recorder = Recorder()

    def connect(*args: Any, **kwargs: Any) -> Recorder:
        return recorder

    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setattr(storage, "local_file", lambda target: Path("meta.duckdb"))
    return recorder


def test_the_lake_secret_is_created_only_when_it_is_not_already_there(
    recorded: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OR REPLACE` is a catalog write per connection, and two at once conflict."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

    duckdb_connect(Location("s3://a-bucket/lake/warehouse/meta.duckdb"))

    create = recorded.statements[-1]
    assert create.startswith(f"CREATE SECRET IF NOT EXISTS {LAKE_SECRET} (")
    assert "OR REPLACE" not in create
    assert "s3.eu-west-1.amazonaws.com" in create


def test_a_local_warehouse_is_told_nothing_about_s3(recorded: Recorder) -> None:
    """The offline path has to stay offline: no extension, no secret, no network."""
    duckdb_connect(Path("warehouse") / "meta.duckdb")

    assert recorded.statements == []


def test_a_write_write_conflict_on_the_secret_is_retried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock is process wide, and the catalog the conflict is in is not."""
    recorder = Recorder(fails=1)
    monkeypatch.setattr(duckdb, "connect", lambda *args, **kwargs: recorder)
    monkeypatch.setattr(storage, "local_file", lambda target: Path("meta.duckdb"))

    duckdb_connect(Location("s3://a-bucket/lake/warehouse/meta.duckdb"))

    assert [statement.split(" (")[0] for statement in recorder.statements] == [
        "INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws",
        f"CREATE SECRET IF NOT EXISTS {LAKE_SECRET}",
        "INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws",
        f"CREATE SECRET IF NOT EXISTS {LAKE_SECRET}",
    ]


def test_two_threads_opening_one_s3_warehouse_both_get_a_usable_connection(
    s3_lake: Location, tmp_path: Path
) -> None:
    """The failure the first live run of the deployed evaluation hit, in miniature.

    The agent runs the tool calls of one model turn in parallel, so two
    `query_marts` calls open two connections to the same database file at the
    same instant. DuckDB gives both of them one catalog, and the secret was
    being written per connection, so one of the two died with a write-write
    conflict and the question came back a 500.

    The download is warmed first and a barrier lines the two threads up, so
    what they race on is the catalog rather than the copy, which is the race
    that failed.
    """
    built = tmp_path / "meta.duckdb"
    with duckdb.connect(str(built)) as seed:
        seed.execute("create table mart_matchups as select 1 as games")
    warehouse = s3_lake / "warehouse" / "meta.duckdb"
    warehouse.upload_file(built)
    local_file(warehouse)

    start = threading.Barrier(2)

    def open_it() -> int:
        start.wait(timeout=30)
        connection = duckdb_connect(warehouse)
        try:
            row = connection.sql("select games from mart_matchups").fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda call: call(), [open_it, open_it])) == [1, 1]

    connection = duckdb_connect(warehouse)
    try:
        secrets = connection.sql("select name from duckdb_secrets()").fetchall()
    finally:
        connection.close()
    assert [name for (name,) in secrets] == [LAKE_SECRET]


def test_table_bytes_is_a_parquet_file(tmp_path: Path) -> None:
    written = table_bytes(TABLE, compression="zstd")
    target = tmp_path / "part-0.parquet"
    target.write_bytes(written)
    assert pa.parquet.read_table(target) == TABLE
