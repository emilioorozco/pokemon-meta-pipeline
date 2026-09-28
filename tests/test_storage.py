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

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest

from pipeline.settings import DATA_DIR_VAR, DataRootError, validate_data_root
from pipeline.storage import (
    Location,
    StorageError,
    duckdb_settings,
    local_file,
    location,
    spark_configuration,
    synced_dir,
    table_bytes,
    tracking_store,
)

TABLE = pa.table({"game_id": ["g1", "g2"], "rows": [1, 2]})


@pytest.fixture(params=["local", "s3"])
def root(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Location]:
    """A lake root of each kind, so every test below runs twice."""
    if request.param == "local":
        yield location(tmp_path / "lake")
        return
    yield request.getfixturevalue("s3_lake")


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


def test_the_tracking_store_yields_a_file_uri_for_an_s3_store(s3_lake: Location) -> None:
    store = s3_lake / "mlruns"
    (store / "0" / "meta.yaml").write_text("experiment_id: 0\n")
    with tracking_store(str(store)) as uri:
        assert uri.startswith("file:")
        assert (Path(uri[len("file:") :]) / "0" / "meta.yaml").is_file()


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


def test_duckdb_is_told_nothing_extra_for_a_local_root(tmp_path: Path) -> None:
    assert duckdb_settings(tmp_path) == []


def test_duckdb_loads_httpfs_and_a_credential_chain_secret_for_an_s3_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    statements = duckdb_settings("s3://a-bucket/lake")
    assert "LOAD httpfs" in statements
    assert any("credential_chain" in item.lower() for item in statements)
    assert any("us-west-2" in item for item in statements)


def test_table_bytes_is_a_parquet_file(tmp_path: Path) -> None:
    written = table_bytes(TABLE, compression="zstd")
    target = tmp_path / "part-0.parquet"
    target.write_bytes(written)
    assert pa.parquet.read_table(target) == TABLE
