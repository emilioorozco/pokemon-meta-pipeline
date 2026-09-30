"""The stages against an `s3://` lake root, on moto's in-process S3.

The claim `pipeline.storage` makes is that a stage does not know which kind of
root it was handed. These tests are how that claim is checked end to end rather
than one helper at a time: the real backfill over the committed fixtures, the
real partition replace, the real single-game merge and the real run-metrics row,
all with `s3://` in place of a directory and nothing else changed.

What is not here, and why:

- Silver. Spark reads and writes through the `s3a://` connector, which is a
  Hadoop jar fetched from Maven at session start and a Java credential chain
  neither moto's botocore hook nor a fake key reaches. `test_silver.py` covers
  the transforms on a local root and `test_storage.py` covers the configuration
  this stage derives; the connector itself is a live check against a real
  bucket.
- Gold. dbt reads silver through DuckDB's `httpfs`, which speaks HTTP from C++
  and is therefore invisible to `mock_aws` for the same reason. `test_gold.py`
  covers the build on a local root, and the S3 path is covered here only where
  it is this repository's code: which locations the stage derives and what it
  uploads afterwards.

Both are named in `docs/stages.md` as the two things the live run has to prove.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from pipeline import backfill as backfill_module
from pipeline import gold, serve, train
from pipeline import run_all as run_all_module
from pipeline.backfill import run_backfill
from pipeline.bronze import (
    BronzeRecord,
    delete_game,
    find_by_source_key,
    read_smoke,
    upsert_records,
    write_partitions,
)
from pipeline.config import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from pipeline.gold import publish_warehouse
from pipeline.observability import RunMetrics, write_run_metrics
from pipeline.quarantine import INVALID_JSON, write_quarantine
from pipeline.settings import Settings
from pipeline.source import LocalSource
from pipeline.storage import Location, duckdb_connect, local_file, location, tracking_store
from tests.conftest import FIXTURES_DIR, INGESTED_AT, TEST_HMAC_KEY
from tests.test_bronze import make_blob

NOW = datetime(2026, 9, 25, 8, 0, tzinfo=UTC)


def record(game_id: str, played_at: str) -> BronzeRecord:
    """One bronze record, built through the contract models like every other test."""
    return BronzeRecord(
        blob=make_blob(game_id, played_at, me="abc123", opponent="def456"),
        source_key=f"parsed/user-1/{game_id}.json",
    )


# ---------------------------------------------------------------- bronze --


def test_a_partition_replace_on_s3_is_idempotent(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    batch = [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-02T09:10:00Z")]

    first = write_partitions(batch, bronze, INGESTED_AT)
    counts = read_smoke(bronze)
    second = write_partitions(batch, bronze, INGESTED_AT)

    assert first == second == {"2026-09-01": 1, "2026-09-02": 1}
    assert counts == read_smoke(bronze) == [("2026-09-01", 1), ("2026-09-02", 1)]
    # One object per partition after two runs: the replace overwrote the key it
    # wrote last time rather than adding a second file beside it.
    assert len(bronze.iter_files("part-0.parquet")) == 2


def test_a_partition_replace_on_s3_drops_what_the_new_batch_does_not_name(
    s3_lake: Location,
) -> None:
    bronze = s3_lake / "lake" / "bronze"
    partition = bronze / "play_date=2026-09-01"
    # A file from an older layout, which the replace has to remove: on a disk
    # the directory is rewritten, on S3 the key has to be deleted by name.
    (partition / "part-1.parquet").write_bytes(b"left over")

    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert [item.name for item in partition.iter_files()] == ["part-0.parquet"]


def test_a_single_game_merge_on_s3_keeps_the_rest_of_the_day(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions(
        [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-01T19:00:00Z")],
        bronze,
        INGESTED_AT,
    )

    landed = upsert_records([record("game-3", "2026-09-01T20:00:00Z")], bronze, NOW)

    assert landed == {"2026-09-01": 1}
    assert read_smoke(bronze) == [("2026-09-01", 3)]


def test_merging_the_same_game_twice_on_s3_leaves_one_row(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    upsert_records([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, NOW)
    upsert_records([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, NOW)

    assert read_smoke(bronze) == [("2026-09-01", 1)]


def test_a_game_can_be_found_and_deleted_on_s3(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions(
        [record("game-1", "2026-09-01T18:22:00.000Z"), record("game-2", "2026-09-01T19:00:00Z")],
        bronze,
        INGESTED_AT,
    )

    found = find_by_source_key(bronze, "parsed/user-1/game-2.json")
    assert found is not None
    assert (found.game_id, found.play_date) == ("game-2", "2026-09-01")
    assert delete_game(bronze, found.play_date, found.game_id) == 1
    assert read_smoke(bronze) == [("2026-09-01", 1)]


def test_emptying_a_partition_on_s3_removes_it(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert delete_game(bronze, "2026-09-01", "game-1") == 1
    assert read_smoke(bronze) == []
    assert not (bronze / "play_date=2026-09-01").is_dir()


# ------------------------------------------------------------ quarantine --


def test_a_rejected_blob_and_its_sidecar_land_on_s3(s3_lake: Location) -> None:
    quarantine = s3_lake / "lake" / "quarantine"

    body = write_quarantine(
        quarantine, "parsed/user-1/game-9.json", b"not json", INVALID_JSON, "JSONDecodeError", NOW
    )

    assert body.read_bytes() == b"not json"
    sidecar = quarantine / INVALID_JSON / "parsed__user-1__game-9.meta.json"
    assert json.loads(sidecar.read_text())["reason"] == INVALID_JSON


# ----------------------------------------------------------- run metrics --


def test_a_run_metrics_row_lands_on_s3(s3_lake: Location) -> None:
    target = s3_lake / "lake" / "run_metrics"
    metrics = RunMetrics(stage="bronze_backfill", run_id="abc123", started_at=NOW)
    metrics.finished_at = NOW
    metrics.rows_in = 3
    metrics.rows_out = 3

    path = write_run_metrics(metrics, target)

    assert path.name == "abc123-bronze_backfill.parquet"
    row = path.read_table().to_pylist()[0]
    assert (row["run_id"], row["stage"], row["rows_out"], row["status"]) == (
        "abc123",
        "bronze_backfill",
        3,
        "ok",
    )


# -------------------------------------------------------------- backfill --


def test_the_backfill_lands_the_fixtures_in_an_s3_root(s3_lake: Location) -> None:
    bronze = s3_lake / "lake" / "bronze"
    settings = Settings(bucket="", hmac_key=TEST_HMAC_KEY)

    summary = run_backfill(
        settings,
        bronze_dir=bronze,
        quarantine_dir=s3_lake / "lake" / "quarantine",
        source=LocalSource(FIXTURES_DIR),
        now=INGESTED_AT,
    )

    assert summary.read == summary.landed > 0
    assert summary.quarantined == {}
    assert sum(count for _, count in read_smoke(bronze)) == summary.landed
    assert len(bronze.iter_files("part-0.parquet")) == len(summary.partitions)


def test_the_backfill_command_takes_an_s3_bronze_directory(
    s3_lake: Location, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole command line, so the argument parser's type is exercised too."""
    monkeypatch.setenv("HANDLE_HMAC_KEY", TEST_HMAC_KEY.decode())
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(s3_lake))

    code = backfill_module.main(
        [
            "--source-dir",
            str(FIXTURES_DIR),
            "--bronze-dir",
            str(s3_lake / "lake" / "bronze"),
            "--quarantine-dir",
            str(s3_lake / "lake" / "quarantine"),
        ]
    )

    assert code == 0
    assert sum(count for _, count in read_smoke(s3_lake / "lake" / "bronze")) > 0
    # The stage wrote its own row into the same lake, because the run was
    # pointed at it and nothing had to be told twice.
    assert len((s3_lake / "lake" / "run_metrics").iter_files(".parquet")) == 1


def test_a_local_root_still_writes_where_it_always_did(tmp_path: Path) -> None:
    """The control: the same call with a directory, asserting the on-disk layout."""
    bronze = tmp_path / "bronze"
    write_partitions([record("game-1", "2026-09-01T18:22:00.000Z")], bronze, INGESTED_AT)

    assert (bronze / "play_date=2026-09-01" / "part-0.parquet").is_file()
    assert list((bronze / "play_date=2026-09-01").glob(".*")) == []


# ------------------------------------------------------------------ gold --


def build_warehouse(path: Path) -> None:
    """A minimal warehouse: one table dbt would have materialized, one view it would not."""
    connection = duckdb.connect(str(path))
    try:
        connection.execute("create table mart_matchups as select 'a' as archetype, 1 as games")
        connection.execute("create view stg_games as select * from mart_matchups")
    finally:
        connection.close()


def test_the_marts_are_written_as_parquet_on_either_root(root: Location, tmp_path: Path) -> None:
    built = tmp_path / "meta.duckdb"
    build_warehouse(built)

    written = publish_warehouse(built, root)

    assert written == ["mart_matchups"]
    exported = (root / "warehouse" / "marts" / "mart_matchups.parquet").read_table()
    assert exported.to_pylist() == [{"archetype": "a", "games": 1}]


def test_the_database_itself_is_uploaded_for_an_s3_root(s3_lake: Location, tmp_path: Path) -> None:
    built = tmp_path / "meta.duckdb"
    build_warehouse(built)

    publish_warehouse(built, s3_lake)

    warehouse = s3_lake / "warehouse" / "meta.duckdb"
    assert warehouse.is_file()
    # And a reader downstream can open the copy it pulls back down.
    connection = duckdb.connect(str(local_file(warehouse)), read_only=True)
    try:
        assert connection.sql("select games from mart_matchups").fetchone() == (1,)
    finally:
        connection.close()


def test_a_local_build_is_not_copied_onto_itself(tmp_path: Path) -> None:
    """The guard that matters: uploading a file over itself would truncate it."""
    root = location(tmp_path)
    (root / "warehouse").mkdir()
    built = (root / "warehouse" / "meta.duckdb").path
    build_warehouse(built)

    publish_warehouse(built, root)

    assert built.stat().st_size > 0


def test_an_s3_root_builds_the_database_locally_and_picks_the_s3_target(
    s3_lake: Location, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The derivation, without dbt: which target, and where the file is built.

    dbt against an S3 lake reads silver through DuckDB's `httpfs`, which speaks
    HTTP from C++ and so cannot see moto's in-process fake. What this repository
    owns is the two decisions above, and those are checked here; the build over
    a real bucket is the live check.
    """
    seen: list[dict[str, str]] = []

    def fake_run(argv: list[str], env: dict[str, str], check: bool) -> Any:
        seen.append({"argv": " ".join(argv), **env})
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(gold.subprocess, "run", fake_run)

    assert gold.run_gold(data_dir=s3_lake) == 0

    assert seen, "dbt was never invoked"
    for call in seen:
        assert f"--target {gold.S3_TARGET}" in call["argv"]
        assert call["PIPELINE_DATA_DIR"] == str(s3_lake)
        # The lake is the bucket and the database file is not: DuckDB cannot
        # write one over object storage.
        assert not call[gold.WAREHOUSE_VAR].startswith("s3://")
        assert call[gold.WAREHOUSE_VAR].endswith("meta.duckdb")


def test_a_local_root_still_picks_the_local_target_and_builds_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, str]] = []

    def fake_run(argv: list[str], env: dict[str, str], check: bool) -> Any:
        seen.append({"argv": " ".join(argv), **env})
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(gold.subprocess, "run", fake_run)

    assert gold.run_gold(data_dir=tmp_path) == 0

    assert f"--target {gold.LOCAL_TARGET}" in seen[0]["argv"]
    assert seen[0][gold.WAREHOUSE_VAR] == str(tmp_path / "warehouse" / "meta.duckdb")


# --------------------------------------------------------------- run_all --


def test_run_all_carries_an_s3_root_to_every_stage(
    s3_lake_server: Location, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner passes the root through as it came, and the backfill lands in it.

    Against the threaded server rather than the in-process fake, because every
    stage is a subprocess and a subprocess cannot see a patched botocore.

    Stopped after the backfill: silver needs a Java Virtual Machine and the
    `s3a://` connector, and gold needs dbt and `httpfs`, which are the two
    things the live check is for.
    """
    monkeypatch.setenv("HANDLE_HMAC_KEY", TEST_HMAC_KEY.decode())

    code = run_all_module.main(
        [
            "--source-dir",
            str(FIXTURES_DIR),
            "--data-dir",
            str(s3_lake_server),
            "--stop-after",
            "backfill",
        ]
    )

    assert code == 0
    assert sum(count for _, count in read_smoke(s3_lake_server / "lake" / "bronze")) > 0
    # Two rows: the backfill's own, written by the child, and the runner's.
    written = (s3_lake_server / "lake" / "run_metrics").iter_files(".parquet")
    assert sorted(item.name.split("-", 1)[1] for item in written) == [
        "bronze_backfill.parquet",
        "run_all.parquet",
    ]


def test_run_all_refuses_an_s3_root_with_no_bucket(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        run_all_module.main(["--data-dir", "s3:///lake"])

    assert raised.value.code == 2
    assert "PIPELINE_DATA_DIR" in capsys.readouterr().err


# ---------------------------------------------------------- mlflow store --


@pytest.mark.ml
def test_a_model_trained_into_an_s3_store_is_loadable_by_the_service(
    s3_lake: Location, tmp_path: Path
) -> None:
    """The real three commands over one lake store: train, move the alias, load the model.

    The store is a directory MLflow stats, lists and renames, none of which
    object storage has, so `pipeline.storage.tracking_store` works on a local
    copy and uploads it at the end. What this asserts is the part that matters
    to the next command: after the stage, the runs and the registered version
    are in the lake and not on a disk that went away with the task, and the
    files they point at are readable from a command that never saw that disk.
    """
    from mlflow.tracking import MlflowClient

    from tests.test_train import synthetic_features

    warehouse = tmp_path / "meta.duckdb"
    connection = duckdb.connect(str(warehouse))
    try:
        connection.register("frame", synthetic_features())
        connection.execute("create table features_turn as select * from frame")
    finally:
        connection.close()

    code = train.main(
        [
            "--warehouse",
            str(warehouse),
            "--tracking-uri",
            str(s3_lake / "mlruns"),
            "--experiment",
            "s3-win-probability",
            "--params",
            "num_boost_round=40",
            "min_data_in_leaf=10",
        ]
    )

    assert code == 0
    landed = [item.key for item in (s3_lake / "mlruns").iter_files()]
    assert any(key.endswith("meta.yaml") for key in landed)
    assert any("/models/" in key or "/win-probability/" in key for key in landed)
    # The metadata is in the store and the files it points at are beside it.
    assert (s3_lake / "mlruns-artifacts").is_dir()

    # What `promote` does, without its verdict: the alias is the address the
    # service loads, and moving it is a registry write in its own command.
    with tracking_store(str(s3_lake / "mlruns")) as tracking_uri:
        MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, "1"
        )

    # And what `serve` does, in a third command with a third temporary
    # directory: load the aliased model and the archetype codes beside it.
    # Before the artifacts moved to S3 this raised, because everything the
    # registry pointed at was under the training command's scratch directory.
    loaded = serve.mlflow_loader(tracking_uri=str(s3_lake / "mlruns"))()
    assert loaded.version == "1"
    assert loaded.alias == PRODUCTION_ALIAS
    assert loaded.codes

    with tracking_store(str(s3_lake / "mlruns"), write_back=False) as tracking_uri:
        client = MlflowClient(tracking_uri=tracking_uri)
        aliased = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        model = client.get_logged_model(str(aliased.source).rsplit("/", 1)[-1])
        assert model.artifact_location.startswith("s3://")


@pytest.mark.ml
def test_artifacts_logged_in_one_command_are_readable_in_the_next(
    s3_lake: Location, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two commands over one lake store: what the first logs, the second can open.

    This is the whole point of putting the artifacts on S3, and it is what the
    synced store could not do before. MLflow's file store writes absolute
    paths, so an experiment created inside the temporary directory of the first
    command recorded that directory as its `artifact_location`, every run under
    it recorded an `artifact_uri` beneath that, and every registered version
    pointed at the same place. The directory is deleted when the command exits,
    so the second command read a store full of runs whose files were nowhere
    and whose run directories had lost the empty subdirectories the file store
    validates, and answered that the run did not exist.

    The two contexts here are two commands. The first stands in for `train`:
    log a file, log a model, register it and move an alias onto it. The second
    stands in for everything that reads afterwards, `promote`, `drift` and
    `serve`: read the run back, download its artifact, and load the model both
    by version and by the alias.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression

    from pipeline.storage import experiment_id, tracking_store

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store = str(s3_lake / "mlruns")
    model_name = "win-probability"

    with tracking_store(store) as tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_id=experiment_id("s3-artifacts", tracking_uri))
        with mlflow.start_run(run_name="lightgbm") as active:
            run_id = active.info.run_id
            mlflow.log_metric("holdout_logloss", 0.25)
            mlflow.log_dict({"charizard": 0}, "archetype_codes.json")
            logged = mlflow.sklearn.log_model(
                LogisticRegression().fit([[0.0], [1.0]], [0, 1]),
                name="model",
                # Named rather than inferred: resolving the environment shells
                # out to the project's own lock file, which is a second of a
                # test that is about where the files land.
                pip_requirements=["scikit-learn"],
            )
        version = mlflow.register_model(logged.model_uri, model_name)
        MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
            model_name, "production", version.version
        )

    with tracking_store(store, write_back=False) as tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        run = client.get_run(run_id)
        assert run.data.metrics["holdout_logloss"] == 0.25
        assert run.info.artifact_uri.startswith("s3://")

        codes = download_artifacts(
            run_id=run_id, artifact_path="archetype_codes.json", tracking_uri=tracking_uri
        )
        assert json.loads(Path(codes).read_text()) == {"charizard": 0}

        aliased = client.get_model_version_by_alias(model_name, "production")
        assert aliased.version == version.version
        # MLflow 3 records a logged model rather than a path as the version's
        # source, and that model is where the `s3://` shows up for serving.
        source = str(aliased.source)
        model = client.get_logged_model(source.rsplit("/", 1)[-1])
        assert model.artifact_location.startswith("s3://")
        assert mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}") is not None
        assert mlflow.sklearn.load_model(f"models:/{model_name}@production") is not None

        experiment = client.get_experiment_by_name("s3-artifacts")
        assert experiment is not None
        assert experiment.artifact_location == str(s3_lake / "mlruns-artifacts" / "s3-artifacts")

    # The model is in the artifact prefix and not in the synced store, so a
    # command that only logs a metric does not download a model to do it.
    assert not any(item.name == "MLmodel" for item in (s3_lake / "mlruns").iter_files())
    assert any(item.name == "MLmodel" for item in (s3_lake / "mlruns-artifacts").iter_files())


@pytest.mark.dbt
def test_gold_builds_over_an_s3_lake_and_publishes_back(
    s3_lake_server: Location, silver_from_fixtures: Path
) -> None:
    """The real dbt build with the sources on S3, against moto's threaded server.

    Possible only through the server: dbt reads silver with DuckDB's `httpfs`,
    which makes its own HTTP calls and never touches botocore, so the in-process
    fake is invisible to it. The fixture silver is uploaded rather than rebuilt,
    because what is under test here is the gold stage over an S3 root and not
    Spark's write to one.

    The last assertion is the one that caught a real bug: the ops models are
    views over the run-metrics Parquet, so a reader of the published warehouse
    has to resolve `s3://` too, and the downloaded copy has to keep the file's
    name or every `meta.main.x` in those views fails to bind.
    """
    pytest.importorskip("dbt.adapters.duckdb", reason="dbt-duckdb is not installed")
    for item in sorted(Path(silver_from_fixtures).rglob("*")):
        if item.is_file():
            relative = item.relative_to(silver_from_fixtures).as_posix()
            (s3_lake_server / relative).upload_file(item)
    # One run-metrics row, so the ops views have something to read.
    metrics = RunMetrics(stage="silver", run_id="abc123", started_at=NOW)
    metrics.finished_at = NOW
    write_run_metrics(metrics, s3_lake_server / "lake" / "run_metrics")

    summary = gold.GoldSummary()
    assert gold.run_gold(data_dir=s3_lake_server, summary=summary) == 0

    assert "mart_matchups" in summary.marts
    assert (s3_lake_server / "warehouse" / "marts" / "mart_matchups.parquet").is_file()
    assert (s3_lake_server / "warehouse" / "meta.duckdb").is_file()
    connection = duckdb_connect(s3_lake_server / "warehouse" / "meta.duckdb")
    try:
        assert connection.sql("select count(*) from mart_pipeline_health").fetchone() is not None
    finally:
        connection.close()
