"""The ops models, built over the run metrics a real backfill and a real silver run wrote.

Marked `dbt` for the same reason `test_gold` is: it starts a Java Virtual
Machine for silver and then shells out to `dbt run` and `dbt test`. Run it with
`uv run pytest -m dbt`.

Two things are being checked, and neither can be stated by a dbt test. The
first is that the loop closes: two stages run under `stage_run`, their rows land
as Parquet in the data directory dbt is pointed at, and `mart_pipeline_health`
comes back with one row per stage carrying the numbers those stages reported.
The second is the empty case, which `test_gold` covers by accident and this
module states on purpose: a data directory that has never run a stage must still
build, because the alternative is a fresh clone where `dbt run` fails on a
directory that does not exist yet.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from pipeline.backfill import run_backfill
from pipeline.gold import run_gold
from pipeline.observability import stage_run
from pipeline.settings import Settings
from pipeline.source import LocalSource
from tests.conftest import CATALOG_PATH, FIXTURES_DIR, INGESTED_AT, TEST_HMAC_KEY

pytestmark = pytest.mark.dbt

RUN_ID = "ops-test-run"
FIXTURE_GAMES = len(sorted(FIXTURES_DIR.glob("*.json")))


@pytest.fixture(scope="module")
def ops_warehouse(
    spark: Any, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    """A data directory with bronze, silver and the two run-metrics rows those runs wrote.

    Built here rather than borrowed from `silver_from_fixtures` because the
    point is the rows the stage wrappers write, and the shared fixture calls
    `run_backfill` and `run_silver` directly, without them. Both stages run
    under one `run_id`, which is what an orchestrated DAG run looks like.
    """
    from pipeline import silver

    root = tmp_path_factory.mktemp("opsdata")
    lake = root / "lake"
    metrics_dir = lake / "run_metrics"

    with stage_run("bronze_backfill", run_id=RUN_ID, directory=metrics_dir) as metrics:
        summary = run_backfill(
            Settings(bucket="", hmac_key=TEST_HMAC_KEY),
            bronze_dir=lake / "bronze",
            quarantine_dir=lake / "quarantine",
            source=LocalSource(FIXTURES_DIR),
            now=INGESTED_AT,
        )
        metrics.rows_in = summary.read
        metrics.rows_out = summary.landed
        metrics.rows_quarantined = sum(summary.quarantined.values())

    with stage_run("silver", run_id=RUN_ID, directory=metrics_dir) as metrics:
        built = silver.run_silver(spark, lake / "bronze", lake / "silver", CATALOG_PATH)
        metrics.rows_in = built.games_in
        metrics.rows_out = built.rows["games"]
        metrics.rows_quarantined = 0

    assert run_gold(data_dir=root) == 0
    connection = duckdb.connect(str(root / "warehouse" / "meta.duckdb"), read_only=True)
    yield connection
    connection.close()


def rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """A query as a list of dictionaries, so an assertion names its columns."""
    result = connection.sql(sql)
    return [dict(zip(result.columns, row, strict=True)) for row in result.fetchall()]


def test_run_metrics_has_one_row_per_stage_of_the_run(
    ops_warehouse: duckdb.DuckDBPyConnection,
) -> None:
    found = rows(
        ops_warehouse,
        "select run_id, stage, status, rows_in, rows_out, rows_quarantined "
        "from run_metrics order by stage",
    )
    assert [row["stage"] for row in found] == ["bronze_backfill", "silver"]
    assert {row["run_id"] for row in found} == {RUN_ID}
    assert {row["status"] for row in found} == {"ok"}
    assert found[0]["rows_in"] == found[0]["rows_out"] == FIXTURE_GAMES
    assert found[0]["rows_quarantined"] == 0
    assert found[1]["rows_in"] == FIXTURE_GAMES


def test_pipeline_health_reports_the_last_run_of_every_stage(
    ops_warehouse: duckdb.DuckDBPyConnection,
) -> None:
    found = rows(ops_warehouse, "select * from mart_pipeline_health order by stage")

    assert [row["stage"] for row in found] == ["bronze_backfill", "silver"]
    for row in found:
        assert row["last_run_id"] == RUN_ID
        assert row["last_status"] == "ok"
        assert row["last_error"] is None
        assert row["runs_considered"] == 1
        assert row["failed_runs"] == 0
        assert row["last_duration_s"] > 0
        assert row["quarantine_rate"] == 0.0
        assert row["quarantine_rate_over_threshold"] is False
    assert found[0]["last_rows_out"] == FIXTURE_GAMES


def test_the_ops_models_build_over_an_empty_run_metrics_directory(
    silver_from_fixtures: Path,
) -> None:
    """A fresh clone that has never completed a stage still gets a warehouse.

    `silver_from_fixtures` is built by calling the stage functions directly, so
    its data directory has no `lake/run_metrics/` at all. That is the case the
    guard in `ops/run_metrics.sql` exists for.
    """
    assert not (silver_from_fixtures / "lake" / "run_metrics").exists()
    assert run_gold(data_dir=silver_from_fixtures) == 0

    connection = duckdb.connect(
        str(silver_from_fixtures / "warehouse" / "meta.duckdb"), read_only=True
    )
    try:
        assert rows(connection, "select * from run_metrics") == []
        assert rows(connection, "select * from mart_pipeline_health") == []
    finally:
        connection.close()
