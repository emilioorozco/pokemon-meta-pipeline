"""Gold end to end: build silver from the committed games, then run dbt over it.

Every test here is marked `dbt` and the default `uv run pytest` skips them, the
same split the Spark tests get and for the same reason: the module builds a
real silver lake (a Java Virtual Machine) and then a real DuckDB warehouse
(`dbt run` and `dbt test`). `uv run pytest -m dbt` runs them.

The point of the module is that the dbt project is exercised as a project.
`run_gold` is the same entry point orchestration will call, the profile is the
committed one, and the 67 generic and singular dbt tests run as part of it, so
a broken key or a broken relationship fails here without a Python assertion
having to name it. The assertions below are the handful of numbers a dbt test
cannot state: the grain against the fixture count, the symmetry of the matchup
mart as an independent query, the bounds on a rate, and the size of
`dim_player` against the tokens silver actually wrote.
"""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from pipeline.gold import run_gold
from tests.conftest import FIXTURES_DIR

pytestmark = pytest.mark.dbt

FIXTURE_GAMES = len(sorted(FIXTURES_DIR.glob("*.json")))
SIDES_PER_GAME = 2


@pytest.fixture(scope="module")
def warehouse(silver_from_fixtures: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """One `dbt run` plus `dbt test` over the fixture silver, then a read-only handle.

    The exit code is asserted here rather than in a test of its own because
    every other test in the module would be meaningless after a failed build,
    and a setup failure says so more clearly than eight identical errors.
    """
    assert run_gold(data_dir=silver_from_fixtures) == 0
    connection = duckdb.connect(
        str(silver_from_fixtures / "warehouse" / "meta.duckdb"), read_only=True
    )
    yield connection
    connection.close()


def scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    """The single number a counting query returns, asserted to be one."""
    row = connection.sql(sql).fetchone()
    assert row is not None
    return int(row[0])


def test_the_build_produced_every_model(warehouse: duckdb.DuckDBPyConnection) -> None:
    built = {name for (name,) in warehouse.sql("select table_name from duckdb_tables()").fetchall()}
    built |= {name for (name,) in warehouse.sql("select view_name from duckdb_views()").fetchall()}
    expected = {
        "stg_games",
        "stg_game_sides",
        "stg_turns",
        "stg_cards_seen",
        "dim_player",
        "dim_archetype",
        "dim_season",
        "dim_format",
        "dim_card",
        "dim_date",
        "fct_game_side",
        "mart_matchups",
        "mart_archetype_weekly",
        "mart_cards_seen",
        "mart_player_summary",
        "ml_split_cutoff",
        "features_turn",
    }
    assert expected <= built


def test_the_fact_has_two_rows_per_fixture_game(warehouse: duckdb.DuckDBPyConnection) -> None:
    assert scalar(warehouse, "select count(*) from fct_game_side") == SIDES_PER_GAME * FIXTURE_GAMES
    assert scalar(warehouse, "select count(distinct game_id) from fct_game_side") == FIXTURE_GAMES


def test_every_fact_row_has_a_key_and_a_dimension(warehouse: duckdb.DuckDBPyConnection) -> None:
    """The relationship tests dbt runs, restated as one query over the built tables."""
    orphans = scalar(
        warehouse,
        """
        select count(*) from fct_game_side f
        left join dim_season s on f.season_key = s.season_key
        left join dim_format t on f.format_key = t.format_key
        left join dim_date d on f.date_key = d.date_key
        where s.season_key is null or t.format_key is null or d.date_key is null
        """,
    )
    assert orphans == 0


def test_mart_matchups_is_symmetric(warehouse: duckdb.DuckDBPyConnection) -> None:
    """A vs B and B vs A exist as a pair and agree, wins against losses included."""
    assert scalar(warehouse, "select count(*) from mart_matchups") > 0
    broken = scalar(
        warehouse,
        """
        select count(*) from mart_matchups a
        left join mart_matchups b
          on a.archetype_key = b.opponent_archetype_key
         and a.opponent_archetype_key = b.archetype_key
        where b.archetype_key is null or a.games <> b.games or a.wins <> b.losses
        """,
    )
    assert broken == 0


def test_seen_rate_is_a_rate(warehouse: duckdb.DuckDBPyConnection) -> None:
    assert scalar(warehouse, "select count(*) from mart_cards_seen") > 0
    out_of_range = scalar(
        warehouse,
        "select count(*) from mart_cards_seen where seen_rate is null or seen_rate <= 0 "
        "or seen_rate > 1 or (inclusion_rate is not null and inclusion_rate not between 0 and 1)",
    )
    assert out_of_range == 0


def test_dim_player_holds_exactly_the_fixture_uploaders(
    warehouse: duckdb.DuckDBPyConnection, silver_from_fixtures: Path
) -> None:
    """Members only: one row per token that holds an uploader seat, strangers absent.

    The expected count is read back out of silver rather than hard coded, so
    the test still means something after the fixtures are refreshed.
    """
    sides = silver_from_fixtures / "lake" / "silver" / "game_sides"
    expected = scalar(
        duckdb.connect(),
        f"select count(distinct player_token) from "
        f"read_parquet('{sides}/**/*.parquet', hive_partitioning = true) "
        f"where is_uploader and player_token is not null",
    )
    assert expected > 0
    assert scalar(warehouse, "select count(*) from dim_player") == expected
    assert scalar(warehouse, "select count(*) from dim_player where player_key is null") == 0


def test_the_weekly_mart_counts_every_in_scope_seat_once(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    """Every seat the marts are allowed to see lands in exactly one week bucket.

    The equality is what would break first if the date dimension gained a
    duplicate or lost a date: a fan-out on the join would inflate the left
    side, a missing date would shrink it.
    """
    weekly_games = scalar(warehouse, "select coalesce(sum(games), 0) from mart_archetype_weekly")
    in_scope = scalar(
        warehouse,
        "select count(*) from fct_game_side "
        "where not excluded_from_stats and archetype_key is not null",
    )
    assert weekly_games == in_scope


def test_features_turn_has_a_row_per_seat_per_turn(warehouse: duckdb.DuckDBPyConnection) -> None:
    """The grain, checked against the games it was built from rather than a constant.

    The fixture corpus is small and most of its seats carry no archetype, so
    the number here is whatever the scope rules leave; what has to hold is the
    shape. Every in-scope game contributes two rows per turn, every turn number
    is inside the game, and turn 1 is the empty board.
    """
    games = scalar(warehouse, "select count(distinct game_id) from features_turn")
    assert games > 0
    expected = scalar(
        warehouse,
        "select coalesce(sum(turn_count), 0) * 2 from ("
        "select distinct game_id, turn_count from features_turn)",
    )
    assert scalar(warehouse, "select count(*) from features_turn") == expected
    outside = scalar(
        warehouse,
        "select count(*) from features_turn where turn_number > turn_count or turn_number < 1",
    )
    assert outside == 0
    assert (
        scalar(
            warehouse,
            "select count(*) from features_turn "
            "where turn_number = 1 and (prize_diff <> 0 or turns_played_self <> 0 "
            "or cards_drawn_self <> 0 or attacks_self <> 0)",
        )
        == 0
    )


def test_features_turn_has_no_nulls_in_any_feature(warehouse: duckdb.DuckDBPyConnection) -> None:
    """Every column of the table is non-null, asserted by reading the columns back.

    dbt already runs a `not_null` test on each of them, and this repeats the
    claim in a form that cannot fall out of step with the model: the column
    list comes from the built table, so a feature added in SQL and forgotten in
    schema.yml is still covered here.
    """
    columns = [
        name
        for (name,) in warehouse.sql(
            "select column_name from information_schema.columns where table_name = 'features_turn'"
        ).fetchall()
    ]
    assert len(columns) > 20
    predicate = " or ".join(f'"{name}" is null' for name in columns)
    assert scalar(warehouse, f"select count(*) from features_turn where {predicate}") == 0


def test_the_split_is_a_date_split(warehouse: duckdb.DuckDBPyConnection) -> None:
    """No game straddles the cutoff, and no training row is on the holdout side of it.

    A random split would pass a null check and fail this, which is the point:
    the rows of one game are near duplicates of each other, so a game on both
    sides of the boundary turns the holdout score into a memory test.
    """
    straddling = scalar(
        warehouse,
        "select count(*) from (select game_id from features_turn "
        "group by game_id having count(distinct split) > 1)",
    )
    assert straddling == 0
    misplaced = scalar(
        warehouse,
        "select count(*) from features_turn f, ml_split_cutoff c "
        "where (f.play_date >= c.holdout_start) <> (f.split = 'holdout')",
    )
    assert misplaced == 0
