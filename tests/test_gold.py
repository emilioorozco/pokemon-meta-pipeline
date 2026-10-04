"""Gold end to end: build silver from the committed games, then run dbt over it.

The silver under it is built from the committed games plus one duplicated
upload (`bronze_with_duplicate_upload`), which is the batch that failed the
first nightly run on dev: every uniqueness test dbt runs below is therefore
also a test that silver's collapse held.

Every test here is marked `dbt` and the default `uv run pytest` skips them, the
same split the Spark tests get and for the same reason: the module builds a
real silver lake (a Java Virtual Machine) and then a real DuckDB warehouse
(`dbt run` and `dbt test`). `uv run pytest -m dbt` runs them.

The point of the module is that the dbt project is exercised as a project.
`run_gold` is the same entry point orchestration will call, the profile is the
committed one, and the 121 generic and singular dbt tests run as part of it, so
a broken key or a broken relationship fails here without a Python assertion
having to name it. The assertions below are the handful of numbers a dbt test
cannot state: the grain against the fixture count, the symmetry of the matchup
mart as an independent query, the bounds on a rate, and the size of
`dim_player` against the tokens silver actually wrote.
"""

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import duckdb
import pytest

from pipeline.config import REPO_ROOT
from tests.conftest import FIXTURES_DIR, DuplicateUpload

pytestmark = pytest.mark.dbt

FIXTURE_GAMES = len(sorted(FIXTURES_DIR.glob("*.json")))
SIDES_PER_GAME = 2
# The `min_games` dbt variable, which every mart carries as `min_games_met`.
MIN_GAMES = 5

# The application's ten pace numbers for both seats of each fixture game, and
# the column of `int_game_side_pace` each one has to equal. The keys are the
# application's own spelling, which is how they are stored.
PACE_DIR: Final = REPO_ROOT / "evals" / "fixtures" / "pace"
PACE_COLUMNS: Final[dict[str, str]] = {
    "firstAttackTurn": "first_attack_turn",
    "turnsWithoutAttackShare": "turns_without_attack_share",
    "energyPerTurn": "energy_per_turn",
    "prizesByTurn4": "prizes_by_turn_4",
    "prizesByTurn6": "prizes_by_turn_6",
    "prizesByTurn8": "prizes_by_turn_8",
    "prizesByTurn10": "prizes_by_turn_10",
    "firstPrizeTurn": "first_prize_turn",
    "firstKnockoutTurn": "first_knockout_turn",
    "concessionTurn": "concession_turn",
}
# `int_game_side_pace` is ephemeral, so it is a common table expression inside
# the mart and never a relation. dbt still compiles it to a file of its own,
# with every `ref` already resolved to a built relation, which is what lets a
# test run the per-seat grain the numbers are checkable at.
COMPILED_PACE: Final = (
    REPO_ROOT
    / "dbt"
    / "target"
    / "compiled"
    / "play_rough_pipeline"
    / "models"
    / "marts"
    / "int_game_side_pace.sql"
)
# Both sides round to three decimals, so the two can only differ by the last
# bit of a double. Anything a person would call a different number is far
# outside this.
PACE_TOLERANCE: Final = 1e-9


@pytest.fixture(scope="module")
def warehouse(gold_from_fixtures: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """A read-only handle on the warehouse the session fixture built.

    The build itself is `gold_from_fixtures` in `conftest.py`, which asserts
    the exit code: every test here would be meaningless after a failed build,
    and one setup failure says so more clearly than eight identical errors.
    It is shared with the agent tests, which query the same tables through the
    agent's SQL tool rather than through a connection of their own.
    """
    connection = duckdb.connect(str(gold_from_fixtures), read_only=True)
    yield connection
    connection.close()


def scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    """The single number a counting query returns, asserted to be one."""
    row = connection.sql(sql).fetchone()
    assert row is not None
    return int(row[0])


def rows_of(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, object]]:
    """Every row of a query as a dictionary, so a test can name its columns."""
    cursor = connection.sql(sql)
    names = [description[0] for description in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def count_of(value: object) -> float:
    """One number out of a row, narrowed: a cell DuckDB typed as anything else is the bug."""
    assert isinstance(value, int | float), value
    return float(value)


def same_number(got: object, want: object) -> bool:
    """Whether two pace numbers agree, with `None` meaning "the game had none"."""
    if got is None or want is None:
        return got is None and want is None
    assert isinstance(got, int | float)
    assert isinstance(want, int | float)
    return math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=PACE_TOLERANCE)


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
        "mart_archetype_pace",
        "mart_cards_seen",
        "mart_player_summary",
        "ml_split_cutoff",
        "features_turn",
    }
    assert expected <= built


def test_the_fact_has_two_rows_per_fixture_game(warehouse: duckdb.DuckDBPyConnection) -> None:
    assert scalar(warehouse, "select count(*) from fct_game_side") == SIDES_PER_GAME * FIXTURE_GAMES
    assert scalar(warehouse, "select count(distinct game_id) from fct_game_side") == FIXTURE_GAMES


def test_a_game_uploaded_twice_reaches_gold_once(
    warehouse: duckdb.DuckDBPyConnection, duplicate_upload: DuplicateUpload
) -> None:
    """PLA-175: the silver bronze under this build holds two blobs for one game.

    Uncollapsed it broke five tests at once, `unique_stg_games_game_id`,
    `unique_fct_game_side_game_side_key`, `unique_features_turn_feature_key`,
    `assert_two_sides_per_game` and `assert_features_turn_covers_both_seats`.
    Those run as part of the build this fixture asserts the exit code of, so
    what is left to state here is the count the duplicate was folded into.
    """
    duplicated = f"where game_id = '{duplicate_upload.game_id}'"
    assert scalar(warehouse, f"select count(*) from stg_games {duplicated}") == 1
    assert scalar(warehouse, f"select upload_count from stg_games {duplicated}") == 2
    assert scalar(warehouse, f"select count(*) from fct_game_side {duplicated}") == SIDES_PER_GAME
    assert scalar(warehouse, "select count(*) from fct_game_side where upload_count = 2") == (
        SIDES_PER_GAME
    )
    assert scalar(warehouse, "select min(upload_count) from stg_games") == 1


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


def test_the_pace_numbers_equal_the_applications_own(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    """The whole point of `mart_archetype_pace`, stated once.

    The application computes ten pace numbers for a member's own seat when the
    game is uploaded, in TypeScript, from the parsed log. The pipeline
    computes the same ten for every seat in the league, in SQL, from silver.
    Two writings of one definition drift, and nothing but a test that runs
    both and subtracts will notice: a comment claiming they agree is worth
    nothing the week somebody changes one of them.

    So this runs the per-seat intermediate over the same ten games the
    application's own snapshot test runs over and asserts every number of
    every seat. The per-seat grain is deliberate: the mart is an average, and
    an average can be right on a corpus where half the rows are wrong.

    The expected values are under `evals/fixtures/pace/`, produced read-only
    by the application's own `analyzeGame` (see the README there). An exclusion
    that drifts shows up here as one named column of one named seat.
    """
    assert COMPILED_PACE.is_file(), (
        f"{COMPILED_PACE} is missing; the gold build compiles it, so this means the build "
        "did not run or the model was renamed"
    )
    rows = {
        (row["game_id"], row["seat"]): row
        for row in rows_of(warehouse, f"select * from ({COMPILED_PACE.read_text('utf-8')})")
    }
    assert len(rows) == SIDES_PER_GAME * FIXTURE_GAMES

    fixtures = sorted(PACE_DIR.glob("game-*.json"))
    assert len(fixtures) == FIXTURE_GAMES, PACE_DIR
    wrong: list[str] = []
    for path in fixtures:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        for seat_text, pace in fixture["seats"].items():
            row = rows[(fixture["game_id"], int(seat_text))]
            for name, column in PACE_COLUMNS.items():
                if not same_number(row[column], pace[name]):
                    wrong.append(
                        f"{path.name} seat {seat_text} {name}: "
                        f"the warehouse says {row[column]!r}, the application {pace[name]!r}"
                    )
    assert not wrong, "\n".join(wrong)


def test_the_pace_mart_is_the_average_of_its_seats(warehouse: duckdb.DuckDBPyConnection) -> None:
    """One archetype's row, recomputed from the seats behind it.

    The mart's own arithmetic rather than its definitions, which the test
    above owns. Two archetypes in the fixture corpus hold two seats each and
    the rest hold one, so this is also the check that an average over more
    than one row is an average and not a last-row-wins.
    """
    seats = rows_of(
        warehouse,
        f"""
        select archetype_key, first_attack_turn, energy_per_turn
        from ({COMPILED_PACE.read_text("utf-8")})
        where not excluded_from_stats
          and archetype_key is not null
          and counted_turns is not null
        """,
    )
    assert seats
    mart = {
        row["archetype_key"]: row for row in rows_of(warehouse, "select * from mart_archetype_pace")
    }
    assert len(mart) == len({seat["archetype_key"] for seat in seats})
    assert max(count_of(row["games"]) for row in mart.values()) > 1

    for key, row in mart.items():
        mine = [seat for seat in seats if seat["archetype_key"] == key]
        assert count_of(row["games"]) == len(mine)
        assert row["min_games_met"] is (len(mine) >= MIN_GAMES)
        for column in ("first_attack_turn", "energy_per_turn"):
            values = [seat[column] for seat in mine if seat[column] is not None]
            expected = sum(count_of(value) for value in values) / len(values) if values else None
            assert same_number(row[column], expected), (key, column)


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
