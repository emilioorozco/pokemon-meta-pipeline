"""The publish stage: the item shapes in the fast suite, the table in the marked one.

Two halves, split the way the module is. The builders are pure functions over
rows of a mart, so they are tested from dictionaries written here: a matchup
with a null win rate, a week in the single digits, a mirror. Those are the
assertions that pin the contract the application is being written against, and
they cost milliseconds, so they run in the default suite.

The second half is marked `dbt`, because it needs a warehouse to read and the
only honest warehouse is the one `pipeline.gold` builds out of the committed
fixtures. Its table is moto's, an in-process DynamoDB that a real boto3 resource
talks to, so the batching, the pagination and the key schema are exercised
without an AWS account. What those tests are for is the part no unit test can
state: that a second publish leaves the table holding exactly the current run's
rows and nothing else.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest
from moto import mock_aws

from pipeline import publish
from pipeline.publish import (
    PK_ARCHETYPE,
    PK_MATCHUP,
    PK_META,
    SK_META,
    WEEKLY_PREFIX,
    Marts,
    archetype_items,
    build_items,
    matchup_item,
    meta_item,
    now_iso,
    rate,
    weekly_item,
)

REGION: Final = "us-west-2"
TABLE: Final = "pra-test-insights"
RUN_ID: Final = "publish-run-1"
PUBLISHED_AT: Final = "2026-09-22T09:00:00Z"
CHARIZARD: Final = "name:charizard-ex"
GARDEVOIR: Final = "name:gardevoir-ex"


def matchup_row(**overrides: Any) -> dict[str, Any]:
    """One `mart_matchups` row, as the reader hands it to the builder."""
    row: dict[str, Any] = {
        "archetype_key": CHARIZARD,
        "archetype_name": "Charizard ex",
        "opponent_archetype_key": GARDEVOIR,
        "opponent_archetype_name": "Gardevoir ex",
        "games": 12,
        "wins": 7,
        "losses": 4,
        "ties": 1,
        "win_rate": 7 / 11,
        "min_games_met": True,
    }
    return {**row, **overrides}


def weekly_row(**overrides: Any) -> dict[str, Any]:
    """One `mart_archetype_weekly` row, as the reader hands it to the builder."""
    row: dict[str, Any] = {
        "archetype_key": CHARIZARD,
        "archetype_name": "Charizard ex",
        "iso_year": 2026,
        "iso_week": 9,
        "week_start": "2026-02-23",
        "games": 6,
        "wins": 4,
        "losses": 2,
        "ties": 0,
        "win_rate": 4 / 6,
        "share_of_week": 0.125,
    }
    return {**row, **overrides}


# ------------------------------------------------------------------ fast --


def test_a_matchup_item_is_keyed_by_the_ordered_pair() -> None:
    item = matchup_item(matchup_row(), run_id=RUN_ID, published_at=PUBLISHED_AT)

    assert item["pk"] == PK_MATCHUP
    assert item["sk"] == f"{CHARIZARD}#{GARDEVOIR}"
    assert item["archetypeKey"] == CHARIZARD
    assert item["archetypeName"] == "Charizard ex"
    assert item["opponentArchetypeKey"] == GARDEVOIR
    assert item["opponentArchetypeName"] == "Gardevoir ex"
    assert item["minGamesMet"] is True
    assert item["runId"] == RUN_ID
    assert item["publishedAt"] == PUBLISHED_AT


def test_every_number_is_a_decimal_and_every_rate_is_a_percentage() -> None:
    """DynamoDB has one numeric type and boto3 refuses a float, so nothing is one."""
    item = matchup_item(matchup_row(), run_id=RUN_ID, published_at=PUBLISHED_AT)

    numbers = {"games", "wins", "losses", "ties", "winRate"}
    assert all(isinstance(item[name], Decimal) for name in numbers), item
    assert not any(isinstance(value, float) for value in item.values())
    assert item["games"] == Decimal("12")
    # 7/11 is 0.636363... in the mart and 63.64 on the wire: the application
    # reads percentages, and the stored value is the one it renders.
    assert item["winRate"] == Decimal("63.64")


def test_a_matchup_with_no_decided_game_omits_the_rate_rather_than_writing_a_null() -> None:
    item = matchup_item(
        matchup_row(wins=0, losses=0, ties=2, games=2, win_rate=None, min_games_met=False),
        run_id=RUN_ID,
        published_at=PUBLISHED_AT,
    )

    assert "winRate" not in item
    assert item["minGamesMet"] is False
    assert item["ties"] == Decimal("2")


def test_a_missing_archetype_name_falls_back_to_the_key() -> None:
    """The dimension join can miss; an item with no name is worse than one named by key."""
    item = matchup_item(
        matchup_row(archetype_name=None, opponent_archetype_name=""),
        run_id=RUN_ID,
        published_at=PUBLISHED_AT,
    )

    assert item["archetypeName"] == CHARIZARD
    assert item["opponentArchetypeName"] == GARDEVOIR


def test_a_weekly_item_is_keyed_by_archetype_and_a_zero_padded_week() -> None:
    """Week 9 has to sort before week 10, which it only does zero padded."""
    single = weekly_item(weekly_row(iso_week=9), run_id=RUN_ID, published_at=PUBLISHED_AT)
    double = weekly_item(weekly_row(iso_week=10), run_id=RUN_ID, published_at=PUBLISHED_AT)

    assert single["pk"] == f"{WEEKLY_PREFIX}{CHARIZARD}"
    assert single["sk"] == "2026-W09"
    assert double["sk"] == "2026-W10"
    assert sorted([double["sk"], single["sk"]]) == ["2026-W09", "2026-W10"]
    assert single["isoYear"] == Decimal("2026")
    assert single["isoWeek"] == Decimal("9")
    assert single["weekStart"] == "2026-02-23"
    assert single["winRate"] == Decimal("66.67")
    assert single["shareOfWeek"] == Decimal("12.5")


def test_a_week_start_that_arrives_as_a_date_is_published_as_a_day() -> None:
    """DuckDB hands back a `date`; the contract says YYYY-MM-DD."""
    from datetime import date

    item = weekly_item(
        weekly_row(week_start=date(2026, 2, 23)), run_id=RUN_ID, published_at=PUBLISHED_AT
    )

    assert item["weekStart"] == "2026-02-23"


def test_a_week_with_no_share_omits_the_attribute() -> None:
    item = weekly_item(
        weekly_row(win_rate=None, share_of_week=None), run_id=RUN_ID, published_at=PUBLISHED_AT
    )

    assert "winRate" not in item
    assert "shareOfWeek" not in item


def test_archetype_rows_are_the_matchup_rows_summed() -> None:
    """One row per archetype, over every opponent, with the mirror counted as it is stored."""
    rows = [
        matchup_row(games=12, wins=7, losses=4, ties=1),
        matchup_row(opponent_archetype_key="name:miraidon-ex", games=8, wins=3, losses=5, ties=0),
        matchup_row(
            archetype_key=GARDEVOIR,
            archetype_name="Gardevoir ex",
            opponent_archetype_key=CHARIZARD,
            opponent_archetype_name="Charizard ex",
            games=12,
            wins=4,
            losses=7,
            ties=1,
        ),
    ]

    items = archetype_items(rows, run_id=RUN_ID, published_at=PUBLISHED_AT)

    assert [item["sk"] for item in items] == [CHARIZARD, GARDEVOIR]
    zard = items[0]
    assert zard["pk"] == PK_ARCHETYPE
    assert zard["archetypeName"] == "Charizard ex"
    assert zard["games"] == Decimal("20")
    assert zard["wins"] == Decimal("10")
    assert zard["losses"] == Decimal("9")
    assert zard["winRate"] == Decimal(str(round(100 * 10 / 19, 2)))
    # Ties are counted into `games` but the archetype row publishes no tie count.
    assert "ties" not in zard


def test_an_archetype_with_no_decided_game_omits_the_rate() -> None:
    items = archetype_items(
        [matchup_row(games=2, wins=0, losses=0, ties=2, win_rate=None)],
        run_id=RUN_ID,
        published_at=PUBLISHED_AT,
    )

    assert "winRate" not in items[0]


def test_the_meta_row_omits_the_model_when_nothing_holds_the_alias() -> None:
    item = meta_item(
        run_id=RUN_ID,
        published_at=PUBLISHED_AT,
        games_total=128,
        matchup_rows=42,
        weekly_rows=17,
        archetype_rows=9,
        model=None,
        source_commit=None,
        min_games=None,
    )

    assert item["pk"] == PK_META
    assert item["sk"] == SK_META
    assert item["modelName"] == "win-probability"
    assert item["gamesTotal"] == Decimal("128")
    assert item["matchupRows"] == Decimal("42")
    assert item["weeklyRows"] == Decimal("17")
    assert item["archetypeRows"] == Decimal("9")
    assert "modelVersion" not in item
    assert "modelAlias" not in item
    assert "sourceCommit" not in item
    assert "minGames" not in item


def test_the_meta_row_names_the_version_and_the_commit_when_there_are_some() -> None:
    item = meta_item(
        run_id=RUN_ID,
        published_at=PUBLISHED_AT,
        games_total=128,
        matchup_rows=42,
        weekly_rows=17,
        archetype_rows=9,
        model=("3", "production"),
        source_commit="0" * 40,
        min_games=5,
    )

    assert item["modelVersion"] == "3"
    assert isinstance(item["modelVersion"], str)
    assert item["modelAlias"] == "production"
    assert item["sourceCommit"] == "0" * 40
    assert item["minGames"] == Decimal("5")


def test_the_min_games_var_is_read_out_of_the_dbt_project() -> None:
    """The threshold is defined once, in `dbt_project.yml`, and published from there."""
    assert publish.dbt_min_games() == 5


def test_an_unreadable_dbt_project_leaves_the_threshold_out(tmp_path: Path) -> None:
    assert publish.dbt_min_games(tmp_path / "nothing.yml") is None


def test_a_rate_is_a_percentage_and_a_non_number_is_no_rate() -> None:
    assert rate(None) is None
    assert rate(float("nan")) is None
    assert rate(0.5) == Decimal("50.0")
    assert rate(1) == Decimal("100")
    assert rate(0) == Decimal("0")


def test_the_published_timestamp_is_iso_utc_to_the_second() -> None:
    assert now_iso(datetime(2026, 9, 22, 9, 0, 0, 123456, tzinfo=UTC)) == "2026-09-22T09:00:00Z"


def test_build_items_carries_one_run_identifier_onto_everything() -> None:
    marts = Marts(matchups=[matchup_row()], weekly=[weekly_row()], games_total=12)

    matchups, weekly, archetypes, meta = build_items(
        marts, run_id=RUN_ID, published_at=PUBLISHED_AT, min_games=5
    )

    everything = [*matchups, *weekly, *archetypes, meta]
    assert len(everything) == 4
    assert {item["runId"] for item in everything} == {RUN_ID}
    assert {item["publishedAt"] for item in everything} == {PUBLISHED_AT}
    assert meta["matchupRows"] == Decimal("1")
    assert meta["gamesTotal"] == Decimal("12")


def test_a_publish_that_was_given_no_table_exits_two_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PRA_INSIGHTS_TABLE", raising=False)

    with pytest.raises(SystemExit) as raised:
        publish.main([])

    assert raised.value.code == 2
    assert "PRA_INSIGHTS_TABLE" in capsys.readouterr().err


# ------------------------------------------------------------------ table --


@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials so a misconfigured run can never reach a real account."""
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SECURITY_TOKEN"):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def table(aws_credentials: None) -> Iterator[Any]:
    """An empty table with the contract's key schema, served in process by moto."""
    import boto3

    with mock_aws():
        resource = boto3.resource("dynamodb", region_name=REGION)
        created = resource.create_table(
            TableName=TABLE,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        created.wait_until_exists()
        yield created


def empty_registry(root: Path) -> str:
    """A tracking URI with no registered model behind it.

    A path under the test's own temporary directory rather than one that does
    not exist: MLflow's file store creates its root as it opens it, so pointing
    at nowhere would either fail or write somewhere real.
    """
    return f"file:{root / 'mlruns'}"


def scan_all(table: Any) -> list[dict[str, Any]]:
    """Every item in the table, which is small enough for one page."""
    items: list[dict[str, Any]] = []
    response = table.scan()
    items += response["Items"]
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items += response["Items"]
    return items


def test_more_than_one_batch_of_items_round_trips(table: Any) -> None:
    """Thirty items is two BatchWriteItem calls, and all thirty have to be there."""
    items = [
        matchup_item(
            matchup_row(opponent_archetype_key=f"name:deck-{index:02d}"),
            run_id=RUN_ID,
            published_at=PUBLISHED_AT,
        )
        for index in range(30)
    ]

    assert publish.write_items(table, items) == 30

    stored = scan_all(table)
    assert len(stored) == 30
    assert {item["sk"] for item in stored} == {item["sk"] for item in items}
    one = next(item for item in stored if item["sk"].endswith("deck-00"))
    assert one["games"] == Decimal("12")
    assert one["winRate"] == Decimal("63.64")
    assert one["minGamesMet"] is True


def test_nothing_to_write_is_not_a_call(table: Any) -> None:
    assert publish.write_items(table, []) == 0
    assert publish.delete_keys(table, []) == 0
    assert scan_all(table) == []


# ------------------------------------------------------------- warehouse --


@pytest.fixture(scope="module")
def built_warehouse(silver_from_fixtures: Path) -> Path:
    """The fixture corpus through the real dbt build, for the tests that read marts."""
    from pipeline.gold import run_gold

    assert run_gold(data_dir=silver_from_fixtures) == 0
    return silver_from_fixtures / "warehouse" / "meta.duckdb"


@pytest.mark.dbt
def test_the_marts_read_out_of_a_real_warehouse(built_warehouse: Path) -> None:
    marts = publish.read_marts(built_warehouse)

    assert marts.matchups, "the fixture corpus has matchups"
    assert marts.weekly, "the fixture corpus spans weeks"
    assert marts.games_total > 0
    assert set(marts.matchups[0]) >= {"archetype_key", "games", "win_rate", "min_games_met"}


@pytest.mark.dbt
def test_a_missing_warehouse_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(publish.PublishError) as raised:
        publish.read_marts(tmp_path / "nothing.duckdb")
    assert "nothing.duckdb" in str(raised.value)


@pytest.mark.dbt
def test_one_publish_writes_the_meta_row_and_every_mart_row(
    built_warehouse: Path, table: Any, tmp_path: Path
) -> None:
    summary = publish.run_publish(
        warehouse=built_warehouse,
        table_name=TABLE,
        region=REGION,
        tracking_uri=empty_registry(tmp_path),
        run_id=RUN_ID,
        table=table,
    )

    stored = scan_all(table)
    assert len(stored) == summary.written
    assert {item["runId"] for item in stored} == {RUN_ID}
    assert summary.matchups > 0 and summary.weekly > 0 and summary.archetypes > 0
    assert summary.deleted == 0

    kinds = {
        PK_MATCHUP: 0,
        PK_ARCHETYPE: 0,
        PK_META: 0,
        WEEKLY_PREFIX: 0,
    }
    for item in stored:
        key = str(item["pk"])
        kinds[key if key in kinds else WEEKLY_PREFIX] += 1
    assert kinds[PK_MATCHUP] == summary.matchups
    assert kinds[WEEKLY_PREFIX] == summary.weekly
    assert kinds[PK_ARCHETYPE] == summary.archetypes
    assert kinds[PK_META] == 1

    meta = next(item for item in stored if item["pk"] == PK_META)
    assert meta["sk"] == SK_META
    assert meta["modelName"] == "win-probability"
    assert meta["matchupRows"] == Decimal(summary.matchups)
    assert meta["weeklyRows"] == Decimal(summary.weekly)
    assert meta["archetypeRows"] == Decimal(summary.archetypes)
    assert meta["gamesTotal"] > Decimal("0")
    assert meta["minGames"] == Decimal("5")
    # No registry at that URI, so the alias is unheld and the row says nothing.
    assert "modelVersion" not in meta


@pytest.mark.dbt
def test_a_second_publish_sweeps_away_what_the_first_one_left(
    built_warehouse: Path, table: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that leaves the mart leaves the table, and the meta row moves on."""
    first = publish.run_publish(
        warehouse=built_warehouse,
        table_name=TABLE,
        region=REGION,
        tracking_uri=empty_registry(tmp_path),
        run_id=RUN_ID,
        table=table,
    )
    dropped = publish.read_marts(built_warehouse).matchups[0]
    dropped_sk = f"{dropped['archetype_key']}#{dropped['opponent_archetype_key']}"
    real_read = publish.read_marts

    def shorter(warehouse: Path) -> Marts:
        marts = real_read(warehouse)
        return Marts(
            matchups=marts.matchups[1:], weekly=marts.weekly, games_total=marts.games_total
        )

    monkeypatch.setattr(publish, "read_marts", shorter)

    second = publish.run_publish(
        warehouse=built_warehouse,
        table_name=TABLE,
        region=REGION,
        tracking_uri=empty_registry(tmp_path),
        run_id="publish-run-2",
        table=table,
    )

    stored = scan_all(table)
    assert {item["runId"] for item in stored} == {"publish-run-2"}
    assert len(stored) == second.written
    assert second.matchups == first.matchups - 1
    assert second.deleted > 0
    assert not [item for item in stored if item["pk"] == PK_MATCHUP and item["sk"] == dropped_sk]
    meta = next(item for item in stored if item["pk"] == PK_META)
    assert meta["runId"] == "publish-run-2"
    assert meta["matchupRows"] == Decimal(second.matchups)


@pytest.mark.dbt
def test_a_dry_run_writes_nothing_and_shows_three_samples(
    built_warehouse: Path, table: Any, tmp_path: Path
) -> None:
    summary = publish.run_publish(
        warehouse=built_warehouse,
        table_name=TABLE,
        region=REGION,
        tracking_uri=empty_registry(tmp_path),
        run_id=RUN_ID,
        dry_run=True,
        table=table,
    )

    assert scan_all(table) == []
    assert summary.written > 0
    assert summary.deleted == 0
    assert summary.samples is not None and len(summary.samples) == 3
    # Archetype names and counts only: no player token reaches this stage, and a
    # sample printed to a terminal is the output a person pastes somewhere else.
    assert not any("player" in name.lower() for item in summary.samples for name in item)
    assert "would write" in str(summary)
