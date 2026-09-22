"""Publish: the marts as rows in the application's DynamoDB table.

This is the step that closes the loop. Everything before it lands in a DuckDB
file on whichever machine ran the pipeline, which is exactly the wrong place for
the web application to read from: it is a single file, it is not in the
application's account, and a reader holding it open blocks the next rebuild. So
the last stage of a run copies the public-safe marts into the one store the
application already reads on every request, keyed the way its access patterns
want them rather than the way the warehouse stores them.

**The shape.** One table, `PRA_INSIGHTS_TABLE`, with the `pk`/`sk` pair every
row carries and four partition-key families over it:

| pk                    | sk                        | what it holds                 |
| --------------------- | ------------------------- | ----------------------------- |
| `MATCHUP`             | `<archetype>#<opponent>`  | one ordered pair of archetypes |
| `WEEKLY#<archetype>`  | `<isoYear>-W<isoWeek>`    | one archetype in one ISO week |
| `ARCHETYPE`           | `<archetype>`             | one archetype, all opponents  |
| `META`                | `LATEST`                  | what the last publish was     |

The matchup and weekly families are `mart_matchups` and `mart_archetype_weekly`
row for row. The archetype family is aggregated here rather than read from a
mart, over the matchup rows for that archetype: summing them counts every seat
the archetype held against a known opponent, mirrors included once per seat, and
it is the same denominator the matchup cells use, so a total and its parts agree.
`mart_archetype_weekly` would have given a different number (it keeps seats whose
opponent archetype is unknown) and `dim_archetype.games_played` a third one (it
counts before the `excluded_from_stats` filter), and three numbers called "games"
is how an application ends up showing two of them on one page.

Attribute names are camelCase because the application is TypeScript and these
rows are read straight into its models; the warehouse's snake_case stops at this
boundary. Numbers are `Decimal`, because DynamoDB has one numeric type and boto3
refuses a float outright rather than rounding one silently. `winRate` and
`shareOfWeek` are percentages, 0 to 100 rounded to two places, which is the
application's wire format for a rate and not the marts' own: the warehouse
divides and gets a fraction, and this is the boundary that scales it. Either way
the value the application renders is the value this wrote, not a base-2
approximation of it.

**Refresh, not update.** A run writes every row it built under its own
`runId` and then deletes, family by family, whatever still carries an older one.
That ordering is deliberate: at no point is the table missing a row it had
before, so a reader mid-publish sees the old row or the new one and never a gap.
The alternative, deleting first, would have shown an empty matchup matrix for
however long the write took.

Finding the weekly partitions to sweep needs both halves of a belt and braces:
the archetype keys just published are the ones that should be there, and a
bounded `Scan` filtered on `begins_with(pk, "WEEKLY#")` finds the ones that
should not, which is exactly the case deletion exists for (an archetype that
dropped out of the corpus keeps its partition for ever otherwise). The Scan is
the one part of this that does not scale for free: it reads the whole table,
which is fine at a few thousand rows and would not be at a few million. The
replacement when that day comes is a global secondary index on `runId`, queried
for the rows of the previous run instead of scanning for all of them.

**Credentials.** Nothing here names a profile or a role: it is boto3's default
chain, the same as every other stage that talks to AWS. What changed for this
ticket is on the other side of that chain, in the application's account: the
role the pipeline assumes was read-only and now carries write on this one table,
and on no other table and no other resource.
"""

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import duckdb
from boto3.dynamodb.conditions import Attr, Key

from pipeline.config import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    REPO_ROOT,
    WAREHOUSE_PATH,
    default_tracking_uri,
)
from pipeline.observability import (
    RunMetrics,
    configure_logging,
    current_run_id,
    emit_summary,
    git_commit,
    stage_run,
)
from pipeline.settings import Settings, SettingsError

if TYPE_CHECKING:  # the boto3 stubs are a dev dependency, not a runtime one
    from mypy_boto3_dynamodb.service_resource import Table

logger = logging.getLogger(__name__)

STAGE: Final = "publish"

#: One row of the table, as boto3's resource interface wants it: native Python
#: values, with `Decimal` for every number.
Item = dict[str, Any]

PK_MATCHUP: Final = "MATCHUP"
PK_ARCHETYPE: Final = "ARCHETYPE"
PK_META: Final = "META"
SK_META: Final = "LATEST"
WEEKLY_PREFIX: Final = "WEEKLY#"

KEY_PK: Final = "pk"
KEY_SK: Final = "sk"
RUN_ID_ATTRIBUTE: Final = "runId"

# BatchWriteItem's own limit, not a tuning choice: DynamoDB refuses 26.
BATCH_SIZE: Final = 25
# Unprocessed items mean the table throttled, so the retry is a back-off rather
# than an immediate resend. Five attempts at the base below is a little over
# three seconds of waiting, which outlasts a burst without hanging a DAG task.
MAX_ATTEMPTS: Final = 5
BACKOFF_BASE_S: Final = 0.2
# Rates cross this boundary as percentages, 0 to 100, because that is the wire
# format the rest of the application uses and a page that renders one number as
# 0.5556 and another as 55.56 is a bug waiting in the presentation layer. Two
# places, so a percentage keeps the four significant figures the fraction had.
RATE_SCALE: Final = 100
RATE_PLACES: Final = 2

DBT_PROJECT: Final = REPO_ROOT / "dbt" / "dbt_project.yml"
# The `min_games` var, read out of the dbt project rather than restated here: a
# second copy of the threshold in Python is the one that would drift. A regular
# expression rather than a YAML parse because the project file is ours, the line
# is two levels deep under `vars:` and nothing else in the pipeline needs a YAML
# dependency to read one integer.
MIN_GAMES_PATTERN: Final = re.compile(r"^\s+min_games:\s*(\d+)\s*$", re.MULTILINE)

MATCHUP_QUERY: Final = """
select
    archetype_key,
    archetype_name,
    opponent_archetype_key,
    opponent_archetype_name,
    games,
    wins,
    losses,
    ties,
    win_rate,
    min_games_met
from mart_matchups
order by archetype_key, opponent_archetype_key
"""

WEEKLY_QUERY: Final = """
select
    archetype_key,
    archetype_name,
    iso_year,
    iso_week,
    week_start,
    games,
    wins,
    losses,
    ties,
    win_rate,
    share_of_week
from mart_archetype_weekly
order by archetype_key, iso_year, iso_week
"""

# The one number the application shows that no mart carries: how many games the
# whole thing was built from, counted the way the marts count, once per game.
GAMES_TOTAL_QUERY: Final = """
select count(distinct game_id)
from fct_game_side
where not excluded_from_stats
"""


class PublishError(RuntimeError):
    """The marts could not be read or the table could not be written; the message says which."""


# ------------------------------------------------------------------ input --


@dataclass(frozen=True)
class Marts:
    """Everything the publish reads, in one object so the reader is one call."""

    matchups: list[dict[str, Any]]
    weekly: list[dict[str, Any]]
    games_total: int

    @property
    def rows(self) -> int:
        """Mart rows read, which is what the `rows_in` of the metrics row means."""
        return len(self.matchups) + len(self.weekly)


def _rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """A query's rows as dictionaries, so the builders below name their columns."""
    result = connection.sql(sql)
    names = list(result.columns)
    return [dict(zip(names, values, strict=True)) for values in result.fetchall()]


def read_marts(warehouse: Path) -> Marts:
    """The two marts and the game count, from a read-only connection.

    Read-only because a publish must never be the thing that changes the
    warehouse, and because the gate that ran just before it may still be
    holding the file.
    """
    if not warehouse.is_file():
        raise PublishError(f"no warehouse at {warehouse}; run `python -m pipeline.gold` first")
    try:
        connection = duckdb.connect(str(warehouse), read_only=True)
    except duckdb.Error as unreadable:
        raise PublishError(f"{warehouse} could not be opened: {unreadable}") from unreadable
    try:
        matchups = _rows(connection, MATCHUP_QUERY)
        weekly = _rows(connection, WEEKLY_QUERY)
        total = connection.sql(GAMES_TOTAL_QUERY).fetchone()
    except duckdb.Error as missing:
        raise PublishError(
            f"{warehouse} has no readable marts; run `python -m pipeline.gold` first ({missing})"
        ) from missing
    finally:
        connection.close()
    return Marts(matchups=matchups, weekly=weekly, games_total=int(total[0]) if total else 0)


def dbt_min_games(project: Path = DBT_PROJECT) -> int | None:
    """The `min_games` dbt var, or None when the project file cannot be read.

    None rather than a guess: a wrong threshold in `META` would have the
    application labelling thin cells as solid ones, which is worse than the
    attribute being absent and the application falling back to its own default.
    """
    try:
        text = project.read_text(encoding="utf-8")
    except OSError as unreadable:
        logger.warning(
            "dbt project unreadable", extra={"path": str(project), "error": str(unreadable)}
        )
        return None
    found = MIN_GAMES_PATTERN.search(text)
    if found is None:
        logger.warning("dbt project names no min_games var", extra={"path": str(project)})
        return None
    return int(found.group(1))


def production_model(tracking_uri: str) -> tuple[str, str] | None:
    """The version holding the `production` alias, or None when there is not one.

    Three ways to have no answer and all of them are normal: MLflow is an
    optional extra and may not be installed, the registry may not exist on this
    machine, and on a small corpus the promotion gate often refuses every
    candidate so the alias is unheld. A publish is still worth doing in all
    three cases; the `META` row simply says nothing about a model.

    `OSError` is caught beside MLflow's own exception because a file store
    creates its directory as it opens it, so an unreachable tracking directory
    fails as an operating-system error rather than as a registry one. Either
    way the answer is the same: this stage publishes marts, and a registry it
    cannot read is not a reason to withhold them.
    """
    try:
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.info("mlflow is not installed, so the published meta names no model version")
        return None
    if tracking_uri.startswith("file:"):
        # The same opt-in the trainer and the promotion step make: a plain
        # directory is a supported store in MLflow 3 only when this is set.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        found = MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
        )
    except (MlflowException, OSError) as unavailable:
        logger.info(
            "no production alias to publish",
            extra={"tracking_uri": tracking_uri, "reason": str(unavailable)},
        )
        return None
    return str(found.version), PRODUCTION_ALIAS


# ----------------------------------------------------------------- values --


def number(value: object) -> Decimal:
    """A count as DynamoDB's one numeric type. Ints only; a rate goes through `rate`."""
    return Decimal(str(int(value)))  # type: ignore[call-overload]


def rate(value: object) -> Decimal | None:
    """A mart's fraction as a percentage to two places, or None when there was nothing to divide.

    The marts compute rates as fractions, 0 to 1, because that is what a
    division is; the application reads percentages, 0 to 100. The conversion
    happens here, at the boundary, rather than in either of the two places it
    could have happened in twice.

    `str()` and not `Decimal(float)`: the latter keeps all fifty-odd digits of
    the binary expansion, which DynamoDB then rejects for exceeding 38 digits
    of precision, and which nobody wanted stored anyway.
    """
    if value is None:
        return None
    number_value = float(value)  # type: ignore[arg-type]
    if number_value != number_value:  # NaN, which a rate over an empty group can be
        return None
    return Decimal(str(round(number_value * RATE_SCALE, RATE_PLACES)))


def _text(value: object, fallback: object) -> str:
    """A name as a non-empty string, falling back to the key when the join found none."""
    if value is None or str(value) == "":
        return str(fallback)
    return str(value)


def _with_rate(item: Item, name: str, value: object) -> Item:
    """Put a rate on an item, or leave the attribute out when there is none.

    Out rather than null: the application accepts either, `attribute_exists` is
    a filter it can use on the first, and a null that means "not enough games"
    is a null every reader has to remember to handle.
    """
    converted = rate(value)
    if converted is not None:
        item[name] = converted
    return item


def now_iso(moment: datetime | None = None) -> str:
    """The publish timestamp, to the second, in UTC with a `Z`."""
    when = moment or datetime.now(UTC)
    return when.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------- builders --


def matchup_item(row: dict[str, Any], *, run_id: str, published_at: str) -> Item:
    """One ordered archetype pair, keyed `MATCHUP` / `<archetype>#<opponent>`."""
    archetype = str(row["archetype_key"])
    opponent = str(row["opponent_archetype_key"])
    item: Item = {
        KEY_PK: PK_MATCHUP,
        KEY_SK: f"{archetype}#{opponent}",
        "archetypeKey": archetype,
        "archetypeName": _text(row.get("archetype_name"), archetype),
        "opponentArchetypeKey": opponent,
        "opponentArchetypeName": _text(row.get("opponent_archetype_name"), opponent),
        "games": number(row["games"]),
        "wins": number(row["wins"]),
        "losses": number(row["losses"]),
        "ties": number(row["ties"]),
        "minGamesMet": bool(row["min_games_met"]),
        RUN_ID_ATTRIBUTE: run_id,
        "publishedAt": published_at,
    }
    return _with_rate(item, "winRate", row.get("win_rate"))


def weekly_item(row: dict[str, Any], *, run_id: str, published_at: str) -> Item:
    """One archetype in one ISO week, keyed `WEEKLY#<archetype>` / `<isoYear>-W<isoWeek>`.

    The week number is zero padded so the sort key sorts: without it week 9 of
    a year comes after week 10, and a `between` over a range of weeks silently
    returns the wrong set.
    """
    archetype = str(row["archetype_key"])
    iso_year = int(row["iso_year"])
    iso_week = int(row["iso_week"])
    item: Item = {
        KEY_PK: f"{WEEKLY_PREFIX}{archetype}",
        KEY_SK: f"{iso_year}-W{iso_week:02d}",
        "archetypeKey": archetype,
        "archetypeName": _text(row.get("archetype_name"), archetype),
        "isoYear": number(iso_year),
        "isoWeek": number(iso_week),
        "weekStart": str(row["week_start"]),
        "games": number(row["games"]),
        "wins": number(row["wins"]),
        "losses": number(row["losses"]),
        "ties": number(row["ties"]),
        RUN_ID_ATTRIBUTE: run_id,
        "publishedAt": published_at,
    }
    _with_rate(item, "winRate", row.get("win_rate"))
    return _with_rate(item, "shareOfWeek", row.get("share_of_week"))


def archetype_items(
    matchups: Sequence[dict[str, Any]], *, run_id: str, published_at: str
) -> list[Item]:
    """One row per archetype, summed over its matchup rows.

    Ties are summed but not published: the contract's archetype row carries
    wins, losses and the rate between them, and a tie is neither. They still
    have to be counted, because `games` includes them and a reader adding wins
    and losses back up would otherwise find the total short.
    """
    totals: dict[str, dict[str, Any]] = {}
    for row in matchups:
        key = str(row["archetype_key"])
        entry = totals.setdefault(
            key,
            {"name": _text(row.get("archetype_name"), key), "games": 0, "wins": 0, "losses": 0},
        )
        entry["games"] += int(row["games"])
        entry["wins"] += int(row["wins"])
        entry["losses"] += int(row["losses"])

    items: list[Item] = []
    for key in sorted(totals):
        entry = totals[key]
        decided = entry["wins"] + entry["losses"]
        item: Item = {
            KEY_PK: PK_ARCHETYPE,
            KEY_SK: key,
            "archetypeKey": key,
            "archetypeName": entry["name"],
            "games": number(entry["games"]),
            "wins": number(entry["wins"]),
            "losses": number(entry["losses"]),
            RUN_ID_ATTRIBUTE: run_id,
            "publishedAt": published_at,
        }
        items.append(_with_rate(item, "winRate", entry["wins"] / decided if decided else None))
    return items


def meta_item(
    *,
    run_id: str,
    published_at: str,
    games_total: int,
    matchup_rows: int,
    weekly_rows: int,
    archetype_rows: int,
    model: tuple[str, str] | None,
    source_commit: str | None,
    min_games: int | None,
) -> Item:
    """The one row that says what the last publish was, and what it was built from.

    The application reads this first: it is how a page can say "128 games, as of
    this morning" without counting anything, and how a stale publish is visible
    rather than inferred from rows that look fine individually.
    """
    item: Item = {
        KEY_PK: PK_META,
        KEY_SK: SK_META,
        RUN_ID_ATTRIBUTE: run_id,
        "publishedAt": published_at,
        "modelName": REGISTERED_MODEL_NAME,
        "gamesTotal": number(games_total),
        "matchupRows": number(matchup_rows),
        "weeklyRows": number(weekly_rows),
        "archetypeRows": number(archetype_rows),
    }
    if model is not None:
        item["modelVersion"], item["modelAlias"] = model
    if source_commit:
        item["sourceCommit"] = source_commit
    if min_games is not None:
        item["minGames"] = number(min_games)
    return item


def build_items(
    marts: Marts,
    *,
    run_id: str,
    published_at: str,
    model: tuple[str, str] | None = None,
    source_commit: str | None = None,
    min_games: int | None = None,
) -> tuple[list[Item], list[Item], list[Item], Item]:
    """Every item one publish writes: matchups, weeks, archetypes, and the meta row."""
    matchups = [
        matchup_item(row, run_id=run_id, published_at=published_at) for row in marts.matchups
    ]
    weekly = [weekly_item(row, run_id=run_id, published_at=published_at) for row in marts.weekly]
    archetypes = archetype_items(marts.matchups, run_id=run_id, published_at=published_at)
    meta = meta_item(
        run_id=run_id,
        published_at=published_at,
        games_total=marts.games_total,
        matchup_rows=len(matchups),
        weekly_rows=len(weekly),
        archetype_rows=len(archetypes),
        model=model,
        source_commit=source_commit,
        min_games=min_games,
    )
    return matchups, weekly, archetypes, meta


# ---------------------------------------------------------------- writing --


def _chunks(items: Sequence[Any], size: int = BATCH_SIZE) -> Iterator[list[Any]]:
    """The sequence in batches of at most `size`."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _send(table: "Table", requests: list[Any]) -> None:
    """One BatchWriteItem, resending whatever the table did not process.

    Unprocessed items are not an error: they are DynamoDB saying it throttled
    part of the batch and expects the rest back. Handing the list to
    `batch_write_item` again with a back-off is the documented answer, and it is
    the reason this is not a loop of `put_item` calls, which would pay a request
    per row and still have to handle the throttle.
    """
    pending: dict[str, Any] = {table.name: requests}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = table.meta.client.batch_write_item(RequestItems=pending)
        unprocessed = response.get("UnprocessedItems") or {}
        left = unprocessed.get(table.name) or []
        if not left:
            return
        logger.warning(
            "the table did not process every item, retrying",
            extra={"left": len(left), "attempt": attempt},
        )
        pending = {table.name: left}
        time.sleep(BACKOFF_BASE_S * attempt)
    raise PublishError(
        f"{table.name} left {len(pending[table.name])} item(s) unprocessed after "
        f"{MAX_ATTEMPTS} attempts"
    )


def write_items(table: "Table", items: Sequence[Item]) -> int:
    """Put every item, twenty-five at a time, and return how many were written."""
    for chunk in _chunks(items):
        _send(table, [{"PutRequest": {"Item": item}} for item in chunk])
    return len(items)


def delete_keys(table: "Table", keys: Sequence[dict[str, str]]) -> int:
    """Delete every key, twenty-five at a time, and return how many were deleted."""
    for chunk in _chunks(keys):
        _send(table, [{"DeleteRequest": {"Key": key}} for key in chunk])
    return len(keys)


def _query_partition(table: "Table", pk: str) -> Iterator[dict[str, Any]]:
    """Every item under one partition key, as `pk`, `sk` and `runId` only.

    Projected rather than whole: the sweep needs the key to delete and the run
    identifier to judge, and reading the rest of a matchup row to throw it away
    would triple the read units this costs.
    """
    condition = Key(KEY_PK).eq(pk)
    projection = f"{KEY_PK}, {KEY_SK}, {RUN_ID_ATTRIBUTE}"
    start: dict[str, Any] | None = None
    while True:
        response = (
            table.query(KeyConditionExpression=condition, ProjectionExpression=projection)
            if start is None
            else table.query(
                KeyConditionExpression=condition,
                ProjectionExpression=projection,
                ExclusiveStartKey=start,
            )
        )
        yield from response.get("Items", [])
        start = response.get("LastEvaluatedKey")
        if not start:
            return


def weekly_partitions(table: "Table") -> set[str]:
    """Every `WEEKLY#...` partition key the table currently holds.

    A Scan, and the module docstring says why it is one and what replaces it at
    scale. It projects the key alone, so the cost is the table's size in keys
    rather than in rows of data.
    """
    found: set[str] = set()
    start: dict[str, Any] | None = None
    condition = Attr(KEY_PK).begins_with(WEEKLY_PREFIX)
    while True:
        response = (
            table.scan(FilterExpression=condition, ProjectionExpression=KEY_PK)
            if start is None
            else table.scan(
                FilterExpression=condition,
                ProjectionExpression=KEY_PK,
                ExclusiveStartKey=start,
            )
        )
        found.update(str(item[KEY_PK]) for item in response.get("Items", []))
        start = response.get("LastEvaluatedKey")
        if not start:
            return found


def stale_keys(
    table: "Table", *, run_id: str, archetype_keys: Iterable[str]
) -> list[dict[str, str]]:
    """Every key in the four families whose `runId` is not this run's."""
    partitions = [PK_MATCHUP, PK_ARCHETYPE, PK_META]
    weekly = {f"{WEEKLY_PREFIX}{key}" for key in archetype_keys} | weekly_partitions(table)
    partitions += sorted(weekly)
    stale: list[dict[str, str]] = []
    for pk in partitions:
        for item in _query_partition(table, pk):
            if str(item.get(RUN_ID_ATTRIBUTE, "")) != run_id:
                stale.append({KEY_PK: str(item[KEY_PK]), KEY_SK: str(item[KEY_SK])})
    return stale


def default_table(name: str, region: str) -> "Table":
    """The table boto3's default credential chain can reach."""
    import boto3

    return boto3.resource("dynamodb", region_name=region).Table(name)


# --------------------------------------------------------------- the stage --


@dataclass
class PublishSummary:
    """What one publish did, per kind, plus what it swept up behind itself."""

    table: str
    run_id: str
    published_at: str
    rows_read: int = 0
    matchups: int = 0
    weekly: int = 0
    archetypes: int = 0
    meta: int = 0
    deleted: int = 0
    dry_run: bool = False
    #: One row per kind, shown by a dry run so a reader can see the shape.
    samples: list[Item] | None = None
    #: The meta row itself, shown by a dry run because there is only ever one of
    #: it and it is the row somebody checks before letting a publish go ahead.
    meta_sample: Item | None = None

    @property
    def written(self) -> int:
        """Items written, which is what the `rows_out` of the metrics row means."""
        return self.matchups + self.weekly + self.archetypes + self.meta

    def counts(self) -> dict[str, int]:
        """Items per partition-key family, for the summary and the metrics row."""
        return {
            "matchup": self.matchups,
            "weekly": self.weekly,
            "archetype": self.archetypes,
            "meta": self.meta,
        }

    def __str__(self) -> str:
        verb = "would write" if self.dry_run else "wrote"
        lines = [f"publish to {self.table} under run {self.run_id} at {self.published_at}", ""]
        lines.append("kind       items")
        for kind, count in self.counts().items():
            lines.append(f"{kind:<10} {count:>5}")
        lines.append("")
        lines.append(
            f"read {self.rows_read} mart row(s), {verb} {self.written} item(s), "
            + (
                "deleted nothing (dry run)"
                if self.dry_run
                else f"deleted {self.deleted} stale row(s)"
            )
        )
        if self.samples:
            lines.append("")
            lines.append("one sample item per kind:")
            lines += [json.dumps(sample, default=str, sort_keys=True) for sample in self.samples]
        if self.meta_sample is not None:
            lines.append("")
            lines.append("the meta row:")
            lines.append(json.dumps(self.meta_sample, default=str, sort_keys=True))
        return "\n".join(lines)


def run_publish(
    *,
    warehouse: Path,
    table_name: str,
    region: str,
    tracking_uri: str,
    run_id: str,
    dry_run: bool = False,
    table: "Table | None" = None,
    now: datetime | None = None,
    metrics: RunMetrics | None = None,
) -> PublishSummary:
    """Read the marts, build the items, write them, sweep the previous run away."""
    marts = read_marts(warehouse)
    published_at = now_iso(now)
    matchups, weekly, archetypes, meta = build_items(
        marts,
        run_id=run_id,
        published_at=published_at,
        model=production_model(tracking_uri),
        source_commit=git_commit(),
        min_games=dbt_min_games(),
    )
    summary = PublishSummary(
        table=table_name,
        run_id=run_id,
        published_at=published_at,
        rows_read=marts.rows,
        matchups=len(matchups),
        weekly=len(weekly),
        archetypes=len(archetypes),
        meta=1,
        dry_run=dry_run,
    )

    if dry_run:
        # Three items and nothing else: the marts carry archetype names and
        # counts and no player token reaches this stage, but a sample printed
        # to a terminal is the one output a person copies into a ticket, so it
        # stays one row per kind rather than a dump.
        summary.samples = [
            sample for sample in (matchups[:1] + weekly[:1] + archetypes[:1]) if sample
        ]
        summary.meta_sample = meta
    else:
        target = table if table is not None else default_table(table_name, region)
        write_items(target, matchups)
        write_items(target, weekly)
        write_items(target, archetypes)
        write_items(target, [meta])
        summary.deleted = delete_keys(
            target,
            stale_keys(
                target,
                run_id=run_id,
                archetype_keys=[str(row["archetype_key"]) for row in marts.matchups],
            ),
        )

    if metrics is not None:
        metrics.rows_in = summary.rows_read
        metrics.rows_out = 0 if dry_run else summary.written
        metrics.rows_quarantined = 0
        metrics.extra = {
            "table": table_name,
            "run_id": run_id,
            "published_at": published_at,
            "items": summary.counts(),
            "deleted_stale": summary.deleted,
            "games_total": marts.games_total,
            "dry_run": dry_run,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    """Publish the marts into the application's table from the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.publish",
        description=(
            "Write the matchup, weekly and archetype marts into the application's "
            "DynamoDB table, then delete the rows the previous run left behind."
        ),
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE_PATH,
        metavar="PATH",
        help="the DuckDB warehouse to read the marts from (default: the configured one)",
    )
    parser.add_argument(
        "--table",
        default=None,
        metavar="NAME",
        help="the DynamoDB table to write (default: $PRA_INSIGHTS_TABLE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every item and print the counts and three samples, writing nothing",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        metavar="URI",
        help="MLflow tracking URI, read for the production version (default: the configured one)",
    )
    args = parser.parse_args(argv)

    configure_logging(STAGE)
    try:
        # No bucket and no anonymization key: this stage reads a warehouse and
        # writes a table, and demanding either would make a publish depend on
        # configuration it never uses.
        settings = Settings.from_env(
            require_bucket=False, require_key=False, require_insights_table=args.table is None
        )
    except SettingsError as unset:
        parser.exit(2, f"{parser.prog}: {unset}\n")

    try:
        with stage_run(STAGE) as metrics:
            summary = run_publish(
                warehouse=args.warehouse,
                table_name=args.table or settings.insights_table,
                region=settings.region,
                tracking_uri=args.tracking_uri or default_tracking_uri(),
                run_id=current_run_id(),
                dry_run=args.dry_run,
                metrics=metrics,
            )
    except PublishError as failure:
        parser.exit(2, f"{parser.prog}: {failure}\n")

    emit_summary(
        logger,
        "publish summary",
        {
            "table": summary.table,
            "run_id": summary.run_id,
            "published_at": summary.published_at,
            "rows_read": summary.rows_read,
            "items": summary.counts(),
            "written": 0 if summary.dry_run else summary.written,
            "deleted_stale": summary.deleted,
            "dry_run": summary.dry_run,
        },
        text=str(summary),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
