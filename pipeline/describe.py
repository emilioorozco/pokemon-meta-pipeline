"""Reading one statement: what it names, and what to call it in a member's words.

Two jobs that are one job. The lower half is the lexical reading the SQL
validator has always done, `strip_literals` and `referenced_tables`, moved
here because it has a second caller now and because neither caller should own
it: `pipeline.agent.check_sql` asks which tables a statement reads, and
`describe_sql` below asks the same question of the same string. One regular
expression for both is the only way the two answers cannot drift apart.

The upper half is `describe_sql`, which is PLA-197. The receipt the
application draws under an answer used to show the statement and a copy
button, which puts `mart_archetype_weekly` and `min_games_met` in front of a
member who did not ask to learn the warehouse. So the panel shows one plain
line per lookup instead, and this is where that line comes from.

**Derived, never asked for.** The description is a pure function of the SQL.
The model is never asked to write it and has no way to put a word in it: a
model that could caption its own receipt could caption a query over the
player table as "a look at the archetype list", and the caption is exactly
the part a reader cannot check. Deriving it costs nothing, it cannot drift
from the statement it describes, and it is the same line every time for the
same query.

**How it is built.** Five pieces, every one of them optional and every one of
them out of a fixed vocabulary:

1. the relations, through `MART_PHRASES`, a phrase per allowlisted table;
2. the aggregate, when it is obvious: a rate column in the select list, or a
   `count`, `avg`, `sum`, `max` or `min` call;
3. the filters, as `for <column words> = <value>`, with the words from
   `COLUMN_WORDS` and the value lifted out of the statement's own literal;
4. a time window, when a date column is filtered;
5. the ordering and the limit, as "top 10 by win rate".

Anything it cannot read falls through to `UNKNOWN_LOOKUP` or to a shorter
line. A description that says less is always better than one that guesses.

**What is never in it.** No relation name, no column name, no SQL keyword in
upper case. That holds by construction rather than by scrubbing: every word
of the output but a filter's value comes from the two tables in this module,
and both are written in a member's words. It is checked anyway, over every
statement in `evals/transcript.yaml` and every statement the golden replay
writes, because the tables are hand written and a new phrase is one careless
"games" away from putting a column name back on the screen. That is also why
the counts here are "matches" and not "games": `games` is a real column of
five of the seven marts.

A refused statement is described the same way as one that ran. The
description says what the lookup was for and stops there; why there are no
rows is `refused_code`, which the application has its own phrasing for
(docs/agent-service.md).
"""

import re
from functools import lru_cache
from typing import Final

from pipeline.prompts import (
    ALLOWED_TABLES,
    COLUMN_SENTENCES,
    SCHEMA_FILES,
    first_sentences,
    read_models,
    warehouse_tables,
)

# The longest line the panel will take. A receipt is a list of one-liners and
# a line that wraps three times is a paragraph, which is the thing this field
# exists to replace.
MAX_DESCRIPTION_CHARS: Final = 160
# How much of a filter's value survives. Archetype names run to about forty
# characters and card names to about thirty, so this cuts a pasted paragraph
# and nothing a member would recognise.
MAX_VALUE_CHARS: Final = 48

# What to call each table the agent may read. One phrase per entry of
# `pipeline.prompts.ALLOWED_TABLES`, which a test asserts, because a table
# added to the allowlist with no phrase here would be a lookup the receipt
# cannot name.
#
# Written against the model descriptions in `dbt/models/marts/schema.yml` and
# then shortened until a member who has never seen the warehouse can read
# them. "Matchup results" rather than "archetype A against archetype B";
# "the calendar" rather than "one row per play date".
MART_PHRASES: Final[dict[str, str]] = {
    "mart_matchups": "matchup results",
    "mart_archetype_weekly": "how each deck did week by week",
    "mart_cards_seen": "which cards showed up",
    "mart_player_summary": "per-player summaries",
    "dim_archetype": "the deck list",
    "dim_card": "card details",
    "dim_date": "the calendar",
}

# What to call each column a description may name. Shorter than the one-liners
# the prompt generates, because these are read inside a sentence rather than
# as a definition, and deliberately free of the warehouse's own words: a count
# of seats is "matches", a flag called `min_games_met` is "enough matches to
# count".
#
# Not every column is here, and it does not have to be. A column with no
# phrase falls through to `column_words`, which tries the description the
# prompt already generated and then gives up, and a filter nothing can name is
# left out of the line rather than guessed at.
COLUMN_WORDS: Final[dict[str, str]] = {
    # who, and against whom
    "archetype_key": "the deck",
    "archetype_name": "the deck",
    "opponent_archetype_key": "the opposing deck",
    "opponent_archetype_name": "the opposing deck",
    "favourite_archetype_key": "the deck played most",
    "favourite_archetype_name": "the deck played most",
    "favourite_archetype_games": "matches with that deck",
    "is_mirror": "both sides on the same deck",
    "player_key": "the member",
    "aliases": "the other labels for a deck",
    "is_name_keyed": "a deck with no shared identifier",
    # which card
    "card_key": "the card",
    "card_name": "the card",
    "catalog_name": "the card",
    "catalog_type": "the kind of card",
    "catalog_set": "the set",
    "set_code": "the set",
    "catalog_hp": "printed hit points",
    "catalog_reg": "the regulation mark",
    "in_catalog": "known to the card catalogue",
    # how many, and how often
    "games": "matches",
    "games_played": "matches",
    "games_uploaded": "matches uploaded",
    "week_games": "matches that week",
    "games_with_card": "matches the card showed up in",
    "decklist_games": "matches with a full deck shared",
    "decklist_games_with_card": "matches with a shared deck holding the card",
    "wins": "matches won",
    "losses": "matches lost",
    "ties": "drawn matches",
    "undecided": "matches with no result",
    "win_rate": "win rate",
    "seen_rate": "how often a card was seen",
    "inclusion_rate": "how often a card was in a shared deck",
    "share_of_week": "share of the week",
    "avg_copies_seen": "copies seen on average",
    "max_copies_seen": "the most copies seen",
    "min_games_met": "enough matches to count",
    # when
    "week_start": "the week beginning",
    "play_date": "the day played",
    "date_key": "the day played",
    "first_seen": "first seen",
    "last_seen": "last seen",
    "first_played": "first played",
    "last_played": "last played",
}

# The columns a filter on is a time window rather than an equality, and the
# word the window opens with.
DATE_COLUMNS: Final[dict[str, str]] = {
    "week_start": "the week beginning",
    "play_date": "",
    "date_key": "",
    "first_seen": "first seen",
    "last_seen": "last seen",
    "first_played": "first played",
    "last_played": "last played",
}

# Date parts, which are filtered as numbers and read as dates. Kept apart from
# `COLUMN_WORDS` because the words for them are the value and not the column:
# `month = 9` is "in September", not "for the month = 9", and `year` and
# `month` are column names a description may not say out loud.
MONTH_NAMES: Final[tuple[str, ...]] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# The rates a select list carries when the question was about a rate, in the
# order a statement that selects two of them is read by.
RATE_HEADS: Final[dict[str, str]] = {
    "win_rate": "Win rate",
    "seen_rate": "How often cards were seen",
    "inclusion_rate": "How often cards were in a shared deck",
    "share_of_week": "Share of the week",
}

# What a statement with no recognisable relation in it is called. Every reason
# a relation is unrecognisable lands here: a name the model invented, a real
# table off the allowlist, and a statement with no FROM clause at all. The
# line says what the lookup was over and nothing about why it failed, because
# why is `refused_code` and the application has its own words for it.
UNKNOWN_LOOKUP: Final = "A lookup over data this tool cannot read"
PLAIN_HEAD: Final = "A lookup"

_LINE_COMMENT: Final = re.compile(r"--[^\n]*")
_BLOCK_COMMENT: Final = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL: Final = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER: Final = re.compile(r'"(?:[^"]|"")*"')
# Whatever follows FROM or JOIN: a bare name, a schema-qualified name, or an
# opening parenthesis for a subquery, which is not a table and is skipped.
_TABLE_REF: Final = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_.$]*)", re.IGNORECASE)
# Names a CTE introduces, which are legal table references even though they are
# not on the allowlist: `with recent as (...) select * from recent`.
_CTE_NAME: Final = re.compile(r"(?:\bwith\b|,)\s+([a-z_][a-z0-9_]*)\s+as\s*\(", re.IGNORECASE)
LIMIT_CLAUSE: Final = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)

_SELECT_LIST: Final = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
_WHERE_KEYWORD: Final = re.compile(r"\bwhere\b", re.IGNORECASE)
_WHERE_CLAUSE: Final = re.compile(
    r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_BY: Final = re.compile(r"\border\s+by\b(.*?)(?:\blimit\b|$)", re.IGNORECASE | re.DOTALL)
_AGGREGATE: Final = re.compile(
    r"\b(count|avg|sum|max|min)\s*\(\s*(distinct\s+)?([a-z_*][a-z0-9_]*)?", re.IGNORECASE
)
_COMPARISON: Final = re.compile(
    r"\b([a-z_][a-z0-9_]*)\s*(=|<>|!=|>=|<=|>|<|ilike|like)\s*"
    r"(?:date\s+|timestamp\s+)?('[^']*'|[-+]?\d+(?:\.\d+)?|true|false)",
    re.IGNORECASE,
)
_BARE_COLUMN: Final = re.compile(r"^([a-z_][a-z0-9_]*)(?:\s+(asc|desc))?$", re.IGNORECASE)
_WORD: Final = re.compile(r"[a-z_][a-z0-9_]*", re.IGNORECASE)

# The comparisons a filter phrase can carry, and the words for them. Equality
# and the two pattern matches read as "=" because that is what the member
# asked for; the prompt tells the model to match a name with `ILIKE` rather
# than `=`, and a receipt that reported the difference would be reporting the
# rule rather than the question.
_OPERATOR_WORDS: Final[dict[str, str]] = {
    "=": "=",
    "ilike": "=",
    "like": "=",
    "!=": "not",
    "<>": "not",
    ">=": "at least",
    "<=": "at most",
    ">": "above",
    "<": "below",
}
_LOWER_BOUNDS: Final[frozenset[str]] = frozenset({">=", ">"})
_UPPER_BOUNDS: Final[frozenset[str]] = frozenset({"<=", "<"})


# ------------------------------------------------- reading the statement --


def strip_literals(sql: str) -> str:
    """The statement with comments, string literals and quoted identifiers blanked.

    Every check in `pipeline.agent.check_sql` is a search for a keyword or a
    name, and all three of these are places a keyword can appear without being
    one: `-- drop this`, `where archetype_name = 'Drop Bear'`, and a column
    deliberately quoted as `"drop"`. Blanking them rather than removing them
    keeps the offsets, so a refusal can still be reasoned about against the
    original string, and so `describe_sql` can find a literal here and read
    its value out of the statement itself.
    """
    blanked = _BLOCK_COMMENT.sub(lambda match: " " * len(match.group(0)), sql)
    blanked = _LINE_COMMENT.sub(lambda match: " " * len(match.group(0)), blanked)
    blanked = _STRING_LITERAL.sub(_blank_inside, blanked)
    return _IDENTIFIER.sub(_blank_inside, blanked)


def _blank_inside(match: re.Match[str]) -> str:
    """A quoted run with its contents replaced by spaces and its quotes kept."""
    text = match.group(0)
    return text[0] + " " * (len(text) - 2) + text[-1]


def referenced_tables(sql: str) -> list[str]:
    """The tables a statement reads, in the order it names them, CTEs left out.

    Three callers need the same answer. `pipeline.agent.check_sql` asks whether
    every name is on the allowlist; `pipeline.eval`'s replay model asks whether
    the system prompt described them, since a model cannot query a table it was
    never told exists; `describe_sql` asks what to call them. Names the
    statement introduces itself with `WITH` are not tables and are dropped here
    rather than at each call site.
    """
    text = strip_literals(sql)
    ctes = {match.group(1).lower() for match in _CTE_NAME.finditer(text)}
    names: list[str] = []
    for match in _TABLE_REF.finditer(text):
        table = match.group(1).lower().split(".")[-1]
        if table not in ctes and table not in names:
            names.append(table)
    return names


# -------------------------------------------------- the words themselves --


@lru_cache(maxsize=1)
def schema_column_words() -> dict[str, str]:
    """The prompt's own column one-liners, for columns `COLUMN_WORDS` has not named.

    The schema listing in the system prompt is generated from
    `dbt/models/marts/schema.yml` so that a renamed column cannot go stale in
    two places, and the same file is the only description of a column this
    project has written down. So a column nobody has given short words to
    borrows the first sentence of its description rather than having its raw
    name spelled out, which is the one thing the line may not do.

    Borrowed under three conditions, because a description is written for a
    reader of the model and not for a receipt: it has to be short enough to
    sit inside a sentence, it may not have been cut short to get there, and it
    may not name a relation or a column. A description that fails any of them
    yields nothing, and the filter it would have described is left out of the
    line rather than guessed at.

    Empty when the dbt project is not beside the warehouse, which is the
    serving image: the hand written table above is the whole vocabulary there,
    and it is the one that covers every column the marts are filtered on.
    """
    forbidden = raw_names()
    words: dict[str, str] = {}
    for model in read_models(SCHEMA_FILES).values():
        columns = model.get("columns") or []
        if not isinstance(columns, list):
            continue
        for column in columns:
            name = str(column.get("name") or "")
            if not name or name in COLUMN_WORDS:
                continue
            sentence = first_sentences(
                str(column.get("description", "")), COLUMN_SENTENCES, MAX_VALUE_CHARS
            )
            phrase = sentence.rstrip(".").strip()
            if not phrase or phrase.endswith("...") or len(phrase) > 36:
                continue
            if any(word.group(0).lower() in forbidden for word in _WORD.finditer(phrase)):
                continue
            words[name] = phrase[0].lower() + phrase[1:]
    return words


def column_words(name: str) -> str:
    """A member's words for one column, or the empty string when there are none."""
    hand = COLUMN_WORDS.get(name.lower())
    if hand:
        return hand
    return schema_column_words().get(name.lower(), "")


@lru_cache(maxsize=1)
def raw_names() -> frozenset[str]:
    """Every relation and column name a description may not contain.

    The seven allowlisted tables, every column of them, and every relation dbt
    builds. This is what the no-raw-names rule is checked against, and it is
    also what keeps a borrowed description in `schema_column_words` from
    putting a column name back on the screen by the back door.
    """
    names = set(ALLOWED_TABLES) | set(warehouse_tables())
    for table, model in read_models(SCHEMA_FILES).items():
        if table not in ALLOWED_TABLES:
            continue
        columns = model.get("columns") or []
        if not isinstance(columns, list):
            continue
        names.update(str(column.get("name") or "").lower() for column in columns)
    names.discard("")
    return frozenset(names)


# ------------------------------------------------------- the description --


@lru_cache(maxsize=256)
def describe_sql(sql: str) -> str:
    """One statement as the line the receipt shows, in a member's words.

    Pure, offline and deterministic: the same statement always produces the
    same line, no model is asked anything, and nothing here opens a file but
    the cached read of the dbt schema the prompt already does. Cached because
    `QueryEvidence.description` is a property and a response body, a log line
    and an evaluation report each ask for it once.
    """
    blanked = strip_literals(sql)
    marts = [MART_PHRASES[table] for table in referenced_tables(sql) if table in MART_PHRASES]
    if not marts:
        return UNKNOWN_LOOKUP
    clauses = [f"{_head(blanked)} over {_join(marts)}"]
    filters, when = _conditions(sql, blanked)
    if filters:
        clauses.append(f"for {_join(filters)}")
    clauses.extend(when)
    ordering = _ordering(blanked)
    if ordering:
        clauses.append(ordering)
    return _cap(", ".join(clauses))


def _head(blanked: str) -> str:
    """What the statement computes, when that is obvious from the select list.

    A rate column first, because a statement that selects `win_rate` is about
    the win rate however many counts it carries beside it, and the rate is the
    thing a member asked for. Then the first aggregate call, named with the
    column it is over when that column has words. Then nothing, which is a
    plain lookup and most of the dimension queries.
    """
    found = _SELECT_LIST.search(blanked)
    if found is None:
        return PLAIN_HEAD
    selected = found.group(1)
    rates = [
        (match.start(), RATE_HEADS[match.group(0).lower()])
        for match in _WORD.finditer(selected)
        if match.group(0).lower() in RATE_HEADS
    ]
    if rates:
        return min(rates)[1]
    call = _AGGREGATE.search(selected)
    if call is None:
        return PLAIN_HEAD
    function = call.group(1).lower()
    over = column_words(call.group(3) or "")
    if function == "count":
        return "A count"
    if function == "avg":
        return f"An average of {over}" if over else "An average"
    if function == "sum":
        return f"A total of {over}" if over else "A total"
    if function == "max":
        return f"The highest {over}" if over else "The highest value"
    return f"The lowest {over}" if over else "The lowest value"


def _conditions(sql: str, blanked: str) -> tuple[list[str], list[str]]:
    """The filters the statement carries, and when it says the matches happened.

    Two lists because they are read differently. A filter is "for <words> =
    <value>" and several of them are one clause; a date is a clause of its
    own, "in September" or "for the week beginning 2026-09-14", because a
    reader takes the time window as a separate fact about the lookup.

    Read out of the first WHERE clause and only when the statement has exactly
    one, because a second one is a subquery or a CTE and this has no way to
    know which of the two a reader is looking at. An `OR` in the clause is the
    other thing that stops it: "for the deck = A and the deck = B" is a lie
    about a statement that asked for either, and a line that says less is
    better than one that says the wrong thing.
    """
    if len(_WHERE_KEYWORD.findall(blanked)) != 1:
        return [], []
    clause = _WHERE_CLAUSE.search(blanked)
    if clause is None or re.search(r"\bor\b", clause.group(1), re.IGNORECASE):
        return [], []
    start = clause.start(1)
    phrases: list[str] = []
    parts: list[str] = []
    dates: dict[str, dict[str, str]] = {}
    for match in _COMPARISON.finditer(clause.group(1)):
        column = match.group(1).lower()
        operator = match.group(2).lower()
        value = _value(sql, start + match.start(3), start + match.end(3))
        if not value:
            continue
        if column in DATE_COLUMNS:
            dates.setdefault(column, {})[operator] = value
            continue
        part = _part_of_date(column, value)
        if part:
            parts.append(part)
            continue
        words = column_words(column)
        if not words:
            continue
        phrases.append(f"{words} {_OPERATOR_WORDS.get(operator, '=')} {value}")
    window = _window(dates)
    return phrases, [*parts, window] if window else parts


def _window(dates: dict[str, dict[str, str]]) -> str:
    """The time window a date filter describes, as one clause or none.

    A column with a floor and a ceiling on it is a range and reads as one; a
    column with a single value on it is a day, or the Monday of a week when
    the column is the week's own start. Two different date columns filtered at
    once is a shape nobody has written, so the first of them is described and
    the rest are left to the rows.
    """
    for column, bounds in dates.items():
        low = next((bounds[sign] for sign in _LOWER_BOUNDS if sign in bounds), "")
        high = next((bounds[sign] for sign in _UPPER_BOUNDS if sign in bounds), "")
        if low and high:
            return f"from {low} to {high}"
        opener = DATE_COLUMNS[column]
        single = bounds.get("=") or low or high
        if not single:
            continue
        return f"for {opener} {single}" if opener else f"for {single}"
    return ""


def _part_of_date(column: str, value: str) -> str:
    """A filter on a part of the calendar, read as the part rather than the number.

    `month = 9` is "in September" and `iso_week = 38` is "in week 38". Written
    out here rather than in `COLUMN_WORDS` for a reason that is not only
    style: `year` and `month` are columns of the calendar as well as ordinary
    English, so a line that said "for the month = 9" would be a line with a
    column name in it.
    """
    if column in {"year", "iso_year"} and value.isdigit():
        return f"in {value}"
    if column == "month" and value.isdigit() and 1 <= int(value) <= 12:
        return f"in {MONTH_NAMES[int(value) - 1]}"
    if column == "iso_week" and value.isdigit():
        return f"in week {value}"
    if column in {"month_name", "day_name"}:
        return f"in {value}"
    return ""


def _value(sql: str, start: int, end: int) -> str:
    """One comparison's value, read out of the original statement.

    `strip_literals` keeps a literal's quotes and its offsets and blanks what
    is between them, so the span a match found in the blanked text is the span
    of the real value in the string the model wrote. Doubled quotes are SQL's
    own escape and come back as one, which is what makes `'Cynthia''s
    Garchomp'` read as a deck rather than as a typing accident.
    """
    raw = sql[start:end].strip()
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        raw = raw[1:-1].replace("''", "'")
    text = " ".join(raw.split())
    if text.lower() == "true":
        return "yes"
    if text.lower() == "false":
        return "no"
    if len(text) > MAX_VALUE_CHARS:
        return text[: MAX_VALUE_CHARS - 3].rstrip() + "..."
    return text


def _ordering(blanked: str) -> str:
    """The ordering and the limit, as the one clause a reader reads them as.

    "Top 10 by win rate" is the whole of what an ORDER BY and a LIMIT mean
    together, and it is the half of a statement a member is most likely to
    want confirmed: that the list in front of them is the top of something and
    how long it is. An ordering by an expression rather than by a column is
    left out, because the honest name for it is the expression.

    The limit here is the model's own. The tool adds one of its own when the
    model leaves it out (`pipeline.agent.with_limit`), after the statement has
    been recorded, so a line that says "at most 50 rows" is a line about a
    choice somebody made rather than about a default.
    """
    found = LIMIT_CLAUSE.search(blanked)
    limit = found.group(1) if found else ""
    clause = _ORDER_BY.search(blanked)
    if clause is None:
        return f"at most {limit} rows" if limit else ""
    first = clause.group(1).split(",")[0].strip()
    bare = _BARE_COLUMN.match(first)
    if bare is None:
        return f"at most {limit} rows" if limit else ""
    words = column_words(bare.group(1))
    if not words:
        return f"at most {limit} rows" if limit else ""
    descending = (bare.group(2) or "").lower() == "desc"
    if limit:
        return f"top {limit} by {words}" if descending else f"the lowest {limit} by {words}"
    return f"ordered by {words}, {'highest' if descending else 'lowest'} first"


def _join(parts: list[str]) -> str:
    """A short list as a reader says it: one, "a and b", or "a, b and c"."""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _cap(text: str) -> str:
    """The line, cut to `MAX_DESCRIPTION_CHARS` at a word boundary if it has to be."""
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    return text[: MAX_DESCRIPTION_CHARS - 3].rsplit(" ", 1)[0].rstrip(",;:") + "..."
